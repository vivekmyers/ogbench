from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field, randomize_bc_goals_from_batch
from utils.networks import (
    GCActor,
    Param,
    StateRepresentation,
    actor_action_stack_kwargs_from_batch,
    actor_action_stack_kwargs_from_value,
    build_gc_actor_init,
)


class TMDDQCAgent(flax.struct.PyTreeNode):
    """TMD with DQC-style decoupled chunk/action critics.

    Chunk-level critic (phi + psi) trained with TMD's contrastive + backup +
    invariance losses on full backup_horizon-length action chunks.  Action-level
    critic (action_phi + shared psi) distilled from chunk distances via
    expectile/quantile regression.  Gaussian actor trained with Q-weighted BC
    scored by the action critic.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    # ------------------------------------------------------------------ #
    #  Distance functions (identical to TMD)                               #
    # ------------------------------------------------------------------ #

    @jax.jit
    def mrn_distance(self, x, y):
        K = self.config['components']
        assert x.shape[-1] % K == 0

        @jax.jit
        def mrn_distance_component(x, y):
            eps = 1e-6
            d = x.shape[-1]
            mask = jnp.arange(d) < d // 2
            max_component = jnp.max(jax.nn.relu((x - y) * mask), axis=-1)
            l2_component = jnp.sqrt(jnp.square((x - y) * (1 - mask)).sum(axis=-1) + eps)
            assert max_component.shape == l2_component.shape
            return max_component + l2_component

        x_split = jnp.stack(jnp.split(x, K, axis=-1), axis=-1)
        y_split = jnp.stack(jnp.split(y, K, axis=-1), axis=-1)
        dists = jax.vmap(mrn_distance_component, in_axes=(-1, -1), out_axes=-1)(
            x_split, y_split
        )
        return dists.mean(axis=-1)

    def iqe_distance(self, x, y):
        k = self.config['components']
        alpha_raw = self.network.select('alpha_raw')()
        alpha = jax.nn.sigmoid(alpha_raw)
        reshape = (x.shape[-1] // k, k)
        x = jnp.reshape(x, (*x.shape[:-1], *reshape))
        y = jnp.reshape(y, (*y.shape[:-1], *reshape))
        valid = x < y
        D = x.shape[-1]
        xy = jnp.concatenate(jnp.broadcast_arrays(x, y), axis=-1)
        ixy = xy.argsort(axis=-1)
        sxy = jnp.take_along_axis(xy, ixy, axis=-1)
        neg_inc_copies = jnp.take_along_axis(valid, ixy % D, axis=-1) * jnp.where(
            ixy < D, -1, 1
        )
        neg_inp_copies = jnp.cumsum(neg_inc_copies, axis=-1)
        neg_f = (neg_inp_copies < 0) * (-1.0)
        neg_incf = jnp.concatenate(
            [neg_f[..., :1], neg_f[..., 1:] - neg_f[..., :-1]], axis=-1
        )
        components = (sxy * neg_incf).sum(-1)
        result = alpha * components.mean(axis=-1) + (1 - alpha) * components.max(
            axis=-1
        )
        return result

    @jax.jit
    def distance(self, x, y):
        x, y = jnp.broadcast_arrays(x, y)
        if self.config['use_iqe']:
            return self.iqe_distance(x, y)
        else:
            return self.mrn_distance(x, y)

    # ------------------------------------------------------------------ #
    #  Losses                                                              #
    # ------------------------------------------------------------------ #

    @jax.jit
    def chunk_critic_loss(self, batch, grad_params):
        """TMD contrastive + backup + invariance on H-step action chunks.

        Per-sample mask: ``high_value_action_chunks`` is always ``H`` steps
        long, but ``high_value_backup_horizon`` (``bh``) is ``min(H, goal-idx)``.
        When ``bh < H`` (goal reachable in fewer than H steps), the trailing
        action slots are terminal-padded duplicates of the goal-reaching
        action, so phi(s, a_{0:H}) is being asked to encode actions that were
        never actually executed before reaching g. We mask those rows out of
        the contrastive / backup / invariance losses so phi only learns from
        chunks that were fully consistent with the bootstrap target.
        """
        batch_size = batch['observations'].shape[0]
        H = int(self.config['backup_horizon'])

        phi = self.network.select('phi')(
            batch['observations'], batch['high_value_action_chunks'], params=grad_params
        )
        psi_s = self.network.select('psi')(batch['observations'], params=grad_params)
        psi_next = self.network.select('psi')(
            batch['high_value_next_observations'], params=grad_params
        )
        psi_g = self.network.select('psi')(
            batch['high_value_goals'], params=grad_params
        )

        if len(phi.shape) == 2:
            phi = phi[None, ...]
            psi_s = psi_s[None, ...]
            psi_next = psi_next[None, ...]
            psi_g = psi_g[None, ...]

        # 1.0 if the action chunk for sample b ran the full H steps before the
        # goal/terminal, 0.0 otherwise. Shape (B,).
        valid_sample = (batch['high_value_backup_horizon'] >= H).astype(jnp.float32)
        valid_count = jnp.maximum(valid_sample.sum(), 1.0)
        e = phi.shape[0]

        # --- Contrastive ---
        dist = self.distance(phi[:, :, None], psi_g[:, None, :])
        logits = -dist / jnp.sqrt(phi.shape[-1])

        I = jnp.eye(batch_size)
        contrastive_per = jax.vmap(
            lambda _logits: optax.softmax_cross_entropy(logits=_logits.T, labels=I),
        )(logits)  # (e, B); each entry is the CE for goal/sample b.
        contrastive_loss = jnp.sum(contrastive_per * valid_sample[None, :]) / (valid_count * e)

        # --- Action invariance --- (psi(s) ~ phi(s, a) — only meaningful when
        # the chunk represents real executed actions.)
        if self.config['stopgrad_phi_invariance']:
            action_dist = self.distance(psi_s, jax.lax.stop_gradient(phi))
        else:
            action_dist = self.distance(psi_s, phi)
        action_invariance_loss = jnp.sum(action_dist * valid_sample[None, :]) / (valid_count * e)

        # --- Backup (variable-horizon discount γ^bh per sample) ---
        dist_next = self.distance(psi_next[:, :, None], psi_g[:, None, :])

        t = self.config['t']
        gamma = self.config['discount'] ** batch['high_value_backup_horizon']
        gamma = gamma[None, :, None]

        if self.config['stopgrad_psi_backup']:
            dist = self.distance(
                phi[:, :, None], jax.lax.stop_gradient(psi_g[:, None, :])
            )
        dist_next = jax.lax.stop_gradient(dist_next)

        delta = dist - dist_next
        delta_mask = delta > t
        delta_clipped = jnp.where(delta_mask, t, delta)
        divergence = jnp.where(delta_mask, delta, gamma * jnp.exp(delta_clipped) - dist)

        dw = self.config['diag_backup']
        divergence = (
            divergence * (1 - dw)
            + jnp.diagonal(divergence, axis1=1, axis2=2)[..., None] * dw
        )
        # Backup row index (axis 1) is the source sample b: mask invalid b's.
        backup_mask = valid_sample[None, :, None]
        backup_denom = valid_count * e * batch_size
        backup_loss = jnp.sum(divergence * backup_mask) / backup_denom

        if self.config['dual_descent']:
            optim_backup = 1 - jax.lax.stop_gradient(dist_next) + jnp.log(gamma)
            optim_backup = (
                optim_backup * (1 - dw)
                + jnp.diagonal(optim_backup, axis1=1, axis2=2)[..., None] * dw
            )
            backup_optim_loss = jnp.sum((divergence - optim_backup) * backup_mask) / backup_denom
            val = jnp.exp(
                -(
                    jax.lax.stop_gradient(backup_optim_loss)
                    + jax.lax.stop_gradient(action_invariance_loss)
                )
            )
            critic_loss = (
                val * contrastive_loss + backup_loss + action_invariance_loss
            )
        else:
            critic_loss = (
                contrastive_loss
                + self.config['zeta'] * action_invariance_loss
                + self.config['zeta'] * backup_loss
            )

        logits_mean = jnp.mean(logits, axis=0)
        correct = jnp.argmax(logits_mean, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits_mean * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits_mean * (1 - I)) / jnp.sum(1 - I)

        return critic_loss, {
            'contrastive_loss': contrastive_loss,
            'action_invariance_loss': action_invariance_loss,
            'backup_loss': backup_loss,
            'critic_loss': critic_loss,
            'binary_accuracy': jnp.mean((logits_mean > 0) == I),
            'categorical_accuracy': jnp.mean(correct),
            'logits_pos': logits_pos,
            'logits_neg': logits_neg,
            'logits': logits_mean.mean(),
            'dist': dist.mean(),
            'biggest_diff_in_dist': jnp.max(dist - dist_next),
            'valid_sample_frac': valid_sample.mean(),
        }

    @jax.jit
    def action_critic_loss(self, batch, grad_params):
        """Distill chunk-level distances into action-level critic.

        Gradient flows only through action_phi; phi and psi are frozen here so
        they are trained exclusively by the chunk critic TMD losses.

        Distillation is per-ensemble-member: action_phi[e] regresses onto
        chunk_dist[e] (NOT the mean over members). Mean-pooling collapses the
        action_phi ensemble to identical predictions, making the ``min(q1, q2)``
        in ``actor_loss`` a no-op.
        """
        action_dim = batch['actions'].shape[-1]
        ac_dim = int(self.config['policy_chunk_size']) * action_dim

        # Chunk distances (frozen target, per-ensemble-member).
        phi_chunk = self.network.select('phi')(
            batch['observations'], batch['high_value_action_chunks']
        )
        psi_g_frozen = self.network.select('psi')(batch['high_value_goals'])

        if len(phi_chunk.shape) == 2:
            phi_chunk = phi_chunk[None, ...]
            psi_g_frozen = psi_g_frozen[None, ...]

        chunk_dist = self.distance(phi_chunk, psi_g_frozen)  # (e, B)
        chunk_dist = jax.lax.stop_gradient(chunk_dist)

        # Action distances (gradient through action_phi only, per-member).
        action_chunk = batch['high_value_action_chunks'][..., :ac_dim]
        action_phi_out = self.network.select('action_phi')(
            batch['observations'], action_chunk, params=grad_params
        )
        psi_g_distill = self.network.select('psi')(batch['high_value_goals'])

        if len(action_phi_out.shape) == 2:
            action_phi_out = action_phi_out[None, ...]
            psi_g_distill = psi_g_distill[None, ...]

        action_dist = self.distance(action_phi_out, psi_g_distill)  # (e, B)

        # Asymmetric regression (lower distance = better).
        kappa = self.config['kappa_d']
        weight = jnp.where(chunk_dist <= action_dist, kappa, 1 - kappa)

        if self.config['distill_method'] == 'expectile':
            distill_loss = (weight * jnp.square(action_dist - chunk_dist)).mean()
        else:
            distill_loss = (weight * jnp.abs(action_dist - chunk_dist)).mean()

        return distill_loss, {
            'distill_loss': distill_loss,
            'chunk_dist_mean': chunk_dist.mean(),
            'action_dist_mean': action_dist.mean(),
            'dist_gap': (action_dist - chunk_dist).mean(),
        }

    @jax.jit
    def actor_loss(self, batch, grad_params, rng=None):
        """Gaussian actor with Q-weighted BC scored by action critic."""
        chunk_len = int(self.config['policy_chunk_size'])
        action_dim = batch['actions'].shape[-1]

        rng = rng if rng is not None else self.rng
        p_bc = float(self.config.get('bc_goal_randomize_prob', 0.0))
        if p_bc > 0.0:
            rng, rng_mix = jax.random.split(rng)
            bc_goals = randomize_bc_goals_from_batch(batch['actor_goals'], rng_mix, p_bc)
        else:
            bc_goals = batch['actor_goals']

        akw = actor_action_stack_kwargs_from_batch(self.config, batch)
        if self.config['use_latent']:
            psi_s = self.network.select('psi')(
                batch['observations'], params=grad_params
            )
            psi_g = self.network.select('psi')(
                batch['actor_goals'], params=grad_params
            )
            if p_bc > 0.0:
                psi_g_bc = self.network.select('psi')(bc_goals, params=grad_params)
            if len(psi_s.shape) == 3:
                psi_s = jnp.mean(psi_s, axis=0)
                psi_g = jnp.mean(psi_g, axis=0)
                if p_bc > 0.0:
                    psi_g_bc = jnp.mean(psi_g_bc, axis=0)
            if self.config['freeze_enc_for_actor_grad']:
                psi_s = jax.lax.stop_gradient(psi_s)
                psi_g = jax.lax.stop_gradient(psi_g)
                if p_bc > 0.0:
                    psi_g_bc = jax.lax.stop_gradient(psi_g_bc)
            dist_out = self.network.select('actor')(
                psi_s, psi_g, params=grad_params, **akw
            )
            if p_bc > 0.0:
                dist_bc = self.network.select('actor')(
                    jax.lax.stop_gradient(psi_s),
                    jax.lax.stop_gradient(psi_g_bc),
                    params=grad_params,
                    **akw,
                )
            else:
                dist_bc = dist_out
        else:
            dist_out = self.network.select('actor')(
                batch['observations'], batch['actor_goals'], params=grad_params, **akw
            )
            if p_bc > 0.0:
                dist_bc = self.network.select('actor')(
                    batch['observations'], bc_goals, params=grad_params, **akw
                )
            else:
                dist_bc = dist_out

        if self.config['const_std']:
            q_flat = jnp.clip(dist_out.mode(), -1, 1)
        else:
            q_flat = jnp.clip(dist_out.sample(seed=rng), -1, 1)

        # Score with action_phi (frozen params — gradient only through actor)
        action_phi_out = self.network.select('action_phi')(
            batch['observations'], q_flat
        )
        psi_g_frozen = self.network.select('psi')(batch['actor_goals'])

        if len(action_phi_out.shape) == 2:
            action_phi_out = action_phi_out[None, ...]
            psi_g_frozen = psi_g_frozen[None, ...]

        q_dists = -self.distance(action_phi_out, psi_g_frozen)  # (e, B)
        q = (
            jnp.minimum(q_dists[0], q_dists[1])
            if q_dists.shape[0] >= 2
            else q_dists[0]
        )

        q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)

        action_targets = batch['action_chunks'].reshape(
            batch['action_chunks'].shape[0], -1
        )
        log_prob = dist_bc.log_prob(action_targets)
        bc_loss = -(self.config['alpha'] * log_prob).mean()

        actor_loss_val = q_loss + bc_loss

        pred = dist_out.mode()
        actor_info = {
            'actor_loss': actor_loss_val,
            'q_loss': q_loss,
            'bc_loss': bc_loss,
            'q_mean': q.mean(),
            'q_abs_mean': jnp.abs(q).mean(),
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((pred - action_targets) ** 2),
            'std': jnp.mean(dist_out.scale_diag),
        }
        if chunk_len > 1:
            actor_info['mse_first'] = jnp.mean(
                (pred[:, :action_dim] - action_targets[:, :action_dim]) ** 2
            )

        return actor_loss_val, actor_info

    # ------------------------------------------------------------------ #
    #  Total loss / update                                                 #
    # ------------------------------------------------------------------ #

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng

        chunk_critic_loss_val, chunk_info = self.chunk_critic_loss(
            batch, grad_params
        )
        for k, v in chunk_info.items():
            info[f'chunk_critic/{k}'] = v

        action_critic_loss_val, action_info = self.action_critic_loss(
            batch, grad_params
        )
        for k, v in action_info.items():
            info[f'action_critic/{k}'] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss_val, actor_info = self.actor_loss(
            batch, grad_params, actor_rng
        )
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = chunk_critic_loss_val + action_critic_loss_val + actor_loss_val
        return loss, info

    @jax.jit
    def update(self, batch, step: int = 0):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(network=new_network, rng=new_rng), info

    # ------------------------------------------------------------------ #
    #  Inference                                                           #
    # ------------------------------------------------------------------ #

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        action_stack=None,
        temperature=1.0,
    ):
        """Returns the full ``policy_chunk_size``-step action chunk reshaped to
        (..., chunk_len, action_dim). The previous implementation discarded
        every action after the first, which silently nullified Q-chunking for
        any caller using this helper. ``server.py`` always reshaped the flat
        actor output itself, so eval was unaffected, but downstream code that
        relied on this method was getting a truncated chunk.
        """
        chunk_len = int(self.config['policy_chunk_size'])
        akw = actor_action_stack_kwargs_from_value(self.config, action_stack)
        if self.config['use_latent']:
            psi_s = self.network.select('psi')(observations)
            psi_g = self.network.select('psi')(goals)
            if len(psi_s.shape) == 2:
                psi_s = jnp.mean(psi_s, axis=0)
                psi_g = jnp.mean(psi_g, axis=0)
            dist_out = self.network.select('actor')(
                psi_s, psi_g, temperature=temperature, **akw
            )
        else:
            dist_out = self.network.select('actor')(
                observations, goals, temperature=temperature, **akw
            )
        actions = dist_out.sample(seed=seed)
        actions = jnp.clip(actions, -1, 1)
        if chunk_len > 1:
            flat_dim = actions.shape[-1]
            ad = flat_dim // chunk_len
            actions = actions.reshape(*actions.shape[:-1], chunk_len, ad)
        return actions

    @jax.jit
    def get_distance(self, observations, goals, actions):
        if self.config['use_action_for_distance']:
            phi = self.network.select('action_phi')(observations, actions)
        else:
            phi = self.network.select('psi')(observations)
        psi = self.network.select('psi')(goals)
        dist = self.distance(phi, psi)
        return dist

    # ------------------------------------------------------------------ #
    #  Construction                                                        #
    # ------------------------------------------------------------------ #

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
        steps=None,
        ex_action_stack=None,
    ):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        action_dim = ex_actions.shape[-1]
        backup_horizon = int(config['backup_horizon'])
        policy_chunk_size = int(config['policy_chunk_size'])

        ex_chunk_H = jnp.zeros((ex_observations.shape[0], backup_horizon * action_dim))
        ex_chunk_K = jnp.zeros((ex_observations.shape[0], policy_chunk_size * action_dim))

        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['actor'] = GCEncoder(concat_encoder=encoder_module())
            encoders['state'] = encoder_module()

        phi_def = StateRepresentation(
            hidden_dims=config['value_hidden_dims'],
            latent_dim=config['latent_dim'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            value_exp=True,
            state_encoder=encoders.get('state'),
        )
        psi_def = StateRepresentation(
            hidden_dims=config['value_hidden_dims'],
            latent_dim=config['latent_dim'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            value_exp=True,
            state_encoder=encoders.get('state'),
        )
        action_phi_def = StateRepresentation(
            hidden_dims=config['value_hidden_dims'],
            latent_dim=config['latent_dim'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            value_exp=True,
            state_encoder=encoders.get('state'),
        )
        actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            chunk_length=policy_chunk_size,
            state_dependent_std=False,
            const_std=config['const_std'],
            gc_encoder=encoders.get('actor'),
        )

        if config['use_latent']:
            embed = jnp.zeros((1, config['latent_dim']))
            actor_init_args = build_gc_actor_init(embed, embed, ex_action_stack)
        else:
            actor_init_args = build_gc_actor_init(ex_observations, ex_goals, ex_action_stack)

        network_info = dict(
            phi=(phi_def, (ex_observations, ex_chunk_H)),
            psi=(psi_def, (ex_goals,)),
            action_phi=(action_phi_def, (ex_observations, ex_chunk_K)),
            actor=(actor_def, actor_init_args),
        )

        if config['use_iqe']:
            network_info['alpha_raw'] = (Param(), ())

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent.
            agent_name='tmd_dqc',
            lr=3e-4,
            components=8,
            batch_size=512,
            actor_hidden_dims=(512, 512, 512),
            value_hidden_dims=(512, 512, 512),
            latent_dim=512,
            layer_norm=True,
            discount=0.99,
            alpha=0.1,
            zeta=0.05,
            t=3.0,
            diag_backup=0.5,
            stopgrad_psi_backup=False,
            stopgrad_phi_invariance=False,
            encoder=ml_collections.config_dict.placeholder(str),
            const_std=True,
            # Dataset.
            dataset_class='CGCDataset',
            value_p_curgoal=0.0,
            value_p_trajgoal=1.0,
            value_p_randomgoal=0.0,
            value_geom_sample=True,
            actor_p_curgoal=0.0,
            actor_p_trajgoal=1.0,
            actor_p_randomgoal=0.0,
            actor_geom_sample=False,
            bc_goal_randomize_prob=0.0,
            gc_negative=False,
            p_aug=0.0,
            # TMD options.
            use_iqe=False,
            use_latent=False,
            freeze_enc_for_actor_grad=False,
            use_action_for_distance=True,
            frame_stack=ml_collections.config_dict.placeholder(int),
            dual_descent=False,
            # DQC decoupling.
            backup_horizon=25,
            policy_chunk_size=1,
            action_chunk_length=1,  # Must equal policy_chunk_size; used by CGCDataset.
            distill_method='expectile',
            kappa_d=0.5,
        )
    )
    return config


# Alias so `train.py --algorithm TMD_DQC` resolves to `agents.tmd_dqc.TMD_DQCAgent`.
TMD_DQCAgent = TMDDQCAgent
