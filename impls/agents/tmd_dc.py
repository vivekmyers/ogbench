"""TMD with a double critic.

Critic 1 (``phi``, ``psi``): identical to ``agents.tmd.TMDAgent`` — contrastive
goal-reaching loss + invariance loss + LINEX backup (+ optional dual-descent).

Critic 2 (``phi_am``, ``psi_am``): a *separate* action-match critic.  It grades
how well an action fits the current state's representation by training
``phi_am(s, a)`` to be close to ``psi_am(s)`` (positive) and far from
``psi_am(s')`` for other states ``s'`` in the batch (negatives), using InfoNCE
on the same MRN/IQE distance machinery.

Together, in the actor, the second critic acts as an action-consistency bonus:

    q = -D(phi(s, a), psi(g))  +  q_match_weight * ( -D(phi_am(s, a), psi_am(s)) )

None of the existing TMD/QC/DQC agent modules are modified.
"""

from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field, randomize_bc_goals_from_batch
from utils.networks import (
    DiscreteStateActionRepresentation,
    GCActor,
    GCDiscreteActor,
    Param,
    StateRepresentation,
    actor_action_stack_kwargs_from_batch,
    actor_action_stack_kwargs_from_value,
    build_gc_actor_init,
)


class TMDDCAgent(flax.struct.PyTreeNode):
    """TMD with a goal-reaching critic and a separate action-match critic."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    # ------------------------------------------------------------------ #
    #  Distance primitives (identical to TMDAgent)                        #
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
        dists = jax.vmap(mrn_distance_component, in_axes=(-1, -1), out_axes=-1)(x_split, y_split)
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
        neg_inc_copies = jnp.take_along_axis(valid, ixy % D, axis=-1) * jnp.where(ixy < D, -1, 1)
        neg_inp_copies = jnp.cumsum(neg_inc_copies, axis=-1)
        neg_f = (neg_inp_copies < 0) * (-1.0)
        neg_incf = jnp.concatenate([neg_f[..., :1], neg_f[..., 1:] - neg_f[..., :-1]], axis=-1)
        components = (sxy * neg_incf).sum(-1)
        result = alpha * components.mean(axis=-1) + (1 - alpha) * components.max(axis=-1)
        return result

    @jax.jit
    def distance(self, x, y):
        x, y = jnp.broadcast_arrays(x, y)
        if self.config['use_iqe']:
            return self.iqe_distance(x, y)
        else:
            return self.mrn_distance(x, y)

    # ------------------------------------------------------------------ #
    #  Critic 1: goal-reaching TMD loss (unchanged from agents/tmd.py)    #
    # ------------------------------------------------------------------ #

    @jax.jit
    def critic_loss(self, batch, grad_params):
        batch_size = batch['observations'].shape[0]
        phi = self.network.select('phi')(batch['observations'], batch['actions'], params=grad_params)
        psi_s = self.network.select('psi')(batch['observations'], params=grad_params)
        psi_next = self.network.select('psi')(batch['next_observations'], params=grad_params)
        psi_g = self.network.select('psi')(batch['value_goals'], params=grad_params)

        if len(phi.shape) == 2:
            phi = phi[None, ...]
            psi_s = psi_s[None, ...]
            psi_next = psi_next[None, ...]
            psi_g = psi_g[None, ...]

        dist = self.distance(phi[:, :, None], psi_g[:, None, :])
        logits = -dist / jnp.sqrt(phi.shape[-1])

        I = jnp.eye(batch_size)
        contrastive_loss = jax.vmap(
            lambda _logits: optax.softmax_cross_entropy(logits=_logits.T, labels=I),
        )(logits)
        contrastive_loss = jnp.mean(contrastive_loss)

        if self.config['stopgrad_phi_invariance']:
            action_dist = self.distance(psi_s, jax.lax.stop_gradient(phi))
        else:
            action_dist = self.distance(psi_s, phi)
        action_invariance_loss = jnp.mean(action_dist)

        dist_next = self.distance(psi_next[:, :, None], psi_g[:, None, :])

        t = self.config['t']
        gamma = self.config['discount']
        if self.config['stopgrad_psi_backup']:
            dist = self.distance(phi[:, :, None], jax.lax.stop_gradient(psi_g[:, None, :]))
        dist_next = jax.lax.stop_gradient(dist_next)

        delta = dist - dist_next
        mask = delta > t
        delta_clipped = jnp.where(mask, t, delta)
        divergence = jnp.where(mask, delta, gamma * jnp.exp(delta_clipped) - dist)

        dw = self.config['diag_backup']
        divergence = divergence * (1 - dw) + jnp.diagonal(divergence, axis1=1, axis2=2)[..., None] * dw
        backup_loss = jnp.mean(divergence)

        if self.config['dual_descent']:
            optim_backup = 1 - jax.lax.stop_gradient(dist_next) + jnp.log(gamma)
            optim_backup = optim_backup * (1 - dw) + jnp.diagonal(optim_backup, axis1=1, axis2=2)[..., None] * dw
            backup_optim_loss = jnp.mean(divergence - optim_backup)
            val = jnp.exp(-(jax.lax.stop_gradient(backup_optim_loss) + jax.lax.stop_gradient(action_invariance_loss)))
            critic_loss_val = val * contrastive_loss + backup_loss + action_invariance_loss
        else:
            critic_loss_val = (
                contrastive_loss
                + self.config['zeta'] * action_invariance_loss
                + self.config['zeta'] * backup_loss
            )

        logits_mean = jnp.mean(logits, axis=0)
        correct = jnp.argmax(logits_mean, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits_mean * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits_mean * (1 - I)) / jnp.sum(1 - I)

        return (
            (contrastive_loss, backup_loss, action_invariance_loss),
            critic_loss_val,
            {
                'contrastive_loss': contrastive_loss,
                'action_invariance_loss': action_invariance_loss,
                'backup_loss': backup_loss,
                'critic_loss': critic_loss_val,
                'binary_accuracy': jnp.mean((logits_mean > 0) == I),
                'categorical_accuracy': jnp.mean(correct),
                'logits_pos': logits_pos,
                'logits_neg': logits_neg,
                'logits': logits_mean.mean(),
                'dist': dist.mean(),
                'biggest_diff_in_dist': jnp.max(dist - dist_next),
            },
        )

    # ------------------------------------------------------------------ #
    #  Critic 2: action-match contrastive loss                            #
    # ------------------------------------------------------------------ #
    #
    # For a batch of (s_i, a_i):
    #   phi_am[i] = phi_am(s_i, a_i)          # "what action was taken from s_i"
    #   psi_am[j] = psi_am(s_j)               # state-only key
    # Positive pair: (i, i) — the action was actually taken from state s_i.
    # Negative pairs: (i, j), j != i — action a_i was not taken from state s_j.
    # We minimise InfoNCE over logits L[i, j] = -D(phi_am[i], psi_am[j]).
    # This gives a score that is high when an action "matches" a state.

    @jax.jit
    def action_match_loss(self, batch, grad_params):
        batch_size = batch['observations'].shape[0]
        phi_am = self.network.select('phi_am')(
            batch['observations'], batch['actions'], params=grad_params
        )
        psi_am = self.network.select('psi_am')(batch['observations'], params=grad_params)

        if len(phi_am.shape) == 2:
            phi_am = phi_am[None, ...]
            psi_am = psi_am[None, ...]

        dist = self.distance(phi_am[:, :, None], psi_am[:, None, :])
        logits = -dist / jnp.sqrt(phi_am.shape[-1])

        I = jnp.eye(batch_size)
        am_contrastive = jax.vmap(
            lambda _logits: optax.softmax_cross_entropy(logits=_logits.T, labels=I),
        )(logits)
        am_contrastive = jnp.mean(am_contrastive)

        logits_mean = jnp.mean(logits, axis=0)
        correct = jnp.argmax(logits_mean, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits_mean * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits_mean * (1 - I)) / jnp.sum(1 - I)

        return am_contrastive, {
            'am_contrastive_loss': am_contrastive,
            'am_categorical_accuracy': jnp.mean(correct),
            'am_logits_pos': logits_pos,
            'am_logits_neg': logits_neg,
            'am_dist': dist.mean(),
        }

    # ------------------------------------------------------------------ #
    #  Actor: DDPG+BC with combined goal-reaching and action-match Q      #
    # ------------------------------------------------------------------ #

    @jax.jit
    def actor_loss(self, batch, grad_params, rng=None):
        chunk_len = int(self.config.get('action_chunk_length', 1))
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
            psi_s = self.network.select('psi')(batch['observations'], params=grad_params)
            psi_g = self.network.select('psi')(batch['actor_goals'], params=grad_params)
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
            dist = self.network.select('actor')(psi_s, psi_g, params=grad_params, **akw)
            if p_bc > 0.0:
                dist_bc = self.network.select('actor')(
                    jax.lax.stop_gradient(psi_s),
                    jax.lax.stop_gradient(psi_g_bc),
                    params=grad_params,
                    **akw,
                )
            else:
                dist_bc = dist
        else:
            dist = self.network.select('actor')(
                batch['observations'], batch['actor_goals'], params=grad_params, **akw
            )
            if p_bc > 0.0:
                dist_bc = self.network.select('actor')(
                    batch['observations'], bc_goals, params=grad_params, **akw
                )
            else:
                dist_bc = dist

        if self.config['const_std']:
            q_flat = jnp.clip(dist.mode(), -1, 1)
        else:
            q_flat = jnp.clip(dist.sample(seed=rng), -1, 1)

        if chunk_len > 1:
            q_actions = q_flat[:, :action_dim]
        else:
            q_actions = q_flat

        # Goal-reaching Q via critic 1.
        phi = self.network.select('phi')(batch['observations'], q_actions)
        psi_goal = self.network.select('psi')(batch['actor_goals'])
        q_goal_pair = -self.distance(phi, psi_goal)
        q1, q2 = q_goal_pair
        q_goal = jnp.minimum(q1, q2)

        # Action-match Q via critic 2: actor wants phi_am(s, a) ~ psi_am(s).
        phi_am = self.network.select('phi_am')(batch['observations'], q_actions)
        psi_state = self.network.select('psi_am')(batch['observations'])
        q_match_pair = -self.distance(phi_am, psi_state)
        qm1, qm2 = q_match_pair
        q_match = jnp.minimum(qm1, qm2)

        w = float(self.config.get('q_match_weight', 1.0))
        q = q_goal + w * q_match

        q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)

        if self.config['discrete']:
            log_prob = dist_bc.log_prob(batch['actions'])
            bc_loss = -(self.config['alpha'] * log_prob).mean()
            actor_loss_val = q_loss + bc_loss
            pred = dist.mode()
            return actor_loss_val, {
                'actor_loss': actor_loss_val,
                'q_loss': q_loss,
                'bc_loss': bc_loss,
                'q_mean': q.mean(),
                'q_goal_mean': q_goal.mean(),
                'q_match_mean': q_match.mean(),
                'q_abs_mean': jnp.abs(q).mean(),
                'bc_log_prob': log_prob.mean(),
                'mse': jnp.mean((pred - batch['actions']) ** 2),
                'std': jnp.mean(dist.scale_diag),
            }

        action_targets = batch['action_chunks'].reshape(batch['action_chunks'].shape[0], -1)
        log_prob = dist_bc.log_prob(action_targets)
        bc_loss = -(self.config['alpha'] * log_prob).mean()
        actor_loss_val = q_loss + bc_loss

        pred = dist.mode()
        actor_info = {
            'actor_loss': actor_loss_val,
            'q_loss': q_loss,
            'bc_loss': bc_loss,
            'q_mean': q.mean(),
            'q_goal_mean': q_goal.mean(),
            'q_match_mean': q_match.mean(),
            'q_abs_mean': jnp.abs(q).mean(),
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((pred - action_targets) ** 2),
            'std': jnp.mean(dist.scale_diag),
        }
        if chunk_len > 1:
            actor_info['mse_first'] = jnp.mean(
                (pred[:, :action_dim] - action_targets[:, :action_dim]) ** 2
            )
        return actor_loss_val, actor_info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None, critic_only=False, step: int = 0):
        info = {}
        rng = rng if rng is not None else self.rng

        (_c1, _bk, _inv), critic_loss_val, critic_info = self.critic_loss(batch, grad_params)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        am_loss_val, am_info = self.action_match_loss(batch, grad_params)
        for k, v in am_info.items():
            info[f'critic/{k}'] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss_val, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        am_weight = float(self.config.get('am_loss_weight', 1.0))
        loss = critic_loss_val + am_weight * am_loss_val + actor_loss_val
        info['critic/am_loss_weighted'] = am_weight * am_loss_val
        return loss, info

    @jax.jit
    def update(self, batch, critic_only=False, step: int = 0):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng, critic_only=critic_only, step=step)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, action_stack=None, temperature=1.0):
        chunk_len = int(self.config.get('action_chunk_length', 1))
        akw = actor_action_stack_kwargs_from_value(self.config, action_stack)
        if self.config['use_latent']:
            psi_s, psi_g = self.network.select('psi')(observations), self.network.select('psi')(goals)
            if len(psi_s.shape) == 2:
                psi_s = jnp.mean(psi_s, axis=0)
                psi_g = jnp.mean(psi_g, axis=0)
            dist = self.network.select('actor')(psi_s, psi_g, temperature=temperature, **akw)
        else:
            dist = self.network.select('actor')(observations, goals, temperature=temperature, **akw)
        actions = dist.sample(seed=seed)
        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)
            if chunk_len > 1:
                flat_dim = actions.shape[-1]
                ad = flat_dim // chunk_len
                actions = actions.reshape(*actions.shape[:-1], chunk_len, ad)
                actions = actions[..., 0, :]
        return actions

    @jax.jit
    def get_distance(self, observations, goals, actions):
        if self.config['use_action_for_distance']:
            phi = self.network.select('phi')(observations, actions)
        else:
            phi = self.network.select('psi')(observations)
        psi = self.network.select('psi')(goals)
        return self.distance(phi, psi)

    @jax.jit
    def get_action_match_score(self, observations, actions):
        """Q-style score for (s, a): higher means 'action matches state' more."""
        phi_am = self.network.select('phi_am')(observations, actions)
        psi_am = self.network.select('psi_am')(observations)
        return -self.distance(phi_am, psi_am)

    # ------------------------------------------------------------------ #
    #  Construction                                                       #
    # ------------------------------------------------------------------ #

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config, steps=None, ex_action_stack=None):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        # Encoders: critic-1 and actor share one encoder_modules family; critic-2
        # gets its *own* state encoder so the two critics learn independent
        # state representations ("completely different" per design).
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['actor'] = GCEncoder(concat_encoder=encoder_module())
            encoders['state'] = encoder_module()     # shared across phi, psi
            encoders['state_am'] = encoder_module()  # shared across phi_am, psi_am

        if config['discrete']:
            phi_def = DiscreteStateActionRepresentation(
                hidden_dims=config['value_hidden_dims'],
                latent_dim=config['latent_dim'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                value_exp=True,
                state_encoder=encoders.get('state'),
                action_dim=action_dim,
            )
            psi_def = DiscreteStateActionRepresentation(
                hidden_dims=config['value_hidden_dims'],
                latent_dim=config['latent_dim'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                value_exp=True,
                state_encoder=encoders.get('state'),
                action_dim=action_dim,
            )
            phi_am_def = DiscreteStateActionRepresentation(
                hidden_dims=config['value_hidden_dims'],
                latent_dim=config['latent_dim'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                value_exp=True,
                state_encoder=encoders.get('state_am'),
                action_dim=action_dim,
            )
            psi_am_def = DiscreteStateActionRepresentation(
                hidden_dims=config['value_hidden_dims'],
                latent_dim=config['latent_dim'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                value_exp=True,
                state_encoder=encoders.get('state_am'),
                action_dim=action_dim,
            )
            actor_def = GCDiscreteActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                gc_encoder=encoders.get('actor'),
            )
        else:
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
            phi_am_def = StateRepresentation(
                hidden_dims=config['value_hidden_dims'],
                latent_dim=config['latent_dim'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                value_exp=True,
                state_encoder=encoders.get('state_am'),
            )
            psi_am_def = StateRepresentation(
                hidden_dims=config['value_hidden_dims'],
                latent_dim=config['latent_dim'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                value_exp=True,
                state_encoder=encoders.get('state_am'),
            )
            actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                chunk_length=int(config.get('action_chunk_length', 1)),
                state_dependent_std=False,
                const_std=config['const_std'],
                gc_encoder=encoders.get('actor'),
            )

        if config['use_iqe']:
            network_info = dict(
                actor=(actor_def, build_gc_actor_init(ex_observations, ex_goals, ex_action_stack)),
                phi=(phi_def, (ex_observations, ex_actions)),
                psi=(psi_def, (ex_goals,)),
                phi_am=(phi_am_def, (ex_observations, ex_actions)),
                psi_am=(psi_am_def, (ex_observations,)),
                alpha_raw=(Param(), ()),
            )
        else:
            if config['use_latent']:
                embed = jnp.zeros((1, config['latent_dim']))
                network_info = dict(
                    actor=(actor_def, build_gc_actor_init(embed, embed, ex_action_stack)),
                    phi=(phi_def, (ex_observations, ex_actions)),
                    psi=(psi_def, (ex_goals,)),
                    phi_am=(phi_am_def, (ex_observations, ex_actions)),
                    psi_am=(psi_am_def, (ex_observations,)),
                )
            else:
                network_info = dict(
                    actor=(actor_def, build_gc_actor_init(ex_observations, ex_goals, ex_action_stack)),
                    phi=(phi_def, (ex_observations, ex_actions)),
                    psi=(psi_def, (ex_goals,)),
                    phi_am=(phi_am_def, (ex_observations, ex_actions)),
                    psi_am=(psi_am_def, (ex_observations,)),
                )

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
            # Agent hyperparameters (critic 1: identical to TMD).
            agent_name='tmd_dc',
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
            actor_log_q=True,
            const_std=True,
            discrete=False,
            # Dataset hyperparameters.
            dataset_class='GCDataset',
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
            use_iqe=False,
            use_latent=False,
            freeze_enc_for_actor_grad=False,
            use_action_for_distance=True,
            frame_stack=ml_collections.config_dict.placeholder(int),
            dual_descent=False,
            action_chunk_length=1,
            # --- Double-critic additions ---
            # Weight of the action-match critic's loss in total_loss.
            am_loss_weight=1.0,
            # Weight of the action-match Q term added to the goal-reaching Q in the actor.
            q_match_weight=1.0,
        )
    )
    return config


# Alias so `train.py --algorithm TMD_DC` resolves to `agents.tmd_dc.TMD_DCAgent`.
TMD_DCAgent = TMDDCAgent
