"""TMD with Q-chunking: a single metric critic φ(s, a₀:K) matching the policy chunk length."""

from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState
from utils.networks import (
    DiscreteStateActionRepresentation,
    GCActor,
    GCDiscreteActor,
    Param,
    StateRepresentation,
)

from .tmd import TMDAgent


class TMDQCAgent(TMDAgent):
    """Temporal Metric Distillation with Q-chunking (single critic on full action chunks).

    Same losses as TMD, but φ takes a flattened K-step action chunk (K = action_chunk_length),
    and the backup compares ψ(s_{t+K}) to ψ(g) with discount γ^K when chunk_backup_discount_k.
    """

    @staticmethod
    def _action_chunk_flat(batch) -> jnp.ndarray:
        return batch['action_chunks'].reshape(batch['action_chunks'].shape[0], -1)

    @staticmethod
    def _chunk_gamma(config: Any) -> jnp.ndarray:
        k = int(config.get('action_chunk_length', 1))
        d = float(config['discount'])
        if config.get('chunk_backup_discount_k', True) and k > 1:
            return jnp.array(d**k, dtype=jnp.float32)
        return jnp.array(d, dtype=jnp.float32)

    @jax.jit
    def critic_loss(self, batch, grad_params):
        batch_size = batch['observations'].shape[0]
        a_flat = self._action_chunk_flat(batch)

        if self.config['encoder'] is not None:
            phi, _ = self.network.select('phi')(batch['observations'], a_flat, params=grad_params)
            psi_s, _ = self.network.select('psi')(batch['observations'], params=grad_params)
            psi_next, _ = self.network.select('psi')(
                batch['chunk_next_observations'], params=grad_params
            )
            psi_g, _ = self.network.select('psi')(batch['value_goals'], params=grad_params)
        else:
            phi = self.network.select('phi')(batch['observations'], a_flat, params=grad_params)
            psi_s = self.network.select('psi')(batch['observations'], params=grad_params)
            psi_next = self.network.select('psi')(batch['chunk_next_observations'], params=grad_params)
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
        gamma = self._chunk_gamma(self.config)
        if self.config['stopgrad_psi_backup']:
            dist = self.distance(phi[:, :, None], jax.lax.stop_gradient(psi_g[:, None, :]))
        dist_next = jax.lax.stop_gradient(dist_next)

        delta = dist - dist_next
        mask = delta > t
        delta_clipped = jnp.where(mask, t, delta)
        divergence = jnp.where(mask, delta, gamma * jnp.exp(delta_clipped) - dist)

        dw = self.config['diag_backup']
        divergence = (
            divergence * (1 - dw)
            + jnp.diagonal(divergence, axis1=1, axis2=2)[..., None] * dw
        )
        backup_loss = jnp.mean(divergence)
        if self.config['dual_descent']:
            optim_backup = 1 - jax.lax.stop_gradient(dist_next) + jnp.log(gamma)
            optim_backup = (
                optim_backup * (1 - dw)
                + jnp.diagonal(optim_backup, axis1=1, axis2=2)[..., None] * dw
            )
            backup_optim_loss = jnp.mean(divergence - optim_backup)
            val = jnp.exp(
                -(
                    jax.lax.stop_gradient(backup_optim_loss)
                    + jax.lax.stop_gradient(action_invariance_loss)
                )
            )
            critic_loss = val * contrastive_loss + backup_loss + action_invariance_loss
        else:
            critic_loss = (
                contrastive_loss
                + self.config['zeta'] * action_invariance_loss
                + self.config['zeta'] * backup_loss
            )

        logits = jnp.mean(logits, axis=0)
        correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

        return (
            (contrastive_loss, backup_loss, action_invariance_loss),
            critic_loss,
            {
                'contrastive_loss': contrastive_loss,
                'action_invariance_loss': action_invariance_loss,
                'backup_loss': backup_loss,
                'critic_loss': critic_loss,
                'binary_accuracy': jnp.mean((logits > 0) == I),
                'categorical_accuracy': jnp.mean(correct),
                'logits_pos': logits_pos,
                'logits_neg': logits_neg,
                'logits': logits.mean(),
                'dist': dist.mean(),
                'biggest_diff_in_dist': jnp.max(dist - dist_next),
            },
        )

    @jax.jit
    def actor_loss(self, batch, grad_params, rng=None):
        chunk_len = int(self.config.get('action_chunk_length', 1))
        action_dim = batch['actions'].shape[-1]

        if self.config['use_latent']:
            psi_s, psi_g = (
                self.network.select('psi')(batch['observations'], params=grad_params),
                self.network.select('psi')(batch['actor_goals'], params=grad_params),
            )
            if len(psi_s.shape) == 3:
                psi_s = jnp.mean(psi_s, axis=0)
                psi_g = jnp.mean(psi_g, axis=0)
            if self.config['freeze_enc_for_actor_grad']:
                psi_s, psi_g = jax.lax.stop_gradient(psi_s), jax.lax.stop_gradient(psi_g)
            dist = self.network.select('actor')(psi_s, psi_g, params=grad_params)
        else:
            dist = self.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)

        if self.config['const_std']:
            q_flat = jnp.clip(dist.mode(), -1, 1)
        else:
            q_flat = jnp.clip(dist.sample(seed=rng), -1, 1)

        # Q-chunking: score the full K-step chunk with φ (same input as critic).
        phi = self.network.select('phi')(batch['observations'], q_flat)
        psi = self.network.select('psi')(batch['actor_goals'])
        q1, q2 = -self.distance(phi, psi)
        q = jnp.minimum(q1, q2)

        q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)

        if self.config['discrete']:
            log_prob = dist.log_prob(batch['actions'])
            bc_loss = -(self.config['alpha'] * log_prob).mean()
            actor_loss = q_loss + bc_loss
            pred = dist.mode()
            return actor_loss, {
                'actor_loss': actor_loss,
                'q_loss': q_loss,
                'bc_loss': bc_loss,
                'q_mean': q.mean(),
                'q_abs_mean': jnp.abs(q).mean(),
                'bc_log_prob': log_prob.mean(),
                'mse': jnp.mean((pred - batch['actions']) ** 2),
                'std': jnp.mean(dist.scale_diag),
            }

        action_targets = batch['action_chunks'].reshape(batch['action_chunks'].shape[0], -1)
        log_prob = dist.log_prob(action_targets)
        bc_loss = -(self.config['alpha'] * log_prob).mean()
        actor_loss = q_loss + bc_loss

        pred = dist.mode()
        actor_info = {
            'actor_loss': actor_loss,
            'q_loss': q_loss,
            'bc_loss': bc_loss,
            'q_mean': q.mean(),
            'q_abs_mean': jnp.abs(q).mean(),
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((pred - action_targets) ** 2),
            'std': jnp.mean(dist.scale_diag),
        }
        if chunk_len > 1:
            actor_info['mse_first'] = jnp.mean(
                (pred[:, :action_dim] - action_targets[:, :action_dim]) ** 2
            )
        return actor_loss, actor_info

    @jax.jit
    def get_distance(self, observations, goals, actions):
        chunk_len = int(self.config.get('action_chunk_length', 1))
        action_dim = actions.shape[-1]
        expected = chunk_len * action_dim
        if self.config['use_action_for_distance']:
            if actions.shape[-1] < expected:
                pad = jnp.zeros((*actions.shape[:-1], expected - actions.shape[-1]), dtype=actions.dtype)
                actions = jnp.concatenate([actions, pad], axis=-1)
            elif actions.shape[-1] > expected:
                actions = actions[..., :expected]
            phi = self.network.select('phi')(observations, actions)
        else:
            phi = self.network.select('psi')(observations)
        psi = self.network.select('psi')(goals)
        return self.distance(phi, psi)

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
        steps=None,
    ):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        k = int(config.get('action_chunk_length', 1))
        if config['discrete'] and k > 1:
            raise ValueError(
                'TMDQCAgent: discrete action space only supports action_chunk_length=1 '
                '(use continuous actions for K>1 Q-chunking).'
            )

        if config['discrete']:
            ex_a_chunk = ex_actions
        else:
            ex_a_chunk = jnp.zeros(
                (ex_observations.shape[0], k * action_dim), dtype=ex_actions.dtype
            )

        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['actor'] = GCEncoder(concat_encoder=encoder_module())
            encoders['state'] = encoder_module()

        if config['discrete']:
            phi_def = DiscreteStateActionRepresentation(
                hidden_dims=config['value_hidden_dims'],
                latent_dim=config['latent_dim'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                value_exp=True,
                state_encoder=encoders.get('state'),
                action_dim=action_dim * k,
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
            actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                chunk_length=k,
                state_dependent_std=False,
                const_std=config['const_std'],
                gc_encoder=encoders.get('actor'),
            )

        if config['use_iqe']:
            network_info = dict(
                actor=(actor_def, (ex_observations, ex_goals)),
                phi=(phi_def, (ex_observations, ex_a_chunk)),
                psi=(psi_def, (ex_goals,)),
                alpha_raw=(Param(), ()),
            )
        else:
            if config['use_latent']:
                embed = jnp.zeros((1, config['latent_dim']))
                network_info = dict(
                    actor=(actor_def, (embed, embed)),
                    phi=(phi_def, (ex_observations, ex_a_chunk)),
                    psi=(psi_def, (ex_goals,)),
                )
            else:
                network_info = dict(
                    actor=(actor_def, (ex_observations, ex_goals)),
                    phi=(phi_def, (ex_observations, ex_a_chunk)),
                    psi=(psi_def, (ex_goals,)),
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
            agent_name='tmd_qc',
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
            dataset_class='GCDataset',
            value_p_curgoal=0.0,
            value_p_trajgoal=1.0,
            value_p_randomgoal=0.0,
            value_geom_sample=True,
            actor_p_curgoal=0.0,
            actor_p_trajgoal=1.0,
            actor_p_randomgoal=0.0,
            actor_geom_sample=False,
            gc_negative=False,
            p_aug=0.0,
            use_iqe=False,
            use_latent=False,
            freeze_enc_for_actor_grad=False,
            use_action_for_distance=True,
            frame_stack=ml_collections.config_dict.placeholder(int),
            dual_descent=False,
            action_chunk_length=1,
            chunk_backup_discount_k=True,
        )
    )
    return config


TMD_QCAgent = TMDQCAgent
