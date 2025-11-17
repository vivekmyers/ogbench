from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import (
    GCActor,
    GCBilinearValue,
    GCDiscreteActor,
    GCDiscreteBilinearCritic,
    GoalDistanceHead,
)


class CRLAgent(flax.struct.PyTreeNode):
    """Contrastive RL (CRL) agent.

    This implementation supports both AWR (actor_loss='awr') and DDPG+BC (actor_loss='ddpgbc') for the actor loss.
    CRL with DDPG+BC only fits a Q function, while CRL with AWR fits both Q and V functions to compute advantages.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def contrastive_loss(self, batch, grad_params, module_name='critic'):
        """Compute the contrastive value loss for the Q or V function."""
        batch_size = batch['observations'].shape[0]

        if module_name == 'critic':
            actions = batch['actions']
        else:
            actions = None
        v, phi, psi = self.network.select(module_name)(
            batch['observations'],
            batch['value_goals'],
            actions=actions,
            info=True,
            params=grad_params,
        )
        if len(phi.shape) == 2:  # Non-ensemble.
            phi = phi[None, ...]
            psi = psi[None, ...]
        
        # Normalize phi and psi to unit vectors to prevent representation collapse
        phi_norm = phi / (jnp.linalg.norm(phi, axis=-1, keepdims=True) + 1e-8)
        psi_norm = psi / (jnp.linalg.norm(psi, axis=-1, keepdims=True) + 1e-8)
        
        # Compute logits: similarity between phi[i] and psi[j] for all pairs
        # Use normalized representations to ensure meaningful similarity scores
        logits = jnp.einsum('eik,ejk->ije', phi_norm, psi_norm)
        # logits.shape is (B, B, e) with one term for positive pair and (B - 1) terms for negative pairs in each row.
        I = jnp.eye(batch_size)
        
        # Use InfoNCE-style contrastive loss with temperature scaling
        # Temperature scaling helps the model learn better separations
        temperature = 0.17  # Standard temperature for contrastive learning
        logits_scaled = logits / temperature
        
        # For each ensemble member, compute InfoNCE loss
        # For each row i, we want logits[i, i] (positive) to be larger than logits[i, j] for j != i (negatives)
        def infonce_loss(_logits):

            row_log_probs = jax.nn.log_softmax(_logits, axis=-1)  # Shape: (B, B)
            
            pos_log_probs = jnp.diag(row_log_probs)  # Shape: (B,)
        
            return -jnp.mean(pos_log_probs)
        
        contrastive_loss = jax.vmap(infonce_loss, in_axes=-1, out_axes=-1)(logits_scaled)
        contrastive_loss = jnp.mean(contrastive_loss)
        
        # Use original logits (before scaling) for statistics
        logits_for_stats = logits

        # Compute additional statistics.
        v = jnp.exp(v)
        logits = jnp.mean(logits_for_stats, axis=-1)  # Average over ensemble for stats
        correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)
        
        # Additional diagnostics: check representation diversity
        phi_mean = jnp.mean(phi_norm, axis=(0, 1))  # Average over ensemble and batch
        psi_mean = jnp.mean(psi_norm, axis=(0, 1))
        phi_std = jnp.std(phi_norm, axis=(0, 1))
        psi_std = jnp.std(psi_norm, axis=(0, 1))
        phi_psi_diff = jnp.mean(jnp.abs(phi_mean - psi_mean))
        
        # Check for representation collapse: if all representations are similar, std will be low
        # Average std across batch dimension for each ensemble member, then average over ensemble
        phi_batch_std = jnp.mean(jnp.std(phi_norm, axis=1))  # Shape: (e, latent_dim) -> mean over e and latent_dim
        psi_batch_std = jnp.mean(jnp.std(psi_norm, axis=1))
        
        # Check if positive pairs are actually more similar than negatives
        pos_similarities = jnp.diag(logits)  # Positive pair similarities (B,)
        neg_similarities = logits * (1 - I)  # All negative pairs (B, B)
        neg_similarities_mean = jnp.sum(neg_similarities) / jnp.sum(1 - I)
        pos_neg_gap = jnp.mean(pos_similarities) - neg_similarities_mean

        distance_weight = float(self.config.get('distance_loss_weight', 0.0))
        distance_loss = 0.0

        stats = {
            'contrastive_loss': contrastive_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
            'binary_accuracy': jnp.mean((logits > 0) == I),
            'categorical_accuracy': jnp.mean(correct),
            'logits_pos': logits_pos,
            'logits_neg': logits_neg,
            'logits': logits.mean(),
            'logits_std': jnp.std(logits),
            'logits_pos_neg_diff': logits_pos - logits_neg,  # Should be positive if working
            'phi_mean_norm': jnp.linalg.norm(phi_mean),
            'psi_mean_norm': jnp.linalg.norm(psi_mean),
            'phi_std_mean': jnp.mean(phi_std),
            'psi_std_mean': jnp.mean(psi_std),
            'phi_psi_diff': phi_psi_diff,
            # Representation collapse diagnostics
            'phi_batch_std': phi_batch_std,  # Low (< 0.05) = representation collapse
            'psi_batch_std': psi_batch_std,  # Low (< 0.05) = representation collapse
            'pos_neg_gap': pos_neg_gap,  # Should be positive and increasing (target: > 0.2)
            'pos_similarity_mean': jnp.mean(pos_similarities),
            'neg_similarity_mean': neg_similarities_mean,
        }

        if (
            module_name == 'critic'
            and distance_weight > 0.0
            and 'distance_head' in grad_params
            and 'value_goal_deltas' in batch
        ):
            phi_mean = jnp.mean(phi, axis=0)
            psi_mean = jnp.mean(psi, axis=0)
            distance_pred = self.network.select('distance_head')(
                phi_mean,
                psi_mean,
                params=grad_params,
            )
            targets = batch['value_goal_deltas']
            mask = batch.get('value_goal_delta_mask')
            if mask is None:
                mask = jnp.ones_like(targets)
            mse = (distance_pred - targets) ** 2
            distance_loss = jnp.sum(mask * mse) / (jnp.sum(mask) + 1e-6)
            contrastive_loss = contrastive_loss + distance_weight * distance_loss

            stats.update(
                {
                    'distance_loss': distance_loss,
                    'distance_pred_mean': jnp.mean(distance_pred),
                    'distance_target_mean': jnp.mean(targets),
                    'distance_mask_fraction': jnp.mean(mask),
                }
            )

        return contrastive_loss, stats

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the actor loss (AWR or DDPG+BC)."""
        if self.config['actor_loss'] == 'awr':
            # AWR loss.
            v = self.network.select('value')(batch['observations'], batch['actor_goals'])
            q1, q2 = self.network.select('critic')(batch['observations'], batch['actor_goals'], batch['actions'])
            q = jnp.minimum(q1, q2)
            adv = q - v

            exp_a = jnp.exp(adv * self.config['alpha'])
            exp_a = jnp.minimum(exp_a, 100.0)

            dist = self.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)
            log_prob = dist.log_prob(batch['actions'])

            pred_actions = dist.mode()

            actor_loss = -(exp_a * log_prob).mean()

            actor_info = {
                'actor_loss': actor_loss,
                'adv': adv.mean(),
                'bc_log_prob': log_prob.mean(),
            }
            if not self.config['discrete']:
                actor_info.update(
                    {
                        'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                        'std': jnp.mean(dist.scale_diag),
                    }
                )

            return actor_loss, actor_info
        elif self.config['actor_loss'] == 'ddpgbc':
            # DDPG+BC loss.
            assert not self.config['discrete']

            dist = self.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)
            if self.config['const_std']:
                q_actions = jnp.clip(dist.mode(), -1, 1)
            else:
                q_actions = jnp.clip(dist.sample(seed=rng), -1, 1)
            q1, q2 = self.network.select('critic')(batch['observations'], batch['actor_goals'], q_actions)
            q = jnp.minimum(q1, q2)

            # Normalize Q values by the absolute mean to make the loss scale invariant.
            q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
            log_prob = dist.log_prob(batch['actions'])

            bc_loss = -(self.config['alpha'] * log_prob).mean()

            actor_loss = q_loss + bc_loss

            return actor_loss, {
                'actor_loss': actor_loss,
                'q_loss': q_loss,
                'bc_loss': bc_loss,
                'q_mean': q.mean(),
                'q_abs_mean': jnp.abs(q).mean(),
                'bc_log_prob': log_prob.mean(),
                'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                'std': jnp.mean(dist.scale_diag),
            }
        else:
            raise ValueError(f'Unsupported actor loss: {self.config["actor_loss"]}')

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        critic_loss, critic_info = self.contrastive_loss(batch, grad_params, 'critic')
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        if self.config['actor_loss'] == 'awr':
            value_loss, value_info = self.contrastive_loss(batch, grad_params, 'value')
            for k, v in value_info.items():
                info[f'value/{k}'] = v
        else:
            value_loss = 0.0

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + value_loss + actor_loss
        return loss, info

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor."""
        dist = self.network.select('actor')(observations, goals, temperature=temperature)
        actions = dist.sample(seed=seed)
        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions. In discrete-action MDPs, this should contain the maximum action value.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic_state'] = encoder_module()
            encoders['critic_goal'] = encoder_module()
            encoders['actor'] = GCEncoder(concat_encoder=encoder_module())
            if config['actor_loss'] == 'awr':
                encoders['value_state'] = encoder_module()
                encoders['value_goal'] = encoder_module()

        # Define value and actor networks.
        if config['discrete']:
            critic_def = GCDiscreteBilinearCritic(
                hidden_dims=config['value_hidden_dims'],
                latent_dim=config['latent_dim'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                value_exp=False,
                state_encoder=encoders.get('critic_state'),
                goal_encoder=encoders.get('critic_goal'),
                action_dim=action_dim,
            )
        else:
            critic_def = GCBilinearValue(
                hidden_dims=config['value_hidden_dims'],
                latent_dim=config['latent_dim'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                value_exp=False,
                state_encoder=encoders.get('critic_state'),
                goal_encoder=encoders.get('critic_goal'),
            )

        if config['actor_loss'] == 'awr':
            # AWR requires a separate V network to compute advantages (Q - V).
            value_def = GCBilinearValue(
                hidden_dims=config['value_hidden_dims'],
                latent_dim=config['latent_dim'],
                layer_norm=config['layer_norm'],
                ensemble=False,
                value_exp=False,
                state_encoder=encoders.get('value_state'),
                goal_encoder=encoders.get('value_goal'),
            )

        if config['discrete']:
            actor_def = GCDiscreteActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                gc_encoder=encoders.get('actor'),
            )
        else:
            actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config['const_std'],
                gc_encoder=encoders.get('actor'),
            )

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_goals, ex_actions)),
            actor=(actor_def, (ex_observations, ex_goals)),
        )
        if config['actor_loss'] == 'awr':
            network_info.update(
                value=(value_def, (ex_observations, ex_goals)),
            )
        if config.get('distance_loss_weight', 0.0) > 0.0:
            distance_head_def = GoalDistanceHead(
                latent_dim=config['latent_dim'],
                hidden_dims=config.get('distance_head_hidden_dims', (256, 256)),
            )
            dummy_latent = np.zeros((1, config['latent_dim']), dtype=ex_observations.dtype)
            network_info['distance_head'] = (
                distance_head_def,
                (dummy_latent, dummy_latent),
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
            # Agent hyperparameters.
            agent_name='crl',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            latent_dim=512,  # Latent dimension for phi and psi.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            actor_loss='ddpgbc',  # Actor loss type ('awr' or 'ddpgbc').
            alpha=0.1,  # Temperature in AWR or BC coefficient in DDPG+BC.
            const_std=True,  # Whether to use constant standard deviation for the actor.
            discrete=False,  # Whether the action space is discrete.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            # Dataset hyperparameters.
            dataset_class='GCDataset',  # Dataset class name.
            value_p_curgoal=0.0,  # Probability of using the current state as the value goal.
            value_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.0,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=False,  # Unused (defined for compatibility with GCDataset).
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
            distance_loss_weight=0.0,
            distance_head_hidden_dims=(256, 256),
        )
    )
    return config