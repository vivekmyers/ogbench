from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
import numpy as np
from typing import Dict
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, GCBilinearValue, GCDiscreteActor, GCDiscreteBilinearCritic


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
        logits = jnp.einsum('eik,ejk->ije', phi, psi) / jnp.sqrt(phi.shape[-1])
        # logits.shape is (B, B, e) with one term for positive pair and (B - 1) terms for negative pairs in each row.
        I = jnp.eye(batch_size)
        contrastive_loss = jax.vmap(
            lambda _logits: optax.sigmoid_binary_cross_entropy(logits=_logits, labels=I),
            in_axes=-1,
            out_axes=-1,
        )(logits)
        contrastive_loss = jnp.mean(contrastive_loss)

        # Compute additional statistics.
        logits = jnp.mean(logits, axis=-1)
        correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

        return contrastive_loss, {
            'contrastive_loss': contrastive_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
            'binary_accuracy': jnp.mean((logits > 0) == I),
            'categorical_accuracy': jnp.mean(correct),
            'logits_pos': logits_pos,
            'logits_neg': logits_neg,
            'logits': logits.mean(),
        }

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the actor loss (AWR or DDPG+BC)."""
        # Maximize log Q if actor_log_q is True (which is default).
        if self.config['actor_log_q']:

            def value_transform(x):
                return jnp.log(jnp.maximum(x, 1e-6))
        else:

            def value_transform(x):
                return x

        if self.config['actor_loss'] == 'awr':
            # AWR loss.
            v = value_transform(self.network.select('value')(batch['observations'], batch['actor_goals']))
            q1, q2 = value_transform(
                self.network.select('critic')(batch['observations'], batch['actor_goals'], batch['actions'])
            )
            q = jnp.minimum(q1, q2)
            adv = q - v

            exp_a = jnp.exp(adv * self.config['alpha'])
            exp_a = jnp.minimum(exp_a, 100.0)

            dist = self.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)
            log_prob = dist.log_prob(batch['actions'])

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
            q1, q2 = value_transform(
                self.network.select('critic')(batch['observations'], batch['actor_goals'], q_actions)
            )
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
            ex_observations: Example observations.
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
                value_exp=True,
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
                value_exp=True,
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
                value_exp=True,
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
            actor_log_q=True,  # Whether to maximize log Q (True) or Q itself (False) in the actor loss.
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
        )
    )
    return config


def sample_batch(
    dataset: Dict[str, np.ndarray], *, batch_size: int, frame_stack: int = 1, config: Dict
) -> Dict[str, jnp.ndarray]:
    """Fast batch sampling for CRL contrastive learning."""
    n = int(dataset['observations'].shape[0] * 0.8)
    
    # CRITICAL FIX: Limit sampling range to leave room for goal offsets
    # Use a much more conservative approach - leave plenty of room
    discount = config.get('discount', 0.95)  # Use the actual discount from config
    geometric_p = 1 - discount
    
    # Use 99.9th percentile + safety margin for maximum offset
    theoretical_max = int(np.log(0.001) / np.log(1 - geometric_p))  # 99.9th percentile
    safety_margin = 100  # Additional safety margin
    max_safe_offset = theoretical_max + safety_margin
    max_safe_offset = min(max_safe_offset, 500)  # Cap at 500 frames
    
    # Sample from a much more conservative range
    safe_n = max(n - max_safe_offset, n // 3)  # Use at most 2/3 of dataset for safety
    
    idx = np.random.choice(safe_n, batch_size, replace=True)
    # Create base batch
    batch = {
        'observations': dataset['observations'][idx],
        'action_labels': dataset['action_labels'][idx],
        'actions': dataset['actions'][idx],
        'next_observations': dataset['next_observations'][idx],
        'rewards': jnp.zeros((batch_size,)),
        'masks': jnp.ones((batch_size,)),
    }
    
    should_debug = False  # Disable debug output for cleaner logs
    
    # Use geometric distribution for goal sampling (proper CRL approach)
    # Sample goal offsets using geometric distribution based on discount factor
    discount = config.get('discount', 0.99)
    geometric_p = 1 - discount  # probability parameter for geometric distribution
    
    # Sample offsets for all batch elements using geometric distribution
    offsets = np.random.geometric(p=geometric_p, size=batch_size)
    
    # Additional safety: clip offsets to reasonable range
    offsets = np.clip(offsets, 1, 2000)  # Between 1 and 200 frames ahead
    
    # CRITICAL FIX: Use bidirectional mapping to maintain temporal structure
    # BUT FALLBACK TO SIMPLE METHOD IF MAPPING IS BROKEN
    use_mapping = ('shuffled_to_original' in dataset and 'original_to_shuffled' in dataset)
    
    # Quick sanity check on mapping if debug is enabled
    if use_mapping:
        mapping_ok = (
            len(dataset['shuffled_to_original']) == len(dataset['observations']) and
            len(dataset['original_to_shuffled']) == len(dataset['observations']) and
            np.max(dataset['shuffled_to_original']) < len(dataset['observations']) and
            np.max(dataset['original_to_shuffled']) < len(dataset['observations'])
        )
        if not mapping_ok:
            use_mapping = False
    
    if use_mapping:
        # Get the original temporal positions of our sampled frames
        original_positions = dataset['shuffled_to_original'][idx]
        
        # Add offsets to get future frames in original temporal order
        goal_original_positions = original_positions + offsets
        
        # Clip to valid range in original timeline
        max_original_pos = len(dataset['original_to_shuffled']) - 1
        goal_original_positions_safe = np.clip(goal_original_positions, 0, max_original_pos)
        
        # Direct lookup: where are these goal positions in the shuffled dataset?
        goal_idx = dataset['original_to_shuffled'][goal_original_positions_safe]
    else:
        # SIMPLE BUT EFFECTIVE FALLBACK: Direct offset-based sampling
        # This maintains temporal relationships without complex mapping
        # Ensure goals don't go out of bounds by using the safe sampling we already did
        goal_idx = np.clip(idx + offsets, 0, safe_n - 1)
    
    # Create goal-conditioned batch
    goal_observations = dataset['observations'][goal_idx]
    

    
    # Debug shape issues that might cause NaN
    obs_shape = dataset['observations'].shape
    goal_obs_shape = goal_observations.shape
    
    if np.random.random() < 0.001:  # Very rare debug print
        print(f"DEBUG sample_batch shapes:")
        print(f"  Original observations: {obs_shape}")
        print(f"  Goal observations: {goal_obs_shape}")
        print(f"  Frame stack: {frame_stack}")
        print(f"  Expected channels per frame: 3")
        print(f"  Expected total channels: {3 * frame_stack}")
    
    # Extract the last frame (last 3 channels) for goal
    if len(goal_observations.shape) == 4 and goal_observations.shape[-1] >= 3:
        base_goals = goal_observations[:, :, :, -3:]  # Last frame's RGB
    else:
        print(f"ERROR: Unexpected observation shape: {goal_observations.shape}")
        print(f"Cannot extract last 3 channels for goal. Using full observation.")
        base_goals = goal_observations
    
    # Check for any NaN/inf values in goals
    if np.any(np.isnan(base_goals)) or np.any(np.isinf(base_goals)):
        print(f"WARNING: NaN or Inf detected in base_goals!")
        print(f"Goal indices: {goal_idx[:5]}...")  # Show first few
        print(f"Data indices: {idx[:5]}...")
        base_goals = np.nan_to_num(base_goals, nan=0.0, posinf=1.0, neginf=-1.0)
    
    goal_stack = jnp.concatenate([base_goals] * frame_stack, axis=-1)

    batch['value_goals'] = goal_stack
    batch['actor_goals'] = goal_stack
    


    # Compute rewards based on temporal distance (offsets), not shuffled index distance
    # Closer goals (smaller temporal offsets) get higher rewards
    batch['rewards'] = -offsets.astype(float) / 20.0  # Negative reward proportional to temporal distance
    batch['masks'] = jnp.ones((batch_size,))  # Keep all transitions active
    

    
    return batch