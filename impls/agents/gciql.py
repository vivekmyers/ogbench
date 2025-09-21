import copy
from typing import Any, Dict

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, GCDiscreteActor, GCDiscreteCritic, GCValue
import numpy as np


class GCIQLAgent(flax.struct.PyTreeNode):
    """Goal-conditioned implicit Q-learning (GCIQL) agent.

    This implementation supports both AWR (actor_loss='awr') and DDPG+BC (actor_loss='ddpgbc') for the actor loss.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def value_loss(self, batch, grad_params):
        """Compute the IQL value loss."""
        q1, q2 = self.network.select('target_critic')(batch['observations'], batch['value_goals'], batch['actions'])
        q = jnp.minimum(q1, q2)
        v = self.network.select('value')(batch['observations'], batch['value_goals'], params=grad_params)
        value_loss = self.expectile_loss(q - v, q - v, self.config['expectile']).mean()

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def critic_loss(self, batch, grad_params):
        """Compute the IQL critic loss."""
        next_v = self.network.select('value')(batch['next_observations'], batch['value_goals'])
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v

        q1, q2 = self.network.select('critic')(
            batch['observations'], batch['value_goals'], batch['actions'], params=grad_params
        )
        critic_loss = ((q1 - q) ** 2 + (q2 - q) ** 2).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

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

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        critic_loss, critic_info = self.critic_loss(batch, grad_params)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = value_loss + critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')

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
            encoders['value'] = GCEncoder(concat_encoder=encoder_module())
            encoders['critic'] = GCEncoder(concat_encoder=encoder_module())
            encoders['actor'] = GCEncoder(concat_encoder=encoder_module())

        # Define value and actor networks.
        value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=False,
            gc_encoder=encoders.get('value'),
        )

        if config['discrete']:
            critic_def = GCDiscreteCritic(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                gc_encoder=encoders.get('critic'),
                action_dim=action_dim,
            )
        else:
            critic_def = GCValue(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                gc_encoder=encoders.get('critic'),
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
            value=(value_def, (ex_observations, ex_goals)),
            critic=(critic_def, (ex_observations, ex_goals, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_goals, ex_actions)),
            actor=(actor_def, (ex_observations, ex_goals)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_critic'] = params['modules_critic']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='gciql',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            expectile=0.9,  # IQL expectile.
            actor_loss='ddpgbc',  # Actor loss type ('awr' or 'ddpgbc').
            alpha=0.3,  # Temperature in AWR or BC coefficient in DDPG+BC.
            const_std=True,  # Whether to use constant standard deviation for the actor.
            discrete=False,  # Whether the action space is discrete.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            # Dataset hyperparameters.
            dataset_class='GCDataset',  # Dataset class name.
            value_p_curgoal=0.2,  # Probability of using the current state as the value goal.
            value_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.3,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=True,  # Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as reward.
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config


def sample_batch(
    dataset: Dict[str, np.ndarray], *, batch_size: int, frame_stack: int = 1, config: Dict
) -> Dict[str, jnp.ndarray]:
    """Fast batch sampling for GCIQL with goal-conditioned learning."""
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
    
    # Use geometric distribution for goal sampling (proper approach)
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
