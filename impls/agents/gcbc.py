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
from utils.networks import GCActor, GCDiscreteActor


class GCBCAgent(flax.struct.PyTreeNode):
    """Goal-conditioned behavioral cloning (GCBC) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def actor_loss(self, batch, grad_params, rng=None):
        dist = self.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)
        log_prob = dist.log_prob(batch['actions'])

        actor_loss = -log_prob.mean()

        actor_info = {
            'actor_loss': actor_loss,
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

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = actor_loss
        return loss, info

    @jax.jit
    def update(self, batch):
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
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        if config['discrete']:
            if config.get('multi_discrete', False):
                # For multi-discrete actions, we don't need action_dim in the traditional sense
                # The actor will handle 3D discrete actions internally
                action_dim = None  # Not used for multi-discrete
            else:
                action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['actor'] = GCEncoder(concat_encoder=encoder_module())

        if config['discrete']:
            if config.get('multi_discrete', False):
                # Use multi-discrete mode with GCDiscreteActor
                actor_def = GCDiscreteActor(
                    hidden_dims=config['actor_hidden_dims'],
                    action_dim=None,  # Not used for multi-discrete
                    num_bins_per_dim=32,
                    num_dims=3,
                    multi_discrete=True,
                    gc_encoder=encoders.get('actor'),
                )
            else:
                # Use regular single discrete mode
                actor_def = GCDiscreteActor(
                    hidden_dims=config['actor_hidden_dims'],
                    action_dim=action_dim,
                    multi_discrete=False,
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
            actor=(actor_def, (ex_observations, ex_goals)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))

def sample_batch(
    dataset: Dict[str, np.ndarray], *, batch_size: int, frame_stack: int = 1, config: Dict, add_action_noise: bool = True
) -> Dict[str, jnp.ndarray]:
    n = len(dataset['observations'])
    
    # Action component upsampling
    upsample_mode = config.get('upsample_mode', 'none')
    if upsample_mode != 'none' and 'actions' in dataset:
        actions = dataset['actions']
        throttle = actions[:, 0]
        steer = actions[:, 1]
        brake = actions[:, 2]
        
        steer_thresh = config.get('steer_thresh', 0.1)
        throttle_thresh = config.get('throttle_thresh', 0.3)
        brake_thresh = config.get('brake_thresh', 0.1)
        upsample_weight = config.get('upsample_weight', 3.0)
        
        if upsample_mode == 'turns_low':
            is_target = np.abs(steer) < steer_thresh
        elif upsample_mode == 'turns_high':
            is_target = np.abs(steer) > steer_thresh
        elif upsample_mode == 'throttle_low':
            is_target = throttle < throttle_thresh
        elif upsample_mode == 'throttle_high':
            is_target = throttle > throttle_thresh
        elif upsample_mode == 'brake_low':
            is_target = brake < brake_thresh
        elif upsample_mode == 'brake_high':
            is_target = brake > brake_thresh
        else:
            is_target = np.zeros(n, dtype=bool)
        
        weights = np.where(is_target, upsample_weight, 1.0).astype(np.float64)
        weights /= weights.sum()
        idx = np.random.choice(n, batch_size, replace=True, p=weights)
    else:
        idx = np.random.choice(n, batch_size, replace=True)
    
    # Sample from CPU numpy arrays, then move to GPU as JAX arrays
    batch = {
        'observations': jnp.array(dataset['observations'][idx]),
        'actions': jnp.array(dataset['actions'][idx]),
        'rewards': jnp.zeros((batch_size,)),
        'masks': jnp.ones((batch_size,)),
    }
    
    # Goal sampling with geometric distribution
    discount = config.get('discount', 0.99)
    geometric_p = 1 - discount
    
    # Use pure geometric sampling (standard for goal-conditioned RL)
    # At 20fps with discount=0.99: E[horizon] = 100 steps = 5 seconds
    offsets = np.random.geometric(p=geometric_p, size=batch_size)
    
    block_size = config.get('block_size', 1000)
    goal_idx = np.zeros_like(idx)
    
    for i, obs_idx in enumerate(idx):
        block_start = (obs_idx // block_size) * block_size
        block_end = block_start + block_size
        
        goal_pos = obs_idx + offsets[i]
        # Clip to both block boundaries AND dataset size
        goal_pos = min(goal_pos, block_end - 1, n - 1)
        goal_idx[i] = goal_pos
    
    # Goals use pre-stacked observations directly (already have correct channel count)
    goal_observations = jnp.array(dataset['observations'][goal_idx])
    
    # Only need actor goals for GCBC
    batch['actor_goals'] = goal_observations
    batch['rewards'] = jnp.zeros((batch_size,))
    batch['masks'] = jnp.ones((batch_size,))
    return batch
    
def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='gcbc',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            discount=0.99,  # Discount factor (unused by default; can be used for geometric goal sampling in GCDataset).
            const_std=True,  # Whether to use constant standard deviation for the actor.
            discrete=True,  # Whether the action space is discrete.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            # Dataset hyperparameters.
            dataset_class='GCDataset',  # Dataset class name.
            value_p_curgoal=0.0,  # Unused (defined for compatibility with GCDataset).
            value_p_trajgoal=1.0,  # Unused (defined for compatibility with GCDataset).
            value_p_randomgoal=0.0,  # Unused (defined for compatibility with GCDataset).
            value_geom_sample=False,  # Unused (defined for compatibility with GCDataset).
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=True,  # Unused (defined for compatibility with GCDataset).
            p_aug=0.5,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
            block_size=1000,  # Block size for goal sampling (trajectory length).
            upsample_mode='none',  # Upsampling mode: 'turns_low', 'turns_high', 'throttle_low', 'throttle_high', 'brake_low', 'brake_high', 'none'
            upsample_weight=3.0,  # Weight multiplier for upsampled samples
            steer_thresh=0.1,  # Steer threshold for low/high detection
            throttle_thresh=0.3,  # Throttle threshold for low/high detection
            brake_thresh=0.1,  # Brake threshold for low/high detection
            cycle_steps=20000,  # Steps per mode in cycle (20k each = 120k full cycle)
        )
    )
    return config


# Metrics to log during training
METRICS_TO_LOG = [
    'actor/actor_loss',
    'actor/bc_log_prob',
    'actor/mse',
    'actor/std',
]
