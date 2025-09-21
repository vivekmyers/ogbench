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
        #print(dist.shape)
        print(batch['actions'].shape)
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
    dataset: Dict[str, np.ndarray], *, batch_size: int, frame_stack: int = 1, config: Dict, use_guided: bool = False
) -> Dict[str, jnp.ndarray]:
    n = len(dataset['observations'])
    idx = np.random.choice(n, batch_size, replace=True)
    
    batch = {
        'observations': dataset['observations'][idx],
        'actions': dataset['actions'][idx],
        'next_observations': dataset['next_observations'][idx],
        'rewards': jnp.zeros((batch_size,)),
        'masks': jnp.ones((batch_size,)),
    }
    if not config['discrete']:
        # Continuous action noise handling
        action_noise_stds = jnp.array([0.005, 0.006, 0.003])
        action_correlation = 0.9
        
        action_shape = batch['actions'].shape
        if len(action_shape) == 2:
            batch_size, action_dim = action_shape
            rng = jax.random.PRNGKey(np.random.randint(0, 2**32))
            noise = jnp.zeros((batch_size, action_dim))
            for dim in range(min(3, action_dim)):
                rng, noise_key = jax.random.split(rng)
                noise = noise.at[:, dim].set(
                    jax.random.normal(noise_key, (batch_size,)) * action_noise_stds[dim]
                )
            batch['actions'] = batch['actions'] + noise
            batch['actions'] = batch['actions'].at[:, 0].set(jnp.clip(batch['actions'][:, 0], 0.0, 1.0))
            batch['actions'] = batch['actions'].at[:, 1].set(jnp.clip(batch['actions'][:, 1], -1.0, 1.0))
            if batch['actions'].shape[1] >= 3:
                batch['actions'] = batch['actions'].at[:, 2].set(jnp.clip(batch['actions'][:, 2], 0.0, 1.0))
        elif len(action_shape) == 3:
            batch_size, seq_len, action_dim = action_shape
            rng = jax.random.PRNGKey(np.random.randint(0, 2**32))
            base_noise = jnp.zeros((seq_len, batch_size, action_dim))
            for dim in range(min(3, action_dim)):
                rng, noise_key = jax.random.split(rng)
                base_noise = base_noise.at[:, :, dim].set(
                    jax.random.normal(noise_key, (seq_len, batch_size)) * action_noise_stds[dim]
                )
            
            def correlate_step(carry, noise):
                prev_noise = carry
                new_noise = action_correlation * prev_noise + (1 - action_correlation) * noise
                return new_noise, new_noise
            
            init_noise = base_noise[0]
            _, correlated_noise = jax.lax.scan(correlate_step, init_noise, base_noise[1:])
            correlated_noise = jnp.concatenate([init_noise[None, ...], correlated_noise], axis=0)
            correlated_noise = jnp.transpose(correlated_noise, (1, 0, 2))
            batch['actions'] = batch['actions'] + correlated_noise
            batch['actions'] = batch['actions'].at[:, :, 0].set(jnp.clip(batch['actions'][:, :, 0], 0.0, 1.0))
            batch['actions'] = batch['actions'].at[:, :, 1].set(jnp.clip(batch['actions'][:, :, 1], -1.0, 1.0))
            if batch['actions'].shape[2] >= 3:
                batch['actions'] = batch['actions'].at[:, :, 2].set(jnp.clip(batch['actions'][:, :, 2], 0.0, 1.0))
    
    # Goal sampling
    discount = config.get('discount', 0.99)
    geometric_p = 1 - discount
    offsets = np.random.geometric(p=geometric_p, size=batch_size)
    offsets = np.clip(offsets, 1, 200)
    goal_idx = idx + offsets
    goal_idx = np.clip(goal_idx, 0, len(dataset['observations']) - 1)
    goal_idx = goal_idx % len(dataset['observations'])
    
    base_goals = dataset['observations'][goal_idx][:, :, :, -3:]
    goal_stack = jnp.concatenate([base_goals] * frame_stack, axis=-1)
    
    batch['value_goals'] = jnp.zeros_like(goal_stack)
    batch['actor_goals'] = jnp.zeros_like(goal_stack)
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
        )
    )
    return config
