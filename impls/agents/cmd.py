from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import (
    GCActor,
    GCDiscreteActor,
    StateRepresentation,
    DiscreteStateActionRepresentation,
)


class CMDAgent(flax.struct.PyTreeNode):
    """Contrastive Metric Distillation (CMD) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @jax.jit
    def mrn_distance(self, x: jnp.ndarray, y: jnp.ndarray):
        K = self.config['mrn_components']
        assert x.shape[-1] % K == 0

        @jax.jit
        def mrn_distance_component(x: jnp.ndarray, y: jnp.ndarray):
            eps = 1e-8
            d = x.shape[-1]
            mask = jnp.arange(d) < d // 2
            max_component: jnp.ndarray = jax.nn.relu(jnp.max((x - y) * mask, axis=-1)) # jax.nn.relu(jax.nn.logsumexp((x - y) * mask, axis=-1, b=1/x.shape[-1])) # jax.nn.relu(jax.nn.logsumexp((x - y) * mask, axis=-1, b=1/x.shape[-1]))#jnp.logsumexp(jax.nn.relu(jnp.max((x - y) * mask , axis=-1)))
            l2_component: jnp.ndarray = jnp.linalg.norm((x - y) * (1 - mask) + eps, axis=-1)
            # assert max_component.shape == l2_component.shape
            return max_component + l2_component

        x_split = jnp.stack(jnp.split(x, K, axis=-1), axis=-1)
        y_split = jnp.stack(jnp.split(y, K, axis=-1), axis=-1)
        dists: jnp.ndarray = jax.vmap(mrn_distance_component, in_axes=(-1, -1), out_axes=-1)(x_split, y_split)

        return dists.mean(axis=-1)

    def contrastive_loss(self, batch, grad_params):
        batch_size = batch["observations"].shape[0]

        phi = self.network.select("critic")(
            batch["observations"], batch["actions"], info=True, params=grad_params
        )
        actions_roll = jnp.roll(batch["actions"], shift=1, axis=0)
        psi = self.network.select("critic")(
            batch["value_goals"],
            actions_roll,
            info=True,
            params=grad_params,
        )
        if len(phi.shape) == 2:  # Non-ensemble
            phi = phi[None, ...]

        dist = self.mrn_distance(phi[:, :, None], psi[:, None, :])
        logits = -dist / jnp.sqrt(phi.shape[-1])
        # logits.shape is (e, B, B) with one term for positive pair and (B - 1) terms for negative pairs in each row.

        I = jnp.eye(batch_size)
        contrastive_loss = jax.vmap(
            lambda _logits: optax.sigmoid_binary_cross_entropy(logits=_logits, labels=I),
        )(logits)
        contrastive_loss = jnp.mean(contrastive_loss)

        logits = jnp.mean(logits, axis=0)
        correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

        return contrastive_loss, {
            "contrastive_loss": contrastive_loss,
            "binary_accuracy": jnp.mean((logits > 0) == I),
            "categorical_accuracy": jnp.mean(correct),
            "logits_pos": logits_pos,
            "logits_neg": logits_neg,
            "logits": logits.mean(),
            "dist": dist.mean(),
        }
    
    @jax.jit
    def get_distance(self, observations, goals, actions):
        #actions not used, will be used for cmd
        if self.config['use_action_for_distance']:
            # psi = self.network.select('psi')(observations)
            phi = self.network.select('critic')(observations, actions)
        else:
            phi = self.network.select('critic')(observations, jnp.zeros_like(actions))
        psi = self.network.select('critic')(goals, jnp.zeros_like(actions))
        return self.mrn_distance(phi, psi)

    def actor_loss(self, batch, grad_params, rng=None):
        # Maximize log Q if actor_log_q is True (which is default).

        dist = self.network.select("actor")(batch["observations"], batch["actor_goals"], params=grad_params)
        if self.config["const_std"]:
            q_actions = jnp.clip(dist.mode(), -1, 1)
        else:
            q_actions = jnp.clip(dist.sample(seed=rng), -1, 1)

        actions_roll = jnp.roll(batch["actions"], shift=1, axis=0)

        assert q_actions.shape == actions_roll.shape

        phi = self.network.select("critic")(batch["observations"], q_actions)
        psi = self.network.select("critic")(batch["actor_goals"], actions_roll)
        q1, q2 = -self.mrn_distance(phi, psi)
        q = jnp.minimum(q1, q2)

        # Normalize Q values by the absolute mean to make the loss scale invariant.
        q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
        log_prob = dist.log_prob(batch["actions"])

        bc_loss = -(self.config["alpha"] * log_prob).mean()

        actor_loss = q_loss + bc_loss

        return actor_loss, {
            "actor_loss": actor_loss,
            "q_loss": q_loss,
            "bc_loss": bc_loss,
            "q_mean": q.mean(),
            "q_abs_mean": jnp.abs(q).mean(),
            "bc_log_prob": log_prob.mean(),
            "mse": jnp.mean((dist.mode() - batch["actions"]) ** 2),
            "std": jnp.mean(dist.scale_diag),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None, critic_only=None, step=None):
        info = {}
        rng = rng if rng is not None else self.rng

        critic_loss, critic_info = self.contrastive_loss(batch, grad_params)
        for k, v in critic_info.items():
            info[f"critic/{k}"] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f"actor/{k}"] = v

        loss = critic_loss + actor_loss
        return loss, info

    @jax.jit
    def update(self, batch, contrastive_only=None, step=None):
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
        dist = self.network.select("actor")(observations, goals, temperature=temperature)
        actions = dist.sample(seed=seed)
        if not self.config["discrete"]:
            actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
        train_steps,
    ):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        if config["discrete"]:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config["encoder"] is not None:
            encoder_module = encoder_modules[config["encoder"]]
            encoders["actor"] = GCEncoder(concat_encoder=encoder_module())
            encoders["state"] = encoder_module()

        if config["discrete"]:
            critic_def = DiscreteStateActionRepresentation(
                hidden_dims=config["value_hidden_dims"],
                latent_dim=config["latent_dim"],
                layer_norm=config["layer_norm"],
                ensemble=True,
                value_exp=True,
                state_encoder=encoders.get("state"),
                action_dim=action_dim,
            )
            actor_def = GCDiscreteActor(
                hidden_dims=config["actor_hidden_dims"],
                action_dim=action_dim,
                gc_encoder=encoders.get("actor"),
            )
        else:
            critic_def = StateRepresentation(
                hidden_dims=config["value_hidden_dims"],
                latent_dim=config["latent_dim"],
                layer_norm=config["layer_norm"],
                ensemble=True,
                value_exp=True,
                state_encoder=encoders.get("state"),
            )
            actor_def = GCActor(
                hidden_dims=config["actor_hidden_dims"],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config["const_std"],
                gc_encoder=encoders.get("actor"),
            )

        network_info = dict(
            actor=(actor_def, (ex_observations, ex_goals)),
            critic=(critic_def, (ex_observations, ex_actions)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name="cmd",  # Agent name.
            lr=3e-4,
            mrn_components=8,  # Number of components in MRN.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            latent_dim=512,  # Latent dimension for phi and psi.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            alpha=0.1,  # Temperature in AWR or BC coefficient in DDPG+BC.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            actor_log_q=True,  # Whether to maximize log Q (True) or Q itself (False) in the actor loss.
            const_std=True,  # Whether to use constant standard deviation for the actor.
            discrete=False,  # Whether the action space is discrete.
            # Dataset hyperparameters.
            dataset_class="GCDataset",  # Dataset class name.
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
            cotrain_steps=1000000,  # Number of steps to train the contrastive loss.
            use_action_for_distance=False,  # Whether to use the action for the distance computation.
        )
    )
    return config
