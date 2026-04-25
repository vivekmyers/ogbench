from typing import Any, Optional, Sequence

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp


def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def ensemblize(cls, num_qs, out_axes=0, **kwargs):
    """Ensemblize a module."""
    return nn.vmap(
        cls,
        variable_axes={'params': 0},
        split_rngs={'params': True},
        in_axes=None,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class Identity(nn.Module):
    """Identity layer."""

    def __call__(self, x):
        return x


class MLP(nn.Module):
    """Multi-layer perceptron.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        activations: Activation function.
        activate_final: Whether to apply activation to the final layer.
        kernel_init: Kernel initializer.
        layer_norm: Whether to apply layer normalization.
    """

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
        return x


class LengthNormalize(nn.Module):
    """Length normalization layer.

    It normalizes the input along the last dimension to have a length of sqrt(dim).
    """

    @nn.compact
    def __call__(self, x):
        return x / jnp.linalg.norm(x, axis=-1, keepdims=True) * jnp.sqrt(x.shape[-1])


class Param(nn.Module):
    """Scalar parameter module."""

    init_value: float = 0.0

    @nn.compact
    def __call__(self):
        return self.param('value', init_fn=lambda key: jnp.full((), self.init_value))


class LogParam(nn.Module):
    """Scalar parameter module with log scale."""

    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_value = self.param('log_value', init_fn=lambda key: jnp.full((), jnp.log(self.init_value)))
        return jnp.exp(log_value)


class TransformedWithMode(distrax.Transformed):
    """Transformed distribution with mode calculation."""

    def mode(self):
        return self.bijector.forward(self.distribution.mode())


class RunningMeanStd(flax.struct.PyTreeNode):
    """Running mean and standard deviation.

    Attributes:
        eps: Epsilon value to avoid division by zero.
        mean: Running mean.
        var: Running variance.
        clip_max: Clip value after normalization.
        count: Number of samples.
    """

    eps: Any = 1e-6
    mean: Any = 1.0
    var: Any = 1.0
    clip_max: Any = 10.0
    count: int = 0

    def normalize(self, batch):
        batch = (batch - self.mean) / jnp.sqrt(self.var + self.eps)
        batch = jnp.clip(batch, -self.clip_max, self.clip_max)
        return batch

    def unnormalize(self, batch):
        return batch * jnp.sqrt(self.var + self.eps) + self.mean

    def update(self, batch):
        batch_mean, batch_var = jnp.mean(batch, axis=0), jnp.var(batch, axis=0)
        batch_count = len(batch)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = m_2 / total_count

        return self.replace(mean=new_mean, var=new_var, count=total_count)


class GCActor(nn.Module):
    """Goal-conditioned actor.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        log_std_min: Minimum value of log standard deviation.
        log_std_max: Maximum value of log standard deviation.
        tanh_squash: Whether to squash the action with tanh.
        state_dependent_std: Whether to use state-dependent standard deviation.
        const_std: Whether to use constant standard deviation.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        gc_encoder: Optional GCEncoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    chunk_length: int = 1
    log_std_min: Optional[float] = -5
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    # When const_std=True, use this as the fixed standard deviation (log_std =
    # log(const_std_val); 1.0 matches the old jnp.zeros_like(means) behavior).
    const_std_val: float = 1.0
    final_fc_init_scale: float = 1e-2
    gc_encoder: nn.Module = None

    def setup(self):
        output_dim = self.action_dim * self.chunk_length
        self.actor_net = MLP(self.hidden_dims, activate_final=True)
        self.mean_net = nn.Dense(output_dim, kernel_init=default_init(self.final_fc_init_scale))
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(output_dim, kernel_init=default_init(self.final_fc_init_scale * 0.1))
        else:
            if not self.const_std:
                self.log_stds = self.param(
                    'log_stds',
                    lambda key, shape: jnp.full(shape, -1.0),
                    (output_dim,),
                )

    def __call__(
        self,
        observations,
        goals=None,
        goal_encoded=False,
        temperature=1.0,
        action_stack=None,
    ):
        """Return the action distribution.

        When ``chunk_length > 1`` the distribution operates on *flattened*
        action vectors of size ``action_dim * chunk_length``.  Callers that
        need ``(B, chunk_length, action_dim)`` should reshape ``dist.mode()``
        or ``dist.sample()`` accordingly.

        When ``action_stack`` is set (shape ``(B, L, action_dim)`` or flattened),
        it is concatenated to the encoder output (past L actions, oldest first).
        """
        if self.gc_encoder is not None:
            inputs = self.gc_encoder(observations, goals, goal_encoded=goal_encoded)
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
            inputs = jnp.concatenate(inputs, axis=-1)
        if action_stack is not None:
            aflat = action_stack.reshape(action_stack.shape[0], -1) if action_stack.ndim > 2 else action_stack
            inputs = jnp.concatenate([inputs, aflat], axis=-1)
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.full_like(means, jnp.log(self.const_std_val))
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            distribution = TransformedWithMode(distribution, distrax.Block(distrax.Tanh(), ndims=1))

        return distribution


class GCDiscreteActor(nn.Module):
    """Goal-conditioned actor for discrete actions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        gc_encoder: Optional GCEncoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    num_bins_per_dim: int = 32
    num_dims: int = 3
    final_fc_init_scale: float = 1e-2
    gc_encoder: nn.Module = None
    multi_discrete: bool = False

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True)
        
        if self.multi_discrete:
            # For multi-discrete: output 96 logits (32 * 3)
            total_logits = self.num_bins_per_dim * self.num_dims
            self.logit_net = nn.Dense(total_logits, kernel_init=default_init(self.final_fc_init_scale))
        else:
            # For single discrete: output action_dim logits
            self.logit_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))

    def __call__(
        self,
        observations,
        goals=None,
        goal_encoded=False,
        temperature=1.0,
        action_stack=None,
    ):
        """Return the action distribution.

        Args:
            observations: Observations.
            goals: Goals (optional).
            goal_encoded: Whether the goals are already encoded.
            temperature: Inverse scaling factor for the logits (set to 0 to get the argmax).
            action_stack: Optional past actions (B, L, A) or (B, L*A) concatenated to encoder output.
        """
        if self.gc_encoder is not None:
            inputs = self.gc_encoder(observations, goals, goal_encoded=goal_encoded)
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
            inputs = jnp.concatenate(inputs, axis=-1)
        if action_stack is not None:
            aflat = action_stack.reshape(action_stack.shape[0], -1) if action_stack.ndim > 2 else action_stack
            inputs = jnp.concatenate([inputs, aflat], axis=-1)
        outputs = self.actor_net(inputs)

        logits = self.logit_net(outputs)
        
        # Clip logits to prevent extreme values that cause numerical instability
        logits = jnp.clip(logits, -10.0, 10.0)
        
        if self.multi_discrete:
            # Reshape logits to (batch_size, num_dims, num_bins_per_dim)
            logits = logits.reshape(-1, self.num_dims, self.num_bins_per_dim)
            
            # Create independent categorical distributions for each dimension
            distributions = []
            for i in range(self.num_dims):
                dist = distrax.Categorical(logits=logits[:, i, :] / jnp.maximum(1e-6, temperature))
                distributions.append(dist)
            
            return MultiDiscreteDistribution(distributions, num_bins=self.num_bins_per_dim)
        else:
            # Single discrete distribution
            distribution = distrax.Categorical(logits=logits / jnp.maximum(1e-6, temperature))
            return distribution


class MultiDiscreteDistribution:
    """3D action distribution backed by 3 independent categoricals over bins.

    Fixed to CARLA-style actions ``[throttle, steer, brake]``:
      - dim 0 (throttle): continuous in [0, 1], bins = linspace(0, 1, num_bins)
      - dim 1 (steer):    continuous in [-1, 1], bins = linspace(-1, 1, num_bins)
      - dim 2 (brake):    continuous in [0, 1], bins = linspace(0, 1, num_bins)

    External API is continuous (callers don't need to know about bins):
      - ``mode()``  -> (B, 3) bin-center of argmax per dim
      - ``sample(seed)`` -> (B, 3) bin-center of sampled index per dim
      - ``mean()``  -> (B, 3) softmax-weighted average of bin centers
      - ``log_prob(actions)`` accepts continuous (B, 3) (or (B, 3*chunk_len)
        with chunk_len==1) and internally bins them before summing the
        per-dim Categorical log-probs. This lets the rest of the pipeline
        keep treating actions as float32 while the head is categorical.
    """

    # CARLA action ranges; the head only ever outputs (throttle, steer, brake).
    _LO = (0.0, -1.0, 0.0)
    _HI = (1.0, 1.0, 1.0)

    def __init__(self, distributions, num_bins=32):
        self.distributions = distributions
        self.num_dims = len(distributions)
        self.num_bins = int(num_bins)
        # Per-dim bin centers stacked along a leading "dim" axis.
        self._centers = jnp.stack(
            [
                jnp.linspace(self._LO[i], self._HI[i], self.num_bins)
                for i in range(self.num_dims)
            ],
            axis=0,
        )  # (num_dims, num_bins)
        self._lo = jnp.asarray(self._LO[: self.num_dims])
        self._hi = jnp.asarray(self._HI[: self.num_dims])

    def _to_bin_indices(self, actions):
        """(B, num_dims) continuous -> (B, num_dims) int32 bin indices."""
        clipped = jnp.clip(actions, self._lo, self._hi)
        normalized = (clipped - self._lo) / (self._hi - self._lo)  # [0, 1]
        idx = jnp.round(normalized * (self.num_bins - 1)).astype(jnp.int32)
        return jnp.clip(idx, 0, self.num_bins - 1)

    def _indices_to_centers(self, indices):
        """(B, num_dims) int -> (B, num_dims) continuous bin-center values."""
        centers_b = jnp.broadcast_to(
            self._centers[None, :, :], (indices.shape[0], self.num_dims, self.num_bins)
        )
        return jnp.take_along_axis(centers_b, indices[:, :, None], axis=-1).squeeze(-1)

    def sample(self, seed=None):
        if seed is not None:
            seeds = jax.random.split(seed, self.num_dims)
        else:
            seeds = [None] * self.num_dims
        indices = jnp.stack(
            [self.distributions[i].sample(seed=seeds[i]) for i in range(self.num_dims)],
            axis=-1,
        )
        return self._indices_to_centers(indices)

    def mode(self):
        indices = jnp.stack([d.mode() for d in self.distributions], axis=-1)
        return self._indices_to_centers(indices)

    def log_prob(self, actions):
        """Log-prob of continuous (B, num_dims) actions, summed across dims.

        A chunk-flattened (B, chunk_len*num_dims) input is tolerated with
        chunk_len==1 (i.e. trailing dim already == num_dims). Longer chunks
        aren't supported by this head; GCBC should enforce chunk_len==1.
        """
        if actions.ndim == 2 and actions.shape[-1] != self.num_dims:
            actions = actions[:, : self.num_dims]
        idx = self._to_bin_indices(actions)  # (B, num_dims) int32
        lps = jnp.stack(
            [self.distributions[d].log_prob(idx[:, d]) for d in range(self.num_dims)],
            axis=-1,
        )
        return jnp.sum(lps, axis=-1)

    def mean(self):
        """Softmax-weighted average of bin centers per dim; continuous (B, num_dims)."""
        probs = jnp.stack(
            [jax.nn.softmax(d.logits, axis=-1) for d in self.distributions], axis=1
        )  # (B, num_dims, num_bins)
        return jnp.sum(probs * self._centers[None, :, :], axis=-1)

    def entropy(self):
        ents = jnp.stack([d.entropy() for d in self.distributions], axis=-1)
        return jnp.sum(ents, axis=-1)

    @property
    def logits(self):
        return jnp.stack([dist.logits for dist in self.distributions], axis=1)


class GCValue(nn.Module):
    """Goal-conditioned value/critic function.

    This module can be used for both value V(s, g) and critic Q(s, a, g) functions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        layer_norm: Whether to apply layer normalization.
        ensemble: Whether to ensemble the value function.
        value_exp: Whether to exponentiate the value. Useful for contrastive learning.
        gc_encoder: Optional GCEncoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    layer_norm: bool = True
    ensemble: bool = True
    value_exp: bool = False
    gc_encoder: nn.Module = None

    def setup(self):
        mlp_module = MLP
        if self.ensemble:
            mlp_module = ensemblize(mlp_module, 2)
        value_net = mlp_module((*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm)

        self.value_net = value_net

    def __call__(self, observations, goals=None, actions=None):
        """Return the value/critic function.

        Args:
            observations: Observations.
            goals: Goals (optional).
            actions: Actions (optional).
        """
        if self.gc_encoder is not None:
            inputs = [self.gc_encoder(observations, goals)]
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        v = self.value_net(inputs).squeeze(-1)

        if self.value_exp:
            v = jnp.exp(v)

        return v


class GCDiscreteCritic(GCValue):
    """Goal-conditioned critic for discrete actions."""

    action_dim: int = None

    def __call__(self, observations, goals=None, actions=None):
        actions = jnp.eye(self.action_dim)[actions]
        return super().__call__(observations, goals, actions)


class GCBilinearValue(nn.Module):
    """Goal-conditioned bilinear value/critic function.

    This module computes the value function as V(s, g) = phi(s)^T psi(g) / sqrt(d) or the critic function as
    Q(s, a, g) = phi(s, a)^T psi(g) / sqrt(d), where phi and psi output d-dimensional vectors.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        latent_dim: Latent dimension.
        layer_norm: Whether to apply layer normalization.
        ensemble: Whether to ensemble the value function.
        value_exp: Whether to exponentiate the value. Useful for contrastive learning.
        state_encoder: Optional state encoder.
        goal_encoder: Optional goal encoder.
    """

    hidden_dims: Sequence[int]
    latent_dim: int
    layer_norm: bool = True
    ensemble: bool = True
    value_exp: bool = False
    state_encoder: nn.Module = None
    goal_encoder: nn.Module = None

    def setup(self) -> None:
        mlp_module = MLP
        if self.ensemble:
            mlp_module = ensemblize(mlp_module, 2)
        #print(*self.hidden_dims, self.latent_dim)
        self.phi = mlp_module((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)
        self.psi = mlp_module((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)

    def __call__(self, observations, goals, actions=None, info=False):
        """Return the value/critic function.

        Args:
            observations: Observations.
            goals: Goals.
            actions: Actions (optional).
            info: Whether to additionally return the representations phi and psi.
        """
        if self.state_encoder is not None:
            observations = self.state_encoder(observations)
            # Clip encoder outputs to prevent numerical instability in cuDNN
            observations = jnp.clip(observations, -50.0, 50.0)
        if self.goal_encoder is not None:
            goals = self.goal_encoder(goals)
            # Clip encoder outputs to prevent numerical instability in cuDNN
            goals = jnp.clip(goals, -50.0, 50.0)

        if actions is None:
            phi_inputs = observations
        else:
            phi_inputs = jnp.concatenate([observations, actions], axis=-1)

        phi = self.phi(phi_inputs)
        psi = self.psi(goals)

        # L2 normalize embeddings so that v and contrastive logits share a scale.
        phi = phi / (jnp.linalg.norm(phi, axis=-1, keepdims=True) + 1e-8)
        psi = psi / (jnp.linalg.norm(psi, axis=-1, keepdims=True) + 1e-8)

        # Temperature-scaled bilinear value.
        temperature = 0.1
        v = (phi * psi).sum(axis=-1) / temperature

        if self.value_exp:
            v = jnp.exp(v)

        if info:
            return v, phi, psi
        else:
            return v


class GCDiscreteBilinearCritic(GCBilinearValue):
    """Goal-conditioned bilinear critic for discrete actions."""

    action_dim: int = None

    def __call__(self, observations, goals=None, actions=None, info=False):
        actions = jnp.eye(self.action_dim)[actions]
        return super().__call__(observations, goals, actions, info)


class GoalDistanceHead(nn.Module):
    """Predicts temporal distance between state and goal embeddings."""

    latent_dim: int
    hidden_dims: Sequence[int] = (256, 256)
    layer_norm: bool = False

    @nn.compact
    def __call__(self, phi, psi):
        """Inputs are (batch, latent_dim)."""
        features = [
            phi,
            psi,
            jnp.abs(phi - psi),
            phi * psi,
        ]
        x = jnp.concatenate(features, axis=-1)
        x = MLP((*self.hidden_dims, 1), layer_norm=self.layer_norm)(x)
        return x.squeeze(-1)


class GCMRNValue(nn.Module):
    """Metric residual network (MRN) value function.

    This module computes the value function as the sum of a symmetric Euclidean distance and an asymmetric
    L^infinity-based quasimetric.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        latent_dim: Latent dimension.
        layer_norm: Whether to apply layer normalization.
        value_exp: Whether to exponentiate the value.
        encoder: Optional state/goal encoder.
    """

    hidden_dims: Sequence[int]
    latent_dim: int
    layer_norm: bool = True
    value_exp: bool = False
    encoder: nn.Module = None

    def setup(self) -> None:
        self.phi = MLP((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)

    def __call__(self, observations, goals, is_phi=False, info=False):
        """Return the MRN value function.

        Args:
            observations: Observations.
            goals: Goals.
            is_phi: Whether the inputs are already encoded by phi.
            info: Whether to additionally return the representations phi_s and phi_g.
        """
        if is_phi:
            phi_s = observations
            phi_g = goals
        else:
            if self.encoder is not None:
                observations = self.encoder(observations)
                goals = self.encoder(goals)
            phi_s = self.phi(observations)
            phi_g = self.phi(goals)

        sym_s = phi_s[..., : self.latent_dim // 2]
        sym_g = phi_g[..., : self.latent_dim // 2]
        asym_s = phi_s[..., self.latent_dim // 2 :]
        asym_g = phi_g[..., self.latent_dim // 2 :]
        squared_dist = ((sym_s - sym_g) ** 2).sum(axis=-1)
        quasi = jax.nn.relu((asym_s - asym_g).max(axis=-1))
        v = jnp.sqrt(jnp.maximum(squared_dist, 1e-12)) + quasi

        if self.value_exp:
            v = jnp.exp(v)

        if info:
            return v, phi_s, phi_g
        else:
            return v


class GCIQEValue(nn.Module):
    """Interval quasimetric embedding (IQE) value function.

    This module computes the value function as an IQE-based quasimetric.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        latent_dim: Latent dimension.
        dim_per_component: Dimension of each component in IQE (i.e., number of intervals in each group).
        layer_norm: Whether to apply layer normalization.
        value_exp: Whether to exponentiate the value.
        encoder: Optional state/goal encoder.
    """

    hidden_dims: Sequence[int]
    latent_dim: int
    dim_per_component: int
    layer_norm: bool = True
    value_exp: bool = False
    encoder: nn.Module = None

    def setup(self) -> None:
        self.phi = MLP((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)
        self.alpha = Param()

    def __call__(self, observations, goals, is_phi=False, info=False):
        """Return the IQE value function.

        Args:
            observations: Observations.
            goals: Goals.
            is_phi: Whether the inputs are already encoded by phi.
            info: Whether to additionally return the representations phi_s and phi_g.
        """
        alpha = jax.nn.sigmoid(self.alpha())
        if is_phi:
            phi_s = observations
            phi_g = goals
        else:
            if self.encoder is not None:
                observations = self.encoder(observations)
                goals = self.encoder(goals)
            phi_s = self.phi(observations)
            phi_g = self.phi(goals)

        x = jnp.reshape(phi_s, (*phi_s.shape[:-1], -1, self.dim_per_component))
        y = jnp.reshape(phi_g, (*phi_g.shape[:-1], -1, self.dim_per_component))
        valid = x < y
        xy = jnp.concatenate(jnp.broadcast_arrays(x, y), axis=-1)
        ixy = xy.argsort(axis=-1)
        sxy = jnp.take_along_axis(xy, ixy, axis=-1)
        neg_inc_copies = jnp.take_along_axis(valid, ixy % self.dim_per_component, axis=-1) * jnp.where(
            ixy < self.dim_per_component, -1, 1
        )
        neg_inp_copies = jnp.cumsum(neg_inc_copies, axis=-1)
        neg_f = -1.0 * (neg_inp_copies < 0)
        neg_incf = jnp.concatenate([neg_f[..., :1], neg_f[..., 1:] - neg_f[..., :-1]], axis=-1)
        components = (sxy * neg_incf).sum(axis=-1)
        v = alpha * components.mean(axis=-1) + (1 - alpha) * components.max(axis=-1)

        if self.value_exp:
            v = jnp.exp(v)

        if info:
            return v, phi_s, phi_g
        else:
            return v

class StateRepresentation(nn.Module):
    """State representation module.
    Attributes:
        hidden_dims: Hidden layer dimensions.
        latent_dim: Latent dimension.
        layer_norm: Whether to apply layer normalization.
        ensemble: Whether to ensemble the value function.
        value_exp: Whether to exponentiate the value. Useful for contrastive learning.
        state_encoder: Optional state encoder.
    """

    hidden_dims: Sequence[int]
    latent_dim: int
    layer_norm: bool = True
    ensemble: bool = True
    value_exp: bool = False
    state_encoder: nn.Module = None

    def setup(self) -> None:
        mlp_module = MLP
        if self.ensemble:
            mlp_module = ensemblize(mlp_module, 2)
        self.phi = mlp_module((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)

    def __call__(self, observations, actions=None, info=False):
        """Return the value/critic function.

        Args:
            observations: Observations.
            goals: Goals.
            actions: Actions (optional).
        """
        if self.state_encoder is not None:
            observations = self.state_encoder(observations)

        if actions is None:
            phi_inputs = observations
        else:
            phi_inputs = jnp.concatenate([observations, actions], axis=-1)

        phi = self.phi(phi_inputs)

        return phi


class DiscreteStateActionRepresentation(StateRepresentation):
    """State representation module for discrete actions."""

    action_dim: int = None

    def __call__(self, observations, actions=None, info=False):
        if self.encoder is not None:
            observations = self.encoder(observations)

        if actions is not None:
            actions = jnp.eye(self.action_dim)[actions]

        return super().__call__(observations, actions, info)


class GoalClassifier(nn.Module):
    """Goal-conditioned success classifier C(s, a, g).
    
    This classifier predicts whether action a from state s will reach goal g within
    a certain distance threshold. Shares the state encoder with the critic for
    efficient feature reuse.
    
    Attributes:
        hidden_dims: Hidden layer dimensions.
        layer_norm: Whether to apply layer normalization.
        state_encoder: Optional state encoder (shared with critic).
        action_dim: Action dimension (for discrete actions, to one-hot encode).
        discrete: Whether actions are discrete.
    """
    
    hidden_dims: Sequence[int]
    layer_norm: bool = True
    state_encoder: nn.Module = None
    action_dim: Optional[int] = None
    discrete: bool = False
    
    def setup(self):
        # Trunk: shared state encoder (if provided)
        # Head: MLP that takes [encoded_state, action, encoded_goal] -> logit
        self.classifier_head = MLP(
            (*self.hidden_dims, 1),
            activate_final=False,
            layer_norm=self.layer_norm,
        )
    
    def __call__(self, observations, actions, goals, goal_encoded=False):
        """Return success logit C(s, a, g).
        
        Args:
            observations: Current states s.
            actions: Actions a (continuous or discrete integers).
            goals: Goals g.
            goal_encoded: Whether goals are already encoded (not used, for compatibility).
        """
        # Encode states and goals using shared encoder
        if self.state_encoder is not None:
            encoded_states = self.state_encoder(observations)
            encoded_goals = self.state_encoder(goals)
        else:
            encoded_states = observations
            encoded_goals = goals
        
        # Handle discrete actions: convert to one-hot
        if self.discrete and self.action_dim is not None:
            # actions are integers, convert to one-hot
            actions = jnp.eye(self.action_dim)[actions]
        
        # Concatenate [encoded_state, action, encoded_goal]
        inputs = jnp.concatenate([encoded_states, actions, encoded_goals], axis=-1)
        
        # Output logit
        logit = self.classifier_head(inputs).squeeze(-1)
        return logit


def build_gc_actor_init(ex_observations, ex_goals, ex_action_stack=None):
    """Keyword args for :class:`GCActor` / :class:`GCDiscreteActor` ``init`` (``ModuleDict`` mapping)."""
    d = dict(observations=ex_observations, goals=ex_goals)
    if ex_action_stack is not None:
        d["action_stack"] = ex_action_stack
    return d


def actor_action_stack_kwargs_from_batch(config, batch):
    """Pass ``action_stack`` to the actor when ``action_stack_length`` > 1."""
    L = int(config.get("action_stack_length", 1) or 1)
    if L <= 1:
        return {}
    ast = batch.get("action_stack")
    if ast is None:
        return {}
    return {"action_stack": ast}


def actor_action_stack_kwargs_from_value(config, action_stack):
    """For ``sample_actions`` / inference when action history is provided explicitly."""
    L = int(config.get("action_stack_length", 1) or 1)
    if L <= 1 or action_stack is None:
        return {}
    return {"action_stack": action_stack}
