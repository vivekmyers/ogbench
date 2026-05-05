import dataclasses
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict


def get_size(data):
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


@partial(jax.jit, static_argnames=('padding',))
def random_crop(img, crop_from, padding):
    padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode='edge')
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


@partial(jax.jit, static_argnames=('padding',))
def batched_random_crop(imgs, crop_froms, padding):
    return jax.vmap(random_crop, (0, 0, None))(imgs, crop_froms, padding)


class Dataset(FrozenDict):
    @classmethod
    def create(cls, freeze=True, **fields):
        data = fields
        assert 'observations' in data
        if freeze:
            def _setflags_if_numpy(arr):
                if isinstance(arr, np.ndarray):
                    arr.setflags(write=False)
            jax.tree_util.tree_map(_setflags_if_numpy, data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        if 'valids' in self._dict:
            (self.valid_idxs,) = np.nonzero(self['valids'] > 0)

    def get_random_idxs(self, num_idxs):
        if 'valids' in self._dict:
            return self.valid_idxs[np.random.randint(len(self.valid_idxs), size=num_idxs)]
        else:
            return np.random.randint(self.size, size=num_idxs)

    def sample(self, batch_size, idxs=None):
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        return self.get_subset(idxs)

    def get_subset(self, idxs):
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if 'next_observations' not in result:
            result['next_observations'] = self._dict['observations'][np.minimum(idxs + 1, self.size - 1)]
        return result


class ReplayBuffer(Dataset):
    @classmethod
    def create(cls, transition, size):
        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)
        buffer_dict = jax.tree_util.tree_map(create_buffer, transition)
        return cls(buffer_dict)

    @classmethod
    def create_from_initial_dataset(cls, init_dataset, size):
        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[: len(init_buffer)] = init_buffer
            return buffer
        buffer_dict = jax.tree_util.tree_map(create_buffer, init_dataset)
        dataset = cls(buffer_dict)
        dataset.size = dataset.pointer = get_size(init_dataset)
        return dataset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_size = get_size(self._dict)
        self.size = 0
        self.pointer = 0

    def add_transition(self, transition):
        def set_idx(buffer, new_element):
            buffer[self.pointer] = new_element
        jax.tree_util.tree_map(set_idx, self._dict, transition)
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = max(self.pointer, self.size)

    def clear(self):
        self.size = self.pointer = 0


@dataclasses.dataclass
class GCDataset:
    dataset: Dataset
    config: Any
    preprocess_frame_stack: bool = True
    # Train split uses `upsample_mode` + optional longitudinal rebalancing to build
    # per-frame importance weights. Val split should *never* resample/upsample
    # frames, even if a code path calls `sample(..., evaluation=False)` by
    # mistake, so it sets `sampling_mode="val"` to disable weight construction.
    sampling_mode: str = "train"

    def __post_init__(self):
        self.size = self.dataset.size

        (self.terminal_locs,) = np.nonzero(self.dataset['terminals'] > 0)
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])
        assert self.terminal_locs[-1] == self.size - 1

        # --- FIX 1: Pre-compute per-index trajectory lookups ---
        # O(1) lookup instead of searchsorted every sample call
        self.idx_to_terminal = np.empty(self.size, dtype=np.int64)
        self.idx_to_initial = np.empty(self.size, dtype=np.int64)
        for i, (start, end) in enumerate(zip(self.initial_locs, self.terminal_locs)):
            self.idx_to_terminal[start:end + 1] = end
            self.idx_to_initial[start:end + 1] = start

        assert np.isclose(
            self.config['value_p_curgoal'] + self.config['value_p_trajgoal'] + self.config['value_p_randomgoal'], 1.0
        )
        assert np.isclose(
            self.config['actor_p_curgoal'] + self.config['actor_p_trajgoal'] + self.config['actor_p_randomgoal'], 1.0
        )
        if self.config.get('frame_stack') is not None:
            K = int(self.config['frame_stack'])
            W_cfg = self.config.get('frame_stack_window')
            W = int(W_cfg) if W_cfg else K
            if W < K:
                raise ValueError(
                    f"frame_stack_window ({W}) must be >= frame_stack ({K})."
                )
            self._frame_stack_window = W
            custom_offsets = self.config.get('frame_offsets')
            self._frame_stack_random = (W > K) and (custom_offsets is None)

        else:
            self._frame_stack_window = 1
            self._frame_stack_random = False
        self._raw_obs_view = None
        self._raw_obs_single_frame = None
        if self.config['frame_stack'] is not None:
            assert 'next_observations' not in self.dataset
            if self.preprocess_frame_stack:
                K = int(self.config['frame_stack'])
                raw_obs = self.dataset['observations']
                if self._needs_separate_goal_temporal_stack():
                    self._raw_obs_single_frame = raw_obs
                stacked_observations = self._preprocess_frame_stack()
                self.dataset = Dataset(self.dataset.copy(dict(observations=stacked_observations)))
                if isinstance(stacked_observations, np.ndarray) and stacked_observations.ndim >= 4:
                    C = int(raw_obs.shape[-1])
                    self._raw_obs_view = stacked_observations.reshape(
                        stacked_observations.shape[0],
                        *stacked_observations.shape[1:-1],
                        K, C,
                    )[..., -1, :]

        elif self._needs_separate_goal_temporal_stack():
            self._raw_obs_single_frame = self.dataset['observations']

        if str(self.sampling_mode).lower() in ("val", "valid", "eval", "validation"):
            self._sampling_weights = None
        else:
            self._sampling_weights = self._build_sampling_weights()

        gk, go = self._resolved_goal_stack_spec()
        if gk is not None:
            print(
                f"[GCDataset] actor goal stack: K={gk} offsets={go} "
                f"({'raw temporal gather (extra RAM)' if self._needs_separate_goal_temporal_stack() else 'fast path = same as observations'})"
            )

    @staticmethod
    def _normalize_frame_offsets(k: int, offsets) -> tuple[int, ...]:
        """Return offsets sorted oldest-first (non-positive, includes 0)."""
        if offsets is not None:
            fo = tuple(int(x) for x in offsets)
            if len(fo) != k:
                raise ValueError(f"frame_offsets has {len(fo)} entries but stack depth is k={k}.")
            if any(o > 0 for o in fo):
                raise ValueError(f"frame_offsets must all be <= 0, got {fo}.")
            if 0 not in fo:
                raise ValueError(f"frame_offsets must include 0 (current frame), got {fo}.")
            return tuple(sorted(fo))
        if k == 1:
            return (0,)
        return tuple(range(-(k - 1), 1))

    def _resolved_obs_stack_spec(self) -> tuple[int | None, tuple[int, ...] | None]:
        k = self.config.get('frame_stack')
        if k is None:
            return None, None
        k = int(k)
        fo = self.config.get('frame_offsets')
        return k, self._normalize_frame_offsets(k, fo)

    def _resolved_goal_stack_spec(self) -> tuple[int | None, tuple[int, ...] | None]:
        """Explicit ``goal_*`` overrides obs stacking; omitting both mirrors obs (train.py default)."""
        gk_raw = self.config.get('goal_frame_stack')
        go_raw = self.config.get('goal_frame_offsets')
        if gk_raw is None and go_raw is None:
            # Legacy checkpoints / hand configs with no goal_* keys behave like train defaults:
            # same temporal pattern as observations.
            return self._resolved_obs_stack_spec()
        if gk_raw is None:
            raise ValueError(
                "goal_frame_offsets is set but goal_frame_stack is missing; "
                "set goal_frame_stack or omit both keys to mirror observations."
            )
        gk = int(gk_raw)
        return gk, self._normalize_frame_offsets(gk, go_raw)

    def _needs_separate_goal_temporal_stack(self) -> bool:
        """True iff actor goals must be gathered from raw frames (not the pre-stacked tensor)."""
        gk, go = self._resolved_goal_stack_spec()
        if gk is None:
            return False
        ok, oo = self._resolved_obs_stack_spec()
        return (gk, go) != (ok, oo)

    def _goal_stack_raw_buffer(self) -> np.ndarray:
        """(N,H,W,C) single-frame observations used to build temporal goal stacks."""
        if getattr(self, '_raw_obs_single_frame', None) is not None:
            return self._raw_obs_single_frame
        obs = self.dataset['observations']
        c = int(self.config.get('obs_c', obs.shape[-1]))
        if obs.shape[-1] == c:
            return obs
        raise RuntimeError(
            "goal_frame_stack requires raw (H,W,C) observations, but the dataset "
            "observations look pre-stacked (last dim != obs_c). "
            "This is an internal error: file a bug with your frame_stack / goal_frame_stack config."
        )

    def _gather_goal_temporal_stack(self, goal_idxs: np.ndarray) -> np.ndarray:
        """Stack ``goal_frame_stack`` raw frames around each goal index (trajectory-clamped)."""
        _, go = self._resolved_goal_stack_spec()
        assert go is not None
        raw = self._goal_stack_raw_buffer()
        # Same lag order as ``_preprocess_frame_stack`` / ``get_stacked_observations``:
        # oldest frame first along the channel axis.
        lags = tuple(sorted((-int(o) for o in go), reverse=True))
        b = len(goal_idxs)
        inits = self.idx_to_initial[goal_idxs]
        parts = []
        for lag in lags:
            cur = np.maximum(goal_idxs - lag, inits)
            parts.append(raw[cur])
        return np.concatenate(parts, axis=-1)

    def _compute_speed_from_position(self) -> np.ndarray | None:
        """Per-frame ego speed (m/s) derived from `position`, clipped at traj boundaries.

        Uses XY-plane finite differences only (Z is noisy on bumps/spawn). Speed at
        a trajectory's terminal frame is copied from the previous step so every
        index has a valid value.
        """
        if 'position' not in self.dataset._dict:
            return None
        pos = np.asarray(self.dataset['position'], dtype=np.float64)
        if pos.ndim != 2 or pos.shape[0] != self.size or pos.shape[1] < 2:
            return None

        fps = float(self.config.get('lb_train_fps', 10.0))
        dx = np.zeros(self.size, dtype=np.float64)
        dy = np.zeros(self.size, dtype=np.float64)
        dx[:-1] = pos[1:, 0] - pos[:-1, 0]
        dy[:-1] = pos[1:, 1] - pos[:-1, 1]
        speed = np.sqrt(dx * dx + dy * dy) * fps

        for start, end in zip(self.initial_locs, self.terminal_locs):
            s, e = int(start), int(end)
            if e > s:
                speed[e] = speed[e - 1]
            else:
                speed[e] = 0.0
        return speed

    def _build_sampling_weights(self):
        """Precompute per-frame sampling weights for action upsampling.

        Two independent knobs:
          1. ``upsample_mode``: a single categorical mode that selects a set of
             "target" frames and boosts them by ``upsample_weight`` (legacy).
          2. ``longitudinal_balance``: when True, applies *multiplicative*
             longitudinal re-weighting on top of (1) to rebalance
             throttle/brake ambiguity. Uses speed derived from ``position``:
             - stopped_with_brake (speed<lb_stopped_speed & brake>lb_brake_high)
               → weight ×lb_w_stopped_brake (downsample, e.g. 0.07)
             - accelerating_from_stop (speed<lb_accel_speed & throttle>lb_thr_hi
               & brake<lb_brake_low) → weight ×lb_w_accel_from_stop (e.g. 8x)
             - mixed_longitudinal (throttle>lb_mix_thr & brake>lb_mix_brk)
               → weight ×lb_w_mixed (downsample)
             - decisive_decel (brake>lb_brake_high & speed>lb_decel_speed)
               → weight ×lb_w_decisive_decel (upsample modestly)
        """
        mode = self.config.get('upsample_mode', 'none')
        lb_on = bool(self.config.get('longitudinal_balance', False))
        if 'actions' not in self.dataset._dict:
            return None
        if mode == 'none' and not lb_on:
            return None

        actions = self.dataset['actions']
        steer = actions[:, 1]
        throttle = actions[:, 0]
        brake = actions[:, 2]

        steer_thresh = self.config.get('steer_thresh', 0.05)
        throttle_thresh = self.config.get('throttle_thresh', 0.3)
        brake_thresh = self.config.get('brake_thresh', 0.1)
        w = float(self.config.get('upsample_weight', 3.0))
        approach_h = int(self.config.get('turn_approach_horizon', 0) or 0)

        is_target = None
        if mode == 'turns_high':
            is_target = np.abs(steer) > steer_thresh
        elif mode == 'turns_high_approach':
            # Upsample frames with strong steering AND the H frames around them
            # (before + after; trajectory-clamped). This biases sampling toward
            # the throttle/brake setup and recovery around a turn, not just the
            # high-curvature frames themselves.
            turn = np.abs(steer) > steer_thresh
            if approach_h <= 0:
                is_target = turn
            else:
                is_target = turn.copy()
                # For each trajectory, mark frames within H steps of any turn
                # frame. We do this with a backward scan (distance to next turn)
                # and a forward scan (distance to previous turn), then OR.
                for start, end in zip(self.initial_locs, self.terminal_locs):
                    sl = slice(int(start), int(end) + 1)
                    dist = approach_h + 1  # >H means "not in window"
                    for i in range(int(end), int(start) - 1, -1):
                        dist = 0 if turn[i] else (dist + 1)
                        if dist <= approach_h:
                            is_target[i] = True

                    dist = approach_h + 1
                    for i in range(int(start), int(end) + 1):
                        dist = 0 if turn[i] else (dist + 1)
                        if dist <= approach_h:
                            is_target[i] = True
        elif mode == 'turns_low':
            is_target = np.abs(steer) < steer_thresh
        elif mode == 'throttle_high':
            is_target = throttle > throttle_thresh
        elif mode == 'throttle_low':
            is_target = throttle < throttle_thresh
        elif mode == 'brake_high':
            is_target = brake > brake_thresh
        elif mode == 'brake_low':
            is_target = brake < brake_thresh
        elif mode != 'none':
            return None

        # Base weights from mode (uniform if mode == 'none').
        if is_target is None:
            weights = np.ones(self.size, dtype=np.float64)
        else:
            weights = np.where(is_target, w, 1.0).astype(np.float64)
            n_target = int(np.sum(is_target))
            print(
                f"[GCDataset] upsample_mode={mode}: {n_target}/{self.size} "
                f"({100.0 * n_target / self.size:.1f}%) frames get weight {w}x"
            )

        # Multiplicative longitudinal rebalancing on top of base.
        if lb_on:
            speed = self._compute_speed_from_position()
            if speed is None:
                print(
                    "[GCDataset] longitudinal_balance=True but 'position' is "
                    "missing from the dataset; skipping longitudinal re-weighting."
                )
            else:
                stopped_speed = float(self.config.get('lb_stopped_speed_thresh', 0.5))
                accel_speed = float(self.config.get('lb_accel_speed_thresh', 2.0))
                decel_speed = float(self.config.get('lb_decel_speed_thresh', 2.0))
                brake_high = float(self.config.get('lb_brake_high', 0.3))
                brake_low = float(self.config.get('lb_brake_low', 0.05))
                throttle_high = float(self.config.get('lb_throttle_high', 0.2))
                mix_thr_lo = float(self.config.get('lb_mixed_thr_lo', 0.05))
                mix_brk_lo = float(self.config.get('lb_mixed_brk_lo', 0.05))

                w_stopped_brake = float(self.config.get('lb_w_stopped_brake', 0.07))
                w_accel_from_stop = float(self.config.get('lb_w_accel_from_stop', 8.0))
                w_mixed = float(self.config.get('lb_w_mixed', 0.1))
                w_decisive_decel = float(self.config.get('lb_w_decisive_decel', 2.5))

                stopped_brake_mask = (speed < stopped_speed) & (brake > brake_high)
                accel_from_stop_mask = (
                    (speed < accel_speed) & (throttle > throttle_high) & (brake < brake_low)
                )
                mixed_mask = (throttle > mix_thr_lo) & (brake > mix_brk_lo)
                decisive_decel_mask = (brake > brake_high) & (speed > decel_speed)

                weights = np.where(stopped_brake_mask, weights * w_stopped_brake, weights)
                weights = np.where(accel_from_stop_mask, weights * w_accel_from_stop, weights)
                weights = np.where(mixed_mask, weights * w_mixed, weights)
                weights = np.where(decisive_decel_mask, weights * w_decisive_decel, weights)

                n_stop = int(np.sum(stopped_brake_mask))
                n_accel = int(np.sum(accel_from_stop_mask))
                n_mix = int(np.sum(mixed_mask))
                n_dec = int(np.sum(decisive_decel_mask))
                pct = lambda n: 100.0 * n / max(self.size, 1)
                print(
                    f"[GCDataset] longitudinal_balance: "
                    f"stopped_with_brake={n_stop} ({pct(n_stop):.1f}%, ×{w_stopped_brake}); "
                    f"accel_from_stop={n_accel} ({pct(n_accel):.1f}%, ×{w_accel_from_stop}); "
                    f"mixed={n_mix} ({pct(n_mix):.1f}%, ×{w_mixed}); "
                    f"decisive_decel={n_dec} ({pct(n_dec):.1f}%, ×{w_decisive_decel}). "
                    f"speed: median={np.median(speed):.2f} m/s, "
                    f"p95={np.percentile(speed, 95):.2f} m/s, "
                    f"frac(speed<{stopped_speed})={np.mean(speed < stopped_speed):.2f}"
                )

        total = weights.sum()
        if total <= 0 or not np.isfinite(total):
            print("[GCDataset] sampling weights collapsed to zero; falling back to uniform.")
            return None
        weights /= total
        return weights

    def _action_stack_len(self) -> int:
        return int(self.config.get("action_stack_length", 1) or 1)

    def _gather_past_action_stack(self, idxs: np.ndarray, stack_len: int) -> np.ndarray:
        """Past ``stack_len`` actions before time ``idxs`` (oldest first), trajectory-clamped.

        Indices use ``[a_{t-L}, ..., a_{t-1}]`` with padding at segment starts by repeating
        the earliest available action in-segment (same convention as early-frame obs stacks).
        """
        initial = self.idx_to_initial[idxs]  # (B,)
        j = np.arange(stack_len, dtype=np.int64)[None, :]  # (1, L)
        raw = idxs[:, None] - stack_len + j  # (B, L) -> t-L ... t-1
        past_upper = np.maximum(idxs - 1, initial)
        src = np.clip(raw, initial[:, None], past_upper[:, None])
        return self.dataset["actions"][src]

    def sample(self, batch_size, idxs=None, evaluation=False):
        if idxs is None:
            if self._sampling_weights is not None and not evaluation:
                idxs = np.random.choice(self.size, batch_size, replace=True, p=self._sampling_weights)
            else:
                idxs = self.dataset.get_random_idxs(batch_size)
        minimal = bool(self.config.get('minimal_batch', False))
        p_rand = float(self.config.get('p_value_randomize_stack', 0.0) or 0.0)
        use_random_this_batch = (
            self._frame_stack_random
            and not evaluation
            and p_rand > 0.0
            and np.random.rand() < p_rand
        )
        batch = self.dataset.sample(batch_size, idxs)
        final_state_idxs = self.idx_to_terminal[idxs]
        if minimal:
            batch.pop('next_observations', None)

        if self.config['frame_stack'] is not None:
            batch['observations'] = self.get_observations(idxs, evaluation=evaluation, randomize=use_random_this_batch,)
            if not minimal:
                next_idxs = np.minimum(idxs + 1, self.size - 1)
                batch['next_observations'] = self.get_observations(next_idxs, evaluation=evaluation, randomize=use_random_this_batch,)

        actor_goal_idxs, actor_goal_sources = self.sample_goals(
            idxs,
            self.config['actor_p_curgoal'],
            self.config['actor_p_trajgoal'],
            self.config['actor_p_randomgoal'],
            self.config['actor_geom_sample'],
            return_sources=True,
        )
        if evaluation:
            batch['actor_goal_sources'] = actor_goal_sources

        if minimal:
            if self._needs_separate_goal_temporal_stack():
                batch['actor_goals'] = self._gather_goal_temporal_stack(actor_goal_idxs)
            else:
                all_goals = self._batch_get_observations(
                    actor_goal_idxs[None, :], evaluation=evaluation, randomize=False,
                )
                batch['actor_goals'] = jax.tree_util.tree_map(lambda x: x[0], all_goals)
            successes = np.zeros(len(idxs), dtype=np.float32)
        else:
            value_goal_idxs = self.sample_goals(
                idxs,
                self.config['value_p_curgoal'],
                self.config['value_p_trajgoal'],
                self.config['value_p_randomgoal'],
                self.config['value_geom_sample'],
            )
            value_goals_arr = self._batch_get_observations(
                value_goal_idxs[None, :], evaluation=evaluation, randomize=use_random_this_batch,
            )
            batch['value_goals'] = jax.tree_util.tree_map(lambda x: x[0], value_goals_arr)
            if self._needs_separate_goal_temporal_stack():
                batch['actor_goals'] = self._gather_goal_temporal_stack(actor_goal_idxs)
            else:
                actor_goals_arr = self._batch_get_observations(
                    actor_goal_idxs[None, :], evaluation=evaluation, randomize=False,
                )
                batch['actor_goals'] = jax.tree_util.tree_map(lambda x: x[0], actor_goals_arr)

            successes = (idxs == value_goal_idxs).astype(np.float32)

        batch['masks'] = 1.0 - successes
        batch['rewards'] = successes - (1.0 if self.config['gc_negative'] else 0.0)

        chunk_len = self.config.get('action_chunk_length', 1)
        if chunk_len > 1:
            offsets = np.arange(chunk_len)
            chunk_idxs = idxs[:, None] + offsets[None, :]
            chunk_idxs = np.minimum(chunk_idxs, final_state_idxs[:, None])
            batch['action_chunks'] = self.dataset['actions'][chunk_idxs]  # (B, chunk_len, action_dim)
        else:
            batch['action_chunks'] = batch['actions'][:, None, :]  # (B, 1, action_dim)

        L_as = self._action_stack_len()
        if L_as > 1:
            batch['action_stack'] = self._gather_past_action_stack(idxs, L_as)
            if not evaluation:
                self.apply_action_stack_noise(batch)

        if not minimal:
            chunk_next_idxs = np.minimum(idxs + chunk_len, final_state_idxs)
            batch['chunk_next_observations'] = self.get_observations(chunk_next_idxs, evaluation=evaluation, randomize=use_random_this_batch,)

        if self.config['p_aug'] is not None and not evaluation:
            if np.random.rand() < self.config['p_aug']:
                aug_keys = ['observations', 'actor_goals']
                if not minimal:
                    aug_keys.extend(['next_observations', 'value_goals', 'chunk_next_observations'])
                self.augment(batch, aug_keys)

        if not evaluation:
            self.apply_action_noise(batch)

        return batch

    def _batch_get_observations(self, idx_sets, evaluation=False, randomize=False):
        flat_idxs = idx_sets.ravel()
        if self.config['frame_stack'] is None:
            flat_obs = jax.tree_util.tree_map(lambda arr: arr[flat_idxs], self.dataset['observations'])
        elif randomize and self._frame_stack_random and not evaluation:
            flat_obs = self._random_stacked_observations(flat_idxs)
        elif self.preprocess_frame_stack:
            flat_obs = jax.tree_util.tree_map(lambda arr: arr[flat_idxs], self.dataset['observations'])
        else:
            flat_obs = self.get_stacked_observations(flat_idxs)
        num_sets, batch_size = idx_sets.shape
        return jax.tree_util.tree_map(lambda arr: arr.reshape(num_sets, batch_size, *arr.shape[1:]), flat_obs)

    # Goal-source tags used by logging (hard val frames) and any future
    # per-source metrics. 0=curgoal, 1=trajgoal, 2=randomgoal.
    GOAL_SOURCE_CUR = 0
    GOAL_SOURCE_TRAJ = 1
    GOAL_SOURCE_RANDOM = 2

    def sample_goals(self, idxs, p_curgoal, p_trajgoal, p_randomgoal, geom_sample, return_sources=False):
        batch_size = len(idxs)

        random_goal_idxs = self.dataset.get_random_idxs(batch_size)

        # --- FIX 1 continued: Use pre-computed lookup instead of searchsorted ---
        final_state_idxs = self.idx_to_terminal[idxs]

        if geom_sample:
            offsets = np.random.geometric(p=1 - self.config['discount'], size=batch_size)
            traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            distances = np.random.rand(batch_size)
            traj_goal_idxs = np.round(
                (np.minimum(idxs + 1, final_state_idxs) * distances + final_state_idxs * (1 - distances))
            ).astype(int)

        if p_curgoal == 1.0:
            goal_idxs = idxs
            if return_sources:
                sources = np.full(batch_size, self.GOAL_SOURCE_CUR, dtype=np.int8)
                return goal_idxs, sources
            return goal_idxs

        # Two-stage draw: first traj-vs-random, then overwrite with cur. Mirror
        # the exact same RNG pattern when building the source tags so sources
        # are consistent with the actual goal_idxs returned.
        traj_vs_rand_draw = np.random.rand(batch_size) < p_trajgoal / (1.0 - p_curgoal)
        goal_idxs = np.where(traj_vs_rand_draw, traj_goal_idxs, random_goal_idxs)
        cur_draw = np.random.rand(batch_size) < p_curgoal
        goal_idxs = np.where(cur_draw, idxs, goal_idxs)

        if return_sources:
            sources = np.where(
                traj_vs_rand_draw,
                np.int8(self.GOAL_SOURCE_TRAJ),
                np.int8(self.GOAL_SOURCE_RANDOM),
            )
            sources = np.where(cur_draw, np.int8(self.GOAL_SOURCE_CUR), sources).astype(np.int8)
            return goal_idxs, sources

        return goal_idxs

    def apply_action_noise(self, batch):
        if bool(self.config.get('multi_discrete', False)):
            return

        std_cfg = self.config.get('action_noise_std', 0.0)
        if isinstance(std_cfg, (list, tuple)):
            std_vec = np.asarray(std_cfg, dtype=np.float32)
        else:
            val = float(std_cfg or 0.0)
            if val <= 0.0:
                return
            std_vec = None  # broadcast scalar below; will materialise on use
            std_scalar = val
        if std_vec is not None and not np.any(std_vec > 0.0):
            return
        clip = bool(self.config.get('action_noise_clip', True))
        corr = float(self.config.get('action_noise_chunk_corr', 0.0) or 0.0)
        corr = max(0.0, min(corr, 0.999))  # keep strictly in [0, 0.999]

        def _scale_noise(unit_noise):
            """Multiply unit-variance noise by the (scalar or per-dim) std."""
            if std_vec is None:
                return unit_noise * std_scalar
            # unit_noise has trailing axis == action_dim
            return unit_noise * std_vec.reshape((1,) * (unit_noise.ndim - 1) + (-1,))

        def _ar1_chunk_noise(shape):
            """Draw AR(1)-correlated unit-variance noise with shape (..., T, A).

            ``T`` is the second-to-last axis.  If T == 1 this reduces to iid.
            """
            xi = np.random.randn(*shape).astype(np.float32)
            if corr <= 0.0 or shape[-2] <= 1:
                return xi
            innov = np.sqrt(1.0 - corr * corr)
            eps = np.empty_like(xi)
            eps[..., 0, :] = xi[..., 0, :]
            for t in range(1, shape[-2]):
                eps[..., t, :] = corr * eps[..., t - 1, :] + innov * xi[..., t, :]
            return eps

        if 'action_chunks' in batch:
            ac = batch['action_chunks']  # (B, C, A)
            noise = _scale_noise(_ar1_chunk_noise(ac.shape)).astype(ac.dtype, copy=False)
            ac = ac + noise
            if clip:
                ac = np.clip(ac, -1.0, 1.0)
            batch['action_chunks'] = ac
            # Preserve invariant actions == action_chunks[:, 0, :].
            batch['actions'] = ac[:, 0, :]
        elif 'actions' in batch:
            a = batch['actions']  # (B, A)
            noise = _scale_noise(np.random.randn(*a.shape).astype(np.float32)).astype(a.dtype, copy=False)
            a = a + noise
            if clip:
                a = np.clip(a, -1.0, 1.0)
            batch['actions'] = a

        if 'high_value_action_chunks' in batch:
            hva = batch['high_value_action_chunks']  # (B, H * A) flattened
            B = hva.shape[0]
            if std_vec is not None:
                A = std_vec.shape[0]
            elif 'action_chunks' in batch:
                A = batch['action_chunks'].shape[-1]
            else:
                A = batch['actions'].shape[-1]
            H = hva.shape[1] // A
            unshaped = hva.reshape(B, H, A)
            noise = _scale_noise(_ar1_chunk_noise(unshaped.shape)).astype(hva.dtype, copy=False)
            unshaped = unshaped + noise
            if clip:
                unshaped = np.clip(unshaped, -1.0, 1.0)
            batch['high_value_action_chunks'] = unshaped.reshape(B, H * A)

    def apply_action_stack_noise(self, batch):
        """Gaussian perturbation of the *input* action history (opt-in).

        Controlled by ``action_stack_noise_std`` (float, default 0.0). Noise is
        sampled i.i.d. as N(0, s) per element of ``batch['action_stack']``.
        """
        if bool(self.config.get('multi_discrete', False)):
            return
        std = float(self.config.get('action_stack_noise_std', 0.0) or 0.0)
        if std <= 0.0:
            return
        ast = batch.get('action_stack', None)
        if ast is None:
            return
        noise = (std * np.random.randn(*ast.shape)).astype(np.float32)
        ast_noised = ast + noise.astype(ast.dtype, copy=False)
        # If this looks like CARLA actions, clip each channel to its valid range.
        if ast_noised.ndim >= 2 and ast_noised.shape[-1] == 3:
            ast_noised[..., 0] = np.clip(ast_noised[..., 0], 0.0, 1.0)
            ast_noised[..., 1] = np.clip(ast_noised[..., 1], -1.0, 1.0)
            ast_noised[..., 2] = np.clip(ast_noised[..., 2], 0.0, 1.0)
        else:
            ast_noised = np.clip(ast_noised, -1.0, 1.0)
        batch['action_stack'] = ast_noised

    def augment(self, batch, keys):
        """Apply image augmentation — stay in JAX, no round-trip."""
        padding = 3
        batch_size = len(batch[keys[0]])
        crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
        crop_froms = np.concatenate([crop_froms, np.zeros((batch_size, 1), dtype=np.int64)], axis=1)
        # --- FIX 3: Convert crop_froms to jnp once, avoid per-key np.array() conversion ---
        crop_froms_jnp = jnp.array(crop_froms)
        for key in keys:
            batch[key] = jax.tree_util.tree_map(
                lambda arr: np.asarray(batched_random_crop(arr, crop_froms_jnp, padding))
                if arr.ndim == 4 else arr,
                batch[key],
            )

    def _preprocess_frame_stack(self):
        """Pre-stack all observations. Oldest-first along channels.

        Uses direct array slicing per trajectory instead of fancy indexing
        over the entire dataset, which avoids multiple full-array copies.
        """
        obs = self.dataset['observations']
        N = self.size
        k = int(self.config['frame_stack'])
        C = obs.shape[-1]
        spatial = obs.shape[1:-1]
        out = np.empty((N, *spatial, C * k), dtype=obs.dtype)

        # Resolve the lag for each channel block. Positive int >= 0: how many
        # dataset steps back from the current frame that slot sources from.
        custom_offsets = self.config.get('frame_offsets') if hasattr(self.config, 'get') else None
        if custom_offsets is not None:
            offs = tuple(int(o) for o in custom_offsets)
            if len(offs) != k:
                raise ValueError(
                    f"frame_offsets has {len(offs)} entries but frame_stack is {k}."
                )
            if any(o > 0 for o in offs):
                raise ValueError(f"frame_offsets must all be <= 0, got {offs}.")
            lags_oldest_first = sorted((-o for o in offs), reverse=True)  # e.g. (0,-20) -> [20, 0]
        else:
            lags_oldest_first = list(range(k - 1, -1, -1))  # [K-1, ..., 1, 0]

        for traj_start, traj_end in zip(self.initial_locs, self.terminal_locs):
            traj_len = traj_end - traj_start + 1
            traj_obs = obs[traj_start:traj_end + 1]          # (L, *spatial, C)
            for j, lag in enumerate(lags_oldest_first):
                ch_slice = slice(j * C, (j + 1) * C)
                # For each frame t in [0, traj_len), source = max(0, t - lag).
                # Frames [0, min(lag, traj_len)) clamp to traj_obs[0]; the rest
                # map 1:1 from traj_obs[:traj_len - lag].
                pad_len = min(lag, traj_len)
                body_start = lag
                if pad_len > 0:
                    out[traj_start:traj_start + pad_len, ..., ch_slice] = traj_obs[0:1]
                if body_start < traj_len:
                    out[traj_start + body_start:traj_end + 1, ..., ch_slice] = \
                        traj_obs[:traj_len - body_start]
        return out

    def get_observations(self, idxs, evaluation=False, randomize=None):
        if self.config['frame_stack'] is None:
            return jax.tree_util.tree_map(lambda arr: arr[idxs], self.dataset['observations'])
        if randomize is None:
            do_random = self._frame_stack_random and not evaluation
        else:
            do_random = bool(randomize) and self._frame_stack_random and not evaluation
        
        if do_random:
            return self._random_stacked_observations(idxs)
        if self.preprocess_frame_stack:
            return jax.tree_util.tree_map(lambda arr: arr[idxs], self.dataset['observations'])
        return self.get_stacked_observations(idxs)

    def get_stacked_observations(self, idxs):
        # --- FIX 1 continued: Use pre-computed lookup ---
        initial_state_idxs = self.idx_to_initial[idxs]
        rets = []
        for i in reversed(range(self.config['frame_stack'])):
            cur_idxs = np.maximum(idxs - i, initial_state_idxs)
            rets.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self.dataset['observations']))
        return jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *rets)

    def _random_stacked_observations(self, idxs):
        """Vectorised K-of-W random frame stack with offset 0 always included.

        For each element ``i`` we sample ``K - 1`` distinct offsets uniformly
        from ``[1, W - 1]`` (where ``W = frame_stack_window``) using the
        argpartition trick on per-row uniform keys, then append offset 0
        (the current frame) and sort descending so the channel order along
        the last axis is ``[oldest, ..., current]`` — matching
        :py:meth:`_preprocess_frame_stack` and :py:meth:`get_stacked_observations`.

        Source frame indices are clamped at ``idx_to_initial[idxs]`` so the
        stack never crosses a trajectory boundary (same convention as the
        canonical path).

        Implementation notes (kept in NumPy because the dataset lives on host):
          * One fancy index over a flat ``(B*K,)`` array followed by a
            ``transpose + reshape`` instead of K separate Python-loop
            indexings + ``np.concatenate``. This avoids ``K`` allocations
            per batch and keeps memory bandwidth O(K * frame_bytes) once.
          * The offset draw is O(B * W) and independent across batch
            elements, so different rows see different random subsets per
            forward pass — strongest setting for breaking causal-confusion
            shortcuts.
          * No JAX device transfer: the underlying observation array is on
            CPU as a NumPy buffer; gather happens in C inside NumPy. The
            downstream JAX-side path (encoder forward, augmentation) is
            untouched and still runs on device.
        """
        K = int(self.config['frame_stack'])
        W = int(self._frame_stack_window)
        B = len(idxs)

        if K == 1:
            offsets = np.zeros((B, 1), dtype=np.int64)
        else:
            # Sample K-1 distinct offsets from [1, W-1] per element via the
            # argpartition trick on uniform keys (vectorised, ~tens of microseconds
            # at B=1024, W<=24). Then prepend offset 0 and sort descending so
            # channels are oldest-first.
            keys = np.random.rand(B, W - 1)
            chosen = np.argpartition(keys, K - 1, axis=1)[:, : K - 1] + 1  # (B, K-1) in [1, W-1]
            offsets = np.concatenate(
                [chosen.astype(np.int64), np.zeros((B, 1), dtype=np.int64)], axis=1
            )  # (B, K)
            offsets = -np.sort(-offsets, axis=1)  # descending: oldest first

        initial = self.idx_to_initial[idxs]
        src = np.maximum(idxs[:, None] - offsets, initial[:, None])  # (B, K)

        # When pre-stacking is on (default with random stacking enabled), the raw
        # current-frame buffer is exposed as a strided view of the pre-stacked
        # array's last C channels. Same gather pattern, no extra storage.
        obs = self._raw_obs_view if self._raw_obs_view is not None else self.dataset['observations']
        gathered = obs[src.ravel()].reshape(B, K, *obs.shape[1:])

        if obs.ndim >= 4:
            spatial = obs.shape[1:-1]  # (H, W) for images, (H, W, D) for 3-D, etc.
            C = obs.shape[-1]
            # (B, K, *spatial, C) -> (B, *spatial, K, C) -> (B, *spatial, K*C)
            perm = (0,) + tuple(range(2, 2 + len(spatial))) + (1, 1 + 1 + len(spatial))
            return np.transpose(gathered, perm).reshape(B, *spatial, K * C)
        # 1-D / state-vector obs: stack along feature axis.
        return gathered.reshape(B, K * obs.shape[-1])


@dataclasses.dataclass
class CGCDataset(GCDataset):
    """Dataset for decoupled chunk-based goal-conditioned RL (TMD-DQC).

    Extends GCDataset with multi-step backup fields:
      - high_value_action_chunks: full backup_horizon-length action chunks
      - high_value_next_observations: observations backup_horizon steps ahead
      - high_value_goals / actor_goals: sampled goal observations
      - high_value_backup_horizon, high_value_masks, high_value_rewards
      - valids: per-timestep validity mask within the chunk
    """

    def __post_init__(self):
        super().__post_init__()
        backup_horizon = int(self.config['backup_horizon'])
        cur_idx = 0
        valid_idxs = []
        for terminal_idx in self.terminal_locs:
            valid_idxs.append(np.arange(cur_idx, terminal_idx + 1 - backup_horizon))
            cur_idx = terminal_idx + 1
        self.dataset.valid_idxs = np.concatenate(valid_idxs)

    def _compute_high_next_idxs(self, idxs, final_state_idxs, high_goal_idxs, backup_horizon):
        batch_size = len(idxs)
        bh = np.full(batch_size, backup_horizon)
        bh = np.minimum(bh, final_state_idxs - idxs)
        diff = high_goal_idxs - idxs
        should_clip = (0 <= diff) & (diff < bh)
        bh = np.where(should_clip, diff, bh)
        return idxs + bh, bh

    def sample(self, batch_size, idxs=None, evaluation=False):
        if idxs is None:
            if self._sampling_weights is not None and not evaluation:
                idxs = np.random.choice(self.size, batch_size, replace=True, p=self._sampling_weights)
            else:
                idxs = self.dataset.get_random_idxs(batch_size)

        p_rand = float(self.config.get("p_value_randomize_stack", 0.0) or 0.0)
        use_random_this_batch = (
            self._frame_stack_random
            and not evaluation
            and p_rand > 0.0
            and np.random.rand() < p_rand
        )

        batch = self.dataset.sample(batch_size, idxs)
        if self.config["frame_stack"] is not None:
            batch["observations"] = self.get_observations(
                idxs, evaluation=evaluation, randomize=use_random_this_batch,
            )
            next_idxs = np.minimum(idxs + 1, self.size - 1)
            batch["next_observations"] = self.get_observations(
                next_idxs, evaluation=evaluation, randomize=use_random_this_batch,
            )

        final_state_idxs = self.idx_to_terminal[idxs]
        backup_horizon = int(self.config["backup_horizon"])
        bsz = len(idxs)

        high_value_goal_idxs = self.sample_goals(
            idxs,
            self.config["value_p_curgoal"],
            self.config["value_p_trajgoal"],
            self.config["value_p_randomgoal"],
            self.config["value_geom_sample"],
        )
        actor_goal_idxs, actor_goal_sources = self.sample_goals(
            idxs,
            self.config["actor_p_curgoal"],
            self.config["actor_p_trajgoal"],
            self.config["actor_p_randomgoal"],
            self.config["actor_geom_sample"],
            return_sources=True,
        )
        if evaluation:
            batch["actor_goal_sources"] = actor_goal_sources

        high_value_next_idxs, high_value_bh = self._compute_high_next_idxs(
            idxs, final_state_idxs, high_value_goal_idxs, backup_horizon,
        )
        # Value / chunk critic ψ: same K-of-W policy as φ(obs); actor goals stay canonical.
        hv_stack = np.stack([high_value_goal_idxs, high_value_next_idxs])
        hv_obs = self._batch_get_observations(
            hv_stack, evaluation=evaluation, randomize=use_random_this_batch,
        )
        batch["high_value_goals"] = jax.tree_util.tree_map(lambda x: x[0], hv_obs)
        batch["value_goals"] = batch["high_value_goals"]
        batch["high_value_next_observations"] = jax.tree_util.tree_map(lambda x: x[1], hv_obs)
        if self._needs_separate_goal_temporal_stack():
            batch["actor_goals"] = self._gather_goal_temporal_stack(actor_goal_idxs)
        else:
            ag_only = self._batch_get_observations(
                actor_goal_idxs[None, :], evaluation=evaluation, randomize=False,
            )
            batch["actor_goals"] = jax.tree_util.tree_map(lambda x: x[0], ag_only)

        chunk_offsets = np.arange(backup_horizon)
        chunk_idxs = np.minimum(idxs[:, None] + chunk_offsets, final_state_idxs[:, None])
        batch["high_value_action_chunks"] = self.dataset["actions"][chunk_idxs].reshape(bsz, -1)
        batch['valids'] = (idxs[:, None] + chunk_offsets <= final_state_idxs[:, None]).astype(np.float32)

        high_value_successes = (high_value_bh < backup_horizon).astype(np.float32)
        batch['high_value_backup_horizon'] = high_value_bh
        batch['high_value_masks'] = 1.0 - high_value_successes
        batch['high_value_rewards'] = (self.config['discount'] ** high_value_bh) * high_value_successes

        successes = (idxs == high_value_goal_idxs).astype(np.float32)
        batch['masks'] = 1.0 - successes
        batch['rewards'] = successes - (1.0 if self.config['gc_negative'] else 0.0)

        chunk_len = self.config.get('action_chunk_length', 1)
        if chunk_len > 1:
            offsets = np.arange(chunk_len)
            actor_chunk_idxs = idxs[:, None] + offsets[None, :]
            actor_chunk_idxs = np.minimum(actor_chunk_idxs, final_state_idxs[:, None])
            batch['action_chunks'] = self.dataset['actions'][actor_chunk_idxs]
        else:
            batch['action_chunks'] = batch['actions'][:, None, :]

        L_as = self._action_stack_len()
        if L_as > 1:
            batch['action_stack'] = self._gather_past_action_stack(idxs, L_as)

        if self.config['p_aug'] is not None and not evaluation:
            if np.random.rand() < self.config['p_aug']:
                self.augment(batch, ['observations', 'next_observations',
                                     'high_value_goals', 'actor_goals',
                                     'high_value_next_observations'])

        if not evaluation:
            self.apply_action_noise(batch)

        return batch


@dataclasses.dataclass
class HGCDataset(GCDataset):

    def sample(self, batch_size, idxs=None, evaluation=False):
        if idxs is None:
            if self._sampling_weights is not None and not evaluation:
                idxs = np.random.choice(self.size, batch_size, replace=True, p=self._sampling_weights)
        else:
            idxs = self.dataset.get_random_idxs(batch_size)

        p_rand = float(self.config.get("p_value_randomize_stack", 0.0) or 0.0)
        use_random_this_batch = (
            self._frame_stack_random
            and not evaluation
            and p_rand > 0.0
            and np.random.rand() < p_rand
        )

        batch = self.dataset.sample(batch_size, idxs)
        if self.config['frame_stack'] is not None:
            batch['observations'] = self.get_observations(
                idxs, evaluation=evaluation, randomize=use_random_this_batch,
            )
            next_idxs = np.minimum(idxs + 1, self.size - 1)
            batch['next_observations'] = self.get_observations(
                next_idxs, evaluation=evaluation, randomize=use_random_this_batch,
            )

        value_goal_idxs = self.sample_goals(
            idxs,
            self.config['value_p_curgoal'],
            self.config['value_p_trajgoal'],
            self.config['value_p_randomgoal'],
            self.config['value_geom_sample'],
        )
        # Value ψ: match observation stacking (including K-of-W when enabled); eval stays canonical.
        batch['value_goals'] = self.get_observations(
            value_goal_idxs, evaluation=evaluation, randomize=use_random_this_batch,
        )

        successes = (idxs == value_goal_idxs).astype(np.float32)
        batch['masks'] = 1.0 - successes
        batch['rewards'] = successes - (1.0 if self.config['gc_negative'] else 0.0)

        # --- FIX 1 continued: Use pre-computed lookup ---
        final_state_idxs = self.idx_to_terminal[idxs]
        low_goal_idxs = np.minimum(idxs + self.config['subgoal_steps'], final_state_idxs)

        if self.config['actor_geom_sample']:
            offsets = np.random.geometric(p=1 - self.config['discount'], size=batch_size)
            high_traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            distances = np.random.rand(batch_size)
            high_traj_goal_idxs = np.round(
                (np.minimum(idxs + 1, final_state_idxs) * distances + final_state_idxs * (1 - distances))
            ).astype(int)
        high_traj_target_idxs = np.minimum(idxs + self.config['subgoal_steps'], high_traj_goal_idxs)

        high_random_goal_idxs = self.dataset.get_random_idxs(batch_size)
        high_random_target_idxs = np.minimum(idxs + self.config['subgoal_steps'], final_state_idxs)

        pick_random = np.random.rand(batch_size) < self.config['actor_p_randomgoal']
        high_goal_idxs = np.where(pick_random, high_random_goal_idxs, high_traj_goal_idxs)
        high_target_idxs = np.where(pick_random, high_random_target_idxs, high_traj_target_idxs)
        if evaluation:
            batch['actor_goal_sources'] = np.where(
                pick_random, np.int8(self.GOAL_SOURCE_RANDOM), np.int8(self.GOAL_SOURCE_TRAJ),
            ).astype(np.int8)

        # --- FIX 2: Batch all observation gathering ---
        # All three are goal-like targets for the high-/low-level actors -> canonical.
        all_idx_sets = np.stack([low_goal_idxs, high_goal_idxs, high_target_idxs])  # (3, batch_size)
        all_obs = self._batch_get_observations(all_idx_sets, evaluation=evaluation, randomize=False)
        batch['low_actor_goals'] = jax.tree_util.tree_map(lambda x: x[0], all_obs)
        batch['high_actor_goals'] = jax.tree_util.tree_map(lambda x: x[1], all_obs)
        batch['high_actor_targets'] = jax.tree_util.tree_map(lambda x: x[2], all_obs)

        chunk_len = self.config.get('action_chunk_length', 1)
        if chunk_len > 1:
            offsets = np.arange(chunk_len)
            chunk_idxs = idxs[:, None] + offsets[None, :]
            chunk_idxs = np.minimum(chunk_idxs, final_state_idxs[:, None])
            batch['action_chunks'] = self.dataset['actions'][chunk_idxs]
        else:
            batch['action_chunks'] = batch['actions'][:, None, :]

        chunk_next_idxs = np.minimum(idxs + chunk_len, final_state_idxs)
        batch['chunk_next_observations'] = self.get_observations(
            chunk_next_idxs, evaluation=evaluation, randomize=use_random_this_batch,
        )

        L_as = self._action_stack_len()
        if L_as > 1:
            batch['action_stack'] = self._gather_past_action_stack(idxs, L_as)

        if self.config['p_aug'] is not None and not evaluation:
            if np.random.rand() < self.config['p_aug']:
                self.augment(
                    batch,
                    ['observations', 'next_observations', 'value_goals', 'low_actor_goals',
                     'high_actor_goals', 'high_actor_targets', 'chunk_next_observations'],
                )

        if not evaluation:
            self.apply_action_noise(batch)

        return batch