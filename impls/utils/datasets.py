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

        if self.config['frame_stack'] is not None:
            assert 'next_observations' not in self.dataset
            if self.preprocess_frame_stack:
                stacked_observations = self._preprocess_frame_stack()
                self.dataset = Dataset(self.dataset.copy(dict(observations=stacked_observations)))

    def sample(self, batch_size, idxs=None, evaluation=False):
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        batch = self.dataset.sample(batch_size, idxs)
        if self.config['frame_stack'] is not None:
            batch['observations'] = self.get_observations(idxs)
            next_idxs = np.minimum(idxs + 1, self.size - 1)
            batch['next_observations'] = self.get_observations(next_idxs)

        value_goal_idxs = self.sample_goals(
            idxs,
            self.config['value_p_curgoal'],
            self.config['value_p_trajgoal'],
            self.config['value_p_randomgoal'],
            self.config['value_geom_sample'],
        )
        actor_goal_idxs = self.sample_goals(
            idxs,
            self.config['actor_p_curgoal'],
            self.config['actor_p_trajgoal'],
            self.config['actor_p_randomgoal'],
            self.config['actor_geom_sample'],
        )

        # --- FIX 2: Batch observation gathering ---
        # Single indexing pass for all goal observations instead of separate tree_maps
        all_goal_idxs = np.stack([value_goal_idxs, actor_goal_idxs])  # (2, batch_size)
        all_goals = self._batch_get_observations(all_goal_idxs)
        batch['value_goals'] = jax.tree_util.tree_map(lambda x: x[0], all_goals)
        batch['actor_goals'] = jax.tree_util.tree_map(lambda x: x[1], all_goals)

        successes = (idxs == value_goal_idxs).astype(np.float32)
        batch['masks'] = 1.0 - successes
        batch['rewards'] = successes - (1.0 if self.config['gc_negative'] else 0.0)

        chunk_len = self.config.get('action_chunk_length', 1)
        if chunk_len > 1:
            offsets = np.arange(chunk_len)
            chunk_idxs = idxs[:, None] + offsets[None, :]
            chunk_idxs = np.minimum(chunk_idxs, self.idx_to_terminal[idxs][:, None])
            batch['action_chunks'] = self.dataset['actions'][chunk_idxs]  # (B, chunk_len, action_dim)
        else:
            batch['action_chunks'] = batch['actions'][:, None, :]  # (B, 1, action_dim)

        if self.config['p_aug'] is not None and not evaluation:
            if np.random.rand() < self.config['p_aug']:
                self.augment(batch, ['observations', 'next_observations', 'value_goals', 'actor_goals'])

        return batch

    def _batch_get_observations(self, idx_sets):
        """Gather observations for multiple index arrays in one pass.

        Args:
            idx_sets: (num_sets, batch_size) array of indices.

        Returns:
            Tree of arrays with shape (num_sets, batch_size, ...).
        """
        flat_idxs = idx_sets.ravel()
        if self.config['frame_stack'] is None or self.preprocess_frame_stack:
            flat_obs = jax.tree_util.tree_map(lambda arr: arr[flat_idxs], self.dataset['observations'])
        else:
            flat_obs = self.get_stacked_observations(flat_idxs)
        num_sets, batch_size = idx_sets.shape
        return jax.tree_util.tree_map(lambda arr: arr.reshape(num_sets, batch_size, *arr.shape[1:]), flat_obs)

    def sample_goals(self, idxs, p_curgoal, p_trajgoal, p_randomgoal, geom_sample):
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
        else:
            goal_idxs = np.where(
                np.random.rand(batch_size) < p_trajgoal / (1.0 - p_curgoal), traj_goal_idxs, random_goal_idxs
            )
            goal_idxs = np.where(np.random.rand(batch_size) < p_curgoal, idxs, goal_idxs)

        return goal_idxs

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

        for traj_start, traj_end in zip(self.initial_locs, self.terminal_locs):
            traj_len = traj_end - traj_start + 1
            traj_obs = obs[traj_start:traj_end + 1]          # (L, *spatial, C)
            for j in range(k):
                lag = k - 1 - j
                src_start = max(0, 0 - lag)                    # always 0 when lag < traj_len
                # For each frame t in [0, traj_len), the source is max(0, t - lag)
                # Build the shifted view: pad the beginning by repeating frame 0
                pad_len = min(lag, traj_len)
                body_start = lag                               # first frame that doesn't need clamping
                ch_slice = slice(j * C, (j + 1) * C)
                # Frames [0, pad_len) all map to traj_obs[0]
                if pad_len > 0:
                    out[traj_start:traj_start + pad_len, ..., ch_slice] = traj_obs[0:1]
                # Frames [pad_len, traj_len) map to traj_obs[t - lag]
                if body_start < traj_len:
                    out[traj_start + body_start:traj_end + 1, ..., ch_slice] = \
                        traj_obs[:traj_len - body_start]
        return out

    def get_observations(self, idxs):
        if self.config['frame_stack'] is None or self.preprocess_frame_stack:
            return jax.tree_util.tree_map(lambda arr: arr[idxs], self.dataset['observations'])
        else:
            return self.get_stacked_observations(idxs)

    def get_stacked_observations(self, idxs):
        # --- FIX 1 continued: Use pre-computed lookup ---
        initial_state_idxs = self.idx_to_initial[idxs]
        rets = []
        for i in reversed(range(self.config['frame_stack'])):
            cur_idxs = np.maximum(idxs - i, initial_state_idxs)
            rets.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self.dataset['observations']))
        return jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *rets)


@dataclasses.dataclass
class HGCDataset(GCDataset):

    def sample(self, batch_size, idxs=None, evaluation=False):
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        batch = self.dataset.sample(batch_size, idxs)
        if self.config['frame_stack'] is not None:
            batch['observations'] = self.get_observations(idxs)
            next_idxs = np.minimum(idxs + 1, self.size - 1)
            batch['next_observations'] = self.get_observations(next_idxs)

        value_goal_idxs = self.sample_goals(
            idxs,
            self.config['value_p_curgoal'],
            self.config['value_p_trajgoal'],
            self.config['value_p_randomgoal'],
            self.config['value_geom_sample'],
        )
        batch['value_goals'] = self.get_observations(value_goal_idxs)

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

        # --- FIX 2: Batch all observation gathering ---
        all_idx_sets = np.stack([low_goal_idxs, high_goal_idxs, high_target_idxs])  # (3, batch_size)
        all_obs = self._batch_get_observations(all_idx_sets)
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

        if self.config['p_aug'] is not None and not evaluation:
            if np.random.rand() < self.config['p_aug']:
                self.augment(
                    batch,
                    ['observations', 'next_observations', 'value_goals', 'low_actor_goals',
                     'high_actor_goals', 'high_actor_targets'],
                )

        return batch