from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Dict, Tuple
from queue import Queue
from threading import Thread

import cv2
import flax.serialization as fxs
from flax.core import freeze, unfreeze
import jax
import jax.numpy as jnp
from jax import tree_util, random
import ml_collections
import numpy as np
import wandb
from tqdm import trange, tqdm

from utils.datasets import Dataset, GCDataset

# =============================
# Frame stacking (block-safe)
# =============================
def create_spaced_frame_stack(observations: np.ndarray, frame_offsets: Tuple[int, ...] = (0, -5, -10, -20), block_size: int = 400,) -> np.ndarray:
    T, H, W, C = observations.shape
    stacked_list = []
    for t in range(T):
        block_start = (t // block_size) * block_size
        frames = []
        for off in frame_offsets:
            idx = t + off
            if idx < block_start:
                idx = block_start
            elif idx >= block_start + block_size:
                idx = block_start + block_size - 1
            frames.append(observations[idx])
        stacked_list.append(np.concatenate(frames, axis=-1))
    return np.array(stacked_list, dtype=np.float32)

def filter_intersection_frames_jax(dataset: Dict[str, jnp.ndarray], throttle_threshold: float = 0.05, brake_threshold: float = 0.1,
    window_size: int = 5,
) -> Dict[str, jnp.ndarray]:
    """Pure JAX version that works on GPU."""
    actions = dataset['actions']
    T = len(actions)
    terminals = dataset.get('terminals', jnp.zeros(T, dtype=bool))
    
    # Find terminal locations using JAX
    terminal_indices = jnp.arange(T)[terminals] if jnp.any(terminals) else jnp.array([], dtype=jnp.int32)
    terminal_locs = terminal_indices
    print(f"\n=== FILTERING DEBUG: BEFORE ===")
    print(f"Total frames: {T}")
    print(f"Terminal frames: {jnp.sum(terminals)}")
    if len(terminal_locs) > 0:
        terminal_locs_arr = jnp.asarray(terminal_locs)
        print(f"Terminal locations (first 10): {jnp.asarray(terminal_locs_arr[:min(10, len(terminal_locs_arr))])}")
        if len(terminal_locs_arr) > 10:
            print(f"Terminal locations (last 10): {jnp.asarray(terminal_locs_arr[-10:])}")
        print(f"Trajectory boundaries: {len(terminal_locs_arr)} trajectories")
        if len(terminal_locs_arr) > 0:
            traj_starts = jnp.concatenate([jnp.array([0]), terminal_locs_arr[:-1] + 1])
            traj_lengths = terminal_locs_arr + 1 - traj_starts
            print(f"Trajectory lengths (first 10): {jnp.asarray(traj_lengths[:min(10, len(traj_lengths))])}")
            print(f"Trajectory lengths (stats): min={jnp.min(traj_lengths)}, max={jnp.max(traj_lengths)}, mean={jnp.mean(traj_lengths):.1f}")
    
    throttle = actions[:, 0]
    brake = actions[:, 2]
    low_throttle = throttle < throttle_threshold
    high_brake = brake > brake_threshold
    intersection_mask = jnp.zeros(T, dtype=bool)
    intersection_mask = intersection_mask | high_brake
    
    # Implement sliding window using JAX operations
    if window_size > 0 and T >= window_size:
        # Use convolution to count low_throttle in each window
        kernel = jnp.ones(window_size, dtype=jnp.float32)
        low_throttle_float = low_throttle.astype(jnp.float32)
        window_sums = jnp.convolve(low_throttle_float, kernel, mode='valid')
        all_low_throttle = (window_sums >= window_size - 1e-6).astype(bool)
        
        # Find window starts where all are low using JAX
        window_starts = jnp.arange(len(all_low_throttle))[all_low_throttle] if jnp.any(all_low_throttle) else jnp.array([], dtype=jnp.int32)
        if len(window_starts) > 0:
            # Create indices for all positions in these windows
            window_starts_expanded = window_starts[:, None] + jnp.arange(window_size)[None, :]
            mark_indices = window_starts_expanded.ravel()
            mark_indices = mark_indices[mark_indices < T]
            intersection_mask = intersection_mask.at[mark_indices].set(True)
    
    # Preserve terminals using JAX operations
    terminal_indices = jnp.arange(T)[terminals] if jnp.any(terminals) else jnp.array([], dtype=jnp.int32)
    terminals_preserved = 0
    for i in range(len(terminal_indices)):
        term_idx = terminal_indices[i]
        term_val = bool(intersection_mask[term_idx])
        if term_val:
            intersection_mask = intersection_mask.at[term_idx].set(False)
            terminals_preserved += 1
            if term_idx > 0:
                intersection_mask = intersection_mask.at[term_idx - 1].set(False)
    
    keep_mask = ~intersection_mask
    n_removed = jnp.sum(intersection_mask)
    n_kept = jnp.sum(keep_mask)
    
    print(f"\n=== FILTERING DEBUG: AFTER ===")
    print(f"Intersection filtering: removed {int(n_removed)} frames ({100.0*float(n_removed)/T:.1f}%), kept {int(n_kept)} frames")
    print(f"Terminals preserved from removal: {terminals_preserved}")
    
    # Filter arrays using boolean indexing
    filtered = {}
    for key, arr in dataset.items():
        if hasattr(arr, 'shape') and len(arr) == T:
            filtered[key] = arr[keep_mask]
        else:
            filtered[key] = arr
    
    # Map terminals using JAX
    if 'terminals' in dataset:
        # Build cumulative sum to map old indices to new indices
        keep_mask_int = keep_mask.astype(jnp.int32)
        cumsum = jnp.cumsum(keep_mask_int) - 1  # -1 because cumsum starts at 1 for first True
        
        # Map terminal indices
        new_terminals = jnp.zeros(int(n_kept), dtype=bool)
        terminals_mapped = 0
        for i in range(len(terminal_indices)):
            old_term_idx = terminal_indices[i]
            if keep_mask[old_term_idx]:
                new_term_idx = cumsum[old_term_idx]
                new_terminals = new_terminals.at[new_term_idx].set(True)
                terminals_mapped += 1
        
        # Ensure last frame is terminal
        if len(new_terminals) > 0:
            new_terminals = new_terminals.at[-1].set(True)
        
        filtered['terminals'] = new_terminals
        
        new_terminal_locs = jnp.arange(int(n_kept))[new_terminals] if jnp.any(new_terminals) else jnp.array([], dtype=jnp.int32)
        print(f"Terminals after mapping: {jnp.sum(new_terminals)} (mapped {terminals_mapped} from {len(terminal_indices)} original)")
        if len(new_terminal_locs) > 0:
            print(f"New terminal locations (first 10): {jnp.asarray(new_terminal_locs[:min(10, len(new_terminal_locs))])}")
            if len(new_terminal_locs) > 0:
                traj_starts = jnp.concatenate([jnp.array([0]), new_terminal_locs[:-1] + 1])
                new_traj_lengths = new_terminal_locs + 1 - traj_starts
                print(f"New trajectory lengths (first 10): {jnp.asarray(new_traj_lengths[:min(10, len(new_traj_lengths))])}")
                print(f"New trajectory lengths (stats): min={jnp.min(new_traj_lengths)}, max={jnp.max(new_traj_lengths)}, mean={jnp.mean(new_traj_lengths):.1f}")
    
    print("=" * 50)
    return filtered


def filter_intersection_frames(dataset: Dict[str, np.ndarray], throttle_threshold: float = 0.05, brake_threshold: float = 0.1,
    window_size: int = 5,
) -> Dict[str, np.ndarray]:
    """Remove frames where the car is stopped/idling (low throttle AND low steer for
    extended windows). Does NOT remove braking frames that co-occur with turning,
    because those are exactly the frames we need to learn turns.
    """
    actions = dataset['actions']
    T = len(actions)
    terminals = dataset.get('terminals', np.zeros(T, dtype=bool))
    
    print(f"\n=== FILTERING DEBUG: BEFORE ===")
    print(f"Total frames: {T}")
    print(f"Terminal frames: {np.sum(terminals)}")
    terminal_locs = np.where(terminals)[0]
    print(f"Terminal locations (first 10): {terminal_locs[:10]}")
    if len(terminal_locs) > 0:
        print(f"Terminal locations (last 10): {terminal_locs[-10:]}")
    print(f"Trajectory boundaries: {len(terminal_locs)} trajectories")
    if len(terminal_locs) > 0:
        traj_lengths = np.diff(np.concatenate([[0], terminal_locs + 1]))
        print(f"Trajectory lengths (first 10): {traj_lengths[:10]}")
        print(f"Trajectory lengths (stats): min={traj_lengths.min()}, max={traj_lengths.max()}, mean={traj_lengths.mean():.1f}")
    
    throttle = actions[:, 0]
    steer = actions[:, 1]
    brake = actions[:, 2]
    low_throttle = throttle < throttle_threshold
    is_turning = np.abs(steer) > 0.03
    # Only mark as intersection/idle if braking AND not turning.
    high_brake_idle = (brake > brake_threshold) & (~is_turning)
    intersection_mask = np.zeros(T, dtype=bool)
    intersection_mask = intersection_mask | high_brake_idle
    from numpy.lib.stride_tricks import sliding_window_view
    try:
        windows = sliding_window_view(low_throttle, window_size)
        all_low_throttle = np.all(windows, axis=1)
        window_starts = np.where(all_low_throttle)[0]
        if len(window_starts) > 0:
            mark_indices = (window_starts[:, None] + np.arange(window_size)[None, :]).ravel()
            mark_indices = mark_indices[mark_indices < T]
            intersection_mask[mark_indices] = True
    except (ImportError, AttributeError):
        print("WARNING: Using slower loop-based filtering (upgrade NumPy >= 1.20 for faster version)")
        for i in range(T - window_size + 1):
            if np.all(low_throttle[i:i+window_size]):
                intersection_mask[i:i+window_size] = True
    terminals_to_remove = np.sum(intersection_mask[terminal_locs])
    print(f"Terminals that would be removed (before preservation): {terminals_to_remove}")
    terminal_indices = np.where(terminals)[0]
    terminals_preserved = 0
    for term_idx in terminal_indices:
        if intersection_mask[term_idx]:
            intersection_mask[term_idx] = False
            terminals_preserved += 1
            if term_idx > 0:
                intersection_mask[term_idx - 1] = False
    
    keep_mask = ~intersection_mask
    
    n_removed = np.sum(intersection_mask)
    n_kept = np.sum(keep_mask)
    
    print(f"\n=== FILTERING DEBUG: AFTER ===")
    print(f"Intersection filtering: removed {n_removed} frames ({100.0*n_removed/T:.1f}%), kept {n_kept} frames")
    print(f"Terminals preserved from removal: {terminals_preserved}")
    filtered = {}
    for key, arr in dataset.items():
        if isinstance(arr, np.ndarray) and len(arr) == T:
            filtered[key] = arr[keep_mask]
        else:
            filtered[key] = arr
    
    # Ensure terminals are properly preserved after filtering
    # Map old terminal indices to new indices after filtering
    if 'terminals' in dataset:
        new_terminals = np.zeros(n_kept, dtype=bool)
        # Build mapping: old_index -> new_index for kept frames
        old_to_new = {}
        new_idx = 0
        for old_idx in range(T):
            if keep_mask[old_idx]:
                old_to_new[old_idx] = new_idx
                new_idx += 1
        
        # Map terminal indices
        terminals_mapped = 0
        for old_term_idx in terminal_indices:
            if old_term_idx in old_to_new:
                new_term_idx = old_to_new[old_term_idx]
                new_terminals[new_term_idx] = True
                terminals_mapped += 1
        
        filtered['terminals'] = new_terminals
        # Ensure last frame is always terminal
        if len(new_terminals) > 0:
            new_terminals[-1] = True
        
        print(f"Terminals after mapping: {np.sum(new_terminals)} (mapped {terminals_mapped} from {len(terminal_indices)} original)")
        new_terminal_locs = np.where(new_terminals)[0]
        print(f"New terminal locations (first 10): {new_terminal_locs[:10]}")
        if len(new_terminal_locs) > 0:
            new_traj_lengths = np.diff(np.concatenate([[0], new_terminal_locs + 1]))
            print(f"New trajectory lengths (first 10): {new_traj_lengths[:10]}")
            print(f"New trajectory lengths (stats): min={new_traj_lengths.min()}, max={new_traj_lengths.max()}, mean={new_traj_lengths.mean():.1f}")
    
    print("=" * 50)
    return filtered

def _maybe_get_terminals_from_source(data_np: Dict[str, np.ndarray]) -> np.ndarray | None:
    """Return terminals array if present in the original dataset."""
    if "terminals" in data_np:
        terminals = np.asarray(data_np["terminals"], dtype=bool).reshape(-1)
        if terminals.size > 0 and terminals.any():
            return terminals
    # Some datasets store terminals under 'dones' or 'episode_ends'.
    for key in ("dones", "episode_ends", "is_terminal"):
        if key in data_np:
            terminals = np.asarray(data_np[key], dtype=bool).reshape(-1)
            if terminals.size > 0 and terminals.any():
                return terminals
    return None


def load_dataset_cpu(path: str | Path, frame_offsets: Tuple[int, ...], block_size: int, chunk_size: int = 50000, use_mmap: bool = False) -> Dict[str, np.ndarray]:
    import time
    start_time = time.time()
    print(f"Loading dataset from {path} ...")
    if use_mmap:
        print("Using memory-mapped loading (saves RAM but may be slower)...")
        data_np = np.load(path, mmap_mode='r')
    else:
        # Note: np.load() loads entire file into memory by default.
        # Set use_mmap=True for true memory-mapped chunking (saves RAM but may be slower).
        data_np = np.load(path)
    print(f"Dataset loaded in {time.time() - start_time:.2f}s")

    # Get total size first (without loading full arrays)
    obs_shape = data_np["observations"].shape
    total_frames = obs_shape[0]
    print(f"\n=== DATASET LOADING DEBUG ===")
    print(f"Total frames in dataset: {total_frames}")
    print(f"Processing in chunks of {chunk_size} frames to save memory...")

    # Load terminals first (small array)
    terminals = _maybe_get_terminals_from_source(data_np)
    if terminals is not None:
        terminals = terminals.astype(bool).copy()
        print(f"Found {int(terminals.sum())} terminal markers in dataset file.")
    else:
        # Fallback: assume fixed-length trajectories (legacy datasets)
        print("WARNING: Dataset missing terminal markers. Falling back to synthetic 1000-step boundaries.")
        terminals = np.zeros(total_frames, dtype=bool)
        trajectory_length = 1000
        terminal_indices = np.arange(trajectory_length - 1, total_frames, trajectory_length, dtype=int)
        terminals[terminal_indices] = True

    # Always mark last frame as terminal to close final trajectory
    terminals[-1] = True

    terminal_indices = np.where(terminals)[0]
    print(f"Terminal count after initialization: {len(terminal_indices)}")
    if len(terminal_indices) > 0:
        print(f"Terminal indices (first 10): {terminal_indices[:10]}")
        if len(terminal_indices) > 10:
            print(f"Terminal indices (last 10): {terminal_indices[-10:]}")
        traj_lengths = np.diff(np.concatenate([[0], terminal_indices + 1]))
        print(
            "Trajectory length stats (before filtering): "
            f"min={traj_lengths.min()}, max={traj_lengths.max()}, mean={traj_lengths.mean():.1f}"
        )

    # Process filtering in chunks, keeping trajectories together
    print(f"Grouping trajectories into chunks of ~{chunk_size} frames...")
    filter_start = time.time()
    
    # Find trajectory boundaries
    terminal_indices = np.where(terminals)[0]
    if len(terminal_indices) == 0:
        # No terminals found, treat as single trajectory
        trajectory_starts = np.array([0])
        trajectory_ends = np.array([total_frames])
        trajectory_lengths = np.array([total_frames])
    else:
        # Calculate trajectory start and end indices
        trajectory_starts = np.concatenate([[0], terminal_indices[:-1] + 1])
        trajectory_ends = terminal_indices + 1
        trajectory_lengths = trajectory_ends - trajectory_starts
    
    num_trajectories = len(trajectory_starts)
    print(f"Found {num_trajectories} trajectories (avg length: {trajectory_lengths.mean():.1f} frames)")
    
    # Group trajectories into chunks
    filtered_chunks = []
    chunk_trajectories = []
    chunk_frame_count = 0
    chunk_idx = 0
    
    for traj_idx in range(num_trajectories):
        traj_start = trajectory_starts[traj_idx]
        traj_end = trajectory_ends[traj_idx]
        traj_length = traj_end - traj_start
        
        # If adding this trajectory would exceed chunk_size, process current chunk first
        if chunk_frame_count > 0 and chunk_frame_count + traj_length > chunk_size:
            # Process current chunk
            chunk_start = trajectory_starts[chunk_trajectories[0]]
            chunk_end = trajectory_ends[chunk_trajectories[-1]]
            chunk_slice = slice(chunk_start, chunk_end)
            
            print(f"Processing chunk {chunk_idx + 1} ({len(chunk_trajectories)} trajectories, {chunk_frame_count} frames)...", end="\r")
            
            # Load chunk from file
            obs_chunk = np.asarray(data_np["observations"][chunk_slice])
            actions_chunk = np.asarray(data_np["actions"][chunk_slice], dtype=np.float32)
            terminals_chunk = terminals[chunk_slice].copy()
            
            # Filter this chunk
            filtered_chunk = filter_intersection_frames({
                "observations": obs_chunk,
                "actions": actions_chunk,
                "terminals": terminals_chunk,
            })
            
            filtered_chunks.append(filtered_chunk)
            
            # Start new chunk
            chunk_trajectories = []
            chunk_frame_count = 0
            chunk_idx += 1
        
        # Add trajectory to current chunk
        chunk_trajectories.append(traj_idx)
        chunk_frame_count += traj_length
    
    # Process final chunk if it has trajectories
    if len(chunk_trajectories) > 0:
        chunk_start = trajectory_starts[chunk_trajectories[0]]
        chunk_end = trajectory_ends[chunk_trajectories[-1]]
        chunk_slice = slice(chunk_start, chunk_end)
        
        print(f"Processing chunk {chunk_idx + 1} ({len(chunk_trajectories)} trajectories, {chunk_frame_count} frames)...", end="\r")
        
        # Load chunk from file
        obs_chunk = np.asarray(data_np["observations"][chunk_slice])
        actions_chunk = np.asarray(data_np["actions"][chunk_slice], dtype=np.float32)
        terminals_chunk = terminals[chunk_slice].copy()
        
        # Filter this chunk
        filtered_chunk = filter_intersection_frames({
            "observations": obs_chunk,
            "actions": actions_chunk,
            "terminals": terminals_chunk,
        })
        
        filtered_chunks.append(filtered_chunk)
    
    print(f"\nFiltering completed in {time.time() - filter_start:.2f}s")
    print(f"Processed {len(filtered_chunks)} chunks (all trajectories kept intact)")
    print("Concatenating filtered chunks...")
    
    # Concatenate all filtered chunks
    filtered = {
        "observations": np.concatenate([chunk["observations"] for chunk in filtered_chunks], axis=0),
        "actions": np.concatenate([chunk["actions"] for chunk in filtered_chunks], axis=0),
        "terminals": np.concatenate([chunk["terminals"] for chunk in filtered_chunks], axis=0),
    }
    obs = filtered["observations"]
    actions = filtered["actions"]
    terminals = filtered["terminals"]
    print(f"\n=== AFTER FILTERING (before frame stacking) ===")
    print(f"Frames remaining: {len(obs)}")
    print(f"Terminals remaining: {np.sum(terminals)}")
    terminal_locs_after = np.where(terminals)[0]
    if len(terminal_locs_after) > 0:
        print(f"Terminal locations (first 10): {terminal_locs_after[:10]}")
        traj_lengths_after = np.diff(np.concatenate([[0], terminal_locs_after + 1]))
        print(f"Trajectory lengths (first 10): {traj_lengths_after[:10]}")
        print(f"Trajectory lengths (stats): min={traj_lengths_after.min()}, max={traj_lengths_after.max()}, mean={traj_lengths_after.mean():.1f}")

    print(f"Loaded (unstacked): obs={obs.shape}, actions={actions.shape}, terminals={terminals.shape}")
    return {
        "observations": obs,
        "actions": actions,
        "terminals": terminals,
        "cpu_mode": True,
    }


def _gather_trajectories(
    dataset: Dict[str, np.ndarray],
    trajectory_ids: np.ndarray,
    trajectory_length: int,
) -> Dict[str, np.ndarray]:
    if len(trajectory_ids) == 0:
        return {
            "observations": np.zeros((0, *dataset["observations"].shape[1:]), dtype=dataset["observations"].dtype),
            "actions": np.zeros((0, *dataset["actions"].shape[1:]), dtype=dataset["actions"].dtype),
            "terminals": np.zeros((0,), dtype=bool),
        }

    obs_list = []
    act_list = []
    term_list = []
    next_list = [] if "next_observations" in dataset else None

    total = len(dataset["observations"])
    for traj_id in trajectory_ids:
        start = traj_id * trajectory_length
        if start >= total:
            continue
        end = min(start + trajectory_length, total)
        obs_chunk = dataset["observations"][start:end]
        act_chunk = dataset["actions"][start:end]
        term_chunk = np.asarray(dataset["terminals"][start:end], dtype=bool)
        term_chunk = term_chunk.reshape(-1).copy()
        if len(term_chunk) == 0:
            continue
        term_chunk[-1] = True
        if next_list is not None:
            next_chunk = dataset["next_observations"][start:end].copy()
            if len(next_chunk) > 0:
                next_chunk[-1] = obs_chunk[-1]
            next_list.append(next_chunk)
        obs_list.append(obs_chunk)
        act_list.append(act_chunk)
        term_list.append(term_chunk)

    if not obs_list:
        return {
            "observations": np.zeros((0, *dataset["observations"].shape[1:]), dtype=dataset["observations"].dtype),
            "actions": np.zeros((0, *dataset["actions"].shape[1:]), dtype=dataset["actions"].dtype),
            "terminals": np.zeros((0,), dtype=bool),
        }

    return {
        "observations": np.concatenate(obs_list, axis=0),
        "actions": np.concatenate(act_list, axis=0),
        "terminals": np.concatenate(term_list, axis=0),
        **(
            {"next_observations": np.concatenate(next_list, axis=0)}
            if next_list is not None and len(next_list) > 0
            else {}
        ),
    }


def split_dataset_by_terminals_jax(
    dataset: Dict[str, jnp.ndarray],
    val_fraction: float,
    seed: int,
) -> Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray] | None]:
    """Split dataset by actual trajectory boundaries (terminals), working with JAX arrays on GPU."""
    print(f"\n=== SPLITTING DATASET BY TERMINALS (JAX) ===")
    terminals = dataset.get('terminals', jnp.zeros(len(dataset["observations"]), dtype=bool))
    
    # Find all terminal locations using JAX
    terminal_locs = jnp.arange(len(terminals))[terminals] if jnp.any(terminals) else jnp.array([], dtype=jnp.int32)
    
    print(f"Total frames: {len(dataset['observations'])}")
    print(f"Terminal locations found: {len(terminal_locs)}")
    
    if len(terminal_locs) == 0:
        print("WARNING: No terminals found! Treating as single trajectory.")
        empty_shape = (0, *dataset["observations"].shape[1:])
        return dataset, {
            "observations": jnp.zeros(empty_shape, dtype=dataset["observations"].dtype),
            "actions": jnp.zeros((0, *dataset["actions"].shape[1:]), dtype=dataset["actions"].dtype),
            "terminals": jnp.zeros((0,), dtype=bool),
        }
    
    # Build trajectory boundaries: [start, end) for each trajectory
    traj_starts = jnp.concatenate([jnp.array([0]), terminal_locs[:-1] + 1])
    traj_ends = terminal_locs + 1
    num_trajectories = len(traj_starts)
    
    traj_lengths = traj_ends - traj_starts
    print(f"Number of trajectories: {num_trajectories}")
    if len(traj_lengths) > 0:
        print(f"Trajectory lengths (first 10): {jnp.asarray(traj_lengths[:min(10, len(traj_lengths))])}")
        print(f"Trajectory lengths (stats): min={jnp.min(traj_lengths)}, max={jnp.max(traj_lengths)}, mean={jnp.mean(traj_lengths):.1f}, median={jnp.median(traj_lengths):.1f}")
    
    # Shuffle trajectory indices using JAX random
    key = random.PRNGKey(seed)
    traj_ids = jnp.arange(num_trajectories)
    traj_ids = random.permutation(key, traj_ids)
    
    # Split trajectories
    num_val = max(1, int(round(val_fraction * num_trajectories)))
    num_val = min(num_val, num_trajectories - 1) if num_trajectories > 1 else num_val
    
    val_ids = traj_ids[:num_val]
    train_ids = traj_ids[num_val:] if num_trajectories > num_val else traj_ids[:1]
    
    # Gather trajectories by actual boundaries - use JAX operations
    train_obs_list = []
    train_act_list = []
    train_term_list = []
    
    val_obs_list = []
    val_act_list = []
    val_term_list = []
    
    for traj_id in train_ids:
        start = traj_starts[traj_id]
        end = traj_ends[traj_id]
        train_obs_list.append(dataset["observations"][start:end])
        train_act_list.append(dataset["actions"][start:end])
        term_chunk = terminals[start:end]
        if len(term_chunk) > 0:
            term_chunk = term_chunk.at[-1].set(True)  # Ensure last frame is terminal
        train_term_list.append(term_chunk)
    
    for traj_id in val_ids:
        start = traj_starts[traj_id]
        end = traj_ends[traj_id]
        val_obs_list.append(dataset["observations"][start:end])
        val_act_list.append(dataset["actions"][start:end])
        term_chunk = terminals[start:end]
        if len(term_chunk) > 0:
            term_chunk = term_chunk.at[-1].set(True)  # Ensure last frame is terminal
        val_term_list.append(term_chunk)
    
    train_data = {
        "observations": jnp.concatenate(train_obs_list, axis=0) if train_obs_list else jnp.zeros((0, *dataset["observations"].shape[1:]), dtype=dataset["observations"].dtype),
        "actions": jnp.concatenate(train_act_list, axis=0) if train_act_list else jnp.zeros((0, *dataset["actions"].shape[1:]), dtype=dataset["actions"].dtype),
        "terminals": jnp.concatenate(train_term_list, axis=0) if train_term_list else jnp.zeros((0,), dtype=bool),
    }
    # Ensure last frame of train set is terminal
    if len(train_data["terminals"]) > 0:
        train_data["terminals"] = train_data["terminals"].at[-1].set(True)
    
    val_data = {
        "observations": jnp.concatenate(val_obs_list, axis=0) if val_obs_list else jnp.zeros((0, *dataset["observations"].shape[1:]), dtype=dataset["observations"].dtype),
        "actions": jnp.concatenate(val_act_list, axis=0) if val_act_list else jnp.zeros((0, *dataset["actions"].shape[1:]), dtype=dataset["actions"].dtype),
        "terminals": jnp.concatenate(val_term_list, axis=0) if val_term_list else jnp.zeros((0,), dtype=bool),
    } if val_obs_list else None
    # Ensure last frame of val set is terminal
    if val_data is not None and len(val_data["terminals"]) > 0:
        val_data["terminals"] = val_data["terminals"].at[-1].set(True)
    
    print(f"\n=== SPLIT COMPLETE ===")
    print(f"Train: {len(train_data['observations'])} frames, {len(train_ids)} trajectories")
    if val_data is not None:
        print(f"Val: {len(val_data['observations'])} frames, {len(val_ids)} trajectories")
        train_terminal_count = jnp.sum(train_data['terminals'])
        val_terminal_count = jnp.sum(val_data['terminals'])
        print(f"Train terminals: {train_terminal_count}, Val terminals: {val_terminal_count}")
    else:
        print(f"Val: 0 frames, 0 trajectories")
    print("=" * 50)
    
    return train_data, val_data


def split_dataset_by_terminals(
    dataset: Dict[str, np.ndarray],
    val_fraction: float,
    seed: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Split dataset by actual trajectory boundaries (terminals), not fixed length."""
    print(f"\n=== SPLITTING DATASET BY TERMINALS ===")
    terminals = dataset.get('terminals', np.zeros(len(dataset["observations"]), dtype=bool))
    
    # Find all terminal locations (trajectory boundaries)
    terminal_locs = np.where(terminals)[0]
    
    print(f"Total frames: {len(dataset['observations'])}")
    print(f"Terminal locations found: {len(terminal_locs)}")
    
    if len(terminal_locs) == 0:
        print("WARNING: No terminals found! Treating as single trajectory.")
        # No terminals found, treat as single trajectory
        return dataset, {k: v[:0] if isinstance(v, np.ndarray) else v for k, v in dataset.items()}
    
    # Build trajectory boundaries: [start, end) for each trajectory
    traj_starts = np.concatenate([[0], terminal_locs[:-1] + 1])
    traj_ends = terminal_locs + 1
    num_trajectories = len(traj_starts)
    
    traj_lengths = traj_ends - traj_starts
    print(f"Number of trajectories: {num_trajectories}")
    print(f"Trajectory lengths (first 10): {traj_lengths[:10]}")
    print(f"Trajectory lengths (stats): min={traj_lengths.min()}, max={traj_lengths.max()}, mean={traj_lengths.mean():.1f}, median={np.median(traj_lengths):.1f}")
    
    # Shuffle trajectory indices
    rng = np.random.default_rng(seed)
    traj_ids = np.arange(num_trajectories)
    rng.shuffle(traj_ids)
    
    # Split trajectories
    num_val = max(1, int(np.round(val_fraction * num_trajectories)))
    num_val = min(num_val, num_trajectories - 1) if num_trajectories > 1 else num_val
    
    val_ids = traj_ids[:num_val]
    train_ids = traj_ids[num_val:] if num_trajectories > num_val else traj_ids[:1]
    
    # Gather trajectories by actual boundaries
    train_obs_list = []
    train_act_list = []
    train_term_list = []
    train_next_list = [] if "next_observations" in dataset else None
    
    val_obs_list = []
    val_act_list = []
    val_term_list = []
    val_next_list = [] if "next_observations" in dataset else None
    
    for traj_id in train_ids:
        start = traj_starts[traj_id]
        end = traj_ends[traj_id]
        train_obs_list.append(dataset["observations"][start:end])
        train_act_list.append(dataset["actions"][start:end])
        term_chunk = terminals[start:end].copy()
        term_chunk[-1] = True  # Ensure last frame is terminal
        train_term_list.append(term_chunk)
        if train_next_list is not None:
            next_chunk = dataset["next_observations"][start:end].copy()
            if len(next_chunk) > 0:
                next_chunk[-1] = dataset["observations"][end-1]
            train_next_list.append(next_chunk)
    
    for traj_id in val_ids:
        start = traj_starts[traj_id]
        end = traj_ends[traj_id]
        val_obs_list.append(dataset["observations"][start:end])
        val_act_list.append(dataset["actions"][start:end])
        term_chunk = terminals[start:end].copy()
        term_chunk[-1] = True  # Ensure last frame is terminal
        val_term_list.append(term_chunk)
        if val_next_list is not None:
            next_chunk = dataset["next_observations"][start:end].copy()
            if len(next_chunk) > 0:
                next_chunk[-1] = dataset["observations"][end-1]
            val_next_list.append(next_chunk)
    
    train_data = {
        "observations": np.concatenate(train_obs_list, axis=0) if train_obs_list else np.zeros((0, *dataset["observations"].shape[1:]), dtype=dataset["observations"].dtype),
        "actions": np.concatenate(train_act_list, axis=0) if train_act_list else np.zeros((0, *dataset["actions"].shape[1:]), dtype=dataset["actions"].dtype),
        "terminals": np.concatenate(train_term_list, axis=0) if train_term_list else np.zeros((0,), dtype=bool),
    }
    # Ensure last frame of train set is terminal
    if len(train_data["terminals"]) > 0:
        train_data["terminals"][-1] = True
    if train_next_list and len(train_next_list) > 0:
        train_data["next_observations"] = np.concatenate(train_next_list, axis=0)
    
    val_data = {
        "observations": np.concatenate(val_obs_list, axis=0) if val_obs_list else np.zeros((0, *dataset["observations"].shape[1:]), dtype=dataset["observations"].dtype),
        "actions": np.concatenate(val_act_list, axis=0) if val_act_list else np.zeros((0, *dataset["actions"].shape[1:]), dtype=dataset["actions"].dtype),
        "terminals": np.concatenate(val_term_list, axis=0) if val_term_list else np.zeros((0,), dtype=bool),
    }
    # Ensure last frame of val set is terminal
    if len(val_data["terminals"]) > 0:
        val_data["terminals"][-1] = True
    if val_next_list and len(val_next_list) > 0:
        val_data["next_observations"] = np.concatenate(val_next_list, axis=0)
    
    print(f"\n=== SPLIT COMPLETE ===")
    print(f"Train: {len(train_data['observations'])} frames, {len(train_ids)} trajectories")
    print(f"Val: {len(val_data['observations'])} frames, {len(val_ids)} trajectories")
    train_terminal_count = np.sum(train_data['terminals'])
    val_terminal_count = np.sum(val_data['terminals'])
    print(f"Train terminals: {train_terminal_count}, Val terminals: {val_terminal_count}")
    print("=" * 50)
    
    return train_data, val_data


def enforce_periodic_terminals(data: Dict[str, np.ndarray], period: int) -> Dict[str, np.ndarray]:
    if period <= 0 or len(data["observations"]) == 0:
        return data
    terminals = np.zeros(len(data["observations"]), dtype=bool)
    indices = np.arange(period - 1, len(terminals), period, dtype=int)
    terminals[indices] = True
    terminals[-1] = True
    data["terminals"] = terminals
    return data


# =============================
# Validation
# =============================
def test_goal_conditioning(agent, val_dataset: GCDataset | None, num_test_samples: int = 50):
    """Measure how strongly the actor responds to goal variations on a fixed observation."""
    if val_dataset is None or val_dataset.size == 0:
        return {"val/goal_conditioning_sensitivity": float("nan")}

    total = val_dataset.size
    num_samples = min(num_test_samples, total)
    if num_samples == 0:
        return {"val/goal_conditioning_sensitivity": float("nan")}

    rng = np.random.default_rng()
    sample_indices = rng.choice(total, num_samples, replace=False)
    num_goals_per_obs = 10

    def gather_obs(indices: np.ndarray) -> np.ndarray:
        return np.asarray(val_dataset.get_observations(np.asarray(indices, dtype=np.int64)))

    sensitivities = []
    random_sensitivities = []

    discount = float(val_dataset.config.get("discount", 0.99))
    geom_p = max(1e-4, 1.0 - discount)

    for idx in sample_indices[: min(10, num_samples)]:
        obs = jnp.asarray(gather_obs(np.asarray([idx]))[0])
        obs_batch = jnp.broadcast_to(obs[None, ...], (num_goals_per_obs, *obs.shape))

        block_idx = int(np.searchsorted(val_dataset.terminal_locs, idx))
        block_start = int(val_dataset.initial_locs[block_idx])
        block_end = int(val_dataset.terminal_locs[block_idx]) + 1

        if block_end - block_start <= 1:
            goal_positions = np.full((num_goals_per_obs,), idx, dtype=int)
        else:
            max_forward = max(1, block_end - idx - 1)
            near_offsets = np.clip(
                rng.geometric(p=geom_p, size=max(1, num_goals_per_obs // 2)),
                1,
                max_forward,
            )
            far_offsets = np.linspace(
                1,
                max_forward,
                num=max(1, num_goals_per_obs - len(near_offsets)),
                dtype=int,
            )
            offsets = np.concatenate([near_offsets, far_offsets])[:num_goals_per_obs]
            goal_positions = np.clip(idx + offsets, block_start, block_end - 1)

        goals_same_block = jnp.asarray(gather_obs(goal_positions))
        actor_dist = agent.network.select("actor")(obs_batch, goals_same_block, params=agent.network.params)
        actions = actor_dist.mode()
        sensitivities.append(float(jnp.std(actions, axis=0).mean()))

        random_goal_indices = rng.integers(0, total, size=num_goals_per_obs)
        goals_random = jnp.asarray(gather_obs(random_goal_indices))
        actor_dist_random = agent.network.select("actor")(obs_batch, goals_random, params=agent.network.params)
        actions_random = actor_dist_random.mode()
        random_sensitivities.append(float(jnp.std(actions_random, axis=0).mean()))

    mean_sensitivity = float(np.mean(sensitivities)) if sensitivities else 0.0
    mean_random_sensitivity = float(np.mean(random_sensitivities)) if random_sensitivities else 0.0

    return {
        "val/goal_conditioning_sensitivity": mean_sensitivity,
        "val/goal_conditioning_sensitivity_random": mean_random_sensitivity,
    }


def compute_validation_loss(agent, val_dataset: GCDataset | None, batch_size: int = 1024):
    """Run a validation pass using the goal-conditioned dataset sampler."""
    if val_dataset is None or val_dataset.size == 0:
        return {}

    actual_batch = min(batch_size, val_dataset.size)
    batch = val_dataset.sample(actual_batch, evaluation=True)

    # Compute actor loss (all agents have this)
    actor_loss, actor_info = agent.actor_loss(batch, agent.network.params)

    # Compute critic loss - check for both contrastive_loss (CRL) and critic_loss (TMD)
    critic_loss = 0.0
    critic_info = {}
    if hasattr(agent, 'contrastive_loss'):
        # CRL agents
        critic_loss, critic_info = agent.contrastive_loss(batch, agent.network.params)
    elif hasattr(agent, 'critic_loss'):
        # TMD agents - returns (losses_tuple, critic_loss, info_dict)
        critic_result = agent.critic_loss(batch, agent.network.params)
        if isinstance(critic_result, tuple) and len(critic_result) == 3:
            _, critic_loss, critic_info = critic_result
        else:
            # Fallback if signature is different
            critic_loss = critic_result[0] if isinstance(critic_result, tuple) else critic_result
            critic_info = critic_result[1] if isinstance(critic_result, tuple) and len(critic_result) > 1 else {}

    metrics: Dict[str, object] = {}

    # Actor metrics: keep only BC loss, Q loss, and MSE.
    if "bc_loss" in actor_info:
        metrics["val/actor_bc_loss"] = float(actor_info["bc_loss"])
    if "q_loss" in actor_info:
        metrics["val/actor_q_loss"] = float(actor_info["q_loss"])
    if "mse" in actor_info:
        metrics["val/actor_mse"] = float(actor_info["mse"])
    if "mse_first" in actor_info:
        metrics["val/actor_mse_first"] = float(actor_info["mse_first"])

    # Critic metrics: log critic loss and other relevant metrics
    if hasattr(agent, "contrastive_loss") or hasattr(agent, "critic_loss"):
        metrics["val/critic_loss"] = float(critic_loss)
        if "categorical_accuracy" in critic_info:
            metrics["val/critic_categorical_accuracy"] = float(critic_info["categorical_accuracy"])
        # TMD-specific metrics
        if "logits_pos" in critic_info:
            metrics["val/critic_logits_pos"] = float(critic_info["logits_pos"])
        if "logits_neg" in critic_info:
            metrics["val/critic_logits_neg"] = float(critic_info["logits_neg"])
        if "dual_descent_val" in critic_info:
            metrics["val/dual_descent_val"] = float(critic_info["dual_descent_val"])
        if "backup_optim_loss" in critic_info:
            metrics["val/backup_optim_loss"] = float(critic_info["backup_optim_loss"])
        if "val_times_contrastive" in critic_info:
            metrics["val/val_times_contrastive"] = float(critic_info["val_times_contrastive"])
    
    return metrics


# =============================
# Training
# =============================
def main(args: argparse.Namespace) -> None:
    agent_module = import_module(f"agents.{args.algorithm.lower()}")
    agent_cls = getattr(agent_module, f"{args.algorithm}Agent")
    get_config = getattr(agent_module, "get_config")

    cfg = get_config()
    cfg.batch_size = args.batch_size
    cfg.discrete = False  # Continuous actions (n x 3)
    cfg.actor_loss = "ddpgbc"
    cfg.expectile = 0.7
    cfg.discount = args.discount
    cfg.alpha = 1.0
    cfg.encoder = "impala_small"
    cfg.lr = 1e-4
    cfg.actor_hidden_dims = (512, 512, 512)
    cfg.value_hidden_dims = (512, 512, 512)
    cfg.latent_dim = 512
    cfg.critic_lr_scale = 1.0
    cfg.actor_lr_scale = 1.0
    if args.frame_offsets is not None:
        cfg.frame_offsets = tuple(args.frame_offsets)
        cfg.frame_stack = len(args.frame_offsets)
    else:
        cfg.frame_offsets = (0, -1, -2)
        cfg.frame_stack = 3
    # Saved in config.json so eval server/client match training geometry (no CLI guessing).
    cfg.obs_h = args.obs_h
    cfg.obs_w = args.obs_w
    cfg.obs_c = args.obs_c
    cfg.block_size = args.block_size
    cfg.p_aug = 0.5
    cfg.distance_loss_weight = 0.05
    cfg.distance_head_hidden_dims = (256, 256)
    cfg.upsample_mode = 'turns_high'
    cfg.upsample_weight = 3.0
    cfg.steer_thresh = 0.05
    cfg.throttle_thresh = 0.3
    cfg.brake_thresh = 0.1
    cfg.p_randomgoal = 0.1
    cfg.p_trajgoal = 0.9
    cfg.p_curgoal = 0.0
    # GCDataset requires these parameters (use same values for value and actor goals)
    cfg.value_p_curgoal = cfg.p_curgoal
    cfg.value_p_trajgoal = cfg.p_trajgoal
    cfg.value_p_randomgoal = cfg.p_randomgoal
    cfg.actor_p_curgoal = cfg.p_curgoal
    cfg.actor_p_trajgoal = cfg.p_trajgoal
    cfg.actor_p_randomgoal = cfg.p_randomgoal
    cfg.value_geom_sample = True  # Use geometric sampling for value goals
    cfg.actor_geom_sample = False  # Use uniform sampling for actor goals
    cfg.gc_negative = True  # Use '0 if s == g else -1' reward format
    cfg.action_chunk_length = args.action_chunk_length
    cfg.use_mrn_metric = args.use_mrn_metric
    if args.mrn_components is not None:
        cfg.mrn_components = args.mrn_components

    np.random.seed(args.seed)

    # Load dataset and keep it in NumPy (CPU), matching original OGBench main.py.
    # All sampling logic in utils/datasets.py expects NumPy arrays and runs on CPU.
    import time
    with tqdm(total=4, desc="Loading dataset (NumPy)") as pbar:
        # 1) Load from disk as NumPy
        pbar.set_description("Loading from disk (NumPy)")
        data_np = np.load(args.dataset_path)
        obs = np.asarray(data_np["observations"])
        actions = np.asarray(data_np["actions"], dtype=np.float32)
        
        pbar.set_description(f"Resizing observations to {args.obs_h}x{args.obs_w}")
        batch_size = 1000
        obs_resized = np.zeros((obs.shape[0], args.obs_h, args.obs_w, 3), dtype=obs.dtype)
        for i in range(0, obs.shape[0], batch_size):
            end_idx = min(i + batch_size, obs.shape[0])
            batch = obs[i:end_idx]
            for j in range(len(batch)):
                obs_resized[i + j] = cv2.resize(batch[j], (args.obs_w, args.obs_h), interpolation=cv2.INTER_AREA)
        obs = obs_resized
        pbar.update(1)
        
        # 2) Get terminals as NumPy
        pbar.set_description("Processing terminals (NumPy)")
        terminals_np = _maybe_get_terminals_from_source(data_np)
        if terminals_np is None:
            total_frames = obs.shape[0]
            terminals = np.zeros(total_frames, dtype=bool)
            terminal_indices = np.arange(999, total_frames, 1000, dtype=int)
            terminals[terminal_indices] = True
        else:
            terminals = terminals_np.astype(bool).copy()
        terminals[-1] = True
        pbar.update(1)
        
        # 3) Optional filtering on CPU
        if not args.no_filter_intersections:
            pbar.set_description("Filtering intersection frames (NumPy)")
            filtered = filter_intersection_frames({
                "observations": obs,
                "actions": actions,
                "terminals": terminals,
            })
            obs = filtered["observations"]
            actions = filtered["actions"]
            terminals = filtered["terminals"]
        pbar.update(1)
        
        # 4) Split dataset on CPU by terminals
        pbar.set_description("Splitting dataset (NumPy)")
        train_data, val_data = split_dataset_by_terminals(
            {"observations": obs, "actions": actions, "terminals": terminals},
            val_fraction=0.2,
            seed=args.seed,
        )
        pbar.update(1)
    
    # Build datasets
    def build_gc_dataset(data: Dict) -> GCDataset | None:
        """Construct a goal-conditioned dataset from raw arrays.

        Important: `GCDataset` is written assuming NumPy arrays (like the original
        OGBench code). Here we explicitly convert the JAX arrays produced by
        `split_dataset_by_terminals_jax` back to NumPy before creating
        the `Dataset`, so that all sampling code runs purely on CPU/NumPy.
        This avoids costly JAX <-> NumPy/device transfers in the data loader.
        """
        if data["observations"].size == 0:
            return None
        dataset_fields = dict(
            observations=np.asarray(data["observations"]),
            actions=np.asarray(data["actions"]),
            terminals=np.asarray(data["terminals"]),
        )
        return GCDataset(Dataset.create(**dataset_fields), cfg)

    print(f"\nBuilding train dataset ({len(train_data['observations'])} frames, frame_stack={cfg.frame_stack})...", flush=True)
    t0 = time.time()
    train_dataset = build_gc_dataset(train_data)
    print(f"  Done in {time.time() - t0:.1f}s", flush=True)

    print(f"Building val dataset ({len(val_data['observations']) if val_data else 0} frames)...", flush=True)
    t0 = time.time()
    val_dataset = build_gc_dataset(val_data) if val_data else None
    print(f"  Done in {time.time() - t0:.1f}s", flush=True)


    example_batch = train_dataset.sample(min(10, cfg.batch_size))
    ex_obs_np = np.asarray(example_batch["observations"])
    ex_act_np = np.asarray(example_batch["actions"])
    print(f"Creating agent with example shapes: obs={ex_obs_np.shape}, actions={ex_act_np.shape}")
    import time
    agent_start = time.time()
    create_kwargs = {
        'seed': args.seed,
        'ex_observations': ex_obs_np,
        'ex_actions': ex_act_np,
        'config': cfg,
    }
    # TMD agent requires steps argument
    if args.algorithm.upper() == 'TMD':
        create_kwargs['steps'] = args.train_steps * args.epochs
    agent = agent_cls.create(**create_kwargs)
    print(f"Agent created successfully in {time.time() - agent_start:.2f}s (JIT compilation may happen on first forward pass)")
    
    # Pre-compile the update function with a dummy batch to avoid JIT compilation overhead during training
    print("Pre-compiling update function...")
    compile_start = time.time()
    example_batch = train_dataset.sample(min(10, cfg.batch_size))
    # Compile by running once (JAX will cache the compiled version)
    agent, _ = agent.update(example_batch)
    print(f"Update function compiled in {time.time() - compile_start:.2f}s")

    # Create checkpoint directory with algorithm and date
    date_str = datetime.now().strftime("%Y%m%d")
    ckpt_dir = Path(args.ckpt_dir) / f"{args.algorithm}_{date_str}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config to checkpoint directory
    config_path = ckpt_dir / "config.json"
    with config_path.open("w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    print(f"Saved config to {config_path}")
    
    wandb.init(project=args.project, config=cfg.to_dict())

    total_steps = 0
    last_ckpt_step = -1
    last_val_step = -1
    last_log_step = -1

    for epoch in range(args.epochs):
        print(f"\nStarting epoch {epoch + 1}/{args.epochs}")
        progress = trange(args.train_steps, dynamic_ncols=True)
        for step in range(args.train_steps):
            total_steps += 1
            t_progress_start = time.time()
            progress.update(1)
            t_progress_end = time.time()
            progress_ms = (t_progress_end - t_progress_start) * 1000.0
            
            # Timing breakdown
            t_sample_start = time.time()
            batch = train_dataset.sample(cfg.batch_size)
            t_sample_end = time.time()
            sample_ms = (t_sample_end - t_sample_start) * 1000.0

            t_update_start = time.time()
            agent, info = agent.update(batch)
            t_update_end = time.time()
            update_ms = (t_update_end - t_update_start) * 1000.0
            
            total_iter_ms = (t_update_end - t_sample_start) * 1000.0

            if total_steps % args.log_every == 0:
                t_wandb_start = time.time()
                print(f"\n[Timing @ step {total_steps}] progress={progress_ms:.2f}ms, sample={sample_ms:.1f}ms, update={update_ms:.1f}ms, total={total_iter_ms:.1f}ms ({1000.0/total_iter_ms:.2f} it/s)")
                log_dict: Dict[str, object] = {}
                metric_map = [
                    ("actor/bc_loss", "train/actor_bc_loss"),
                    ("actor/bc_log_prob", "train/actor_bc_log_prob"),
                    ("actor/q_loss", "train/actor_q_loss"),
                    ("actor/mse", "train/actor_mse"),
                    ("actor/mse_std", "train/actor_mse_std"),
                    ("actor/mse_max", "train/actor_mse_max"),
                    ("actor/mse_first", "train/actor_mse_first"),
                    ("actor/std", "train/actor_std"),
                    ("critic/contrastive_loss", "train/critic_loss"),
                    ("critic/critic_loss", "train/critic_loss"),  # TMD uses this key
                    ("critic/categorical_accuracy", "train/critic_categorical_accuracy"),
                    # DEBUG: Check if embeddings are collapsed
                    ("critic/phi_psi_similarity_raw", "train/critic_phi_psi_similarity_raw"),
                    ("critic/phi_batch_std_raw", "train/critic_phi_batch_std_raw"),
                    ("critic/psi_batch_std_raw", "train/critic_psi_batch_std_raw"),
                    ("critic/logits_pos_neg_diff", "train/critic_logits_pos_neg_diff"),
                    ("critic/logits_pos", "train/critic_logits_pos"),
                    ("critic/logits_neg", "train/critic_logits_neg"),
                    # DEBUG: Network output statistics
                    ("critic/phi_mean", "train/critic_phi_mean"),
                    ("critic/phi_std", "train/critic_phi_std"),
                    ("critic/psi_mean", "train/critic_psi_mean"),
                    ("critic/psi_std", "train/critic_psi_std"),
                    ("critic/phi_positive_frac", "train/critic_phi_positive_frac"),
                    ("critic/psi_positive_frac", "train/critic_psi_positive_frac"),
                    ("critic/v_mean_before_exp", "train/critic_v_mean_before_exp"),
                    ("critic/pos_neg_diff_raw", "train/critic_pos_neg_diff_raw"),
                    # Raw logit statistics for scaling diagnosis
                    ("critic/logits_raw_mean", "train/critic_logits_raw_mean"),
                    ("critic/logits_raw_std", "train/critic_logits_raw_std"),
                    ("critic/logits_raw_pos", "train/critic_logits_raw_pos"),
                    ("critic/logits_raw_neg", "train/critic_logits_raw_neg"),
                    ("critic/logits_raw_diff", "train/critic_logits_raw_diff"),
                ]
                for src_key, dst_name in metric_map:
                    if src_key in info:
                        log_dict[dst_name] = float(info[src_key])

                wandb.log(log_dict, step=total_steps)
                t_wandb_end = time.time()
                wandb_ms = (t_wandb_end - t_wandb_start) * 1000.0
                print(f"[Wandb logging] took {wandb_ms:.1f}ms")
                last_log_step = total_steps

            t_ckpt_start = time.time()
            if args.ckpt_every and total_steps % args.ckpt_every == 0:
                ckpt_path = ckpt_dir / f"agent_step{total_steps}.pkl"
                with ckpt_path.open("wb") as f:
                    f.write(fxs.to_bytes(agent))
                last_ckpt_step = total_steps
            t_ckpt_end = time.time()
            ckpt_ms = (t_ckpt_end - t_ckpt_start) * 1000.0
            if ckpt_ms > 1.0 and total_steps % args.log_every == 0:
                print(f"[Checkpoint] took {ckpt_ms:.1f}ms")

            # validation (use total_steps, not step, so validation works across epochs)
            t_val_check_start = time.time()
            if val_dataset is not None and total_steps % args.val_every == 0:
                t_val_start = time.time()
                val_metrics = compute_validation_loss(agent, val_dataset, batch_size=cfg.batch_size)
                t_val_end = time.time()
                val_ms = (t_val_end - t_val_start) * 1000.0
                wandb.log(val_metrics, step=total_steps)
                # Filter out non-numeric values (like wandb.Table) when printing
                numeric_metrics = {k: v for k, v in val_metrics.items() if isinstance(v, (int, float, np.number))}
                print(f"\nValidation @ step {total_steps}: " + ", ".join(f"{k}={v:.4f}" for k, v in numeric_metrics.items()))
                print(f"[Validation timing] took {val_ms:.1f}ms ({val_ms/1000:.2f}s)")
                last_val_step = total_steps
            t_val_check_end = time.time()
            val_check_ms = (t_val_check_end - t_val_check_start) * 1000.0
            if val_check_ms > 1.0 and total_steps % args.log_every == 0:
                print(f"[Validation check] took {val_check_ms:.1f}ms")
            
            # Total step time including everything
            t_step_end = time.time()
            total_step_ms = (t_step_end - t_progress_start) * 1000.0
            if total_steps % args.log_every == 0:
                print(f"[Total step time] {total_step_ms:.1f}ms ({1000.0/total_step_ms:.2f} it/s including all overhead)")

        # After epoch ends, checkpoint/validate/log if we haven't already this step
        if args.ckpt_every and total_steps != last_ckpt_step:
            ckpt_path = ckpt_dir / f"agent_step{total_steps}.pkl"
            with ckpt_path.open("wb") as f:
                f.write(fxs.to_bytes(agent))
            print(f"Checkpoint saved @ step {total_steps} (end of epoch {epoch + 1})")
            last_ckpt_step = total_steps

        if val_dataset is not None and total_steps != last_val_step:
            val_metrics = compute_validation_loss(agent, val_dataset, batch_size=cfg.batch_size)
            wandb.log(val_metrics, step=total_steps)
            numeric_metrics = {k: v for k, v in val_metrics.items() if isinstance(v, (int, float, np.number))}
            print(f"\nValidation @ step {total_steps} (end of epoch {epoch + 1}): " + ", ".join(f"{k}={v:.4f}" for k, v in numeric_metrics.items()))
            wandb.log(val_metrics, step=total_steps)
            last_val_step = total_steps

        # Log at end of epoch if we haven't already this step
        if total_steps != last_log_step:
            batch = train_dataset.sample(cfg.batch_size)
            _, info = agent.update(batch)

            log_dict: Dict[str, object] = {}
            key_subset = (
                "actor/bc_loss",
                "actor/q_loss",
                "actor/mse",
                "critic/contrastive_loss",
                "critic/categorical_accuracy",
            )
            for metric_key in key_subset:
                if metric_key in info:
                    log_dict[f"train/{metric_key.replace('/', '_')}"] = float(info[metric_key])
            
            wandb.log(log_dict, step=total_steps)
            last_log_step = total_steps

    # save final
    final_model_path = ckpt_dir / "final_model.pkl"
    print(f"Saving final model to {final_model_path}")
    with final_model_path.open("wb") as f:
        f.write(fxs.to_bytes(agent))

    # final validation
    if val_dataset is not None:
        final_val = compute_validation_loss(agent, val_dataset, batch_size=cfg.batch_size)
        # Filter out non-numeric values (like wandb.Table) when printing
        numeric_final_val = {k: float(v) for k, v in final_val.items() if isinstance(v, (int, float, np.number))}
        print("Final validation:", numeric_final_val)
        wandb.log(final_val, step=total_steps)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CRL/GCBC offline training (clean CPU-bypass design)")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to .npz offline dataset")
    parser.add_argument("--train_steps", type=int, default=800_000, help="Total gradient steps")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--actor_loss", choices=["awr", "ddpgbc"], default="ddpgbc")
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument(
        "--project",
        default=None,
        help="W&B project name (default: <algorithm>-training, e.g. CRL-training)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=100)  # Log more frequently for debugging
    parser.add_argument("--ckpt_every", type=int, default=50_000)
    parser.add_argument("--ckpt_dir", default="checkpoints")
    parser.add_argument("--algorithm", default="CRL")
    parser.add_argument("--block_size", type=int, default=400, help="Block size for block-aware frame stacking and shuffling")
    parser.add_argument("--chunk_size", type=int, default=50000, help="Process dataset in chunks of ~this many frames (trajectories are kept intact, not split)")
    parser.add_argument("--use_mmap", action="store_true", help="Use memory-mapped file loading (saves RAM but may be slower)")
    parser.add_argument("--val_every", type=int, default=500)
    parser.add_argument("--obs_h", type=int, default=100)
    parser.add_argument("--obs_w", type=int, default=100)
    parser.add_argument("--obs_c", type=int, default=3)
    parser.add_argument("--frame_offsets", nargs="*", type=int, default=None, help="e.g., --frame_offsets 0 -5 -10 -20")
    parser.add_argument("--action_chunk_length", type=int, default=1, help="Number of consecutive actions to predict (1 = no chunking)")
    parser.add_argument("--no_filter_intersections", action="store_true", help="Disable filtering of intersection/stationary frames")
    parser.add_argument("--use_mrn_metric", action="store_true", help="Enable MRN distance inside CRL contrastive loss")
    parser.add_argument("--mrn_components", type=int, default=None, help="Number of MRN components (requires --use_mrn_metric)")

    args = parser.parse_args()
    if args.project is None:
        args.project = f"{args.algorithm}-training"
    main(args)
