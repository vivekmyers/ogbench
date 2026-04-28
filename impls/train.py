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

from utils.datasets import CGCDataset, Dataset, GCDataset

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

    if len(terminal_locs) > 0:
        terminals_to_remove = int(jnp.sum(intersection_mask[terminal_locs]))
    else:
        terminals_to_remove = 0
    print(f"Terminals that would be removed (before preservation): {terminals_to_remove}")

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
    """Remove intersection / idle-y frames: high brake, or sustained low throttle.

    Downweights ambiguous stationary / crawling modes in BC. Terminal indices are
    preserved, then remapped after compaction.
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
    brake = actions[:, 2]
    low_throttle = throttle < throttle_threshold
    high_brake = brake > brake_threshold
    intersection_mask = np.zeros(T, dtype=bool)
    intersection_mask = intersection_mask | high_brake
    for i in range(T - window_size + 1):
        if np.all(low_throttle[i:i+window_size]):
            intersection_mask[i:i+window_size] = True

    # Count how many terminals would be removed BEFORE preserving them
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
    # print(f"Dataset loaded in {time.time() - start_time:.2f}s")

    # Get total size first (without loading full arrays)
    obs_shape = data_np["observations"].shape
    total_frames = obs_shape[0]
    # print(f"\n=== DATASET LOADING DEBUG ===")
    # print(f"Total frames in dataset: {total_frames}")
    # print(f"Processing in chunks of {chunk_size} frames to save memory...")

    # Load terminals first (small array)
    terminals = _maybe_get_terminals_from_source(data_np)
    if terminals is not None:
        terminals = terminals.astype(bool).copy()
        # print(f"Found {int(terminals.sum())} terminal markers in dataset file.")
    else:
        # Fallback: assume fixed-length trajectories (legacy datasets)
        # print("WARNING: Dataset missing terminal markers. Falling back to synthetic 1000-step boundaries.")
        terminals = np.zeros(total_frames, dtype=bool)
        trajectory_length = 1000
        terminal_indices = np.arange(trajectory_length - 1, total_frames, trajectory_length, dtype=int)
        terminals[terminal_indices] = True

    # Always mark last frame as terminal to close final trajectory
    terminals[-1] = True

    terminal_indices = np.where(terminals)[0]
    # print(f"Terminal count after initialization: {len(terminal_indices)}")
    if len(terminal_indices) > 0:
        # print(f"Terminal indices (first 10): {terminal_indices[:10]}")
        #if len(terminal_indices) > 10:
            # print(f"Terminal indices (last 10): {terminal_indices[-10:]}")
        traj_lengths = np.diff(np.concatenate([[0], terminal_indices + 1]))
        # print(
        #     "Trajectory length stats (before filtering): "
        #     f"min={traj_lengths.min()}, max={traj_lengths.max()}, mean={traj_lengths.mean():.1f}"
        # )

    # Process filtering in chunks, keeping trajectories together
    # print(f"Grouping trajectories into chunks of ~{chunk_size} frames...")
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
    # print(f"Found {num_trajectories} trajectories (avg length: {trajectory_lengths.mean():.1f} frames)")
    
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
            
            # print(f"Processing chunk {chunk_idx + 1} ({len(chunk_trajectories)} trajectories, {chunk_frame_count} frames)...", end=\"\\r\")
            
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
        
        # print(f"Processing chunk {chunk_idx + 1} ({len(chunk_trajectories)} trajectories, {chunk_frame_count} frames)...", end=\"\\r\")
        
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
    
    # print(f\"\nFiltering completed in {time.time() - filter_start:.2f}s\")
    # print(f\"Processed {len(filtered_chunks)} chunks (all trajectories kept intact)\")
    # print(\"Concatenating filtered chunks...\")
    
    # Concatenate all filtered chunks
    filtered = {
        "observations": np.concatenate([chunk["observations"] for chunk in filtered_chunks], axis=0),
        "actions": np.concatenate([chunk["actions"] for chunk in filtered_chunks], axis=0),
        "terminals": np.concatenate([chunk["terminals"] for chunk in filtered_chunks], axis=0),
    }
    obs = filtered["observations"]
    actions = filtered["actions"]
    terminals = filtered["terminals"]
    # print(f\"\n=== AFTER FILTERING (before frame stacking) ===\")
    # print(f\"Frames remaining: {len(obs)}\")
    # print(f\"Terminals remaining: {np.sum(terminals)}\")
    terminal_locs_after = np.where(terminals)[0]
    if len(terminal_locs_after) > 0:
        # print(f\"Terminal locations (first 10): {terminal_locs_after[:10]}\")
        traj_lengths_after = np.diff(np.concatenate([[0], terminal_locs_after + 1]))
        # print(f\"Trajectory lengths (first 10): {traj_lengths_after[:10]}\")
        # print(f\"Trajectory lengths (stats): min={traj_lengths_after.min()}, max={traj_lengths_after.max()}, mean={traj_lengths_after.mean():.1f}\")

    # print(f\"Loaded (unstacked): obs={obs.shape}, actions={actions.shape}, terminals={terminals.shape}\")
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
    
    # Any length-T array besides the explicitly-handled ones (observations,
    # actions, terminals, next_observations) gets carried through verbatim per
    # trajectory. This lets callers pipe auxiliary per-frame fields (e.g.
    # `position`) into GCDataset without touching this function again.
    T = len(dataset["observations"])
    explicit_keys = {"observations", "actions", "terminals", "next_observations"}
    extra_keys = [
        k for k, v in dataset.items()
        if k not in explicit_keys
        and isinstance(v, np.ndarray)
        and len(v) == T
    ]

    # Gather trajectories by actual boundaries
    train_obs_list = []
    train_act_list = []
    train_term_list = []
    train_next_list = [] if "next_observations" in dataset else None
    train_extra_lists: Dict[str, list] = {k: [] for k in extra_keys}

    val_obs_list = []
    val_act_list = []
    val_term_list = []
    val_next_list = [] if "next_observations" in dataset else None
    val_extra_lists: Dict[str, list] = {k: [] for k in extra_keys}

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
        for k in extra_keys:
            train_extra_lists[k].append(dataset[k][start:end])

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
        for k in extra_keys:
            val_extra_lists[k].append(dataset[k][start:end])

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
    for k in extra_keys:
        if train_extra_lists[k]:
            train_data[k] = np.concatenate(train_extra_lists[k], axis=0)
        else:
            train_data[k] = np.zeros((0, *dataset[k].shape[1:]), dtype=dataset[k].dtype)

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
    for k in extra_keys:
        if val_extra_lists[k]:
            val_data[k] = np.concatenate(val_extra_lists[k], axis=0)
        else:
            val_data[k] = np.zeros((0, *dataset[k].shape[1:]), dtype=dataset[k].dtype)
    
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


def log_hard_val_frames(agent, val_dataset, batch_size: int, k: int):
    """Sample a clean val batch, score per-example MSE, and package the
    top-`k` hardest examples as a list of wandb.Images (current-obs | goal).

    Returns a dict like {"val/hard_frames": [wandb.Image, ...]} or {} when
    logging isn't applicable (no dataset, k<=0, non-image obs, or the actor
    can't be called). Goal frames come from `batch['actor_goals']`; both
    obs and goal are sliced to their last `obs_c` channels (current frame of
    the stack) so the figure shows "what the agent saw right now" next to
    "where it was told to go".
    """
    if val_dataset is None or k <= 0 or val_dataset.size == 0:
        return {}
    actual_batch = min(batch_size, val_dataset.size)
    batch = val_dataset.sample(actual_batch, evaluation=True)

    obs = batch.get('observations')
    goals = batch.get('actor_goals')
    if obs is None or goals is None:
        return {}
    obs_np = np.asarray(obs)
    goals_np = np.asarray(goals)
    if obs_np.ndim < 4:
        return {}

    try:
        dist = agent.network.select('actor')(obs, goals, params=agent.network.params)
        predicted = np.asarray(dist.mode())
    except Exception:
        return {}

    if 'action_chunks' in batch:
        targets_raw = np.asarray(batch['action_chunks'])
    elif 'actions' in batch:
        targets_raw = np.asarray(batch['actions'])
    else:
        return {}
    targets = targets_raw.reshape(predicted.shape[0], -1)
    if targets.shape != predicted.shape:
        min_d = min(targets.shape[-1], predicted.shape[-1])
        targets = targets[:, :min_d]
        predicted = predicted[:, :min_d]

    # Rank by mse_first (first action in chunk), not full flattened mse.
    # For chunk_len==1 this matches per-step MSE. For chunk_len>1 it focuses the
    # "what should I do right now?" error signal.
    if targets_raw.ndim == 3:
        action_dim = int(targets_raw.shape[-1])
    elif "actions" in batch:
        action_dim = int(np.asarray(batch["actions"]).shape[-1])
    else:
        action_dim = 3
    action_dim = max(1, action_dim)
    pred_first = predicted[:, :action_dim]
    tgt_first = targets[:, :action_dim]
    mse_first_per = np.mean((pred_first - tgt_first) ** 2, axis=-1)
    k_eff = int(min(k, predicted.shape[0]))
    top = np.argsort(-mse_first_per)[:k_eff]

    obs_c = int(val_dataset.config.get('obs_c', 3))
    obs_c = max(1, min(obs_c, obs_np.shape[-1]))
    cur_obs = obs_np[..., -obs_c:]
    cur_goal = goals_np[..., -obs_c:] if goals_np.shape[-1] >= obs_c else goals_np

    def _to_uint8(a):
        if a.dtype == np.uint8:
            return a
        return np.clip(a, 0, 255).astype(np.uint8)

    # Pull per-sample goal source tags (added by GCDataset.sample when
    # evaluation=True). 0=cur, 1=traj, 2=random; fall back to 'unknown' if
    # the dataset didn't provide them (e.g. HGCDataset in old configs).
    _src_names = {0: "cur", 1: "traj", 2: "rand"}
    sources_np = batch.get('actor_goal_sources')
    if sources_np is not None:
        sources_np = np.asarray(sources_np).astype(np.int8)

    images = []
    rows = []
    for i in top:
        i = int(i)
        panel = np.concatenate([cur_obs[i], cur_goal[i]], axis=1)
        panel = _to_uint8(panel)
        if panel.shape[-1] == 1:
            panel = np.repeat(panel, 3, axis=-1)
        tgt3 = np.round(targets[i][:action_dim], 3).tolist()
        prd3 = np.round(predicted[i][:action_dim], 3).tolist()
        src_tag = (
            _src_names.get(int(sources_np[i]), "?") if sources_np is not None else "?"
        )
        cap = (
            f"[src={src_tag}] mse_first={float(mse_first_per[i]):.3f} | "
            f"tgt(t0)={tgt3} | pred(t0)={prd3}"
        )
        img = wandb.Image(panel, caption=cap)
        images.append(img)
        rows.append([i, float(mse_first_per[i]), src_tag, tgt3, prd3, img])

    # NOTE: Logging a list of images under the same key can make W&B spawn a new
    # media panel on every call. A Table is much more stable in the UI.
    out: Dict[str, object] = {"val/hard_frames": images}
    out["val/hard_frames_table"] = wandb.Table(
        columns=["val_idx", "mse_first", "goal_source", "tgt_t0", "pred_t0", "image"],
        data=rows,
    )

    # Per-source mean MSE_FIRST over the full val batch — cheap, and tells you at a
    # glance whether the hardness is concentrated in random-goal samples or
    # traj-goal samples.
    if sources_np is not None:
        for src_id, name in _src_names.items():
            mask = sources_np == src_id
            if np.any(mask):
                out[f"val/mse_first_by_goal_source/{name}"] = float(mse_first_per[mask].mean())
                out[f"val/count_by_goal_source/{name}"] = int(mask.sum())
    return out


def compute_validation_loss(agent, val_dataset: GCDataset | None, batch_size: int = 1024):
    """Run a validation pass using the goal-conditioned dataset sampler."""
    if val_dataset is None or val_dataset.size == 0:
        return {}

    actual_batch = min(batch_size, val_dataset.size)
    # evaluation=True: no p_aug, no action_noise — val loss matches clean targets.
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
    # GCBC uses different naming: actor_loss is the BC NLL (=-log_prob.mean()).
    # Log it under the same key so dashboards stay consistent across agents.
    elif "actor_loss" in actor_info:
        metrics["val/actor_bc_loss"] = float(actor_info["actor_loss"])
    if "bc_log_prob" in actor_info:
        metrics["val/actor_bc_log_prob"] = float(actor_info["bc_log_prob"])
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
    # multi_discrete: 3 independent 32-bin categorical heads over (throttle,
    # steer, brake). Handles throttle/brake multimodality that a single Gaussian
    # averages into "mild acceleration". Continuous from the outside -- the
    # actor still takes and returns continuous actions; only the loss and output
    # head are categorical internally. Forces action_chunk_length=1 since the
    # current head only outputs num_bins*num_dims logits (no chunk support).
    cfg.multi_discrete = bool(args.multi_discrete)
    cfg.num_bins_per_dim = 32
    cfg.multi_discrete_num_dims = 3
    cfg.discrete = bool(args.multi_discrete)  # Gate inside GCBCAgent.
    if args.multi_discrete and int(args.action_chunk_length) != 1:
        print(
            f"[train] multi_discrete head only supports action_chunk_length=1; "
            f"overriding requested value ({int(args.action_chunk_length)}) to 1."
        )
    cfg.actor_loss = "ddpgbc"
    cfg.discount = args.discount
    cfg.alpha = 0.01
    cfg.encoder = "impala_small"
    cfg.lr = 3e-4
    cfg.actor_hidden_dims = (2048, 2048, 2048)
    cfg.value_hidden_dims = (2048, 2048, 2048)
    cfg.latent_dim = 512
    cfg.critic_lr_scale = 1.0
    cfg.actor_lr_scale = 1.0
    if args.frame_offsets is not None:
        cfg.frame_offsets = tuple(args.frame_offsets)
        cfg.frame_stack = len(args.frame_offsets)
    else:
        cfg.frame_offsets = (0, -1)
        cfg.frame_stack = 2
    # Default window == K for canonical single-frame observations.
    _default_frame_stack_window = int(cfg.frame_stack)
    cfg.frame_stack_window = (
        int(args.frame_stack_window) if args.frame_stack_window is not None else _default_frame_stack_window
    )
    if cfg.frame_stack_window > cfg.frame_stack:
        print(
            f"[train] frame_stack_window={cfg.frame_stack_window} > frame_stack={cfg.frame_stack}: "
            f"per-batch random K-of-W offset sampling at train time (current frame always "
            f"included). Validation uses canonical consecutive stack."
        )
    # Saved in config.json so eval server/client match training geometry (no CLI guessing).
    cfg.obs_h = args.obs_h
    cfg.obs_w = args.obs_w
    cfg.obs_c = args.obs_c
    # Actor goal images: default to the *same* temporal pattern as observations.
    # Override with --goal-frame-stack / --goal-frame-offsets only when goals need
    # a different stack (then GCDataset may retain a raw single-frame buffer).
    cfg.goal_frame_stack = int(cfg.frame_stack)
    cfg.goal_frame_offsets = tuple(cfg.frame_offsets)
    if args.goal_frame_offsets is not None:
        cfg.goal_frame_offsets = tuple(int(x) for x in args.goal_frame_offsets)
        if args.goal_frame_stack is None:
            cfg.goal_frame_stack = len(cfg.goal_frame_offsets)
        else:
            cfg.goal_frame_stack = int(args.goal_frame_stack)
        if len(cfg.goal_frame_offsets) != int(cfg.goal_frame_stack):
            raise ValueError(
                f"--goal-frame-offsets has {len(cfg.goal_frame_offsets)} entries but "
                f"--goal-frame-stack={cfg.goal_frame_stack}."
            )
    elif args.goal_frame_stack is not None:
        cfg.goal_frame_stack = int(args.goal_frame_stack)
        # Depth changed vs observation offsets → use consecutive offsets for this K only.
        if cfg.goal_frame_stack != len(tuple(cfg.frame_offsets)):
            cfg.goal_frame_offsets = None
    cfg.block_size = args.block_size
    cfg.p_aug = 0.8
    # action_noise_std: scalar for shared std, sequence for per-dim stds.
    _ans = list(args.action_noise_std)
    cfg.action_noise_std = float(_ans[0]) if len(_ans) == 1 else tuple(float(x) for x in _ans)
    cfg.action_noise_chunk_corr = float(args.action_noise_chunk_corr)
    cfg.action_noise_clip = not args.no_action_noise_clip
    cfg.action_stack_noise_std = float(getattr(args, "action_stack_noise_std", 0.0) or 0.0)
    # Downsample high-brake frames: there are many "stopped at light" frames in
    # the dataset that don't matter at eval. Mode picks the target set,
    # `upsample_weight` is literal weight multiplier -- <1 means "pick these
    # less often" (w/brake_thresh=0.1 → brake>0.1 ~stops get weight 0.3 vs 1.0
    # for everything else, after normalization).
    cfg.upsample_mode = "turns_high_approach"
    cfg.turn_approach_horizon = 12
    cfg.upsample_weight = 15.0
    cfg.steer_thresh = 0.1
    cfg.throttle_thresh = 0.3
    cfg.brake_thresh = 0.3
    # delta for Huber loss when actor_loss == "huber".
    cfg.huber_delta = 0.1
    cfg.longitudinal_balance = not bool(getattr(args, "no_longitudinal_balance", False))
    cfg.lb_train_fps = 10.0  # dataset cadence (Hz) used to convert Δposition → m/s
    cfg.lb_stopped_speed_thresh = 0.5  # m/s: "stopped"
    cfg.lb_accel_speed_thresh = 2.0    # m/s: "still low speed" for accel-from-stop
    cfg.lb_decel_speed_thresh = 2.0    # m/s: moving fast enough to be decisive decel
    cfg.lb_brake_high = 0.3            # "strong brake"
    cfg.lb_brake_low = 0.05            # "not braking"
    cfg.lb_throttle_high = 0.2         # "applying throttle"
    cfg.lb_mixed_thr_lo = 0.05         # mixed: both throttle & brake above these
    cfg.lb_mixed_brk_lo = 0.05
    cfg.lb_w_stopped_brake = 0.07      # downsample stopped-with-brake heavily
    cfg.lb_w_accel_from_stop = 8.0     # upsample accel-from-stop
    cfg.lb_w_mixed = 0.1               # downsample noisy/mixed longitudinal frames
    cfg.lb_w_decisive_decel = 2.5      # modestly upsample decisive decel
    # Value (critic) goals: must be reachable from s for contrastive positives to
    # be valid. TMD's phi(s,a) <-> psi(g) target is meaningless if g is from a
    # different trajectory, so keep value_p_randomgoal small (or 0).
    cfg.value_p_curgoal = 0.0
    cfg.value_p_trajgoal = float(args.value_p_trajgoal)
    cfg.value_p_randomgoal = float(args.value_p_randomgoal)
    cfg.value_geom_sample = True

    # Actor goals: random goals across the dataset are useful for goal-conditioning
    # generalisation in BC; the actor doesn't need the goal to be reachable from s.
    cfg.actor_p_curgoal = 0.0
    cfg.actor_p_trajgoal = float(args.actor_p_trajgoal)
    cfg.actor_p_randomgoal = float(args.actor_p_randomgoal)
    cfg.actor_geom_sample = True
    cfg.gc_negative = True  # Use '0 if s == g else -1' reward format
    cfg.action_chunk_length = 1 if args.multi_discrete else args.action_chunk_length
    cfg.action_stack_length = int(args.action_stack_length)
    cfg.use_mrn_metric = args.use_mrn_metric
    if args.mrn_components is not None:
        cfg.mrn_components = args.mrn_components
        cfg.components = args.mrn_components

    if args.algorithm.upper() == 'TMD_DQC':
        cfg.policy_chunk_size = args.action_chunk_length
        cfg.action_chunk_length = args.action_chunk_length
        cfg.backup_horizon = args.backup_horizon

    if args.algorithm.upper() in ('TMD', 'TMD_QC', 'TMD_DQC', 'TMD_DC'):
        cfg.bc_goal_randomize_prob = float(args.bc_goal_randomize_prob)

    # GCBC and BC don't read next_observations / value_goals / chunk_next_observations.
    # Flipping minimal_batch on cuts ~3 of 5 per-batch fancy-index gathers,
    # which is the bulk of sample() time once frame_stack_window > frame_stack.
    cfg.minimal_batch = args.algorithm.upper() in ('GCBC', 'BC')
    if cfg.minimal_batch:
        print(
            f"[train] minimal_batch=True ({args.algorithm}): GCDataset.sample() will "
            f"skip next_observations / value_goals / chunk_next_observations gathers."
        )

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
        position = np.asarray(data_np["position"], dtype=np.float32) if "position" in data_np.files else None
        
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
            filter_inputs = {
                "observations": obs,
                "actions": actions,
                "terminals": terminals,
            }
            if position is not None:
                filter_inputs["position"] = position
            filtered = filter_intersection_frames(filter_inputs)
            obs = filtered["observations"]
            actions = filtered["actions"]
            terminals = filtered["terminals"]
            position = filtered.get("position", position)
        pbar.update(1)
        
        # 4) Split dataset on CPU by terminals
        pbar.set_description("Splitting dataset (NumPy)")
        split_inputs = {"observations": obs, "actions": actions, "terminals": terminals}
        if position is not None:
            split_inputs["position"] = position
        train_data, val_data = split_dataset_by_terminals(
            split_inputs,
            val_fraction=0.1,
            seed=args.seed,
        )
        pbar.update(1)
    
    # Build datasets
    def build_gc_dataset(data: Dict, *, for_validation: bool = False) -> GCDataset | CGCDataset | None:
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
        # Optional auxiliary per-frame fields (e.g. `position` for longitudinal
        # rebalancing in GCDataset.) Carry them into the Dataset only when
        # present; sampling ignores them unless a feature explicitly reads them.
        if "position" in data and isinstance(data["position"], np.ndarray) and data["position"].size > 0:
            dataset_fields["position"] = np.asarray(data["position"])
        ds = Dataset.create(**dataset_fields)
        if args.algorithm.upper() == 'TMD_DQC':
            return CGCDataset(ds, cfg, sampling_mode=("val" if for_validation else "train"))
        return GCDataset(ds, cfg, sampling_mode=("val" if for_validation else "train"))

    _ans_cfg = cfg.action_noise_std
    _noise_on = (isinstance(_ans_cfg, (list, tuple)) and any(float(x) > 0.0 for x in _ans_cfg)) \
                or (not isinstance(_ans_cfg, (list, tuple)) and float(_ans_cfg) > 0.0)
    if _noise_on:
        print(
            f"[train] action_noise_std={_ans_cfg} "
            f"chunk_corr={cfg.action_noise_chunk_corr} "
            f"clip={cfg.action_noise_clip} — "
            f"Gaussian perturbation will be applied to actions and action_chunks "
            f"on every (non-evaluation) batch.",
            flush=True,
        )
    print(
        f"\nBuilding train dataset ({len(train_data['observations'])} frames, "
        f"frame_stack={cfg.frame_stack}, frame_stack_window={cfg.frame_stack_window}, "
        f"action_stack_length={cfg.action_stack_length})...",
        flush=True,
    )
    t0 = time.time()
    train_dataset = build_gc_dataset(train_data)
    # print(f"  Done in {time.time() - t0:.1f}s", flush=True)

    print(f"Building val dataset ({len(val_data['observations']) if val_data else 0} frames)...", flush=True)
    t0 = time.time()
    val_dataset = build_gc_dataset(val_data, for_validation=True) if val_data else None
    # print(f"  Done in {time.time() - t0:.1f}s", flush=True)


    example_batch = train_dataset.sample(min(10, cfg.batch_size))
    ex_obs_np = np.asarray(example_batch["observations"])
    ex_act_np = np.asarray(example_batch["actions"])
    ex_goals_np = np.asarray(example_batch["actor_goals"])
    ex_stack_msg = (
        f", action_stack={np.asarray(example_batch['action_stack']).shape}"
        if int(cfg.action_stack_length) > 1
        else ""
    )
    print(
        f"Creating agent with example shapes: obs={ex_obs_np.shape}, "
        f"goals={ex_goals_np.shape}, actions={ex_act_np.shape}{ex_stack_msg}"
    )
    import time
    agent_start = time.time()
    create_kwargs = {
        'seed': args.seed,
        'ex_observations': ex_obs_np,
        'ex_actions': ex_act_np,
        'config': cfg,
    }
    if args.algorithm.upper() == "GCBC":
        create_kwargs["ex_goals"] = ex_goals_np
    # TMD agent requires steps argument
    if args.algorithm.upper() in ('TMD', 'TMD_QC', 'TMD_DQC', 'TMD_DC'):
        create_kwargs['steps'] = args.steps * args.epochs
    if int(cfg.action_stack_length) > 1:
        B_ex = int(ex_obs_np.shape[0])
        ex_as = example_batch.get("action_stack")
        if ex_as is not None:
            # Must match ex_observations / ex_goals batch dim for Flax init concat.
            create_kwargs["ex_action_stack"] = np.asarray(ex_as)[:B_ex]
        else:
            a_dim = int(ex_act_np.shape[-1])
            create_kwargs["ex_action_stack"] = np.zeros(
                (B_ex, int(cfg.action_stack_length), a_dim), dtype=np.float32
            )
    else:
        create_kwargs["ex_action_stack"] = None
    agent = agent_cls.create(**create_kwargs)
    # print(f"Agent created successfully in {time.time() - agent_start:.2f}s (JIT compilation may happen on first forward pass)")
    
    # Pre-compile the update function with a dummy batch to avoid JIT compilation overhead during training
    # print("Pre-compiling update function...")
    compile_start = time.time()
    example_batch = train_dataset.sample(min(10, cfg.batch_size))
    # Compile by running once (JAX will cache the compiled version)
    agent, _ = agent.update(example_batch)
    # print(f"Update function compiled in {time.time() - compile_start:.2f}s")

    # Create checkpoint directory with algorithm and date
    date_str = datetime.now().strftime("%Y%m%d")
    ckpt_dir = Path(args.ckpt_dir) / f"{args.algorithm}_{date_str}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config to checkpoint directory
    config_path = ckpt_dir / "config.json"
    with config_path.open("w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    # print(f"Saved config to {config_path}")
    
    wandb.init(project=args.project, config=cfg.to_dict())

    total_steps = 0
    last_ckpt_step = -1
    last_val_step = -1
    last_log_step = -1

    for epoch in range(args.epochs):
        # print(f"\nStarting epoch {epoch + 1}/{args.epochs}")
        progress = trange(args.steps, dynamic_ncols=True)
        for step in range(args.steps):
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
                # print(f"\n[Timing @ step {total_steps}] progress={progress_ms:.2f}ms, sample={sample_ms:.1f}ms, update={update_ms:.1f}ms, total={total_iter_ms:.1f}ms ({1000.0/total_iter_ms:.2f} it/s)")
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
                # print(f"[Wandb logging] took {wandb_ms:.1f}ms")
                last_log_step = total_steps

            t_ckpt_start = time.time()
            if args.ckpt_every and total_steps % args.ckpt_every == 0:
                ckpt_path = ckpt_dir / f"agent_step{total_steps}.pkl"
                with ckpt_path.open("wb") as f:
                    f.write(fxs.to_bytes(agent))
                last_ckpt_step = total_steps
            t_ckpt_end = time.time()
            ckpt_ms = (t_ckpt_end - t_ckpt_start) * 1000.0
            # if ckpt_ms > 1.0 and total_steps % args.log_every == 0:
            #     print(f"[Checkpoint] took {ckpt_ms:.1f}ms")

            # validation (use total_steps, not step, so validation works across epochs)
            t_val_check_start = time.time()
            if val_dataset is not None and total_steps % args.val_every == 0:
                t_val_start = time.time()
                val_metrics = compute_validation_loss(agent, val_dataset, batch_size=cfg.batch_size)
                t_val_end = time.time()
                val_ms = (t_val_end - t_val_start) * 1000.0
                wandb.log(val_metrics, step=total_steps)
                hard_frames = log_hard_val_frames(
                    agent, val_dataset, cfg.batch_size, args.log_hard_frames_k,
                )
                if hard_frames:
                    wandb.log(hard_frames, step=total_steps)
                # Filter out non-numeric values (like wandb.Table) when printing
                numeric_metrics = {k: v for k, v in val_metrics.items() if isinstance(v, (int, float, np.number))}
                # print(f"\nValidation @ step {total_steps}: " + ", ".join(f"{k}={v:.4f}" for k, v in numeric_metrics.items()))
                # print(f"[Validation timing] took {val_ms:.1f}ms ({val_ms/1000:.2f}s)")
                last_val_step = total_steps
            t_val_check_end = time.time()
            val_check_ms = (t_val_check_end - t_val_check_start) * 1000.0
            # if val_check_ms > 1.0 and total_steps % args.log_every == 0:
            #     print(f"[Validation check] took {val_check_ms:.1f}ms")
            
            # Total step time including everything
            t_step_end = time.time()
            total_step_ms = (t_step_end - t_progress_start) * 1000.0
            # if total_steps % args.log_every == 0:
            #     print(f"[Total step time] {total_step_ms:.1f}ms ({1000.0/total_step_ms:.2f} it/s including all overhead)")

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
            hard_frames = log_hard_val_frames(
                agent, val_dataset, cfg.batch_size, args.log_hard_frames_k,
            )
            if hard_frames:
                wandb.log(hard_frames, step=total_steps)
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
    parser.add_argument("--steps", type=int, default=800_000, help="Total gradient steps")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--actor_loss", choices=["awr", "ddpgbc", "huber"], default="ddpgbc")
    parser.add_argument("--discount", type=float, default=0.95)
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
    parser.add_argument("--obs_h", type=int, default=64)
    parser.add_argument("--obs_w", type=int, default=64)
    parser.add_argument("--obs_c", type=int, default=3)
    parser.add_argument("--frame_offsets", nargs="*", type=int, default=None, help="e.g., --frame_offsets 0 -5 -10 -20")
    parser.add_argument(
        "--goal-frame-stack",
        type=int,
        default=None,
        help=(
            "Actor goal temporal depth K. Default: same as observation frame_stack. "
            "Pass a different K only if goal images should use another stack length; "
            "if K != len(--frame_offsets), offsets default to consecutive lags for that K. "
            "When goals use a different pattern than observations, training keeps one extra "
            "single-frame observation copy in RAM."
        ),
    )
    parser.add_argument(
        "--goal-frame-offsets",
        nargs="*",
        type=int,
        default=None,
        help=(
            "Same convention as --frame_offsets (<=0, must include 0). Default: copy "
            "--frame_offsets. Infer K from len(offsets) if --goal-frame-stack omitted."
        ),
    )
    parser.add_argument(
        "--frame_stack_window",
        type=int,
        default=None,
        help=(
            "Lookback window (dataset cadence) for building the frame stack. Default 30: with "
            "the default K=3 stack, offset 0 is always the current frame and the other K-1 "
            "history slots are sampled uniformly without replacement from [1, W-1] at train time. "
            "Set equal to frame_stack for canonical consecutive stacking only (no random subsampling)."
        ),
    )
    parser.add_argument("--action_chunk_length", type=int, default=1, help="Number of consecutive actions to predict (1 = no chunking)")
    parser.add_argument(
        "--action_stack_length",
        type=int,
        default=1,
        help=(
            "Concatenate this many past actions (oldest first) to the actor MLP after the image encoder. "
            "1 = disabled (default). Independent of --action_chunk_length (future action chunking)."
        ),
    )
    parser.add_argument(
        "--backup_horizon",
        type=int,
        default=25,
        help="Chunk critic horizon H for TMD_DQC (must be <= trajectory length; CGCDataset samples only valid starts).",
    )
    parser.add_argument(
        "--bc_goal_randomize_prob",
        type=float,
        default=1.0,
        help="TMD / TMD_QC / TMD_DQC only: per-batch-row probability of replacing the goal with another goal from the same batch for the BC log-likelihood only (critic and Q path unchanged).",
    )
    parser.add_argument(
        "--log_hard_frames_k", type=int, default=8,
        help=(
            "At each validation step, log the top-K highest-MSE val frames "
            "(current observation | actor goal) as a W&B image gallery under "
            "'val/hard_frames'. Set to 0 to disable."
        ),
    )
    parser.add_argument("--no_filter_intersections", action="store_true", help="Disable filtering of intersection/stationary frames")
    parser.add_argument("--use_mrn_metric", action="store_true", help="Enable MRN distance inside CRL contrastive loss")
    parser.add_argument("--mrn_components", type=int, default=None, help="Number of MRN components (requires --use_mrn_metric)")
    parser.add_argument(
        # Default back ON (CARLA): mild per-dim noise [throttle, steer, brake].
        "--action_noise_std", type=float, nargs="+", default=[0.00, 0.00, 0.00],
    )
    parser.add_argument(
        "--action_stack_noise_std",
        type=float,
        default=0.0,
        help="Stddev of i.i.d. Gaussian noise N(0,s) added to batch['action_stack'] during training (default 0 = off).",
    )
    parser.add_argument(
        "--action_noise_chunk_corr", type=float, default=0.3,
    )
    parser.add_argument("--no_action_noise_clip", action="store_true",
                        help="Disable clipping of noised actions to [-1, 1] (default: clip on).")
    parser.add_argument(
        "--value_p_trajgoal", type=float, default=0.9,
    )
    parser.add_argument(
        "--value_p_randomgoal", type=float, default=0.1,
    )
    parser.add_argument(
        "--actor_p_trajgoal", type=float, default=0.5,
    )
    parser.add_argument(
        "--actor_p_randomgoal", type=float, default=0.5,
    )
    parser.add_argument(
        "--longitudinal_balance",
        action="store_true",
        help=(
            "[DEPRECATED: now on by default] Apply multiplicative longitudinal rebalancing on top of "
            "upsample_mode: downsample stopped-with-brake and mixed "
            "throttle+brake frames, upsample accel-from-stop and decisive "
            "deceleration. Requires 'position' in the dataset .npz."
        ),
    )
    parser.add_argument(
        "--no_longitudinal_balance",
        action="store_true",
        help="Disable longitudinal rebalancing (enabled by default).",
    )
    parser.add_argument(
        "--multi_discrete",
        action="store_true",
        help=(
            "GCBC only: model actions as 3 independent 32-bin categoricals "
            "over (throttle, steer, brake). Handles the throttle/brake "
            "multimodality that a Gaussian cannot (e.g. 'floor it' vs "
            "'slam brakes' at the same visual). Forces action_chunk_length=1."
        ),
    )

    args = parser.parse_args()
    if args.project is None:
        args.project = f"{args.algorithm}-training"
    main(args)
