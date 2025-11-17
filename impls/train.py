from __future__ import annotations

import argparse
import time
from importlib import import_module
from pathlib import Path
from typing import Dict, Tuple

import flax.serialization as fxs
from flax.core import freeze, unfreeze
import jax
import jax.numpy as jnp
from jax import tree_util
import ml_collections
import numpy as np
import wandb
from tqdm import trange

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

def filter_intersection_frames(dataset: Dict[str, np.ndarray], throttle_threshold: float = 0.05, brake_threshold: float = 0.1,
    window_size: int = 5,
) -> Dict[str, np.ndarray]:
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
    
    # CRITICAL: Preserve terminal markers - don't remove terminal frames
    # If a terminal frame would be removed, keep it and mark the frame before as terminal instead
    terminal_indices = np.where(terminals)[0]
    
    # Mark terminal frames as "must keep" to preserve trajectory boundaries
    terminals_preserved = 0
    for term_idx in terminal_indices:
        if intersection_mask[term_idx]:
            # Terminal frame would be removed - keep it to preserve boundary
            intersection_mask[term_idx] = False
            terminals_preserved += 1
            # Also ensure the frame before terminal is kept (if it exists)
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


def load_dataset_cpu(path: str | Path, frame_offsets: Tuple[int, ...], block_size: int) -> Dict[str, np.ndarray]:
    print(f"Loading dataset from {path} ...")
    data_np = np.load(path)

    obs = np.asarray(data_np["observations"])
    actions = np.asarray(data_np["actions"], dtype=np.float32)
    
    print(f"\n=== DATASET LOADING DEBUG ===")
    print(f"Total frames in dataset: {len(obs)}")

    terminals = _maybe_get_terminals_from_source(data_np)
    if terminals is not None:
        terminals = terminals.astype(bool).copy()
        print(f"Found {int(terminals.sum())} terminal markers in dataset file.")
    else:
        # Fallback: assume fixed-length trajectories (legacy datasets)
        print("WARNING: Dataset missing terminal markers. Falling back to synthetic 1000-step boundaries.")
        terminals = np.zeros(len(obs), dtype=bool)
        trajectory_length = 1000
        terminal_indices = np.arange(trajectory_length - 1, len(obs), trajectory_length, dtype=int)
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

    # Always filter intersection frames (will preserve terminal markers)
    filtered = filter_intersection_frames({
        "observations": obs,
        "actions": actions,
        "terminals": terminals,
    })
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


def numpy_to_jax(batch: Dict[str, np.ndarray]) -> Dict[str, jnp.ndarray]:
    return tree_util.tree_map(
        lambda x: jnp.asarray(x) if isinstance(x, np.ndarray) else x,
        batch,
    )


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
    batch_np = val_dataset.sample(actual_batch, evaluation=True)
    batch = numpy_to_jax(batch_np)

    for k in ("observations", "actions", "actor_goals", "value_goals"):
        if k in batch and jnp.any(jnp.isnan(batch[k])):
            print(f"WARNING: NaN detected in validation field '{k}'")

    # Compute actor loss (all agents have this)
    actor_loss, actor_info = agent.actor_loss(batch, agent.network.params)

    # Compute critic loss only if the agent has it (CRL agents)
    if hasattr(agent, 'contrastive_loss'):
        critic_loss, critic_info = agent.contrastive_loss(batch, agent.network.params)
    else:
        critic_loss = 0.0
        critic_info = {}

    metrics: Dict[str, object] = {}

    # ============================
    # Validation action distributions
    # ============================
    # Log predicted vs target action distributions as histograms for diagnostics.
    if "pred_actions" in actor_info and "target_actions" in actor_info:
        pred_actions = np.asarray(actor_info["pred_actions"])
        target_actions = np.asarray(actor_info["target_actions"])
        if pred_actions.ndim == 2 and pred_actions.shape == target_actions.shape:
            action_names = ["steer", "throttle", "brake"]
            action_dim = pred_actions.shape[1]
            for i in range(action_dim):
                name = action_names[i] if i < len(action_names) else f"action_{i}"
                metrics[f"val/pred_{name}_hist"] = wandb.Histogram(pred_actions[:, i])
                metrics[f"val/target_{name}_hist"] = wandb.Histogram(target_actions[:, i])

    # Actor metrics: keep only BC loss, Q loss, and MSE.
    if "bc_loss" in actor_info:
        metrics["val/actor_bc_loss"] = float(actor_info["bc_loss"])
    if "q_loss" in actor_info:
        metrics["val/actor_q_loss"] = float(actor_info["q_loss"])
    if "mse" in actor_info:
        metrics["val/actor_mse"] = float(actor_info["mse"])

    # Critic metrics: keep only contrastive loss and categorical accuracy.
    if hasattr(agent, "contrastive_loss"):
        metrics["val/critic_loss"] = float(critic_loss)
        metrics["val/critic_categorical_accuracy"] = float(
            critic_info.get("categorical_accuracy", 0.0)
        )

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
    cfg.alpha = 0.5
    cfg.encoder = "impala_large"
    cfg.lr = 3e-4  # Increased LR to help critic learn (was 1e-4)
    cfg.actor_hidden_dims = (512, 512, 512)
    cfg.value_hidden_dims = (512, 512, 512)
    cfg.latent_dim = 512
    cfg.critic_lr_scale = 1.0
    cfg.actor_lr_scale = 1.0
    cfg.frame_stack = None  # Stacking handled on-the-fly via frame_offsets.
    cfg.block_size = args.block_size
    cfg.frame_offsets = tuple(args.frame_offsets if args.frame_offsets else [0, -10, -20, -50, -80])
    cfg.p_aug = 0.25
    cfg.distance_loss_weight = 0.05
    cfg.distance_head_hidden_dims = (256, 256)
    cfg.upsample_mode = 'none'  # Disable action component upsampling (was causing instability)
    cfg.upsample_weight = 3.0
    cfg.steer_thresh = 0.1
    cfg.throttle_thresh = 0.3
    cfg.brake_thresh = 0.1
    cfg.p_curgoal = 0.025
    cfg.use_mrn_metric = args.use_mrn_metric
    if args.mrn_components is not None:
        cfg.mrn_components = args.mrn_components

    np.random.seed(args.seed)

    cpu_dataset = load_dataset_cpu(
        args.dataset_path,
        frame_offsets=cfg.frame_offsets,
        block_size=args.block_size,
    )

    # Split by actual trajectory boundaries (using terminal markers, not fixed length)
    # This will respect the variable-length trajectories after filtering
    train_split, val_split = split_dataset_by_terminals(
        cpu_dataset,
        val_fraction=0.2,
        seed=args.seed,
    )

    def build_gc_dataset(split: Dict[str, np.ndarray]) -> GCDataset | None:
        if split["observations"].size == 0:
            return None
        dataset_fields = dict(
            observations=split["observations"],
            actions=split["actions"],
            terminals=split["terminals"],
        )
        if "next_observations" in split:
            dataset_fields["next_observations"] = split["next_observations"]
        return GCDataset(Dataset.create(**dataset_fields), cfg)

    train_dataset = build_gc_dataset(train_split)
    val_dataset = build_gc_dataset(val_split)

    if train_dataset is None or train_dataset.size == 0:
        raise ValueError("Training dataset is empty after preprocessing.")

    print(
        f"Dataset summary: train={train_dataset.size} frames, "
        f"val={(val_dataset.size if val_dataset is not None else 0)} frames"
    )

    example_batch = train_dataset.sample(min(10, cfg.batch_size))
    ex_obs_np = np.asarray(example_batch["observations"])
    ex_act_np = np.asarray(example_batch["actions"])
    print(f"Creating agent with example shapes: obs={ex_obs_np.shape}, actions={ex_act_np.shape}")

    agent = agent_cls.create(
        seed=args.seed,
        ex_observations=ex_obs_np,
        ex_actions=ex_act_np,
        config=cfg,
    )
    print("Agent created successfully")

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    wandb.init(project=args.project, config=cfg.to_dict())

    total_steps = 0
    last_ckpt_step = -1
    last_val_step = -1
    last_log_step = -1

    for epoch in range(args.epochs):
        print(f"\nStarting epoch {epoch + 1}/{args.epochs}")
        if float(agent.config['alpha']) != float(cfg.alpha):
            config_dict = unfreeze(agent.config)
            config_dict['alpha'] = float(cfg.alpha)
            agent = agent.replace(config=freeze(config_dict))

        progress = trange(args.steps, dynamic_ncols=True)
        for step in progress:
            total_steps += 1
            t0 = time.time()
            batch_np = train_dataset.sample(cfg.batch_size)
            batch = numpy_to_jax(batch_np)
            batch_ms = (time.time() - t0) * 1000.0

            t1 = time.time()
            agent, info = agent.update(batch)
            upd_ms = (time.time() - t1) * 1000.0

            # Logging (use total_steps, not step, so logging works across epochs)
            if total_steps % args.log_every == 0:
                log_dict: Dict[str, object] = {}

                # ============================
                # Actor action distributions (train)
                # ============================
                # Recompute predictions here to avoid changing the agent implementation.
                actor_dist = agent.network.select("actor")(
                    batch["observations"],
                    batch["actor_goals"],
                    params=agent.network.params,
                )
                pred_actions = np.asarray(actor_dist.mode())
                target_actions = np.asarray(batch["actions"])
                if pred_actions.ndim == 2 and pred_actions.shape == target_actions.shape:
                    action_names = ["steer", "throttle", "brake"]
                    action_dim = pred_actions.shape[1]
                    for i in range(action_dim):
                        name = action_names[i] if i < len(action_names) else f"action_{i}"
                        log_dict[f"train/pred_{name}_hist"] = wandb.Histogram(pred_actions[:, i])
                        log_dict[f"train/target_{name}_hist"] = wandb.Histogram(target_actions[:, i])

                # Log only the requested scalar metrics under clear names.
                metric_map = [
                    ("actor/bc_loss", "train/actor_bc_loss"),
                    ("actor/q_loss", "train/actor_q_loss"),
                    ("actor/mse", "train/actor_mse"),
                    ("critic/contrastive_loss", "train/critic_loss"),
                    ("critic/categorical_accuracy", "train/critic_categorical_accuracy"),
                ]
                for src_key, dst_name in metric_map:
                    if src_key in info:
                        log_dict[dst_name] = float(info[src_key])

                wandb.log(log_dict, step=total_steps)
                last_log_step = total_steps

            # Progress bar: show primary loss metrics
            postfix = {}
            if "actor/bc_loss" in info:
                postfix["bc"] = float(info["actor/bc_loss"])
            if "actor/q_loss" in info:
                postfix["q"] = float(info["actor/q_loss"])
            if "critic/contrastive_loss" in info:
                postfix["critic"] = float(info["critic/contrastive_loss"])
            if postfix:
                progress.set_postfix(**postfix)

            # checkpoints (use total_steps, not step, so checkpoints work across epochs)
            if args.ckpt_every and total_steps % args.ckpt_every == 0:
                ckpt_path = ckpt_dir / f"agent_step{total_steps}.pkl"
                with ckpt_path.open("wb") as f:
                    f.write(fxs.to_bytes(agent))
                last_ckpt_step = total_steps

            # validation (use total_steps, not step, so validation works across epochs)
            if val_dataset is not None and total_steps % args.val_every == 0:
                val_metrics = compute_validation_loss(agent, val_dataset, batch_size=cfg.batch_size)
                # Filter out non-numeric values (like wandb.Table) when printing
                numeric_metrics = {k: v for k, v in val_metrics.items() if isinstance(v, (int, float, np.number))}
                print(f"\nValidation @ step {total_steps}: " + ", ".join(f"{k}={v:.4f}" for k, v in numeric_metrics.items()))
                wandb.log(val_metrics, step=total_steps)
                last_val_step = total_steps

        # After epoch ends, checkpoint/validate/log if we haven't already this step
        if args.ckpt_every and total_steps != last_ckpt_step:
            ckpt_path = ckpt_dir / f"agent_step{total_steps}.pkl"
            with ckpt_path.open("wb") as f:
                f.write(fxs.to_bytes(agent))
            print(f"Checkpoint saved @ step {total_steps} (end of epoch {epoch + 1})")
            last_ckpt_step = total_steps

        if val_dataset is not None and total_steps != last_val_step:
            val_metrics = compute_validation_loss(agent, val_dataset, batch_size=cfg.batch_size)
            numeric_metrics = {k: v for k, v in val_metrics.items() if isinstance(v, (int, float, np.number))}
            print(f"\nValidation @ step {total_steps} (end of epoch {epoch + 1}): " + ", ".join(f"{k}={v:.4f}" for k, v in numeric_metrics.items()))
            wandb.log(val_metrics, step=total_steps)
            last_val_step = total_steps

        # Log at end of epoch if we haven't already this step
        if total_steps != last_log_step:
            batch_np = train_dataset.sample(cfg.batch_size)
            batch = numpy_to_jax(batch_np)
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
    final_model_path = Path(args.ckpt_dir) / "final_model.pkl"
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
    parser.add_argument("--actor_loss", choices=["awr", "ddpgbc"], default="ddpgbc")
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--project", default="crl_training")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=1000)
    parser.add_argument("--ckpt_every", type=int, default=50_000)
    parser.add_argument("--ckpt_dir", default="checkpoints")
    parser.add_argument("--algorithm", default="CRL")
    parser.add_argument("--block_size", type=int, default=400, help="Block size for block-aware frame stacking and shuffling")
    parser.add_argument("--val_every", type=int, default=500)
    parser.add_argument("--obs_h", type=int, default=100)
    parser.add_argument("--obs_w", type=int, default=100)
    parser.add_argument("--obs_c", type=int, default=3)
    parser.add_argument("--frame_offsets", nargs="*", type=int, default=None, help="e.g., --frame_offsets 0 -5 -10 -20")
    parser.add_argument("--no_filter_intersections", action="store_true", help="Disable filtering of intersection/stationary frames")
    parser.add_argument("--use_mrn_metric", action="store_true", help="Enable MRN distance inside CRL contrastive loss")
    parser.add_argument("--mrn_components", type=int, default=None, help="Number of MRN components (requires --use_mrn_metric)")

    args = parser.parse_args()
    main(args)
