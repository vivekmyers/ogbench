"""
Simple script to show the basic structure of a CARLA HDF5 dataset
and infer terminal states using multiple detection methods
"""

import h5py
import numpy as np
import sys
import os

def explore_simple_hdf5(filepath):
    """Show basic structure of HDF5 dataset focusing on RL data components"""
    print(f"Loading HDF5 dataset from {filepath}")
    
    try:
        with h5py.File(filepath, 'r') as f:
            print("\n=== Basic Dataset Structure ===")
            print(f"Top-level keys: {list(f.keys())}")
            
            # Look for important RL data components
            rl_components = ['observations', 'actions', 'rewards', 'terminals', 'next_observations', 
                           'dones', 'infos', 'states', 'next_states', 'timeouts', 'episode']
            
            found_components = {}
            
            # Simple traversal function to find RL components
            def find_components(name, obj):
                if isinstance(obj, h5py.Dataset):
                    name_lower = name.lower()
                    for component in rl_components:
                        if component in name_lower:
                            found_components[component] = {
                                'path': name,
                                'shape': obj.shape,
                                'dtype': obj.dtype
                            }
                            # Show sample if not too large
                            if len(obj.shape) == 0 or (len(obj.shape) == 1 and obj.shape[0] < 10):
                                found_components[component]['sample'] = obj[...]
                            elif len(obj.shape) >= 1:
                                found_components[component]['sample_first'] = obj[0]
                return None
            
            f.visititems(find_components)
            
            # Show what we found
            print("\n=== Found RL Data Components ===")
            for component, info in found_components.items():
                print(f"{component}:")
                print(f"  Path: {info['path']}")
                print(f"  Shape: {info['shape']}")
                print(f"  Dtype: {info['dtype']}")
                if 'sample' in info:
                    print(f"  Sample: {info['sample']}")
                elif 'sample_first' in info:
                    sample = info['sample_first']
                    if isinstance(sample, np.ndarray) and sample.size > 10:
                        print(f"  First sample shape: {sample.shape}")
                    else:
                        print(f"  First sample: {sample}")
                print()
            
            # Check for terminal states using multiple methods
            print("\n=== Terminal State Detection ===")
            
            # Method 1: Check for explicit done/terminal flags
            terminal_indices = []
            if 'terminals' in found_components:
                terminals_path = found_components['terminals']['path']
                terminals = f[terminals_path][...]
                terminal_indices = np.where(terminals)[0]
                print(f"Method 1: Found {len(terminal_indices)} terminal states using 'terminals' flag")
                if len(terminal_indices) > 0:
                    print(f"  Terminal indices: {terminal_indices[:5]} ... (total: {len(terminal_indices)})")
            
            if 'dones' in found_components and not terminal_indices:
                dones_path = found_components['dones']['path']
                dones = f[dones_path][...]
                terminal_indices = np.where(dones)[0]
                print(f"Method 1: Found {len(terminal_indices)} terminal states using 'dones' flag")
                if len(terminal_indices) > 0:
                    print(f"  Terminal indices: {terminal_indices[:5]} ... (total: {len(terminal_indices)})")
            
            # Method 2: Use rewards to detect terminal states
            if 'rewards' in found_components and not terminal_indices:
                rewards_path = found_components['rewards']['path']
                rewards = f[rewards_path][...]
                
                # Look for large negative rewards (collisions) or large positive rewards (task completion)
                reward_mean = np.mean(rewards)
                reward_std = np.std(rewards)
                
                large_rewards = np.where(np.abs(rewards - reward_mean) > 3 * reward_std)[0]
                print(f"Method 2: Found {len(large_rewards)} potential terminal states using reward outliers")
                if len(large_rewards) > 0:
                    print(f"  Potential terminal indices: {large_rewards[:5]} ... (total: {len(large_rewards)})")
                    
                    # Detect episodes by grouping consecutive indices
                    episode_ends = []
                    for i in range(len(large_rewards) - 1):
                        if large_rewards[i+1] - large_rewards[i] > 1:  # Non-consecutive
                            episode_ends.append(large_rewards[i])
                    if len(large_rewards) > 0:
                        episode_ends.append(large_rewards[-1])
                    
                    print(f"  Estimated episode ends: {episode_ends[:5]} ... (total: {len(episode_ends)})")
                    terminal_indices = episode_ends if not terminal_indices else terminal_indices
            
            # Method 3: Look for position discontinuities in observations
            if 'observations' in found_components and not terminal_indices and 'next_observations' in found_components:
                obs_path = found_components['observations']['path']
                next_obs_path = found_components['next_observations']['path']
                
                # Sample a subset of the data to check for position discontinuities
                # (checking all data would be slow)
                sample_size = min(1000, f[obs_path].shape[0])
                sample_indices = np.linspace(0, f[obs_path].shape[0]-1, sample_size, dtype=int)
                
                try:
                    # Try to extract position data - this is dataset-specific
                    # For CARLA, position might be in the first few elements of the observation
                    obs_sample = f[obs_path][sample_indices]
                    next_obs_sample = f[next_obs_path][sample_indices]
                    
                    # If observations are images, we can't easily extract position
                    if len(obs_sample.shape) > 2 and obs_sample.shape[-1] in [1, 3, 4]:
                        print("Method 3: Can't detect position discontinuities - observations are images")
                    else:
                        # Try to find position data - this is a guess and might need adjustment
                        position_diff = np.sum(np.abs(next_obs_sample - obs_sample), axis=1)
                        position_diff_mean = np.mean(position_diff)
                        position_diff_std = np.std(position_diff)
                        
                        # Large position differences might indicate resets
                        large_diffs = np.where(position_diff > position_diff_mean + 3 * position_diff_std)[0]
                        large_diff_indices = sample_indices[large_diffs]
                        
                        print(f"Method 3: Found {len(large_diff_indices)} potential terminal states using position discontinuities")
                        if len(large_diff_indices) > 0:
                            print(f"  Potential terminal indices: {large_diff_indices[:5]} ... (total: {len(large_diff_indices)})")
                            terminal_indices = large_diff_indices if not terminal_indices else terminal_indices
                except Exception as e:
                    print(f"Method 3: Error detecting position discontinuities: {e}")
            
            # Use the terminal indices to calculate episode statistics
            if len(terminal_indices) > 0:
                print("\n=== Episode Statistics ===")
                # Add first index for first episode
                episode_starts = np.concatenate([[0], terminal_indices[:-1] + 1])
                episode_lengths = terminal_indices - episode_starts + 1
                
                print(f"Total episodes: {len(terminal_indices)}")
                print(f"Episode lengths: min={np.min(episode_lengths)}, max={np.max(episode_lengths)}, mean={np.mean(episode_lengths)}")
                print(f"First few episode lengths: {episode_lengths[:5]}")
            else:
                print("\nNo terminal states detected. The dataset might be a single long episode or terminal states are not marked.")
                if 'observations' in found_components:
                    print(f"Dataset contains {f[found_components['observations']['path']].shape[0]} transitions")
    
    except Exception as e:
        print(f"Error exploring dataset: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = "/home/achyuth/carla_town_flat-v0.hdf5"
    
    if not os.path.exists(filepath):
        print(f"Error: Dataset file {filepath} does not exist.")
        sys.exit(1)
        
    explore_simple_hdf5(filepath) 