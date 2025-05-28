"""
Script to explore the structure of the CARLA offline dataset
"""

import numpy as np
import sys
import os

def explore_dataset(dataset_path):
    """Explore and print details of the dataset"""
    print(f"Loading dataset from {dataset_path}")
    
    try:
        data = np.load(dataset_path, allow_pickle=True)
        
        print("\n=== Dataset Keys ===")
        print(list(data.keys()))
        
        print("\n=== Data Shapes ===")
        for key in data.keys():
            print(f"{key}: {data[key].shape} (dtype: {data[key].dtype})")
        
        print("\n=== Basic Statistics ===")
        for key in data.keys():
            if data[key].dtype == np.float32 or data[key].dtype == np.float64:
                print(f"{key}:")
                print(f"  Min: {np.min(data[key])}")
                print(f"  Max: {np.max(data[key])}")
                print(f"  Mean: {np.mean(data[key])}")
                print(f"  Std: {np.std(data[key])}")
            elif key == 'terminals':
                print(f"{key}:")
                print(f"  True count: {np.sum(data[key])}")
                print(f"  False count: {len(data[key]) - np.sum(data[key])}")
        
        # Sample data examples
        print("\n=== Sample Data ===")
        if 'observations' in data:
            print("First observation shape:", data['observations'][0].shape)
            if len(data['observations'][0].shape) == 3:  # Image data
                print("  This appears to be image data (height, width, channels)")
        
        if 'actions' in data:
            print("First few actions:", data['actions'][:5])
            
        if 'rewards' in data:
            print("Reward distribution:")
            print("  First few rewards:", data['rewards'][:5])
            
        if 'terminals' in data:
            terminal_indices = np.where(data['terminals'])[0]
            print(f"Terminal states at indices: {terminal_indices[:5]}... (total: {len(terminal_indices)})")
            
            # Calculate episode lengths
            if len(terminal_indices) > 0:
                episode_starts = np.concatenate([[0], terminal_indices[:-1] + 1])
                episode_lengths = terminal_indices - episode_starts + 1
                print(f"Episode lengths: {episode_lengths[:5]}... (total episodes: {len(episode_lengths)})")
                print(f"  Min: {np.min(episode_lengths)}")
                print(f"  Max: {np.max(episode_lengths)}")
                print(f"  Mean: {np.mean(episode_lengths)}")
                
        # Check for NaN values
        print("\n=== Data Quality Checks ===")
        for key in data.keys():
            if data[key].dtype == np.float32 or data[key].dtype == np.float64:
                nan_count = np.isnan(data[key]).sum()
                print(f"{key} NaN count: {nan_count}")
    
    except Exception as e:
        print(f"Error exploring dataset: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        dataset_path = "/home/achyuth/CarlaOfflineV0_vision_data.npz"
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file {dataset_path} does not exist.")
        sys.exit(1)
        
    explore_dataset(dataset_path) 