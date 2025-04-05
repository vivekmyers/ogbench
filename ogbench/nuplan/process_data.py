import numpy as np

def process_nuplan_data(input_path, output_path):
    """Process NuPlan dataset to add required fields for training.
    
    Args:
        input_path: Path to input .npz file
        output_path: Path to save processed .npz file
    """
    # Load raw data
    print("Loading data from:", input_path)
    data = dict(np.load(input_path))
    
    # Print shapes of arrays
    print("\nOriginal data shapes:")
    for k, v in data.items():
        print(f"{k}: {v.shape} ({v.dtype})")
    
    # Add required fields
    # For goal-conditioned learning, we use next_observations as goals
    data['value_goals'] = data['next_observations'].copy()
    data['actor_goals'] = data['next_observations'].copy()
    
    # Ensure all arrays have correct dtypes
    data['observations'] = data['observations'].astype(np.float32)
    data['next_observations'] = data['next_observations'].astype(np.float32)
    data['actions'] = data['actions'].astype(np.float32)
    data['rewards'] = data['rewards'].astype(np.float32)
    data['terminals'] = data['terminals'].astype(bool)
    data['value_goals'] = data['value_goals'].astype(np.float32)
    data['actor_goals'] = data['actor_goals'].astype(np.float32)
    
    print("\nProcessed data shapes:")
    for k, v in data.items():
        print(f"{k}: {v.shape} ({v.dtype})")
    
    # Save processed data
    print(f"\nSaving processed data to: {output_path}")
    np.savez(output_path, **data)
    print("Done!") 