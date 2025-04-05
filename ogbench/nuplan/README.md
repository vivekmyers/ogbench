# NuPlan Environment Wrapper

This module provides a Python wrapper for the NuPlan environment, allowing you to easily load, visualize, and interact with NuPlan data in a Gymnasium-compatible environment.

## Features

- **Environment Wrapper**: A Gymnasium-compatible environment for NuPlan data
- **Data Loader**: A flexible loader for NuPlan datasets
- **Visualization**: Tools for visualizing NuPlan data, including:
  - Static visualizations of the current state
  - Video generation of trajectories
  - Animated GIFs of trajectories
- **Rendering Modes**: Support for multiple rendering modes:
  - `rgb_array`: Returns an RGB array for the current state
  - `video`: Generates a video of the episode
  - `animation`: Generates an animated GIF of the episode

## Installation

The NuPlan environment wrapper is part of the OGBench package. To install it, follow the instructions in the main OGBench README.

## Usage

### Basic Usage

```python
import gymnasium as gym
from ogbench.nuplan import NuplanEnv, NuplanLoader

# Create environment
env = gym.make('nuplan-v0', dataset_path='path/to/nuplan_dataset.npz')

# Reset environment
obs, info = env.reset()

# Run a few steps
for _ in range(10):
    # Sample random action
    action = env.action_space.sample()
    
    # Step environment
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Break if episode is done
    if terminated or truncated:
        break

# Close environment
env.close()
```

### Using the Loader for Visualization

```python
from ogbench.nuplan import NuplanLoader
import matplotlib.pyplot as plt

# Create loader
loader = NuplanLoader('path/to/data/directory')

# Load dataset
data = loader.load_dataset('nuplan_dataset.npz')

# Create visualization
observation = data['observations'][0]
action = data['actions'][0]
goal = np.array([5.0, 5.0])  # Example goal

# Create visualization
img = loader.create_visualization(observation, action, goal)

# Save visualization
plt.figure(figsize=(10, 8))
plt.imshow(img)
plt.axis('off')
plt.savefig('nuplan_visualization.png')
plt.close()

# Create video
loader.create_video(
    data['observations'][:50], 
    data['actions'][:50], 
    goal,
    'nuplan_video.mp4'
)

# Create animation
loader.create_animation(
    data['observations'][:50], 
    data['actions'][:50], 
    goal,
    'nuplan_animation.gif'
)
```

### Using Different Rendering Modes

```python
# Create environment with video rendering
env = gym.make('video-nuplan-v0', dataset_path='path/to/nuplan_dataset.npz')

# Reset environment
obs, info = env.reset()

# Run a few steps
for _ in range(10):
    # Sample random action
    action = env.action_space.sample()
    
    # Step environment
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Render (will create video at the end of the episode)
    env.render()
    
    # Break if episode is done
    if terminated or truncated:
        break

# Close environment
env.close()
```

## Data Format

The NuPlan environment expects data in the following format:

- `observations`: Array of shape `(n_steps, n_features)` containing the observations
- `actions`: Array of shape `(n_steps, n_actions)` containing the actions
- `terminals`: Array of shape `(n_steps,)` containing boolean flags indicating episode termination
- `rewards`: Array of shape `(n_steps,)` containing the rewards (optional)

## Configuration

The NuPlan environment can be configured using the `config` parameter:

```python
config = {
    'frame_stack': 1,  # Number of frames to stack
    'action_repeat': 1,  # Number of times to repeat each action
    # ... other configuration parameters
}

env = gym.make('nuplan-v0', dataset_path='path/to/nuplan_dataset.npz', config=config)
```

## Example

See `impls/examples/nuplan_example.py` for a complete example of how to use the NuPlan environment wrapper. 