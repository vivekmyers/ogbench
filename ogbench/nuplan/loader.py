import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import cv2

class NuplanLoader:
    """Loader for NuPlan environment data.
    
    This class handles loading and preprocessing NuPlan data for use in the environment.
    It provides methods for loading data from various sources and formats.
    """
    
    def __init__(self, data_dir: str, config: Optional[Dict[str, Any]] = None):
        """Initialize the NuPlan loader.
        
        Args:
            data_dir: Directory containing NuPlan data
            config: Configuration dictionary
        """
        self.data_dir = data_dir
        self.config = config or {}
        self.frame_stack = self.config.get('frame_stack', 1)
        
    def load_dataset(self, dataset_name: str) -> Dict[str, np.ndarray]:
        """Load a dataset from the data directory.
        
        Args:
            dataset_name: Name of the dataset to load
            
        Returns:
            Dictionary containing the dataset arrays
        """
        dataset_path = os.path.join(self.data_dir, dataset_name)
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
            
        # Load data from file
        data = np.load(dataset_path)
        
        # Process data
        processed_data = self._process_data(dict(data))
        
        return processed_data
    
    def _process_data(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Process the dataset arrays.
        
        Args:
            data: Dictionary containing the dataset arrays
            
        Returns:
            Processed dataset arrays
        """
        # Ensure arrays are float32
        processed_data = {}
        for key, value in data.items():
            processed_data[key] = value.astype(np.float32)
            
        # Add rewards if not present
        if 'rewards' not in processed_data:
            processed_data['rewards'] = np.zeros(len(processed_data['observations']), dtype=np.float32)
            
        # Compute next observations
        processed_data['next_observations'] = np.roll(processed_data['observations'], -1, axis=0)
        
        # Handle terminal states
        if 'terminals' in processed_data:
            terminal_mask = processed_data['terminals'].astype(bool)
            processed_data['next_observations'][terminal_mask] = 0.0
            
        return processed_data
    
    def create_visualization(self, 
                            observation: np.ndarray, 
                            action: Optional[np.ndarray] = None,
                            goal: Optional[np.ndarray] = None,
                            width: int = 800,
                            height: int = 600) -> np.ndarray:
        """Create a visualization of the current state.
        
        Args:
            observation: Current observation
            action: Current action (optional)
            goal: Goal position (optional)
            width: Width of the visualization
            height: Height of the visualization
            
        Returns:
            RGB image of the visualization
        """
        # Create a blank image
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Extract position and orientation from observation
        # Assuming first 2 dimensions are x,y position and next 2 are orientation
        pos_x, pos_y = observation[0], observation[1]
        heading = np.arctan2(observation[3], observation[2]) if len(observation) > 3 else 0
        
        # Scale coordinates to image space
        scale = 20  # pixels per meter
        center_x, center_y = width // 2, height // 2
        img_x = int(pos_x * scale + center_x)
        img_y = int(pos_y * scale + center_y)
        
        # Draw vehicle
        vehicle_length = 4.5  # meters
        vehicle_width = 1.8  # meters
        
        # Convert to pixels
        length_px = int(vehicle_length * scale)
        width_px = int(vehicle_width * scale)
        
        # Create a rectangle for the vehicle
        rect = patches.Rectangle(
            (img_x - length_px // 2, img_y - width_px // 2),
            length_px,
            width_px,
            angle=np.degrees(heading),
            rotation_point=(img_x, img_y),
            facecolor='blue',
            alpha=0.7
        )
        
        # Draw the rectangle on the image
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
        ax.imshow(img)
        ax.add_patch(rect)
        
        # Draw heading direction
        heading_length = length_px // 2
        heading_x = img_x + int(heading_length * np.cos(heading))
        heading_y = img_y + int(heading_length * np.sin(heading))
        ax.plot([img_x, heading_x], [img_y, heading_y], 'r-', linewidth=2)
        
        # Draw goal if provided
        if goal is not None:
            goal_x = int(goal[0] * scale + center_x)
            goal_y = int(goal[1] * scale + center_y)
            ax.plot(goal_x, goal_y, 'g*', markersize=15)
            
        # Draw action if provided
        if action is not None:
            action_length = 30  # pixels
            action_x = img_x + int(action_length * action[0])
            action_y = img_y + int(action_length * action[1])
            ax.plot([img_x, action_x], [img_y, action_y], 'y-', linewidth=2, alpha=0.7)
            
        # Set axis limits and remove ticks
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Convert to numpy array
        plt.tight_layout()
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        return img
    
    def create_video(self, 
                    observations: np.ndarray, 
                    actions: np.ndarray,
                    goal: Optional[np.ndarray] = None,
                    output_path: str = 'nuplan_video.mp4',
                    fps: int = 10) -> None:
        """Create a video from a sequence of observations and actions.
        
        Args:
            observations: Sequence of observations
            actions: Sequence of actions
            goal: Goal position (optional)
            output_path: Path to save the video
            fps: Frames per second
        """
        # Create visualizations for each frame
        frames = []
        for i in range(len(observations)):
            frame = self.create_visualization(
                observations[i], 
                actions[i] if i < len(actions) else None,
                goal
            )
            frames.append(frame)
            
        # Create video
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            # Convert from RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
            
        out.release()
        
    def create_animation(self, 
                        observations: np.ndarray, 
                        actions: np.ndarray,
                        goal: Optional[np.ndarray] = None,
                        output_path: str = 'nuplan_animation.gif',
                        fps: int = 10) -> None:
        """Create an animated GIF from a sequence of observations and actions.
        
        Args:
            observations: Sequence of observations
            actions: Sequence of actions
            goal: Goal position (optional)
            output_path: Path to save the animation
            fps: Frames per second
        """
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create initial visualization
        img = self.create_visualization(observations[0], actions[0] if len(actions) > 0 else None, goal)
        im = ax.imshow(img)
        
        # Set axis limits and remove ticks
        ax.set_xlim(0, img.shape[1])
        ax.set_ylim(img.shape[0], 0)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Update function for animation
        def update(frame):
            img = self.create_visualization(
                observations[frame], 
                actions[frame] if frame < len(actions) else None,
                goal
            )
            im.set_array(img)
            return [im]
        
        # Create animation
        anim = FuncAnimation(
            fig, update, frames=len(observations),
            interval=1000/fps, blit=True
        )
        
        # Save animation
        anim.save(output_path, writer='pillow', fps=fps)
        plt.close(fig) 