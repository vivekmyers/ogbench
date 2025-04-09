"""
Training script for CARLA environment using CRL.
"""

import os
from absl import app, flags
from ml_collections import config_flags

# Import the main training function and its flags
from main import main as train_main, FLAGS
import ogbench.carla

# Define CARLA-specific flags
flags.DEFINE_string('dataset_path', None, 'Path to the offline dataset.')

def main(_):
    # Set CARLA-specific default values for existing flags
    FLAGS.set_default('env_name', 'CarlaOfflineV0')
    FLAGS.set_default('run_group', 'CARLA-CRL')
    FLAGS.set_default('train_steps', 500000)
    FLAGS.set_default('eval_interval', 10000)
    FLAGS.set_default('save_interval', 50000)
    FLAGS.set_default('log_interval', 1000)
    FLAGS.set_default('eval_episodes', 5)
    FLAGS.set_default('video_episodes', 2)

    # Ensure dataset path is provided
    if FLAGS.dataset_path is None:
        raise ValueError("Please provide --dataset_path")
        
    # Set agent config if not provided
    if FLAGS.agent is None:
        FLAGS.agent = {
            'agent_name': 'crl',
            'dataset_class': 'GCDataset',
            'frame_stack': 1,
            'batch_size': 256,
            'discrete': False,
            'actor_loss': 'ddpgbc',
            'actor_log_q': True,
            'alpha': 2.5,
            'const_std': True,
            # Network architecture
            'encoder_type': 'resnet18',
            'hidden_dims': [256, 256],
            'latent_dim': 512,
        }
    
    # Run the main training loop
    train_main(None)

if __name__ == '__main__':
    app.run(main) 