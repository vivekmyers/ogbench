"""
Training script for CARLA environment using CRL.
"""

import os
from absl import app, flags
from ml_collections import config_flags

# Import the main training function
from main import main as train_main

FLAGS = flags.FLAGS

# Override default flags for CARLA
flags.DEFINE_string('env_name', 'carla-offline-v0', 'Environment name.')
flags.DEFINE_string('run_group', 'CARLA-CRL', 'Run group name.')
flags.DEFINE_integer('train_steps', 500000, 'Number of training steps.')
flags.DEFINE_integer('eval_interval', 10000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 50000, 'Save interval.')
flags.DEFINE_integer('log_interval', 1000, 'Log interval.')
flags.DEFINE_integer('eval_episodes', 5, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 2, 'Number of video episodes to record.')
flags.DEFINE_string('dataset_path', None, 'Path to the offline dataset.')

# CRL specific configuration
config_flags.DEFINE_config_file(
    'agent',
    None,
    'Agent configuration file.',
    lock_config=False,
)

def main(_):
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