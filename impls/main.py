import json
import os
import random
import time
from collections import defaultdict

import jax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags
from utils.datasets import Dataset, GCDataset, HGCDataset
from utils.env_utils import make_env_and_datasets
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb
from agents import agents

# Import and register the NuPlan environment
import gymnasium
from ogbench.nuplan import NuPlanEnv
from ogbench.nuplan.dataset import NuPlanDataset

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'nuplan-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

# Add NuPlan-specific flags
flags.DEFINE_string('dataset_path', None, 'Path to the NuPlan dataset.')
flags.DEFINE_string('render_mode', 'rgb_array', 'Rendering mode for NuPlan environment.')

flags.DEFINE_integer('train_steps', 1000000, 'Number of training steps.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 1000000, 'Saving interval.')

flags.DEFINE_integer('eval_tasks', None, 'Number of tasks to evaluate (None for all).')
flags.DEFINE_integer('eval_episodes', 20, 'Number of episodes for each task.')
flags.DEFINE_float('eval_temperature', 0, 'Actor temperature for evaluation.')
flags.DEFINE_float('eval_gaussian', None, 'Action Gaussian noise for evaluation.')
flags.DEFINE_integer('video_episodes', 1, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')
flags.DEFINE_integer('eval_on_cpu', 1, 'Whether to evaluate on CPU.')

config_flags.DEFINE_config_file('agent', 'agents/gciql.py', lock_config=False)


def create_nuplan_env_and_datasets(dataset_path, frame_stack=None, config=None):
    """Create NuPlan environment and datasets.
    
    Args:
        dataset_path: Path to the NuPlan dataset.
        frame_stack: Number of frames to stack.
        config: Configuration dictionary.
        
    Returns:
        A tuple of the environment, training dataset, and validation dataset.
    """
    # Create the environment
    env = NuPlanEnv(
        dataset_path=dataset_path, 
        render_mode=FLAGS.render_mode, 
        config=config
    )
    
    # Load the dataset
    dataset = NuPlanDataset(dataset_path)
    data = dataset.data
    
    # Create training and validation datasets
    # For simplicity, we'll use the same data for both
    # In a real scenario, you would split the data
    train_dataset = {
        'observations': data['observations'],
        'actions': data['actions'],
        'rewards': data['rewards'],
        'next_observations': data['next_observations'],
        'terminals': data['terminals'],
        'value_goals': data['value_goals'],
        'actor_goals': data['actor_goals'],
    }
    
    val_dataset = train_dataset.copy()
    
    # Apply frame stacking if requested
    if frame_stack is not None and frame_stack > 1:
        from utils.env_utils import FrameStackWrapper
        env = FrameStackWrapper(env, frame_stack)
    
    # Reset the environment
    env.reset()
    
    return env, train_dataset, val_dataset


def main(_):
    # Set up logger.
    exp_name = get_exp_name(FLAGS.seed)
    setup_wandb(project='OGBench', group=FLAGS.run_group, name=exp_name)

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    
    # Get flag dict and filter out non-serializable values
    flag_dict = get_flag_dict()
    # Remove the agent config which might contain non-serializable objects
    if 'agent' in flag_dict:
        del flag_dict['agent']
    
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    # Set up environment and dataset.
    config = FLAGS.agent
    
    # Special handling for NuPlan environment
    if FLAGS.env_name == 'nuplan-v0':
        # Check if dataset path is provided
        if FLAGS.dataset_path is None:
            # Use default path
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'nuplan')
            dataset_path = os.path.join(data_dir, 'nuplan_dataset.npz')
        else:
            dataset_path = FLAGS.dataset_path
        
        # Create environment and datasets
        print(f"Creating NuPlan environment and datasets from {dataset_path}...")
        env, train_dataset, val_dataset = create_nuplan_env_and_datasets(
            dataset_path, 
            frame_stack=config['frame_stack'],
            config=config
        )
    else:
        # Standard environment creation
        env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name, frame_stack=config['frame_stack'])

        dataset_class = {
            'GCDataset': GCDataset,
            'HGCDataset': HGCDataset,
        }[config['dataset_class']]
        train_dataset = dataset_class(Dataset.create(**train_dataset), config)
        if val_dataset is not None:
            val_dataset = dataset_class(Dataset.create(**val_dataset), config)

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Handle different dataset formats
    if FLAGS.env_name == 'nuplan-v0':
        example_batch = {
            'observations': train_dataset['observations'][:1],
            'actions': train_dataset['actions'][:1],
        }
    else:
        example_batch = train_dataset.sample(1)
        
    if config['discrete']:
        # Fill with the maximum action to let the agent know the action space size.
        example_batch['actions'] = np.full_like(example_batch['actions'], env.action_space.n - 1)

    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    # Restore agent.
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # Train agent.
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()
    for i in tqdm.tqdm(range(1, FLAGS.train_steps + 1), smoothing=0.1, dynamic_ncols=True):
        # Update agent.
        if FLAGS.env_name == 'nuplan-v0':
            # Sample from NuPlan dataset
            batch_size = min(config['batch_size'], len(train_dataset['observations']))
            indices = np.random.randint(0, len(train_dataset['observations']), size=batch_size)
            
            batch = {
                'observations': train_dataset['observations'][indices],
                'actions': train_dataset['actions'][indices],
                'rewards': train_dataset['rewards'][indices],
                'next_observations': train_dataset['next_observations'][indices],
                'terminals': train_dataset['terminals'][indices],
                'value_goals': train_dataset['value_goals'][indices],
                'actor_goals': train_dataset['actor_goals'][indices],
            }
        else:
            # Standard dataset sampling
            batch = train_dataset.sample(config['batch_size'])
            
        agent, update_info = agent.update(batch)

        # Log metrics.
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if val_dataset is not None:
                if FLAGS.env_name == 'nuplan-v0':
                    # Sample from NuPlan validation dataset
                    val_batch_size = min(config['batch_size'], len(val_dataset['observations']))
                    val_indices = np.random.randint(0, len(val_dataset['observations']), size=val_batch_size)
                    
                    val_batch = {
                        'observations': val_dataset['observations'][val_indices],
                        'actions': val_dataset['actions'][val_indices],
                        'rewards': val_dataset['rewards'][val_indices],
                        'next_observations': val_dataset['next_observations'][val_indices],
                        'terminals': val_dataset['terminals'][val_indices],
                        'value_goals': val_dataset['value_goals'][val_indices],
                        'actor_goals': val_dataset['actor_goals'][val_indices],
                    }
                else:
                    # Standard validation dataset sampling
                    val_batch = val_dataset.sample(config['batch_size'])
                    
                _, val_info = agent.total_loss(val_batch, grad_params=None)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        if i == 1 or i % FLAGS.eval_interval == 0:
            if FLAGS.eval_on_cpu:
                eval_agent = jax.device_put(agent, device=jax.devices('cpu')[0])
            else:
                eval_agent = agent
            renders = []
            eval_metrics = {}
            overall_metrics = defaultdict(list)
            task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
            num_tasks = FLAGS.eval_tasks if FLAGS.eval_tasks is not None else len(task_infos)
            for task_id in tqdm.trange(1, num_tasks + 1):
                task_name = task_infos[task_id - 1]['task_name']
                eval_info, trajs, cur_renders = evaluate(
                    agent=eval_agent,
                    env=env,
                    task_id=task_id,
                    config=config,
                    num_eval_episodes=FLAGS.eval_episodes,
                    num_video_episodes=FLAGS.video_episodes,
                    video_frame_skip=FLAGS.video_frame_skip,
                    eval_temperature=FLAGS.eval_temperature,
                    eval_gaussian=FLAGS.eval_gaussian,
                )
                renders.extend(cur_renders)
                metric_names = ['success']
                eval_metrics.update(
                    {f'evaluation/{task_name}_{k}': v for k, v in eval_info.items() if k in metric_names}
                )
                for k, v in eval_info.items():
                    if k in metric_names:
                        overall_metrics[k].append(v)
            for k, v in overall_metrics.items():
                eval_metrics[f'evaluation/overall_{k}'] = np.mean(v)

            if FLAGS.video_episodes > 0:
                video = get_wandb_video(renders=renders, n_cols=num_tasks)
                eval_metrics['video'] = video

            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

        # Save agent.
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    app.run(main)