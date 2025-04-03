from typing import Dict, Any

def get_nuplan_config() -> Dict[str, Any]:
    """Get default configuration for the nuplan environment.
    
    Returns:
        Dictionary containing configuration parameters
    """
    return {
        # Environment parameters
        'env_name': 'nuplan',
        'frame_stack': 1,
        'action_repeat': 1,
        'seed': 0,
        
        # Dataset parameters
        'batch_size': 256,
        'num_trajectories': 1000,
        'max_trajectory_length': 1000,
        
        # Training parameters
        'train_steps': 1000000,
        'log_interval': 1000,
        'eval_interval': 10000,
        'save_interval': 100000,
        
        # Evaluation parameters
        'eval_tasks': None,  # None means evaluate all tasks
        'eval_episodes': 50,
        'video_episodes': 1,
        
        # CRL-specific parameters
        'agent_name': 'crl',
        'learning_rate': 3e-4,
        'bc_epochs': 50,
        'bc_batch_size': 256,
        'bc_learning_rate': 3e-4,
        'bc_weight_decay': 0.0,
        'bc_dropout': 0.1,
        'bc_hidden_dims': [256, 256],
        'bc_activation': 'relu',
        'bc_normalize_inputs': True,
        'bc_normalize_outputs': True,
        'bc_use_batch_norm': True,
        'bc_use_layer_norm': False,
        'bc_use_residual': False,
        'bc_use_skip': False,
        'bc_use_attention': False,
        'bc_use_transformer': False,
        'bc_use_conv': False,
        'bc_use_lstm': False,
        'bc_use_gru': False,
        'bc_use_rnn': False,
        'bc_use_mlp': True,
        'bc_use_cnn': False,
        'bc_use_resnet': False,
        'bc_use_densenet': False,
        'bc_use_vgg': False,
        'bc_use_inception': False,
        'bc_use_mobilenet': False,
        'bc_use_efficientnet': False,
        'bc_use_transfer': False,
        'bc_use_pretrained': False,
        'bc_use_finetune': False,
        'bc_use_ensemble': False,
        'bc_use_bootstrap': False,
    } 