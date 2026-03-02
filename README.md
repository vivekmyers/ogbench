# Training and Evaluation Guide

This repository contains training and evaluation scripts for offline goal-conditioned RL agents.

## Training (`impls/train.py`)

Train offline RL agents on pre-collected datasets.

### Basic Usage

```bash
cd impls
python train.py --dataset_path /path/to/dataset.npz --agent CRL
```

**Note:** Run from the `impls/` directory, or adjust the import paths accordingly.

### Required Arguments

- `--dataset_path`: Path to the `.npz` dataset file (required)

### Common Arguments

- `--agent`: Agent/algorithm to train (`CRL`, `GCBC`, `CMD`, `GCIQL`, etc.) (default: `CRL`)
- `--steps`: Total number of gradient steps (default: `800000`)
- `--epochs`: Number of epochs to train (default: `2`)
- `--batch_size`: Batch size for training (default: `256`)
- `--actor_loss`: Actor loss type for CRL (`awr` or `ddpgbc`) (default: `ddpgbc`)
- `--discount`: Discount factor (default: `0.99`)
- `--seed`: Random seed (default: `0`)

### Logging and Checkpointing

- `--project`: Wandb project name (default: `crl_training`)
- `--log_every`: Log metrics every N steps (default: `100`)
- `--ckpt_every`: Save checkpoint every N steps (default: `50000`)
- `--ckpt_dir`: Directory to save checkpoints (default: `checkpoints`)

**Note:** Checkpoints are saved as `agent_step{total_steps}.pkl` in the checkpoint directory.

### Observation Configuration

- `--obs_h`: Observation height (default: `100`)
- `--obs_w`: Observation width (default: `100`)
- `--obs_c`: Observation channels (default: `3`)
- `--frame_offsets`: Frame offsets for frame stacking, e.g., `--frame_offsets 0 -5 -10 -20` (default: `[0]` if not specified, meaning single frame)
- `--block_size`: Block size for block-aware frame stacking and shuffling (default: `400`)

### Dataset Processing

- `--chunk_size`: Process dataset in chunks of ~this many frames (default: `50000`)
- `--use_mmap`: Use memory-mapped file loading (saves RAM but may be slower)
- `--no_filter_intersections`: Disable filtering of intersection/stationary frames
- `--val_every`: Run validation every N steps (default: `500`)

### Advanced Options

- `--use_mrn_metric`: Enable MRN distance inside CRL contrastive loss
- `--mrn_components`: Number of MRN components (requires `--use_mrn_metric`)

### Example Commands

```bash
# Basic CRL training
python train.py --dataset_path /path/to/dataset.npz --agent CRL --steps 1000000

# Training with custom frame stacking
python train.py --dataset_path /path/to/dataset.npz --agent CRL \
    --frame_offsets 0 -5 -10 -20 --block_size 400

# Training with memory mapping for large datasets
python train.py --dataset_path /path/to/large_dataset.npz --agent CRL \
    --use_mmap --chunk_size 100000

# GCBC training
python train.py --dataset_path /path/to/dataset.npz --agent GCBC \
    --batch_size 512 --steps 500000
```

## Evaluation Server (`impls/server.py`)

Run an evaluation server that loads a trained model and serves action predictions for CARLA evaluation.

### Basic Usage

```bash
cd impls
python server.py --agent crl --model_path /path/to/model.pkl --dataset_path /path/to/goals.npz
```

**Note:** Run from the `impls/` directory. The `--agent` argument accepts lowercase (e.g., `crl`, `gcbc`) and is converted to the appropriate agent class internally.

### Required Arguments

- `--agent`: Agent type (`crl`, `cmd`, `gcbc`, `gciql`, `tmd`) (default: `gcbc`)
- `--model_path`: Path to trained model checkpoint (default: `/global/scratch/users/achyuthkv76/tmd_models/run2.pkl`)
- `--dataset_path`: Path to goals dataset for goal sampling (default: `/global/scratch/users/achyuthkv76/carla_test_scripts/goals.npz`)

### Server Configuration

- `--host`: Server host address (default: `localhost`)
- `--port`: Server port (default: `5050`)
- `--goal_frame_index`: Index of goal frame in dataset (default: `1`)

### Observation Configuration

- `--obs_h`: Observation height (default: `100`)
- `--obs_w`: Observation width (default: `100`)
- `--obs_c`: Observation channels (default: `3`)
- `--frame_offsets`: Frame offsets for frame stacking, e.g., `--frame_offsets 0 -5 -10 -20` (default: `[0, -1]` if not specified)
  - **⚠️ Important:** If you trained with custom `--frame_offsets`, you MUST use the same values when running the server, otherwise the model will fail to load correctly
- `--block_size`: Block size for block-aware frame stacking (default: `400`)

### Action Configuration

- `--action_chunk_length`: Number of actions to predict in sequence (1 = disabled, typical: 4-16) (default: `1`)
- `--n_actions`: Use only the first N actions from the chunk (default: `4`)
- `--use_discrete`: Use discrete actions (multi-discrete mode: discretize throttle/steer/brake into 32 bins each)

### Example Commands

```bash
# Start server with CRL agent
python server.py --agent crl --model_path checkpoints/crl_model.pkl \
    --dataset_path goals.npz --host 0.0.0.0 --port 5050

# Start server with custom frame stacking
python server.py --agent crl --model_path checkpoints/crl_model.pkl \
    --dataset_path goals.npz --frame_offsets 0 -5 -10 -20

# Start server with discrete actions
python server.py --agent gcbc --model_path checkpoints/gcbc_model.pkl \
    --dataset_path goals.npz --use_discrete
```

## Evaluation Client (`impls/client.py`)

Connect to CARLA simulator and evaluation server to run online evaluation.

### Basic Usage

```bash
cd impls
python client.py --server-host localhost --server-port 5050
```

**Note:** Run from the `impls/` directory. Ensure CARLA simulator is running before starting the client.

### CARLA Configuration

- `--carla-host`: CARLA simulator host (default: `localhost`)
- `--carla-port`: CARLA simulator port (default: `2000`)

### Server Configuration

- `--server-host`: Evaluation server host (default: `localhost`)
- `--server-port`: Evaluation server port (default: `5050`)

### Evaluation Configuration

- `--frames`: Number of frames to evaluate (default: `600`)
- `--print-every`: Print metrics every N frames (default: `20`)
- `--no-video`: Disable video recording

### Wandb Logging

- `--wandb-project`: Wandb project name (default: `cmd_carla_eval`)
- `--wandb-run`: Wandb run name (default: `sync_run`)

### Example Commands

```bash
# Basic evaluation
python client.py --server-host localhost --server-port 5050 --frames 1000

# Evaluation with remote server
python client.py --server-host 192.168.1.100 --server-port 5050 \
    --carla-host localhost --carla-port 2000

# Evaluation without video recording
python client.py --server-host localhost --server-port 5050 --no-video

# Evaluation with custom Wandb project
python client.py --server-host localhost --server-port 5050 \
    --wandb-project my_eval_project --wandb-run test_run_1
```

## Complete Evaluation Workflow

1. **Train a model:**
   ```bash
   cd impls
   python train.py --dataset_path /path/to/dataset.npz --agent CRL \
       --steps 1000000 --ckpt_dir checkpoints
   ```
   This will save checkpoints as `checkpoints/agent_step{step}.pkl` (e.g., `checkpoints/agent_step1000000.pkl`)

2. **Start the evaluation server** (in a separate terminal):
   ```bash
   cd impls
   python server.py --agent crl --model_path checkpoints/agent_step1000000.pkl \
       --dataset_path goals.npz --host 0.0.0.0 --port 5050
   ```
   **Important:** The server must use the same observation configuration (frame stacking, etc.) as the training script.

3. **Start CARLA simulator** (if not already running):
   ```bash
   # CARLA should be running on localhost:2000
   # The client will connect to this automatically
   ```

4. **Run the evaluation client:**
   ```bash
   cd impls
   python client.py --server-host localhost --server-port 5050 \
       --frames 1000 --wandb-project evaluation
   ```

## Important Notes

- **Working Directory:** All scripts should be run from the `impls/` directory
- **CARLA:** Make sure CARLA simulator is running (default: `localhost:2000`) before starting the evaluation client
- **Configuration Matching:** The server must use the same observation configuration (frame stacking, `--frame_offsets`, `--block_size`, etc.) as was used during training
- **Checkpoint Format:** Checkpoints are saved as `agent_step{total_steps}.pkl` in the directory specified by `--ckpt_dir`
- **Agent Names:** 
  - For `train.py`: Use uppercase (e.g., `CRL`, `GCBC`, `CMD`, `GCIQL`)
  - For `server.py`: Use lowercase (e.g., `crl`, `gcbc`, `cmd`, `gciql`)
- **Network Configuration:** For distributed evaluation, ensure proper network configuration for the server host/port
- **Frame Stacking:** If `--frame_offsets` is not specified:
  - `train.py` defaults to `[0]` (single frame)
  - `server.py` defaults to `[0, -1]` (two frames). Make sure to match training configuration!
- **JAX/XLA Memory:** The server sets default JAX memory settings (`XLA_PYTHON_CLIENT_PREALLOCATE=false`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5`). You can override these with environment variables if needed.
