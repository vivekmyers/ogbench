# Conda Environment Files

This directory contains conda environment YAML files for different components of the OGBench project.

## Environments

### `train.yml`
Environment for training offline RL agents using `impls/train.py`.

**Usage:**
```bash
conda env create -f train.yml
conda activate train
```

**Key dependencies:**
- JAX/JAXlib (with CUDA support)
- Flax
- Optax
- Wandb
- Gymnasium
- Other ML/RL libraries

### `carla_client.yml`
Environment for running the CARLA evaluation client (`impls/client.py`).

**Usage:**
```bash
conda env create -f carla_client.yml
conda activate carla_client
```

**Key dependencies:**
- CARLA Python API
- OpenCV
- Socket libraries
- Wandb (for logging)

### `eval_server.yml`
Environment for running the evaluation server (`impls/server.py`).

**Usage:**
```bash
conda env create -f eval_server.yml
conda activate eval_server
```

**Key dependencies:**
- JAX/JAXlib (with CUDA support)
- Flax
- Socket libraries
- Model loading utilities

## Updating Environments

To update an environment file after making changes to a conda environment:

```bash
conda env export -n <env_name> --no-builds > <env_name>.yml
```

The `--no-builds` flag removes build strings, making the files more portable across different systems.

## Notes

- These files were exported with `--no-builds` to improve portability
- You may need to adjust CUDA versions or other system-specific packages based on your hardware
- Some packages may need to be installed from specific channels (e.g., JAX CUDA packages)
