import flax.serialization as fxs
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import unfreeze
from agents.crl import CRLAgent, get_config

# ----------------------------------------
# Dummy config setup (must match training)
# ----------------------------------------
config = get_config()
config.encoder = None
config.frame_stack = 1
config.discrete = False
config.actor_loss = 'ddpgbc'

# ----------------------------------------
# Dummy inputs for agent structure
# ----------------------------------------
obs_shape = (48, 48, 3)
act_shape = (2,)
dummy_obs = jnp.zeros((1, np.prod(obs_shape)))  # Flattened obs
dummy_act = jnp.zeros((1, *act_shape))

agent = CRLAgent.create(
    seed=0,
    ex_observations=dummy_obs,
    ex_actions=dummy_act,
    config=config
)

# ----------------------------------------
# Load serialized model
# ----------------------------------------
with open("/global/scratch/users/achyuthkv76/final_model.pkl", "rb") as f:
    byte_data = f.read()
    model_state = fxs.from_bytes(agent, byte_data)

# ----------------------------------------
# Structure summary (2–3 levels)
# ----------------------------------------
def summarize(obj, path="root", depth=0, visited=None, max_depth=3):
    if visited is None:
        visited = set()
    if depth > max_depth:
        print("  " * depth + f"{path}: <max depth reached>")
        return
    obj_id = id(obj)
    if obj_id in visited:
        print("  " * depth + f"{path}: <already visited>")
        return
    visited.add(obj_id)

    indent = "  " * depth
    if isinstance(obj, dict) or hasattr(obj, "keys"):
        try:
            items = obj.items()
        except Exception:
            print(f"{indent}{path}: <uninspectable dict>")
            return
        print(f"{indent}{path}: dict with {len(list(items))} keys")
        for k, v in items:
            summarize(v, f"{path}['{k}']", depth + 1, visited)
    elif hasattr(obj, "shape") and hasattr(obj, "dtype"):
        print(f"{indent}{path}: array, shape={obj.shape}, dtype={obj.dtype}")
    elif hasattr(obj, '__dict__'):
        print(f"{indent}{path}: object of type {type(obj).__name__}")
        for k, v in vars(obj).items():
            summarize(v, f"{path}.{k}", depth + 1, visited)
    else:
        print(f"{indent}{path}: {type(obj).__name__}")

# ----------------------------------------
# Run summary
# ----------------------------------------
print("🔍 Model Structure Summary:")
summarize(unfreeze(model_state))
