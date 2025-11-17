import socket
import pickle
import numpy as np
import jax
import jax.numpy as jnp
from flax.serialization import from_bytes
from collections import deque
import argparse
from importlib import import_module

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, required=True, help="Algorithm name (e.g., CRL, GCIQL, SAC)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model .pkl file")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset .npz file")
    parser.add_argument("--goal_index", type=int, default=100, help="Goal frame index from dataset")
    parser.add_argument("--frame_stack", type=int, default=4)
    args = parser.parse_args()

    # === Load Model ===
    print(f"Loading {args.algorithm} agent...")
    agent_module = import_module(f"agents.{args.algorithm.lower()}")
    agent_cls = getattr(agent_module, f"{args.algorithm}Agent")
    get_config = getattr(agent_module, "get_config")
    
    config = get_config()
    config.encoder = "impala_large"  # Match training encoder
    config.frame_stack = args.frame_stack
    config.layer_norm = False  # Match training config
    config.frame_offsets = [0, -5, -10, -15]  # Match training frame offsets
    
    # Algorithm-specific configs
    if args.algorithm.upper() == "CRL":
        config.actor_loss = "ddpgbc"
    config.discrete = False

    obs_shape = (100, 100, 3)  # Match training observation shape
    act_shape = (3,)  # Match training action shape (throttle, steer, brake)
    stacked_obs_shape = (100, 100, obs_shape[2] * config.frame_stack)

    # Dummy inputs
    dummy_obs = jnp.zeros((1, *stacked_obs_shape))
    dummy_act = jnp.zeros((1, *act_shape))

    agent = agent_cls.create(
        seed=0,
        ex_observations=dummy_obs,
        ex_actions=dummy_act,
        config=config,
    )

    with open(args.model_path, "rb") as f:
        byte_data = f.read()
        agent = from_bytes(agent, byte_data)

    print("Model loaded and agent initialized")

    # === Load goal ===
    dataset = np.load(args.dataset_path)
    goal = dataset["observations"][args.goal_index].astype(np.float32)
    goal_stacked = np.concatenate([goal] * config.frame_stack, axis=-1)
    print("Goal loaded")

    # === Socket setup ===
    HOST = "localhost"
    PORT = 5050
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.bind((HOST, PORT))
    server_sock.listen(1)
    print(f"Listening on {HOST}:{PORT}...")
    conn, addr = server_sock.accept()
    print(f"Connection established with client: {addr}")

    try:
        while True:
            length_bytes = conn.recv(4)
            if not length_bytes:
                break
            msg_len = int.from_bytes(length_bytes, 'big')
            data = b''
            while len(data) < msg_len:
                chunk = conn.recv(msg_len - len(data))
                if not chunk:
                    raise ConnectionError("Connection lost during image reception")
                data += chunk

            # Deserialize and reshape image
            img = pickle.loads(data)
            obs = jnp.asarray(img).reshape(1, *obs_shape) # (1, 48, 48, 3)
            obs_stacked = jnp.concatenate([obs] * config.frame_stack, axis=-1)

            # Get deterministic action (mean of distribution)
            dist = agent.network.select('actor')(obs_stacked, goal_stacked[None])
            act = dist.mean()

            act_np = np.array(act[0], dtype=np.float32)
            
            # Debug: Print action values
            print(f"Action: throttle={act_np[0]:.3f}, steer={act_np[1]:.3f}, brake={act_np[2]:.3f}")
            
            # If simulator expects 2D actions, drop the brake dimension
            if len(act_np) == 3:
                # For CARLA, might want to use: [throttle, steer] or [throttle-brake, steer]
                act_np_2d = np.array([act_np[0], act_np[1]], dtype=np.float32)  # throttle, steer
                conn.sendall(act_np_2d.tobytes())
            else:
                conn.sendall(act_np.tobytes())

    except KeyboardInterrupt:
        print("Interrupted by User")
    finally:
        conn.close()
        server_sock.close()
        print("Server shutdown cleanly")

if __name__ == "__main__":
    main()
