"""Test if the agent actually uses goal information."""
import numpy as np
import jax.numpy as jnp
import argparse
from pathlib import Path
import flax.serialization as fxs
from importlib import import_module

def test_goal_sensitivity(checkpoint_path, algorithm='CMD'):
    """Test if agent's actions change with different goals."""
    
    # Load agent
    agent_module = import_module(f'agents.{algorithm.lower()}')
    
    with open(checkpoint_path, 'rb') as f:
        agent = fxs.from_bytes(agent_module.__dict__[f'{algorithm}Agent'], f.read())
    
    # Create a single observation (random for testing)
    obs = jnp.ones((1, 100, 100, 12)) * 0.5  # Assuming 4-frame stack, 3 channels each
    
    # Create different goals
    goal1 = jnp.ones((1, 100, 100, 12)) * 0.2  # Dark goal
    goal2 = jnp.ones((1, 100, 100, 12)) * 0.8  # Bright goal
    goal3 = jnp.zeros((1, 100, 100, 12))        # Black goal
    
    # Get actions for same observation but different goals
    dist1 = agent.network.select('actor')(obs, goal1, params=agent.network.params)
    dist2 = agent.network.select('actor')(obs, goal2, params=agent.network.params)
    dist3 = agent.network.select('actor')(obs, goal3, params=agent.network.params)
    
    action1 = dist1.mode()
    action2 = dist2.mode()
    action3 = dist3.mode()
    
    print("\n" + "="*60)
    print("GOAL SENSITIVITY TEST")
    print("="*60)
    print(f"\nSame observation with different goals:")
    print(f"  Dark goal  -> Action: {action1}")
    print(f"  Bright goal-> Action: {action2}")
    print(f"  Black goal -> Action: {action3}")
    
    # Compute differences
    diff_1_2 = float(jnp.mean(jnp.abs(action1 - action2)))
    diff_1_3 = float(jnp.mean(jnp.abs(action1 - action3)))
    diff_2_3 = float(jnp.mean(jnp.abs(action2 - action3)))
    
    print(f"\nAction differences (mean absolute):")
    print(f"  Dark vs Bright: {diff_1_2:.6f}")
    print(f"  Dark vs Black:  {diff_1_3:.6f}")
    print(f"  Bright vs Black: {diff_2_3:.6f}")
    
    avg_diff = (diff_1_2 + diff_1_3 + diff_2_3) / 3
    
    print(f"\nAverage difference: {avg_diff:.6f}")
    
    if avg_diff < 0.01:
        print("\n❌ PROBLEM: Agent is NOT goal-conditioned!")
        print("   Actions barely change with different goals.")
        return False
    else:
        print("\n✓ Agent appears to use goal information.")
        return True


def test_action_diversity(checkpoint_path, algorithm='CMD'):
    """Test if agent produces diverse actions or just drives forward."""
    
    agent_module = import_module(f'agents.{algorithm.lower()}')
    
    with open(checkpoint_path, 'rb') as f:
        agent = fxs.from_bytes(agent_module.__dict__[f'{algorithm}Agent'], f.read())
    
    # Create random observations and goals
    np.random.seed(42)
    n_samples = 100
    obs = jnp.array(np.random.randn(n_samples, 100, 100, 12).astype(np.float32) * 0.1 + 0.5)
    goals = jnp.array(np.random.randn(n_samples, 100, 100, 12).astype(np.float32) * 0.1 + 0.5)
    
    # Get actions
    dist = agent.network.select('actor')(obs, goals, params=agent.network.params)
    actions = dist.mode()
    
    print("\n" + "="*60)
    print("ACTION DIVERSITY TEST")
    print("="*60)
    print(f"\nStatistics over {n_samples} random (obs, goal) pairs:")
    print(f"  Throttle: mean={float(jnp.mean(actions[:, 0])):.3f}, std={float(jnp.std(actions[:, 0])):.3f}")
    print(f"  Steer:    mean={float(jnp.mean(actions[:, 1])):.3f}, std={float(jnp.std(actions[:, 1])):.3f}")
    print(f"  Brake:    mean={float(jnp.mean(actions[:, 2])):.3f}, std={float(jnp.std(actions[:, 2])):.3f}")
    
    # Check if actions are too similar (trivial solution)
    steer_std = float(jnp.std(actions[:, 1]))
    throttle_mean = float(jnp.mean(actions[:, 0]))
    
    if steer_std < 0.05:
        print("\n❌ PROBLEM: Steer has very low variance!")
        print("   Agent might be always steering the same amount.")
    
    if throttle_mean > 0.7:
        print("\n❌ PROBLEM: Throttle is very high on average!")
        print("   Agent might be learning 'always drive forward'.")
        return False
    else:
        print("\n✓ Actions show reasonable diversity.")
        return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to agent checkpoint')
    parser.add_argument('--algorithm', type=str, default='CMD', help='Algorithm name')
    args = parser.parse_args()
    
    print("\nTesting agent from:", args.checkpoint)
    
    goal_ok = test_goal_sensitivity(args.checkpoint, args.algorithm)
    action_ok = test_action_diversity(args.checkpoint, args.algorithm)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    if goal_ok and action_ok:
        print("✓ Agent appears to be working correctly!")
    else:
        print("❌ Agent has issues - see problems above.")
        print("\nLikely causes:")
        print("  1. Goal encoding is not informative")
        print("  2. Dataset is too easy (all 'drive forward')")
        print("  3. Training instability preventing learning")
        print("  4. CMD contrastive loss not working properly")


