# CMD Training Issues and Fixes

## Problems Identified

### 1. **Periodic Training Instability** (Q-mean drops every 200k steps)
**Cause**: Random reshuffling at every epoch boundary with different random seeds
- Each epoch, blocks are shuffled with a NEW random seed (line 145: `np.random.default_rng()`)
- CMD's contrastive learning is sensitive to data ordering
- When blocks are reshuffled, Q-values learned on old adjacencies don't match new data
- Results in periodic Q-mean drops at epoch boundaries

**Fix**: Use fixed shuffle seed (line 145)
```python
rng = np.random.default_rng(seed=42)  # Fixed seed
```

### 2. **BC Loss Dominance** (Agent ignores goals, just imitates)
**Cause**: `alpha=1.0` makes behavioral cloning loss equal to Q-loss
- With alpha=1.0, BC loss dominates the training signal
- Agent learns to imitate expert actions without considering goals
- Results in "always drive forward" behavior

**Fix**: Reduce alpha to 0.1 (line 276)
```python
cfg.alpha = 0.1  # Let Q-learning influence policy more
```

### 3. **Training Instability and Early Collapse** (Jump at 300k steps)
**Cause**: Learning rate too high combined with unstable Q-values
- LR of 3e-4 is too aggressive for large vision networks
- Combined with periodic reshuffling shocks
- Q-values collapse after ~300k steps

**Fix**: Lower learning rate (line 278)
```python
cfg.lr = 1e-4  # More stable training
```

### 4. **Potential CMD Architecture Issue**
**Issue**: CMD's contrastive loss uses rolled actions (line 100-105 in cmd.py):
```python
actions_roll = jnp.roll(batch["actions"], shift=1, axis=0)
phi = self.network.select("critic")(batch["observations"], q_actions)
psi = self.network.select("critic")(batch["actor_goals"], actions_roll)
```

The rolled action comes from a DIFFERENT transition, which may not be appropriate for goal-conditioned learning. However, this might be by design in CMD's contrastive framework.

**Action**: Monitor if the fixes above resolve the issue. If not, consider switching to CRL or GCIQL which have more standard goal-conditioned architectures.

## Changes Made

1. ✅ Fixed shuffle seed to prevent periodic distribution shifts
2. ✅ Reduced alpha from 1.0 to 0.1 to reduce BC dominance  
3. ✅ Reduced learning rate from 3e-4 to 1e-4 for stability
4. ✅ Created diagnostic script to test goal-conditioning

## Next Steps

### 1. Test existing checkpoint before retraining:
```bash
cd /nfs/kun2/users/achyuth/ogbench/impls
python test_goal_conditioning.py \
  --checkpoint /nfs/kun2/users/achyuth/checkpoints/cmd_10.24_mini_99_disc/agent_step280000.pkl \
  --algorithm CMD
```

This will tell you if the current agent is using goals at all.

### 2. Retrain with new settings:
```bash
CUDA_VISIBLE_DEVICES=7 python train.py \
  --dataset_path /nfs/kun2/users/achyuth/train_data.npz \
  --project cmd_fixed \
  --ckpt_dir /nfs/kun2/users/achyuth/checkpoints/cmd_fixed \
  --ckpt_every 20_000 \
  --frame_stack 4 \
  --algorithm CMD \
  --chunk_size 2100 \
  --batch_size 64 \
  --steps 100000 \
  --epochs 25 \
  --block_size 150
```

### 3. Monitor these metrics during training:
- `train/actor_q_mean` - should be stable without periodic drops
- `train/actor_bc_loss` vs `train/actor_q_loss` - Q-loss should be significant
- `train/actor_mse` - should decrease over time
- `val/actor_loss` - should decrease smoothly

### 4. Expected behavior after fixes:
- No periodic Q-mean drops
- Actor loss should decrease and stay stable
- Agent should show goal-conditioned behavior (test with diagnostic script)

## Alternative Algorithms to Consider

If CMD continues to have issues, consider:

1. **CRL** (Contrastive RL):
   - More standard goal-conditioned architecture
   - Uses standard Q-learning without rolled actions
   - Set `--algorithm CRL`

2. **GCIQL** (Goal-Conditioned IQL):
   - In-sample learning, more stable
   - No contrastive learning complications
   - Set `--algorithm GCIQL`

## Understanding the Periodic Pattern

The periodic spikes in your losses were occurring at **every 2 epochs** (200k steps) because:
- 100k steps per epoch
- Shuffle happens at start of each epoch
- But the pattern shows up every 200k steps

**Why 200k instead of 100k?**
Likely the model needs ~1 epoch to adapt to new shuffle, so the impact shows up in metrics at the END of the adjustment period (i.e., at the start of the NEXT epoch).

The fixed shuffle seed eliminates this entirely - all epochs will see the same data ordering.


