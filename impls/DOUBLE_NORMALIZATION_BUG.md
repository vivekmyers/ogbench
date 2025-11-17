# Critical Bug: Double Normalization Destroying Goal-Conditioning

## The Bug

**Your agent wasn't learning goal-conditioning because observations were being normalized TWICE**, crushing the visual signal to near-zero values where everything looks identical.

### Data Flow (BEFORE FIX):

```
1. Load dataset: uint8 [0, 255]
2. train.py line 103: / 255.0  →  float32 [0, 1]        ← NORMALIZE #1
3. Frame stacking: stays [0, 1]
4. encoders.py line 84: / 255.0  →  float32 [0, 0.004]  ← NORMALIZE #2 ❌
5. Encoder processes: all inputs ≈ 0, can't distinguish anything
```

### Why This Breaks Goal-Conditioning:

- All observations look nearly black (values ≈ 0.002)
- All goals look nearly black (values ≈ 0.002)  
- Encoder cannot distinguish between different observations or goals
- Agent learns to ignore goals and just output constant actions ("drive forward")
- No matter what goal you give it, the encoder sees the same thing

### Data Flow (AFTER FIX):

```
1. Load dataset: uint8 [0, 255]
2. train.py line 104: keep uint8  →  uint8 [0, 255]     ← No normalization
3. Convert to float: astype(float32)  →  float32 [0, 255]
4. Frame stacking: stays [0, 255]
5. encoders.py line 84: / 255.0  →  float32 [0, 1]      ← Single normalize ✓
6. Encoder processes: full visual range, can distinguish features
```

## How to Verify the Fix

### Step 1: Check that old checkpoints were affected

```bash
cd /nfs/kun2/users/achyuth/ogbench/impls

# This should show double normalization (values ≈ 0.002)
python verify_normalization.py --dataset_path /nfs/kun2/users/achyuth/train_data.npz
```

If you see "❌ PROBLEM DETECTED: Double normalization!", that confirms all your previous checkpoints are unusable.

### Step 2: Test an old checkpoint to confirm it's not goal-conditioned

```bash
python test_goal_conditioning.py \
  --checkpoint /nfs/kun2/users/achyuth/checkpoints/cmd_10.24_mini_99_disc/agent_step280000.pkl \
  --algorithm CMD
```

Expected output:
```
❌ PROBLEM: Agent is NOT goal-conditioned!
   Actions barely change with different goals.
```

### Step 3: Retrain from scratch with the fix

**IMPORTANT**: You MUST retrain from scratch! Old checkpoints are corrupted by double normalization and cannot be fixed.

```bash
CUDA_VISIBLE_DEVICES=7 python train.py \
  --dataset_path /nfs/kun2/users/achyuth/train_data.npz \
  --project cmd_fixed_normalization \
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

### Step 4: After training, test the new checkpoint

```bash
python test_goal_conditioning.py \
  --checkpoint /nfs/kun2/users/achyuth/checkpoints/cmd_fixed/agent_step100000.pkl \
  --algorithm CMD
```

Expected output:
```
✓ Agent appears to use goal information.
✓ Actions show reasonable diversity.
```

## Why This Bug Was Hard to Find

1. **Training still "worked"**: Loss decreased, no NaN values, gradients were stable
2. **Encoder structure looked correct**: It's supposed to normalize inputs
3. **No error messages**: Everything ran without crashes
4. **Gradual degradation**: Agent learned *something*, just not goal-conditioning
5. **Other issues masked it**: The periodic Q-drops and other instabilities distracted from the root cause

## Impact on Your Results

**ALL your previous experiments were affected by this bug**, including:
- 40k dataset run (jump at 750k steps)
- 900k dataset run (jump at 300k steps)  
- All checkpoints from these runs

The periodic instabilities and training collapses you saw were likely SYMPTOMS of this bug, not the root cause. When the visual signal is crushed to [0, 0.004], the model:
- Can't learn meaningful features
- Overfits to noise
- Becomes unstable after some training
- Learns trivial solutions

## What to Expect After the Fix

With proper normalization, you should see:
1. **No more double normalization** - observations in [0, 1] range at encoder input
2. **Goal-conditioned behavior** - agent actions change based on goals
3. **More stable training** - fewer periodic collapses (though you should still fix the random shuffle issue)
4. **Better performance** - agent can actually solve tasks in the dataset

## Additional Recommended Fixes

While fixing double normalization is critical, you should also address:

1. **Fixed shuffle seed** (prevents periodic Q-drops):
```python
# Line 145 in train.py
rng = np.random.default_rng(seed=42)
```

2. **Lower alpha** (reduces BC dominance):
```python
# Line 276 in train.py
cfg.alpha = 0.3  # Balance BC and Q-learning
```

3. **Lower learning rate** (more stable):
```python
# Line 278 in train.py
cfg.lr = 1e-4
```

## Verification Checklist

- [ ] Run `verify_normalization.py` to confirm fix
- [ ] Retrain from scratch (old checkpoints unusable)
- [ ] Run `test_goal_conditioning.py` on new checkpoint
- [ ] Check that actions change with different goals
- [ ] Verify agent can solve tasks from the dataset
- [ ] Monitor for periodic Q-drops (should be reduced with fixed shuffle)

## Bottom Line

**This was THE bug preventing goal-conditioning.** Everything else (periodic drops, training collapses, trivial solutions) were downstream effects. Fix this and retrain from scratch.


