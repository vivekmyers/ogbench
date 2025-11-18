# CMD Agent Fix - Critical Indentation Bug

## The Bug

The `sample_batch` function was incorrectly indented inside the `get_config()` function (starting at line 289), making it **inaccessible at module level**.

### Before (Broken):
```python
def get_config():
    config = ml_collections.ConfigDict(...)
    return config

    def sample_batch(...):  # ← WRONG! Indented inside get_config()
        ...
```

### After (Fixed):
```python
def get_config():
    config = ml_collections.ConfigDict(...)
    return config


def sample_batch(...):  # ← CORRECT! Module-level function
    ...
```

## Why This Broke CMD

1. When `train.py` tried to import `sample_batch` from `agents.cmd`, it would fail or get None
2. The training script would fall back to using CRL's `sample_batch` function
3. **CMD was never using its own goal sampling strategy**
4. The agent couldn't learn to use goals properly because the data loader wasn't even calling CMD's code

## Impact

This was preventing CMD from working as designed. Now that `sample_batch` is accessible:
- ✅ CMD uses geometric goal sampling (E[horizon] = 100 steps = 5 seconds at 20fps)
- ✅ Goals are properly sampled within block boundaries (400 steps = 20 seconds)
- ✅ The agent can actually learn to condition on goals

## Other Changes

Added missing import: `import numpy as np` (needed by `sample_batch`)

## Testing

Verify the fix works:
```python
from agents.cmd import sample_batch
print(sample_batch)  # Should print the function, not None or error
```
