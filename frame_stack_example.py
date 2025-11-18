#!/usr/bin/env python3
"""
Example of how to use the new spaced frame stacking feature.

This shows different frame stacking patterns you can use.
"""

# Example configurations for different frame stacking patterns:

# 1. Default consecutive frames (same as before)
consecutive_config = {
    'frame_stack': 4,
    'frame_offsets': None  # Will use [0, -1, -2, -3]
}

# 2. Your suggested spaced pattern
spaced_config = {
    'frame_stack': 4,
    'frame_offsets': [0, -5, -10, -15]  # Uses [idx, idx-5, idx-10, idx-15]
}

# 3. Every 2 frames
every_two_config = {
    'frame_stack': 4,
    'frame_offsets': [0, -2, -4, -6]  # Uses [idx, idx-2, idx-4, idx-6]
}

# 4. Every 10 frames (for very long-term patterns)
long_term_config = {
    'frame_stack': 4,
    'frame_offsets': [0, -10, -20, -30]  # Uses [idx, idx-10, idx-20, idx-30]
}

# 5. Mixed spacing (exponential-like)
mixed_config = {
    'frame_stack': 4,
    'frame_offsets': [0, -1, -3, -7]  # Uses [idx, idx-1, idx-3, idx-7]
}

# 6. Recent + very old (for temporal context)
temporal_context_config = {
    'frame_stack': 4,
    'frame_offsets': [0, -1, -25, -50]  # Mix of recent and old frames
}

print("Frame stacking configuration examples:")
print(f"Consecutive: {consecutive_config}")
print(f"Spaced (your suggestion): {spaced_config}")
print(f"Every 2 frames: {every_two_config}")
print(f"Long term: {long_term_config}")
print(f"Mixed spacing: {mixed_config}")
print(f"Temporal context: {temporal_context_config}")





















