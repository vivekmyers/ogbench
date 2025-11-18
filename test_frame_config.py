#!/usr/bin/env python3
"""
Test configuration for the new frame stacking setup.
"""

# Your new configuration:
config = {
    'frame_stack': 2,
    'frame_offsets': [0, -5],
    'batch_size': 1024,
}

print("New CRL Configuration:")
print(f"  Frame stack: {config['frame_stack']}")
print(f"  Frame offsets: {config['frame_offsets']}")
print(f"  Batch size: {config['batch_size']}")
print()
print("What this means:")
print("  - Each observation will contain 2 frames:")
print("    * Current frame (offset 0)")
print("    * Frame from 5 steps ago (offset -5)")
print("  - Batch size of 1024 should provide good contrastive learning signal")
print("  - Much simpler than 4-frame stacking, easier to debug")
print()
print("Expected benefits:")
print("  - Temporal context without overfitting to consecutive frames")
print("  - Simpler model (2 frames vs 4)")
print("  - Better contrastive learning with larger batch size")
print("  - More stable training without gradient accumulation")



















