#!/usr/bin/env python3
import sys, os
import numpy as np

def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_npz.py <src.npz> [dst.npz]")
        sys.exit(1)

    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(src)[0] + "_trim.npz"

    d = np.load(src)
    obs = d.get("observations", d.get("frames"))
    if obs is None:
        raise KeyError("Neither 'observations' nor 'frames' found in source NPZ")

    actions = d["actions"]  # must exist
    N = obs.shape[0]
    terminals = np.zeros((N,), dtype=np.bool_)  # all zeros as requested

    np.savez_compressed(dst, observations=obs, actions=actions, terminals=terminals)
    out = np.load(dst)
    print("Wrote", dst, {k: out[k].shape for k in out.files})

if __name__ == "__main__":
    main()
