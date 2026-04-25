#!/usr/bin/env python3
"""Inspect a saved agent .pkl: print embedded config (if any) and infer
training-time hyperparameters from network parameter shapes.

Usage:
    python impls/inspect_ckpt.py --path /global/scratch/users/achyuthkv76/gcbc_models/run27.pkl
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from flax.serialization import msgpack_restore


def _walk(d: Any, prefix: str = "") -> dict[str, tuple]:
    """Flatten a nested dict of arrays into {dotted.path: shape}."""
    out: dict[str, tuple] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_walk(v, f"{prefix}.{k}" if prefix else k))
    elif hasattr(d, "shape"):
        out[prefix] = tuple(d.shape)
    else:
        out[prefix] = ("scalar", type(d).__name__)
    return out


def _maybe_unpickle(raw: bytes) -> tuple[bytes, dict | None]:
    """Return (flax_bytes, embedded_config_or_None)."""
    try:
        obj = pickle.loads(raw)
    except Exception:
        return raw, None
    if isinstance(obj, dict) and "agent" in obj:
        return obj["agent"], obj.get("config")
    return raw, None


def _infer_obs_geometry(shapes: dict[str, tuple]) -> dict:
    """Look at the first conv kernel of state/value/actor encoders to infer
    obs_c * frame_stack_k. Conv kernels are (kh, kw, in_channels, out_channels).
    """
    encoder_first_convs = {}
    for path, shp in shapes.items():
        if not isinstance(shp, tuple) or len(shp) != 4:
            continue
        # heuristic: very first conv inside an encoder, kernel name == 'kernel',
        # and it's nested under something with "encoder" in the path
        if "encoder" not in path.lower():
            continue
        if not path.endswith("kernel"):
            continue
        # take the conv with the smallest depth in its module path (first conv)
        module_depth = path.count(".")
        encoder_first_convs.setdefault(path.split(".encoder")[0], (module_depth, path, shp))
        cur = encoder_first_convs[path.split(".encoder")[0]]
        if module_depth < cur[0]:
            encoder_first_convs[path.split(".encoder")[0]] = (module_depth, path, shp)

    seen_in_channels = set()
    for _, (_, _, shp) in encoder_first_convs.items():
        seen_in_channels.add(int(shp[2]))

    info: dict = {"encoder_first_conv_in_channels": sorted(seen_in_channels)}
    if len(seen_in_channels) == 1:
        c = next(iter(seen_in_channels))
        info["inferred_obs_c_times_frame_stack"] = c
        # Common assumption: 3-channel RGB.
        if c % 3 == 0:
            info["plausible_frame_stack (assuming obs_c=3)"] = c // 3
    return info


def _infer_action_chunk_length(shapes: dict[str, tuple]) -> dict:
    """Find the actor's final output Dense kernel; its out_features == action_dim
    * action_chunk_length (if the actor outputs a chunk).
    The trunk output dim is the second-to-last MLP layer size.
    """
    actor_dense_outs: list[tuple[str, tuple]] = []
    for path, shp in shapes.items():
        if not isinstance(shp, tuple) or len(shp) != 2:
            continue
        if "actor" not in path.lower():
            continue
        if not path.endswith("kernel"):
            continue
        actor_dense_outs.append((path, shp))

    info: dict = {"actor_dense_kernels": actor_dense_outs[-6:]}
    if not actor_dense_outs:
        return info

    # Heuristic: last actor Dense kernel = (hidden, action_dim * chunk_len)
    # Try to recognise (..., 3) -> chunk=1, (..., 6) -> chunk=2, (..., 9) -> 3, ...
    final_path, final_shape = actor_dense_outs[-1]
    out_dim = int(final_shape[-1])
    info["actor_final_layer"] = (final_path, final_shape)
    for ad in (3,):  # CARLA action_dim is 3 (throttle, steer, brake)
        if out_dim % ad == 0:
            info[f"chunk_len_assuming_action_dim={ad}"] = out_dim // ad
    # If the head produces a Tanh-Normal, sometimes there's a separate log_std
    # layer; show the previous layer too to give the human a hint.
    if len(actor_dense_outs) >= 2:
        info["actor_penultimate_layer"] = actor_dense_outs[-2]
    return info


def _list_top_level_modules(shapes: dict[str, tuple]) -> list[str]:
    tops: set[str] = set()
    for path in shapes:
        head = path.split(".")[0]
        # peel one more level for "modules_*" containers used by flax_utils
        parts = path.split(".")
        if head == "params" and len(parts) >= 2:
            tops.add(f"params.{parts[1]}")
        elif head in ("network",):
            if len(parts) >= 3:
                tops.add(".".join(parts[:3]))
            else:
                tops.add(head)
        else:
            tops.add(head)
    return sorted(tops)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--max-shapes", type=int, default=40,
                    help="How many param shapes to print after the inferred summary.")
    args = ap.parse_args()

    p = Path(args.path)
    print(f"== Inspecting {p} ({p.stat().st_size/1e6:.1f} MB) ==")

    raw = p.read_bytes()
    flax_bytes, embedded_config = _maybe_unpickle(raw)

    if embedded_config is not None:
        print("\n-- Embedded config (from pickle wrapper) --")
        try:
            print(json.dumps(embedded_config, indent=2, default=str))
        except Exception:
            for k, v in (embedded_config.items() if isinstance(embedded_config, dict) else []):
                print(f"  {k}: {v}")
    else:
        print("\n-- No embedded config in pickle wrapper (raw flax bytes or no 'config' key). --")

    print("\n-- Restoring msgpack state dict --")
    state = msgpack_restore(flax_bytes)
    if not isinstance(state, dict):
        print(f"Restored object is not a dict (type={type(state).__name__}); aborting shape walk.")
        return

    shapes = _walk(state)
    print(f"  total leaves: {len(shapes)}")

    print("\n-- Top-level modules --")
    for top in _list_top_level_modules(shapes):
        print(f"  {top}")

    print("\n-- Inferred obs geometry --")
    for k, v in _infer_obs_geometry(shapes).items():
        print(f"  {k}: {v}")

    print("\n-- Inferred actor head / chunk length --")
    for k, v in _infer_action_chunk_length(shapes).items():
        print(f"  {k}: {v}")

    print(f"\n-- First {args.max_shapes} param shapes (alphabetical) --")
    for path in sorted(shapes)[: args.max_shapes]:
        print(f"  {path}: {shapes[path]}")


if __name__ == "__main__":
    main()
