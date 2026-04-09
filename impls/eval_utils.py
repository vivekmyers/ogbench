"""Pure numpy + stdlib helpers for eval server/client (no JAX)."""

from __future__ import annotations

import pickle
import socket

import numpy as np


def send_len_pickled(conn: socket.socket, obj) -> None:
    """Send one message: [4-byte big-endian length] + pickle(obj)."""
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    conn.sendall(len(payload).to_bytes(4, "big") + payload)


def recvall(conn: socket.socket, n: int, *, timeout: float | None = None) -> bytes:
    if timeout is not None:
        conn.settimeout(timeout)
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Connection lost while receiving payload")
        buf += chunk
    return buf


def fit_to_hw(img: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    Center-crop to (H,W) if larger; pad with edge pixels if smaller.
    Keeps channels untouched. No resampling libs needed.
    """
    h, w = img.shape[:2]
    y0 = max(0, (h - H) // 2)
    x0 = max(0, (w - W) // 2)
    cropped = img[y0 : min(y0 + H, h), x0 : min(x0 + W, w)]
    pad_h = H - cropped.shape[0]
    pad_w = W - cropped.shape[1]
    if pad_h > 0 or pad_w > 0:
        if cropped.ndim == 2:
            cropped = cropped[:, :, None]
        cropped = np.pad(
            cropped,
            ((0, max(0, pad_h)), (0, max(0, pad_w)), (0, 0)),
            mode="edge",
        )
        cropped = cropped[:H, :W, :]
    return cropped


def dataset_image_stack(dataset) -> np.ndarray:
    """Training .npz usually has ``observations``; older eval files may use ``frames``."""
    if "observations" in dataset:
        return dataset["observations"]
    if "frames" in dataset:
        return dataset["frames"]
    raise KeyError("Dataset must contain 'observations' or 'frames' (RGB per timestep)")


def get_xy_from_dataset(dataset, frame_idx: int) -> tuple[float, float] | None:
    for k in ("position", "poses_xy", "poses", "goal_xy", "loc_xy", "xy", "ego_xy"):
        if k in dataset:
            arr = np.asarray(dataset[k][frame_idx]).reshape(-1)
            if arr.size >= 2:
                return float(arr[0]), float(arr[1])
    return None


def get_yaw_deg_from_dataset(dataset, frame_idx: int) -> float | None:
    for k in ("yaw", "heading", "rotation"):
        if k in dataset:
            return float(np.asarray(dataset[k][frame_idx]).reshape(-1)[0])
    if "position" in dataset:
        arr = np.asarray(dataset["position"][frame_idx]).reshape(-1)
        if arr.size >= 4:
            return float(arr[3])
    return None
