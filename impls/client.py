#!/usr/bin/env python3
from __future__ import annotations

import os, sys, math, time, random, socket, pickle, argparse
import datetime
from pathlib import Path
from collections import deque
from queue import Queue, Empty, Full

import numpy as np
import cv2
import carla
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

try:
    import wandb
except Exception:
    wandb = None

# ================= CARLA helpers =================

def clear_dynamic_actors(world: carla.World):
    for pat in ("vehicle.*", "walker.*", "sensor.*"):
        for a in world.get_actors().filter(pat):
            try: a.destroy()
            except Exception: pass
    world.tick()
    print("[CARLA] ✓ Sim clean.")


def set_weather(world: carla.World, *, quiet: bool = False) -> None:
    """Force CARLA 0.9.15 into clear midday lighting with all scene lamps off.

    Uses the built-in ClearNoon preset (sun_altitude_angle=90 = midday in 0.9.15,
    zero clouds/fog/rain) then disables the automatic day-night lamp cycle and turns
    every scene light off so city lamps don't bloom the image.
    """
    world.set_weather(carla.WeatherParameters.ClearNoon)

    # Two ticks so the Unreal sky catches up before the first camera frame.
    if not quiet:
        world.tick()
        world.tick()
        print("[CARLA] Weather → ClearNoon.")


def _push_latest(q: Queue, image):
    """Non-blocking camera callback: keep only the newest frame."""
    try:
        q.put_nowait(image)
    except Full:
        try: q.get_nowait()
        except Empty: pass
        try: q.put_nowait(image)
        except Full: pass


def _make_collision_callback(state: dict):
    """Return a callback that records the first 'real' collision into ``state``.

    ``state`` keys we touch:
      - 'collided' (bool): latched True on first accepted event
      - 'other' (str): other_actor.type_id of the offender
      - 'impulse' (float): magnitude of the collision impulse (N·s)
      - 'frame' (int): CARLA world frame at which collision was recorded
      - 'ignore_until_frame' (int): warm-up; events strictly before this are dropped
                                    (e.g. to skip the spawn-drop 'static.road' hit)

    The CARLA collision sensor fires on its own thread, so we keep writes minimal
    and check a latch so the first event wins.
    """
    def _cb(event):
        if state.get('collided'):
            return  # already latched; ignore rest of the shower
        frame = int(getattr(event, 'frame', 0))
        if frame < int(state.get('ignore_until_frame', 0)):
            return
        imp = event.normal_impulse
        mag = (imp.x * imp.x + imp.y * imp.y + imp.z * imp.z) ** 0.5
        other = getattr(event.other_actor, 'type_id', 'unknown')
        state['collided'] = True
        state['other'] = str(other)
        state['impulse'] = float(mag)
        state['frame'] = frame
    return _cb

def rgb_from_image(img: carla.Image) -> np.ndarray:
    # CARLA gives BGRA bytes; convert to RGB uint8 and copy
    arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)[:, :, :3]
    return arr[:, :, ::-1].copy()

def tick_and_grab(world: carla.World, q: Queue, timeout: float = 3.0):
    """Advance one tick and return camera image with the SAME frame id.
       Falls back to freshest frame (or None) if exact match doesn't arrive."""
    world.tick()
    target = world.get_snapshot().frame
    deadline = time.time() + timeout
    freshest = None
    while time.time() < deadline:
        try:
            im = q.get(timeout=max(0.0, deadline - time.time()))
            freshest = im
            if im.frame == target:
                # drain leftovers so we don't lag
                while True:
                    try: q.get_nowait()
                    except Empty: break
                return im
        except Empty:
            pass
    return freshest  # may be None

# ================= Socket helpers =================

def connect_eval(host: str, port: int, timeout: float = 10.0) -> socket.socket:
    print(f"[NET] Connecting to eval server {host}:{port} …")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect((host, port))
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print("[NET] ✓ Connected.")
    return s

def send_with_len(sock: socket.socket, payload: bytes):
    sock.sendall(len(payload).to_bytes(4, "big") + payload)

def recvall(sock: socket.socket, n: int, timeout: float = 10.0) -> bytes:
    sock.settimeout(timeout)
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed while receiving")
        buf += chunk
    return buf


def recv_one_pickled(sock: socket.socket, timeout: float = 30.0):
    """Read one length-prefixed message and unpickle."""
    n = int.from_bytes(recvall(sock, 4, timeout=timeout), "big")
    payload = recvall(sock, n, timeout=timeout)
    return pickle.loads(payload)


def send_episode_end(sock: socket.socket, reason: str, frames_used: int) -> None:
    """Tell server this episode finished (goal, timeout, stuck, etc.)."""
    msg = pickle.dumps(
        {"type": "episode_end", "reason": reason, "frames_used": int(frames_used)},
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    send_with_len(sock, msg)


def recv_action_chunk(sock: socket.socket, timeout: float = 10.0) -> None:
    """Receive and discard one action chunk (e.g. neutral chunk after a new header)."""
    chunk_length_bytes = recvall(sock, 4, timeout=timeout)
    chunk_length = int.from_bytes(chunk_length_bytes, "big")
    action_chunk_size = chunk_length * 3 * 4
    recvall(sock, action_chunk_size, timeout=timeout)


def parse_eval_header_obj(obj) -> dict:
    """Parse a pickled header dict into the same structure as try_receive_goal_header."""
    out: dict = {
        "goal_img": None,
        "goal_xy": None,
        "start_xy": None,
        "start_yaw_deg": None,
        "start_frame_index": None,
        "goal_frame_index": None,
        "obs_h": None,
        "obs_w": None,
        "obs_c": None,
        "frame_stack": None,
        "frames_per_episode": None,
        "num_episodes": None,
        "episode_index": None,
        "goal_xy_mse_threshold": None,
    }
    goal_img = None
    if isinstance(obj, dict):
        if "goal_img" in obj and isinstance(obj["goal_img"], dict) and "data" in obj["goal_img"]:
            gi_meta = obj["goal_img"]
            shape = tuple(int(x) for x in gi_meta.get("shape", []))
            flat = gi_meta.get("data", [])
            try:
                arr = np.array(flat, dtype=np.uint8).reshape(shape)
                goal_img = arr
            except Exception as e:
                print(f"[NET] Failed to reconstruct goal_img from header: {e}")
                goal_img = None
        elif "goal_img" in obj:
            goal_img = obj["goal_img"]

        if "goal_xy" in obj and obj["goal_xy"] is not None:
            try:
                out["goal_xy"] = (float(obj["goal_xy"][0]), float(obj["goal_xy"][1]))
            except Exception:
                pass
        if "start_xy" in obj and obj["start_xy"] is not None:
            try:
                out["start_xy"] = (float(obj["start_xy"][0]), float(obj["start_xy"][1]))
            except Exception:
                pass
        if obj.get("start_yaw_deg") is not None:
            try:
                out["start_yaw_deg"] = float(obj["start_yaw_deg"])
            except Exception:
                pass
        for k in (
            "start_frame_index",
            "goal_frame_index",
            "obs_h",
            "obs_w",
            "obs_c",
            "frame_stack",
            "frames_per_episode",
            "num_episodes",
            "episode_index",
        ):
            if k in obj and obj[k] is not None:
                try:
                    out[k] = int(obj[k])
                except Exception:
                    pass
        if obj.get("goal_xy_mse_threshold") is not None:
            try:
                out["goal_xy_mse_threshold"] = float(obj["goal_xy_mse_threshold"])
            except Exception:
                pass
    elif isinstance(obj, (list, tuple, np.ndarray)) and len(obj) >= 2:
        try:
            out["goal_xy"] = (float(obj[0]), float(obj[1]))
        except Exception:
            pass

    if isinstance(goal_img, np.ndarray):
        if goal_img.dtype != np.uint8:
            g = goal_img.astype(np.float32)
            mx = float(np.nanmax(g)) if g.size else 1.0
            if mx <= 1.0:
                g = np.clip(g * 255.0, 0.0, 255.0)
            goal_img = g.astype(np.uint8)
        print(
            f"[NET] goal_img shape={goal_img.shape}, dtype={goal_img.dtype} "
            f"min/max=({goal_img.min()},{goal_img.max()})"
        )
    else:
        print("[NET] No goal image in header.")

    out["goal_img"] = goal_img
    print(
        f"[NET] goal_xy={out['goal_xy']} start_xy={out['start_xy']} start_yaw_deg={out['start_yaw_deg']} "
        f"obs_hw=({out['obs_h']},{out['obs_w']}) frame_stack={out['frame_stack']} "
        f"frames/ep={out.get('frames_per_episode')} goal_mse_thr={out.get('goal_xy_mse_threshold')}"
    )
    return out


def try_receive_goal_header(sock: socket.socket, timeout: float = 2.0) -> dict:
    """Read one-time header from server.

    Returns a dict with optional keys:
      goal_img (uint8 H,W,3), goal_xy, start_xy, start_yaw_deg,
      start_frame_index, goal_frame_index (ints),
      obs_h, obs_w, obs_c, frame_stack (from server training config).
    """
    print("[NET] Waiting for one-time eval header …")
    sock.settimeout(timeout)

    # Try to peek 4-byte length (if MSG_PEEK exists)
    try:
        hdr = sock.recv(4, socket.MSG_PEEK)
    except (AttributeError, OSError):
        hdr = sock.recv(4)

    if len(hdr) < 4:
        print("[NET] No header available yet.")
        return {}

    # Consume header + payload
    n = int.from_bytes(recvall(sock, 4, timeout=timeout), "big")
    t0 = time.time()
    payload = recvall(sock, n, timeout=timeout)
    dt_ms = (time.time() - t0) * 1000.0
    print(f"[NET] Goal header length = {n} bytes.  [NET] Goal header received in {dt_ms:.1f}ms.")

    obj = pickle.loads(payload)
    return parse_eval_header_obj(obj)


def spawn_ego_at_dataset_xy(world, ego_bp, start_xy, start_yaw_deg=None):
    """Spawn ego at dataset (x,y). Snap Z using CARLA map waypoint; yaw from header or road."""
    carla_map = world.get_map()
    x, y = float(start_xy[0]), float(start_xy[1])
    probe = carla.Location(x=x, y=y, z=500.0)
    wp = carla_map.get_waypoint(
        probe, project_to_road=True, lane_type=carla.LaneType.Driving
    )
    if wp is None:
        print(f"[CARLA] No driving waypoint near ({x:.2f},{y:.2f}); spawn failed.")
        return None, None
    tf = wp.transform
    tf.location.z += 0.35
    if start_yaw_deg is not None:
        tf.rotation = carla.Rotation(pitch=0.0, yaw=float(start_yaw_deg), roll=0.0)
    ego = world.try_spawn_actor(ego_bp, tf)
    return ego, tf


def wandb_trajectory_map_image(
    world: carla.World,
    xs: list[float],
    ys: list[float],
    goal_xy,
    start_xy=None,
) -> np.ndarray | None:
    """Matplotlib map overlay + trajectory; returns RGB uint8 HxWx3, or None if no points.

    If ``start_xy`` is set (dataset spawn target), draws a green square so you can compare
    to the first trajectory point (ego after tick), e.g. when CARLA snaps to a waypoint.
    """
    pts = [(float(x), float(y)) for x, y in zip(xs, ys) if np.isfinite(x) and np.isfinite(y)]
    if not pts:
        return None
    rows = [[x, y, "traj"] for x, y in pts]
    if goal_xy is not None and np.all(np.isfinite(goal_xy)):
        rows.append([float(goal_xy[0]), float(goal_xy[1]), "goal"])
    try:
        carla_map = world.get_map()
        waypoints = carla_map.generate_waypoints(distance=2.0)
        rows.extend(
            [[float(wp.transform.location.x), float(wp.transform.location.y), "map"] for wp in waypoints]
        )
    except Exception as e:
        print(f"[W&B] [WARN] map waypoints: {e}")

    map_pts = [(x, y) for x, y, s in rows if s == "map"]
    traj_pts = [(x, y) for x, y, s in rows if s == "traj"]
    goal_pts = [(x, y) for x, y, s in rows if s == "goal"]

    fig = Figure(figsize=(6, 6), dpi=120)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)

    if map_pts:
        mx, my = zip(*map_pts)
        ax.scatter(mx, my, s=2, c="#cccccc", alpha=0.3, label="map")
    if traj_pts:
        tx, ty = zip(*traj_pts)
        ax.plot(tx, ty, c="#4e79a7", linewidth=0.5, alpha=0.8, label="traj", zorder=2)
        if len(traj_pts) > 1:
            arrow_interval = max(1, len(traj_pts) // 10)
            for j in range(0, len(traj_pts) - 1, arrow_interval):
                x1, y1 = traj_pts[j]
                x2, y2 = traj_pts[j + 1]
                dx = x2 - x1
                dy = y2 - y1
                if abs(dx) > 0.01 or abs(dy) > 0.01:
                    ax.annotate(
                        "",
                        xy=(x2, y2),
                        xytext=(x1, y1),
                        arrowprops=dict(
                            arrowstyle="->",
                            lw=2.5,
                            color="#4e79a7",
                            alpha=0.9,
                            mutation_scale=25,
                        ),
                        zorder=3,
                    )
    if goal_pts:
        gx, gy = zip(*goal_pts)
        ax.scatter(gx, gy, marker="*", s=220, c="#f28e2b", label="goal", zorder=4)
    if start_xy is not None and np.all(np.isfinite(start_xy)):
        ax.scatter(
            [float(start_xy[0])],
            [float(start_xy[1])],
            s=100,
            c="#2ca02c",
            marker="s",
            label="dataset_start_xy",
            zorder=5,
        )

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best")
    fig.tight_layout()
    canvas.draw()
    buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    wh = fig.canvas.get_width_height()[::-1]
    return buf.reshape(wh + (3,))


def wandb_log_episode(
    *,
    ep_idx: int,
    hdr: dict,
    reason: str,
    frames_used: int,
    success_rate: float,
    xs: list[float],
    ys: list[float],
    goal_xy,
    world: carla.World,
    log_trajectory_map: bool,
    map_img: np.ndarray | None = None,
    collided: bool = False,
    collision_rate: float | None = None,
    category: str | None = None,
    cat_success_rate: float | None = None,
    cat_collision_rate: float | None = None,
) -> None:
    """Per-episode metrics; optional trajectory map. Combined video is logged once at end of run."""
    if wandb is None:
        return
    sfi = hdr.get("start_frame_index")
    gfi = hdr.get("goal_frame_index")
    cat_suffix = f" cat={category}" if category else ""
    cap = f"reason={reason} frames={frames_used} start_f={sfi} goal_f={gfi}{cat_suffix}"
    to_log: dict = {}
    try:
        if log_trajectory_map:
            if map_img is None:
                map_img = wandb_trajectory_map_image(
                    world, xs, ys, goal_xy, start_xy=hdr.get("start_xy")
                )
            if map_img is not None:
                to_log[f"eval/episode_{ep_idx}/trajectory_map"] = wandb.Image(map_img, caption=cap)
        # Single series vs episode step (not per-episode keys, so one chart in W&B).
        to_log["eval/frames_used"] = int(frames_used)
        to_log["eval/success_rate"] = float(success_rate)
        to_log["eval/collided"] = int(bool(collided))
        if collision_rate is not None:
            to_log["eval/collision_rate"] = float(collision_rate)
        # Per-category series: each category draws its own line in W&B because
        # the metric key includes the category name. Only log if we know the
        # category and have a rate to report (avoids spurious NaNs otherwise).
        if category:
            to_log[f"eval/by_category/{category}/collided"] = int(bool(collided))
            if cat_success_rate is not None:
                to_log[f"eval/by_category/{category}/success_rate"] = float(cat_success_rate)
            if cat_collision_rate is not None:
                to_log[f"eval/by_category/{category}/collision_rate"] = float(cat_collision_rate)
        if to_log:
            wandb.log(to_log, step=ep_idx)
    except Exception as e:
        print(f"[W&B] [WARN] episode {ep_idx} log failed: {e}")


# ================= Main =================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--carla-host", type=str, default="localhost")
    ap.add_argument("--carla-port", type=int, default=2000)
    ap.add_argument("--server-host", type=str, default="localhost")
    ap.add_argument("--server-port", type=int, default=5050)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument(
        "--algorithm",
        type=str,
        default="gcbc",
        help="Training/eval algorithm name; default W&B project is '<algorithm>-eval' (e.g. gcbc -> gcbc-eval).",
    )
    ap.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="W&B project (default: <algorithm>-eval from --algorithm).",
    )
    ap.add_argument("--wandb-run", type=str, default="sync_run")
    ap.add_argument(
        "--wandb-map-every",
        type=int,
        default=10,
        help="W&B only: log trajectory map every N episodes by episode_index (10=~10%% of maps; 1=all; 0=off). Plots-dir always saves every episode.",
    )
    ap.add_argument(
        "--wandb-video-every",
        type=int,
        default=10,
        help="W&B only: use every Nth simulator frame in the combined camera video (1=all frames). Plots-dir unchanged.",
    )
    ap.add_argument(
        "--plots-dir",
        type=str,
        default=None,
        help="Directory for trajectory-map PNGs (default: plots/<algorithm>_<YYYYMMDD>_<HHMMSS>). Ignored with --no-plot-files.",
    )
    ap.add_argument(
        "--no-plot-files",
        action="store_true",
        help="Do not write trajectory map PNGs to disk.",
    )
    ap.add_argument("--print-every", type=int, default=20,
                    help="Print log line every N POLICY steps (one policy step = action_repeat sim ticks).")
    ap.add_argument(
        "--no-collision-end",
        action="store_true",
        help="Do NOT end the episode on collision (still log them). Default: end episode on first accepted collision.",
    )
    ap.add_argument(
        "--collision-ignore-frames",
        type=int,
        default=5,
        help="Ignore collision events in the first N simulator frames after spawn (default 5). "
             "CARLA often emits a spurious 'static.road' hit when the ego settles after spawn.",
    )
    ap.add_argument(
        "--train-fps",
        type=int,
        default=10,
        help="FPS of the training dataset. Used (with --sim-fps) to compute --action-repeat "
             "if the latter is not given. Our CARLA dumps were recorded at 10 Hz, so default=10.",
    )
    ap.add_argument(
        "--sim-fps",
        type=int,
        default=60,
        help="CARLA simulator FPS in synchronous mode (fixed_delta_seconds = 1/sim_fps). "
             "Higher = smoother physics/video; the policy still runs at sim_fps/action_repeat Hz.",
    )
    ap.add_argument(
        "--action-repeat",
        type=int,
        default=1,
        help="Hold each predicted action for N consecutive sim ticks before consuming the next "
             "chunk slot. Default 1: policy runs once per sim tick (no skipping). Train-time "
             "frame_stack_window already exposes the model to longer / variable-spaced histories, "
             "so eval no longer needs to throttle to dataset cadence. Pass e.g. "
             "round(sim_fps / train_fps) (=6 for 60/10) to recover the old behavior.",
    )
    # ---- Stuck-recovery overrides (policy sometimes just stops; nudge it forward) ----
    ap.add_argument(
        "--no-unstick",
        action="store_true",
        help="Disable stuck-recovery override. Default: enabled (when stationary too long, "
             "apply a fixed throttle with zero steer/brake until the ego starts moving again).",
    )
    ap.add_argument(
        "--unstick-after-sec",
        type=float,
        default=1.5,
        help="Engage unstick recovery after the ego has been stationary for this many wall-clock "
             "seconds (measured via the existing stuck counter at sim_fps).",
    )
    ap.add_argument(
        "--unstick-warmup-sec",
        type=float,
        default=0.5,
        help="Don't engage unstick recovery in the first N seconds of an episode (gives the "
             "policy a chance to start driving naturally from spawn).",
    )
    ap.add_argument(
        "--unstick-throttle",
        type=float,
        default=0.4,
        help="Throttle value applied while unstick recovery is active (steer=0, brake=0).",
    )
    ap.add_argument(
        "--unstick-release-speed",
        type=float,
        default=0.6,
        help="Disengage unstick once the ego's speed (m/s) exceeds this threshold; policy resumes.",
    )
    args = ap.parse_args()
    if args.wandb_project is None:
        alg = (args.algorithm or "gcbc").strip().lower()
        args.wandb_project = f"{alg}-eval" if alg else "carla_eval"
    print(f"[W&B] project={args.wandb_project} (algorithm={args.algorithm})")

    plots_path: Path | None = None
    if not args.no_plot_files:
        if args.plots_dir is None:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            alg_safe = (args.algorithm or "gcbc").strip().lower().replace(os.sep, "_").replace("/", "_")
            plots_path = (Path("plots") / f"{alg_safe}_{ts}").resolve()
        else:
            plots_path = Path(args.plots_dir).expanduser().resolve()
        plots_path.mkdir(parents=True, exist_ok=True)
        print(f"[PLOTS] trajectory PNGs -> {plots_path}")

    # ---- CARLA ----
    print(f"[CARLA] Connecting to {args.carla_host}:{args.carla_port} …")
    client = carla.Client(args.carla_host, args.carla_port)
    client.set_timeout(120.0)
    world = client.get_world()
    original_settings = world.get_settings()
    bp = world.get_blueprint_library()

    clear_dynamic_actors(world)

    ego_bp = bp.find("vehicle.tesla.model3")
    if ego_bp.has_attribute("role_name"):
        ego_bp.set_attribute("role_name", "hero")

    # ---- Server: connect + first header (start_xy / goal from same .npz as training) ----
    sock = connect_eval(args.server_host, args.server_port, timeout=10.0)
    hdr = try_receive_goal_header(sock, timeout=30.0)
    use_eval_protocol = hdr.get("frames_per_episode") is not None

    # ---- W&B ----
    wb_active = False
    if wandb is not None:
        run = wandb.init(project=args.wandb_project, name=args.wandb_run)
        wb_active = True
        try:
            print(f"[W&B] run: {run.url}")
        except Exception:
            pass
    else:
        print("[W&B] wandb not installed; running without logging.")

    goal_xy = None
    video_frames_all: list[np.ndarray] = []
    successes: list[int] = []
    # Per-category bookkeeping so W&B shows separate success rates for
    # same_traj_easy / same_traj_hard / random_goal. "unknown" catches pairs
    # that came from an old server without a category field (legacy configs).
    cat_successes: dict[str, list[int]] = {
        "same_traj_easy": [],
        "same_traj_hard": [],
        "random_goal": [],
        "unknown": [],
    }
    cat_collisions: dict[str, int] = {k: 0 for k in cat_successes}

    vel_thresh, pos_thresh = 0.15, 0.10
    fps_out = 50
    episode_count = 0

    # ------------------------------------------------------------------
    # Action-repeat setup. With train-time frame_stack_window > frame_stack the
    # policy already saw randomized longer histories, so we no longer throttle
    # eval to dataset cadence by default (action_repeat=1 -> one policy decision
    # per sim tick at args.sim_fps). Pass --action-repeat N to restore the old
    # "hold each action for N ticks" behavior; we just print the effective rate.
    # ------------------------------------------------------------------
    action_repeat = max(1, int(args.action_repeat))
    eff_policy_hz = float(args.sim_fps) / float(action_repeat)
    print(
        f"[CTRL] action_repeat={action_repeat} (sim_fps={args.sim_fps}, train_fps="
        f"{args.train_fps}); effective policy rate ≈ {eff_policy_hz:.2f} Hz."
    )

    # Stuck-recovery: convert wall-clock seconds to sim-tick counts at sim_fps.
    # `stuck_counter` already increments once per sim tick while ego is ~still.
    unstick_enabled = (not args.no_unstick)
    unstick_after_ticks = int(round(max(0.0, args.unstick_after_sec) * float(args.sim_fps)))
    unstick_warmup_ticks = int(round(max(0.0, args.unstick_warmup_sec) * float(args.sim_fps)))
    if unstick_enabled:
        print(
            f"[STUCK] Unstick recovery ON: after_ticks={unstick_after_ticks} "
            f"(={args.unstick_after_sec}s), warmup_ticks={unstick_warmup_ticks} "
            f"(={args.unstick_warmup_sec}s), throttle={args.unstick_throttle}, "
            f"release_speed={args.unstick_release_speed} m/s."
        )
    else:
        print("[STUCK] Unstick recovery OFF (--no-unstick).")

    cam_bp = bp.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", "256")
    cam_bp.set_attribute("image_size_y", "256")
    cam_bp.set_attribute("fov", "70")
    cam_bp.set_attribute("sensor_tick", "0.0")

    col_bp = bp.find("sensor.other.collision")
    collision_episodes = 0  # count for W&B summary

    # Synchronous mode before any spawn so every episode's ego pose is valid.
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / float(args.sim_fps)
    world.apply_settings(settings)
    fps_out = int(round(1.0 / settings.fixed_delta_seconds))
    print(f"[CARLA] Synchronous mode fps≈{fps_out} "
          f"(policy decisions every {action_repeat} ticks ≈ {fps_out / action_repeat:.2f} Hz).")
    set_weather(world, quiet=False)

    try:
        while True:
            # Re-apply each episode: some maps / resets can leave night lighting otherwise.
            set_weather(world, quiet=True)
            goal_xy = hdr.get("goal_xy")
            start_xy = hdr.get("start_xy")
            start_yaw_deg = hdr.get("start_yaw_deg")
            send_obs_w = hdr.get("obs_w")
            send_obs_h = hdr.get("obs_h")
            if send_obs_w is None or send_obs_h is None:
                send_obs_w, send_obs_h = 100, 100
                print(f"[NET] Header missing obs_h/obs_w; using {send_obs_w}x{send_obs_h}.")
            else:
                print(f"[NET] Server expects camera frames at {send_obs_w}x{send_obs_h}.")

            # Frame stacking on the client is what keeps eval temporally
            # consistent with training: training stacked K frames at
            # train_fps cadence (one per policy step), so on eval we maintain
            # a deque of single-frame obs that advances ONCE per policy step
            # (i.e. once per action_repeat sim ticks), regardless of action
            # chunk size. The server bypasses its own HISTORY when it sees
            # an already-stacked channel dim.
            client_frame_stack_k = max(1, int(hdr.get("frame_stack") or 1))
            client_obs_c = max(1, int(hdr.get("obs_c") or 3))
            hdr_frame_offsets = hdr.get("frame_offsets")
            if hdr_frame_offsets is not None:
                try:
                    client_frame_offsets = tuple(int(x) for x in hdr_frame_offsets)
                except Exception:
                    client_frame_offsets = None
            else:
                client_frame_offsets = None
            if client_frame_offsets is None or len(client_frame_offsets) != client_frame_stack_k:
                # Fallback: canonical consecutive offsets (oldest -> current).
                client_frame_offsets = tuple(range(-(client_frame_stack_k - 1), 1))
            client_frame_offsets = tuple(sorted(client_frame_offsets))
            if 0 not in client_frame_offsets:
                client_frame_offsets = tuple(list(client_frame_offsets[:-1]) + [0])
                client_frame_offsets = tuple(sorted(client_frame_offsets))

            # --- FPS-aware stacking ---
            # Dataset was collected at 10 Hz. Each unit in `frame_offsets` is a
            # 10 Hz step, but eval history advances once per sim tick at
            # `sim_fps`, so scale by `sim_fps / 10` to preserve physical spacing
            # without holding actions longer. Example: (0, -1) with sim_fps=60
            # → (0, -6) sim ticks back.
            k = max(1, int(round(float(args.sim_fps) / 10.0)))
            k = 1
            client_frame_offsets_sim = tuple(int(o) * k for o in client_frame_offsets)

            max_lag = max(0, max((-o for o in client_frame_offsets_sim if o < 0), default=0))
            history_maxlen = int(max_lag + 1)
            print(
                f"[NET] Client-side frame stack: K={client_frame_stack_k}, obs_c={client_obs_c} "
                f"→ stacked channel dim={client_frame_stack_k * client_obs_c} "
                f"(advances once per policy step); frame_offsets(sim)={client_frame_offsets_sim} "
                f"(history_maxlen={history_maxlen})."
            )

            frames_cap = int(hdr.get("frames_per_episode") or args.frames)
            mse_thr = hdr.get("goal_xy_mse_threshold")

            ego, chosen_tf = None, None
            if start_xy is not None:
                ego, chosen_tf = spawn_ego_at_dataset_xy(world, ego_bp, start_xy, start_yaw_deg)
                if ego is None:
                    print("[CARLA] Dataset spawn failed; falling back to random spawn.")
            if ego is None:
                spawns = world.get_map().get_spawn_points()
                if not spawns:
                    raise RuntimeError("No spawn points available")
                for tf in random.sample(spawns, k=min(20, len(spawns))):
                    ego = world.try_spawn_actor(ego_bp, tf)
                    if ego is not None:
                        chosen_tf = tf
                        break
            if not ego:
                raise RuntimeError("Failed to spawn ego vehicle (tesla.model3)")
            print(
                f"[CARLA] ✓ Ego spawned at "
                f"({chosen_tf.location.x:.2f},{chosen_tf.location.y:.2f}) yaw={chosen_tf.rotation.yaw:.4f}"
            )

            cam_tf = carla.Transform(carla.Location(x=1.5, z=2.4))
            camera = world.spawn_actor(
                cam_bp, cam_tf, attach_to=ego,
                attachment_type=carla.AttachmentType.Rigid
            )

            q = Queue(maxsize=1)
            camera.listen(lambda im: _push_latest(q, im))

            # Collision sensor — fires on any collision the ego participates in.
            # Warm-up window skips the 'static.road' hit that sometimes fires as the
            # ego settles from the +0.35 m spawn drop.
            spawn_frame = int(world.get_snapshot().frame)
            collision_state: dict = {
                'collided': False,
                'other': None,
                'impulse': 0.0,
                'frame': 0,
                'ignore_until_frame': spawn_frame + int(max(0, args.collision_ignore_frames)),
            }
            collision_sensor = world.spawn_actor(
                col_bp, carla.Transform(), attach_to=ego,
                attachment_type=carla.AttachmentType.Rigid,
            )
            collision_sensor.listen(_make_collision_callback(collision_state))

            # One tick so transforms are committed before we read pose (otherwise
            # get_location() can be stale / identical across respawns). Keep the
            # frame: we use it as the first observation sent to the server.
            world.tick()
            last_rgb = None
            try:
                im_init = q.get(timeout=2.0)
                last_rgb = rgb_from_image(im_init)
            except Empty:
                print("[CARLA] [WARN] No camera frame within 2 s of spawn; first obs will be black.")
            if last_rgb is None:
                last_rgb = np.zeros((256, 256, 3), np.uint8)

            start_loc = ego.get_location()
            xs: list[float] = [float(start_loc.x)]
            ys: list[float] = [float(start_loc.y)]
            stuck_counter = 0
            prev_loc = start_loc
            current_action_chunk = None
            chunk_index = 0
            video_frames: list[np.ndarray] = []
            # Unstick state, reset per-episode. `unstick_active` is latched so the
            # override persists across policy steps until we actually pick up speed.
            unstick_active = False
            unstick_activations = 0
            unstick_override_ticks = 0

            # Pre-fill the policy-step frame-stack history with the spawn frame so
            # the very first observation sent looks like (frame_t0, frame_t0, ..)
            # instead of zero-padded.
            # History stores single frames at policy-step cadence. We keep a
            # longer buffer when offsets request older frames (e.g. -40).
            obs_history: deque = deque(maxlen=history_maxlen)
            init_resized = cv2.resize(
                last_rgb, (int(send_obs_w), int(send_obs_h)), interpolation=cv2.INTER_AREA
            ).astype(np.uint8)
            for _ in range(history_maxlen):
                obs_history.append(init_resized)

            # frames_cap is interpreted as POLICY STEPS (training-cadence steps).
            # Total simulator ticks per episode ≤ frames_cap * action_repeat.
            print(
                f"[RUN] Episode {episode_count + 1} (server ep {hdr.get('episode_index')}) "
                f"max {frames_cap} policy steps × action_repeat={action_repeat} = "
                f"{frames_cap * action_repeat} sim ticks "
                f"(≈ {frames_cap * action_repeat / max(1, fps_out):.1f} s); "
                f"goal_mse_thr={mse_thr}",
                flush=True,
            )

            reason = "timeout"
            frames_used = 0  # accumulator: total sim ticks consumed
            for policy_step in range(frames_cap):
                # Capture the current single-frame obs and push it into the
                # policy-step frame-stack history. This advances ONCE per policy
                # step regardless of action_repeat or chunk length, matching the
                # 1-frame-per-training-step cadence used to build the dataset.
                img_resized = cv2.resize(
                    last_rgb, (int(send_obs_w), int(send_obs_h)), interpolation=cv2.INTER_AREA
                ).astype(np.uint8)
                obs_history.append(img_resized)

                if client_frame_stack_k > 1:
                    # Build explicit-offset stack oldest-first. Clamp to the
                    # oldest available history element when we're early in the
                    # episode (trajectory-start clamp analog).
                    hist_list = list(obs_history)
                    n_hist = len(hist_list)
                    parts = []
                    for off in client_frame_offsets_sim:
                        idx = (n_hist - 1) + int(off)
                        idx = 0 if idx < 0 else idx
                        parts.append(hist_list[idx])
                    img_send = np.concatenate(parts, axis=-1)
                else:
                    img_send = img_resized

                if current_action_chunk is None or chunk_index >= len(current_action_chunk):
                    payload = pickle.dumps(img_send, protocol=pickle.HIGHEST_PROTOCOL)
                    try:
                        send_with_len(sock, payload)
                        chunk_length_bytes = recvall(sock, 4, timeout=10.0)
                        chunk_length = int.from_bytes(chunk_length_bytes, "big")
                        action_chunk_size = chunk_length * 3 * 4
                        action_chunk_flat = np.frombuffer(
                            recvall(sock, action_chunk_size, timeout=10.0), dtype=np.float32
                        )
                        current_action_chunk = action_chunk_flat.reshape(chunk_length, 3)
                        chunk_index = 0
                        print(f"[NET] Received new action chunk: shape={current_action_chunk.shape}")
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, socket.timeout) as e:
                        print(f"[NET] IO failed at policy step {policy_step}: {e} → reconnecting…")
                        try:
                            sock.close()
                        except Exception:
                            pass
                        sock = connect_eval(args.server_host, args.server_port, timeout=10.0)
                        send_with_len(sock, payload)
                        chunk_length_bytes = recvall(sock, 4, timeout=10.0)
                        chunk_length = int.from_bytes(chunk_length_bytes, "big")
                        action_chunk_size = chunk_length * 3 * 4
                        action_chunk_flat = np.frombuffer(
                            recvall(sock, action_chunk_size, timeout=10.0), dtype=np.float32
                        )
                        current_action_chunk = action_chunk_flat.reshape(chunk_length, 3)
                        chunk_index = 0

                action = current_action_chunk[chunk_index]
                chunk_index += 1

                thr_raw = float(np.clip(action[0], 0.0, 1.0))
                steer = float(np.clip(action[1], -1.0, 1.0))
                brk_raw = float(np.clip(action[2], 0.0, 1.0))
                thr, brk = thr_raw, brk_raw
                #if thr_raw >= brk_raw:
                #    thr, brk = thr_raw, 0.0
                #else:
                #    thr, brk = 0.0, brk_raw

                # Stuck-recovery override. Engage (and latch) if the ego has been
                # stationary for >= unstick_after_ticks past the warmup window.
                # Released inside the sub-tick loop below once vnorm clears
                # --unstick-release-speed (or at episode end). This preempts the
                # policy's brake/zero-throttle outputs, which is the whole point.
                if (
                    unstick_enabled
                    and not unstick_active
                    and frames_used >= unstick_warmup_ticks
                    and stuck_counter >= unstick_after_ticks
                ):
                    unstick_active = True
                    unstick_activations += 1
                    print(
                        f"[STUCK] Engaging unstick recovery at step {policy_step+1} "
                        f"(tick {frames_used}, stuck_counter={stuck_counter}): "
                        f"throttle={args.unstick_throttle}, steer=0, brake=0."
                    )
                if unstick_active:
                    thr = float(args.unstick_throttle)
                    steer = 0.0
                    brk = 0.0

                ctl = carla.VehicleControl(
                    throttle=thr, steer=steer, brake=brk, hand_brake=False
                )
                ego.apply_control(ctl)

                # Hold this control for `action_repeat` simulator ticks. Each
                # sub-tick advances physics by 1/sim_fps s, refreshes the camera
                # (so video / pose tracking stay smooth), and is the granularity
                # at which we check collision / goal / stuck so we can break out
                # mid-hold without overshooting.
                early_break_reason: str | None = None
                for sub in range(action_repeat):
                    im_sub = tick_and_grab(world, q, timeout=3.0)
                    if im_sub is not None:
                        last_rgb = rgb_from_image(im_sub)
                    sub_resized = cv2.resize(
                        last_rgb, (int(send_obs_w), int(send_obs_h)), interpolation=cv2.INTER_AREA
                    )
                    video_frames.append(sub_resized)
                    frames_used += 1

                    loc = ego.get_location()
                    xs.append(float(loc.x))
                    ys.append(float(loc.y))
                    vel = ego.get_velocity()
                    vnorm = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
                    dloc = math.sqrt((loc.x - prev_loc.x) ** 2 + (loc.y - prev_loc.y) ** 2)
                    prev_loc = loc
                    stuck_counter = (
                        stuck_counter + 1 if (vnorm < vel_thresh or dloc < pos_thresh) else 0
                    )

                    if unstick_active:
                        unstick_override_ticks += 1
                        if vnorm >= float(args.unstick_release_speed):
                            print(
                                f"[STUCK] Released unstick recovery (vnorm={vnorm:.2f} m/s "
                                f"≥ {args.unstick_release_speed}). Handing control back to policy."
                            )
                            unstick_active = False

                    if mse_thr is not None and goal_xy is not None:
                        gx, gy = float(goal_xy[0]), float(goal_xy[1])
                        mse_xy = 0.5 * ((loc.x - gx) ** 2 + (loc.y - gy) ** 2)
                        if mse_xy <= float(mse_thr):
                            early_break_reason = "goal"
                            print(f"[RUN] Goal reached mse_xy={mse_xy:.6f} <= {mse_thr}")
                            break

                    if collision_state.get('collided') and not args.no_collision_end:
                        early_break_reason = "collision"
                        print(
                            f"[RUN] Collision with {collision_state.get('other')} "
                            f"(impulse {collision_state.get('impulse'):.1f} N·s at frame "
                            f"{collision_state.get('frame')}); ending episode."
                        )
                        break

                    if stuck_counter >= 500 and frames_used > 500:
                        early_break_reason = "stuck"
                        print("[RUN] Stuck too long; ending episode.")
                        break

                if (policy_step + 1) % args.print_every == 0:
                    print(
                        f"[step {policy_step+1}/{frames_cap} | tick {frames_used}] "
                        f"thr={thr:.3f} steer={steer:.3f} brake={brk:.3f} "
                        f"loc=({loc.x:.2f},{loc.y:.2f})",
                        flush=True,
                    )

                if early_break_reason is not None:
                    reason = early_break_reason
                    break

            # Collisions still counted even with --no-collision-end.
            if collision_state.get('collided'):
                collision_episodes += 1

            if unstick_enabled and unstick_activations > 0:
                print(
                    f"[STUCK] Episode summary: unstick engaged {unstick_activations}x, "
                    f"override active for {unstick_override_ticks} sim ticks "
                    f"(~{unstick_override_ticks / max(1, fps_out):.1f} s)."
                )

            successes.append(1 if reason == "goal" else 0)
            success_rate = float(np.mean(successes))
            video_frames_all.extend(video_frames)

            # Per-category bookkeeping. Server tags each header with a category
            # in {same_traj_easy, same_traj_hard, random_goal}; legacy servers
            # send no tag and we fall back to "unknown".
            category = str(hdr.get("category") or "unknown")
            if category not in cat_successes:
                # Defensive: accept new categories gracefully without crashing.
                cat_successes[category] = []
                cat_collisions[category] = 0
            cat_successes[category].append(1 if reason == "goal" else 0)
            if collision_state.get('collided'):
                cat_collisions[category] += 1
            cat_n = len(cat_successes[category])
            cat_success_rate = float(np.mean(cat_successes[category])) if cat_n else None
            cat_collision_rate = float(cat_collisions[category] / cat_n) if cat_n else None

            ep_idx = int(hdr.get("episode_index") if hdr.get("episode_index") is not None else episode_count)
            log_traj_map = args.wandb_map_every > 0 and (ep_idx % args.wandb_map_every == 0)
            need_map_disk = plots_path is not None
            need_map_wandb = wb_active and log_traj_map
            map_img = None
            if need_map_disk or need_map_wandb:
                map_img = wandb_trajectory_map_image(
                    world, xs, ys, goal_xy, start_xy=hdr.get("start_xy")
                )
            if map_img is not None and need_map_disk and plots_path is not None:
                try:
                    out_png = plots_path / f"episode_{ep_idx:04d}.png"
                    cv2.imwrite(str(out_png), cv2.cvtColor(map_img, cv2.COLOR_RGB2BGR))
                except Exception as e:
                    print(f"[PLOTS] [WARN] failed to write {out_png}: {e}")

            if wb_active:
                # episode_count was bumped to +1 for this finished episode further down;
                # compute the collision rate including this episode.
                cur_eps = max(1, episode_count + 1)
                cur_collision_rate = float(collision_episodes / cur_eps)
                wandb_log_episode(
                    ep_idx=ep_idx,
                    hdr=hdr,
                    reason=reason,
                    frames_used=frames_used,
                    success_rate=success_rate,
                    xs=xs,
                    ys=ys,
                    goal_xy=goal_xy,
                    world=world,
                    log_trajectory_map=log_traj_map,
                    map_img=map_img if log_traj_map else None,
                    collided=bool(collision_state.get('collided')),
                    collision_rate=cur_collision_rate,
                    category=category,
                    cat_success_rate=cat_success_rate,
                    cat_collision_rate=cat_collision_rate,
                )

            try:
                camera.stop()
            except Exception:
                pass
            try:
                collision_sensor.stop()
            except Exception:
                pass
            try:
                collision_sensor.destroy()
            except Exception:
                pass
            try:
                ego.destroy()
            except Exception:
                pass

            episode_count += 1

            if use_eval_protocol:
                send_episode_end(sock, reason, frames_used)
                reply = recv_one_pickled(sock, timeout=30.0)
                if isinstance(reply, dict) and reply.get("type") == "eval_done":
                    print("[NET] eval_done — all episodes finished.")
                    break
                hdr = parse_eval_header_obj(reply)
                recv_action_chunk(sock)
            else:
                break

    finally:
        try:
            world.apply_settings(original_settings)
        except Exception:
            pass
        try:
            sock.close()
        except Exception:
            pass

    if wb_active:
        if not args.no_video and video_frames_all:
            try:
                vstep = max(1, int(args.wandb_video_every))
                frames_wb = video_frames_all[::vstep]
                if not frames_wb:
                    frames_wb = video_frames_all[-1:]
                vid = np.stack(frames_wb, axis=0).astype(np.uint8)
                vid = np.moveaxis(vid, -1, 1)
                # Keep playback duration ~ wall time: fewer frames → lower fps by same factor.
                eff_fps = max(1.0, float(fps_out) / float(vstep))
                n_eps = len(successes)
                wandb.log(
                    {"eval/camera_video": wandb.Video(vid, fps=eff_fps, format="mp4")},
                    step=n_eps,
                )
            except Exception as e:
                print(f"[W&B] [WARN] combined camera video log failed: {e}")
        if successes:
            try:
                wandb.summary["eval/success_rate"] = float(np.mean(successes))
                wandb.summary["eval/collision_episodes"] = int(collision_episodes)
                wandb.summary["eval/collision_rate"] = float(collision_episodes / max(1, len(successes)))
                # Per-category summary: headline rate + counts per bucket. Drop
                # empty buckets so the summary stays clean for runs that don't
                # exercise every category.
                for cat, lst in cat_successes.items():
                    if not lst:
                        continue
                    n = len(lst)
                    wandb.summary[f"eval/by_category/{cat}/success_rate"] = float(np.mean(lst))
                    wandb.summary[f"eval/by_category/{cat}/n"] = int(n)
                    wandb.summary[f"eval/by_category/{cat}/collision_rate"] = float(
                        cat_collisions.get(cat, 0) / n
                    )
            except Exception:
                pass
        wandb.finish()
        print("[W&B] ✓ Logged to Weights & Biases.")

    # Always print a final breakdown, even without W&B, so the run log
    # carries the per-category numbers.
    if successes:
        print("[EVAL] per-category final:")
        for cat, lst in cat_successes.items():
            if not lst:
                continue
            n = len(lst)
            print(
                f"  {cat}: n={n} success_rate={float(np.mean(lst)):.3f} "
                f"collisions={cat_collisions.get(cat, 0)}"
            )

    print("[DONE] ◠‿◠")


if __name__ == "__main__":
    # For slow networks you can run offline and sync later:
    #   WANDB_MODE=offline python client.py
    #   wandb sync wandb/offline-run-*/
    main()

