#!/usr/bin/env python3
from __future__ import annotations

import os, sys, math, time, random, socket, pickle, argparse
import datetime
from pathlib import Path
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

def _push_latest(q: Queue, image):
    """Non-blocking camera callback: keep only the newest frame."""
    try:
        q.put_nowait(image)
    except Full:
        try: q.get_nowait()
        except Empty: pass
        try: q.put_nowait(image)
        except Full: pass

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
) -> None:
    """Per-episode metrics; optional trajectory map. Combined video is logged once at end of run."""
    if wandb is None:
        return
    sfi = hdr.get("start_frame_index")
    gfi = hdr.get("goal_frame_index")
    cap = f"reason={reason} frames={frames_used} start_f={sfi} goal_f={gfi}"
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
    ap.add_argument("--print-every", type=int, default=20)
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

    vel_thresh, pos_thresh = 0.15, 0.10
    fps_out = 50
    episode_count = 0

    cam_bp = bp.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", "256")
    cam_bp.set_attribute("image_size_y", "256")
    cam_bp.set_attribute("fov", "70")
    cam_bp.set_attribute("sensor_tick", "0.0")

    # Synchronous mode before any spawn so every episode's ego pose is valid.
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / 50.0
    world.apply_settings(settings)
    fps_out = int(round(1.0 / settings.fixed_delta_seconds))
    print(f"[CARLA] Synchronous mode fps≈{fps_out}.")

    try:
        while True:
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

            # One tick so transforms are committed before we read pose (otherwise
            # get_location() can be stale / identical across respawns).
            world.tick()
            try:
                q.get(timeout=2.0)
            except Empty:
                pass
            start_loc = ego.get_location()
            xs: list[float] = [float(start_loc.x)]
            ys: list[float] = [float(start_loc.y)]
            last_rgb = None
            stuck_counter = 0
            prev_loc = start_loc
            current_action_chunk = None
            chunk_index = 0
            video_frames: list[np.ndarray] = []

            print(
                f"[RUN] Episode {episode_count + 1} (server ep {hdr.get('episode_index')}) "
                f"max {frames_cap} frames; goal_mse_thr={mse_thr}",
                flush=True,
            )

            reason = "timeout"
            frames_used = 0
            for i in range(frames_cap):
                t0 = time.time()
                im = tick_and_grab(world, q, timeout=3.0)
                t_cam = time.time()

                if im is None:
                    rgb = last_rgb if last_rgb is not None else np.zeros((256, 256, 3), np.uint8)
                else:
                    rgb = rgb_from_image(im)
                    last_rgb = rgb

                img_resized = cv2.resize(
                    rgb, (int(send_obs_w), int(send_obs_h)), interpolation=cv2.INTER_AREA
                )
                video_frames.append(img_resized)
                img_send = img_resized.astype(np.uint8)

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
                        print(f"[NET] IO failed at step {i}: {e} → reconnecting…")
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
                if thr_raw >= brk_raw:
                    thr, brk = thr_raw, 0.0
                else:
                    thr, brk = 0.0, brk_raw

                ego.apply_control(carla.VehicleControl(throttle=thr, steer=steer, brake=brk, hand_brake=False))
                t_step = time.time()
                frames_used = i + 1

                loc = ego.get_location()
                xs.append(float(loc.x))
                ys.append(float(loc.y))
                vel = ego.get_velocity()
                vnorm = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
                dloc = math.sqrt((loc.x - prev_loc.x) ** 2 + (loc.y - prev_loc.y) ** 2)
                prev_loc = loc
                stuck_counter = stuck_counter + 1 if (vnorm < vel_thresh or dloc < pos_thresh) else 0

                if (i + 1) % args.print_every == 0:
                    print(
                        f"[{i+1}/{frames_cap}] thr={thr:.3f} steer={steer:.3f} brake={brk:.3f} "
                        f"loc=({loc.x:.2f},{loc.y:.2f})",
                        flush=True,
                    )

                if mse_thr is not None and goal_xy is not None:
                    gx, gy = float(goal_xy[0]), float(goal_xy[1])
                    mse_xy = 0.5 * ((loc.x - gx) ** 2 + (loc.y - gy) ** 2)
                    if mse_xy <= float(mse_thr):
                        reason = "goal"
                        print(f"[RUN] Goal reached mse_xy={mse_xy:.6f} <= {mse_thr}")
                        break

                if stuck_counter >= 500 and i > 500:
                    reason = "stuck"
                    print("[RUN] Stuck too long; ending episode.")
                    break

            successes.append(1 if reason == "goal" else 0)
            success_rate = float(np.mean(successes))
            video_frames_all.extend(video_frames)

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
                )

            try:
                camera.stop()
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
            except Exception:
                pass
        wandb.finish()
        print("[W&B] ✓ Logged to Weights & Biases.")

    print("[DONE] ◠‿◠")


if __name__ == "__main__":
    # For slow networks you can run offline and sync later:
    #   WANDB_MODE=offline python client.py
    #   wandb sync wandb/offline-run-*/
    main()

