#!/usr/bin/env python3
import os, sys, math, time, random, socket, pickle, argparse
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
        for k in ("start_frame_index", "goal_frame_index", "obs_h", "obs_w", "obs_c", "frame_stack"):
            if k in obj and obj[k] is not None:
                try:
                    out[k] = int(obj[k])
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
        f"obs_hw=({out['obs_h']},{out['obs_w']}) frame_stack={out['frame_stack']}"
    )
    return out


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
    ap.add_argument("--print-every", type=int, default=20)
    args = ap.parse_args()
    if args.wandb_project is None:
        alg = (args.algorithm or "gcbc").strip().lower()
        args.wandb_project = f"{alg}-eval" if alg else "carla_eval"
    print(f"[W&B] project={args.wandb_project} (algorithm={args.algorithm})")

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

    # ---- Server header first (start_xy / goal from same .npz as training) ----
    sock = connect_eval(args.server_host, args.server_port, timeout=10.0)
    hdr = try_receive_goal_header(sock, timeout=30.0)
    goal_img = hdr.get("goal_img")
    goal_xy = hdr.get("goal_xy")
    start_xy = hdr.get("start_xy")
    start_yaw_deg = hdr.get("start_yaw_deg")
    send_obs_w = hdr.get("obs_w")
    send_obs_h = hdr.get("obs_h")
    if send_obs_w is None or send_obs_h is None:
        send_obs_w, send_obs_h = 100, 100
        print(f"[NET] Header missing obs_h/obs_w; resizing camera frames to {send_obs_w}x{send_obs_h} (legacy default).")
    else:
        print(f"[NET] Server expects camera frames at {send_obs_w}x{send_obs_h} (from config).")

    ego, chosen_tf = None, None
    if start_xy is not None:
        ego, chosen_tf = spawn_ego_at_dataset_xy(world, ego_bp, start_xy, start_yaw_deg)
        if ego is None:
            print("[CARLA] Dataset spawn failed; falling back to random spawn point.")
    if ego is None:
        spawns = world.get_map().get_spawn_points()
        print(f"[CARLA] #spawn points = {len(spawns)}")
        if not spawns:
            raise RuntimeError("No spawn points available")
        for tf in random.sample(spawns, k=min(20, len(spawns))):
            ego = world.try_spawn_actor(ego_bp, tf)
            if ego is not None:
                chosen_tf = tf
                break
    if not ego:
        raise RuntimeError("Failed to spawn ego vehicle (tesla.model3)")
    print(f"[CARLA] ✓ Ego vehicle spawned: {ego.type_id} at "
          f"Location({chosen_tf.location.x:.6f}, {chosen_tf.location.y:.6f}, {chosen_tf.location.z:.6f}) "
          f"yaw={chosen_tf.rotation.yaw:.8f}")

    # Camera in front (so the car never appears)
    cam_bp = bp.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", "256")
    cam_bp.set_attribute("image_size_y", "256")
    cam_bp.set_attribute("fov", "70")              # a bit narrower helps keep the car out
    cam_bp.set_attribute("sensor_tick", "0.0")     # one frame per tick

    front = float(ego.bounding_box.extent.x) + 0.30  # 30 cm ahead of bumper
    height = 1.40
    #cam_tf = carla.Transform(
    #    carla.Location(x=front, y=0.0, z=height),
    #    carla.Rotation(pitch=0.0)
    #)
    cam_tf = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(
        cam_bp, cam_tf, attach_to=ego,
        attachment_type=carla.AttachmentType.Rigid
    )
    print(f"[CARLA] Camera TF: loc=({cam_tf.location.x:.2f}, {cam_tf.location.y:.2f}, {cam_tf.location.z:.2f}) "
          f"pitch={cam_tf.rotation.pitch:.1f} fov={cam_bp.get_attribute('fov').as_float()}")

    # Synchronous world
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / 50.0  # 50 FPS
    world.apply_settings(settings)
    fps_out = int(round(1.0 / settings.fixed_delta_seconds))
    print(f"[CARLA] ✓ Synchronous mode set (dt={settings.fixed_delta_seconds}, fps≈{fps_out}).")

    # Camera queue
    q = Queue(maxsize=1)
    camera.listen(lambda im: _push_latest(q, im))
    print("[CARLA] Camera listener started.")

    # ---- W&B ----
    wb_active = False
    if wandb is not None:
        run = wandb.init(project=args.wandb_project, name=args.wandb_run)
        wb_active = True
        try: print(f"[W&B] run: {run.url}")
        except Exception: pass
    else:
        print("[W&B] wandb not installed; running without logging.")

    # ---- buffers ----
    video_frames: list[np.ndarray] = []
    xs, ys = [], []

    # seed trajectory with initial pose (useful even if car doesn't move right away)
    start_loc = ego.get_location()
    print(f"[DBG] Start location: ({start_loc.x:.2f}, {start_loc.y:.2f})")
    xs.append(float(start_loc.x)); ys.append(float(start_loc.y))
    last_rgb = None

    # stuck heuristic
    vel_thresh, pos_thresh = 0.15, 0.10
    stuck_counter = 0
    prev_loc = start_loc

    # Action chunking state
    current_action_chunk = None
    chunk_index = 0

    print(f"[RUN] → Running up to {args.frames} frames …", flush=True)

    try:
        for i in range(args.frames):
            t0 = time.time()

            # tick + image
            im = tick_and_grab(world, q, timeout=3.0)
            t_cam = time.time()

            if im is None:
                rgb = last_rgb if last_rgb is not None else np.zeros((256, 256, 3), np.uint8)
                print("im is None")
            else:
                rgb = rgb_from_image(im); last_rgb = rgb

            # Resize to match server's expected obs size (from training config via header).
            # Send as uint8 [0,255] — ImpalaEncoder normalizes internally.
            img_resized = cv2.resize(
                rgb, (int(send_obs_w), int(send_obs_h)), interpolation=cv2.INTER_AREA
            )
            video_frames.append(img_resized)
            img_send = img_resized.astype(np.uint8)

            # Request new action chunk if we've exhausted the current one
            if current_action_chunk is None or chunk_index >= len(current_action_chunk):
                payload = pickle.dumps(img_send, protocol=pickle.HIGHEST_PROTOCOL)

                # send → recv with auto-reconnect; re-send same payload on success
                try:
                    send_with_len(sock, payload)
                    # Receive action chunk: [4-byte chunk_length] + [chunk_length * 3 * float32]
                    chunk_length_bytes = recvall(sock, 4, timeout=10.0)
                    chunk_length = int.from_bytes(chunk_length_bytes, "big")
                    action_chunk_size = chunk_length * 3 * 4  # chunk_length * 3 dims * 4 bytes per float32
                    action_chunk_flat = np.frombuffer(recvall(sock, action_chunk_size, timeout=10.0), dtype=np.float32)
                    current_action_chunk = action_chunk_flat.reshape(chunk_length, 3)  # (chunk_length, 3)
                    chunk_index = 0
                    print(f"[NET] Received new action chunk: shape={current_action_chunk.shape}")
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, socket.timeout) as e:
                    print(f"[NET] IO failed at step {i}: {e} → reconnecting…")
                    try: sock.close()
                    except Exception: pass
                    sock = connect_eval(args.server_host, args.server_port, timeout=10.0)
                    send_with_len(sock, payload)
                    chunk_length_bytes = recvall(sock, 4, timeout=10.0)
                    chunk_length = int.from_bytes(chunk_length_bytes, "big")
                    action_chunk_size = chunk_length * 3 * 4
                    action_chunk_flat = np.frombuffer(recvall(sock, action_chunk_size, timeout=10.0), dtype=np.float32)
                    current_action_chunk = action_chunk_flat.reshape(chunk_length, 3)  # (chunk_length, 3)
                    chunk_index = 0

            # Use next action from current chunk
            action = current_action_chunk[chunk_index]
            chunk_index += 1
            
            thr_raw = float(np.clip(action[0], 0.0, 1.0))
            steer   = float(np.clip(action[1], -1.0, 1.0))
            brk_raw = float(np.clip(action[2], 0.0, 1.0))
            # Mutual exclusion: you're either on gas or brake, not both.
            if thr_raw >= brk_raw:
                thr, brk = thr_raw, 0.0
            else:
                thr, brk = 0.0, brk_raw

            # Apply control
            if i < 0:
                print("manually controlling")
                ego.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0, brake=0.0, hand_brake=False))
            else:
                ego.apply_control(carla.VehicleControl(throttle=thr, steer=steer, brake=brk, hand_brake=False))
            t_step = time.time()

            # trajectory + stuck every 5 ticks
            if i % 5 == 0:
                loc = ego.get_location()
                xs.append(float(loc.x)); ys.append(float(loc.y))
                vel = ego.get_velocity()
                vnorm = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
                dloc = math.sqrt((loc.x - prev_loc.x)**2 + (loc.y - prev_loc.y)**2)
                prev_loc = loc
                stuck_counter = stuck_counter + 1 if (vnorm < vel_thresh or dloc < pos_thresh) else 0

            # periodic prints
            if (i + 1) % args.print_every == 0:
                t1 = time.time()
                print(
                    f"[{i+1}/{args.frames}] "
                    f"thr={thr:.3f} steer={steer:.3f} brake={brk:.3f}  "
                    f"loc=({prev_loc.x:.2f},{prev_loc.y:.2f})  "
                    f"cam={(t_cam - t0)*1000:.1f}ms step={(t_step - t0)*1000:.1f}ms",
                    flush=True
                )

            # optional early stop if really stuck
            if stuck_counter >= 500 and i > 500:
                print("[RUN] Stuck too long; ending early.")
                break

        print("[RUN] ✓ Loop finished.")
    finally:
        # Cleanup CARLA + socket
        try: camera.stop()
        except Exception: pass
        try: ego.destroy()
        except Exception: pass
        try: world.apply_settings(original_settings)
        except Exception: pass
        try: sock.close()
        except Exception: pass

    # ---- W&B: video + trajectory scatter (single shot) ----
    if wb_active:
        # video
        if not args.no_video:
            try:
                vid = np.stack(video_frames, axis=0).astype(np.uint8)   # (N,100,100,3)
                vid = np.moveaxis(vid, -1, 1)                           # (N,C,H,W)
                wandb.log({"camera_video": wandb.Video(vid, fps=fps_out, format="mp4")},
                          step=len(video_frames))
            except Exception as e:
                print(f"[W&B] [WARN] video log failed: {e}")
        # --- XY scatter (traj + goal as star) with map overlay ---
        pts = [(float(x), float(y)) for x, y in zip(xs, ys)
               if np.isfinite(x) and np.isfinite(y)]

        if pts:
            rows = [[x, y, "traj"] for x, y in pts]
            if goal_xy is not None and np.all(np.isfinite(goal_xy)):
                rows.append([float(goal_xy[0]), float(goal_xy[1]), "goal"])

            # Get map waypoints for overlay
            try:
                carla_map = world.get_map()
                waypoints = carla_map.generate_waypoints(distance=2.0)  # Sample every 2m
                map_rows = [[float(wp.transform.location.x), float(wp.transform.location.y), "map"] 
                           for wp in waypoints]
                rows.extend(map_rows)
            except Exception as e:
                print(f"[W&B] [WARN] Failed to get map waypoints: {e}")

            # Build matplotlib overlay (map gray, traj blue, goal orange)
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
                # Draw trajectory as a thin line
                ax.plot(tx, ty, c="#4e79a7", linewidth=0.5, alpha=0.8, label="traj", zorder=2)
                
                # Add arrows to show direction (every 5th point, or at least every 10 points if trajectory is short)
                if len(traj_pts) > 1:
                    arrow_interval = max(1, len(traj_pts) // 10)  # Show ~10 arrows total
                    for i in range(0, len(traj_pts) - 1, arrow_interval):
                        x1, y1 = traj_pts[i]
                        x2, y2 = traj_pts[i + 1]
                        dx = x2 - x1
                        dy = y2 - y1
                        # Only draw arrow if there's meaningful movement
                        if abs(dx) > 0.01 or abs(dy) > 0.01:
                            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                                       arrowprops=dict(arrowstyle='->', lw=2.5, 
                                                      color='#4e79a7', alpha=0.9,
                                                      mutation_scale=25),  # Large arrow head (default is ~10)
                                       zorder=3)
            if goal_pts:
                gx, gy = zip(*goal_pts)
                ax.scatter(gx, gy, marker="*", s=220, c="#f28e2b", label="goal", zorder=4)

            ax.set_aspect("equal")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.legend(loc="best")

            # Tight layout and render to array
            fig.tight_layout()
            canvas.draw()
            map_img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            map_img = map_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            try:
                wandb.log({"trajectory_map": wandb.Image(map_img)}, step=len(video_frames))
            except Exception as e:
                print(f"[W&B] [WARN] map image log failed: {e}")
        else:
            print("[W&B] [WARN] No valid trajectory points; skipping scatter.")

        # Goal frame (if any)
        try:
            if isinstance(goal_img, np.ndarray):
                wandb.log({"goal_frame": wandb.Image(goal_img)})
        except Exception as e:
            print(f"[W&B] [WARN] goal image log failed: {e}")
        wandb.finish()
        print("[W&B] ✓ Logged to Weights & Biases.")

    print("[DONE] ◠‿◠")


if __name__ == "__main__":
    # For slow networks you can run offline and sync later:
    #   WANDB_MODE=offline python client.py
    #   wandb sync wandb/offline-run-*/
    main()

