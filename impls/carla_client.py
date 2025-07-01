import socket
import pickle
import time
import numpy as np
import imageio
import wandb
import carla
import cv2
from collections import deque

# === Socket info ===
HOST = 'localhost'
PORT = 5050

# === Goal setup (same as on server) ===
GOAL = None  # set externally by loading the same dataset + index
GOAL_IDX = 0  # index into dataset

# === Logging setup ===
wandb.init(project="crl_carla_eval", name="eval_run_with_video")
video_frames = []
goal_distances = []
at_goal_frames = 0
all_steers = []
all_throttles = []

# === Connect to eval_server.py ===
client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_sock.connect((HOST, PORT))
print(f"🔗 Connected to eval_server at {HOST}:{PORT}")

# === CARLA setup ===
carla_client = carla.Client('localhost', 2000)
carla_client.set_timeout(10.0)
world = carla_client.get_world()
bp_lib = world.get_blueprint_library()

vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
spawn_point = world.get_map().get_spawn_points()[0]
vehicle = world.spawn_actor(vehicle_bp, spawn_point)
vehicle.set_autopilot(False)
print("✅ Vehicle spawned.")

# === Camera setup ===
camera_bp = bp_lib.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '256')
camera_bp.set_attribute('image_size_y', '256')
camera_bp.set_attribute('fov', '90')

camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
print("📸 Camera attached.")

# === Buffer to hold latest action ===
action_buffer = deque(maxlen=1)

def send_recv_action(img_48):
    img_48 = img_48.astype(np.float32) / 255.0
    data = pickle.dumps(img_48, protocol=pickle.HIGHEST_PROTOCOL)
    client_sock.sendall(len(data).to_bytes(4, 'big') + data)

    # Receive 2 float32s
    action_bytes = b''
    while len(action_bytes) < 8:
        chunk = client_sock.recv(8 - len(action_bytes))
        if not chunk:
            raise ConnectionError("Lost connection to eval_server")
        action_bytes += chunk

    action = np.frombuffer(action_bytes, dtype=np.float32)
    return action

# === Camera callback ===
def camera_callback(image):
    global at_goal_frames
    print("camera callback fired")
    img = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[:, :, :3]
    img_48 = cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA)
    print("[DEBUG] Sending image to server...")
    action = send_recv_action(img_48)
    print("[DEBUG] Received action:", action)
    action_buffer.append(action)
    print("[DEBUG] Appended frame, len(video_frames) =", len(video_frames))

    throttle = float(np.clip(action[0], 0.0, 1.0))
    steer = float(np.clip(action[1], -1.0, 1.0))
    vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer))
    
    all_throttles.append(throttle)
    all_steers.append(steer)

    # === Logging frame ===
    video_frames.append(img_48)

    # === Goal distance ===
    if GOAL is not None:
        dist = np.linalg.norm(img_48.astype(np.float32) - GOAL.astype(np.float32)) / 48.0
        goal_distances.append(dist)
        if dist < 0.05:
            at_goal_frames += 1

# === Load goal ===
goal_npz = np.load("/global/scratch/users/achyuthkv76/CarlaOfflineV0_vision_data.npz")
GOAL = goal_npz["observations"][GOAL_IDX]
print("🎯 Loaded goal frame from dataset")

# === Begin driving ===
action_buffer.clear()
camera.listen(camera_callback)
print("🚘 Driving with CRL policy via socket...")

start = time.time()
while time.time() - start < 300:
    print(f"[DEBUG] Client running, t={time.time() - start:.1f}s")
    time.sleep(2)

# === Cleanup ===
camera.stop()
vehicle.destroy()
client_sock.close()
print("🧹 Cleaned up and closed connection.")

# === Save and log video to wandb ===
video_array = np.stack(video_frames).astype(np.uint8)
video_array = np.moveaxis(video_array, -1, 1)
#video_array = np.expand_dims(video_array, 0)  # add batch dim
print(f"video_array shape: {video_array.shape}")
wandb.log({
    "eval_video": wandb.Video(video_array, fps=10, format="mp4"),
    "avg_goal_distance": float(np.mean(goal_distances)),
    "at_goal_percent": float(100.0 * at_goal_frames / len(video_frames)),
    "avg_throttle": float(np.mean(all_throttles)),
    "avg_steer": float(np.mean(all_steers)),
    "steer_std": float(np.std(all_steers)),
    "throttle_std": float(np.std(all_throttles)),
})
wandb.finish()
