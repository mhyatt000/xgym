import numpy as np
import cv2
import argparse
from collections import deque
from webpolicy.deploy.client import WebsocketClientPolicy
from xgym.lrbt.from_mano_npz import postprocess

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--window-size", type=int, default=10, help="how many future steps to show")
parser.add_argument("--port", type=int, default=8003, help="port for hamer server")
args = parser.parse_args()

# config
ep_path = "/home/jgoldberg/bela/sweep_mano/ep75_20250523_114526.npz"
save_video_path = "trajectory_overlay.mp4"
window_size = args.window_size

# load frames
episode = np.load(ep_path, allow_pickle=True)
frames = episode["low"]
height, width, _ = frames[0].shape

# connect to hamer
client = WebsocketClientPolicy(host="0.0.0.0", port=args.port)

# predict wrist keypoints
predicted_wrist_by_frame = []   

for i, frame in enumerate(frames):
    out = client.infer({"img": frame})

    data = postprocess(out, frame)
    if "kp2d" not in data:
        predicted_wrist_by_frame.append(None)
        continue
    wrist = data["kp2d"][9]
    predicted_wrist_by_frame.append(np.round(wrist).astype(int))

# overlay sliding trajectories
output_frames = []

for t in range(len(frames) - window_size):
    base = frames[t].copy()
    traj = deque()

    for future in range(1, window_size + 1):
        kp = predicted_wrist_by_frame[t + future]
        if kp is not None:
            traj.append(kp)

    for i in range(1, len(traj)):
        p1, p2 = tuple(traj[i - 1]), tuple(traj[i])
        if all(0 <= v < width for v in [p1[0], p2[0]]) and all(0 <= v < height for v in [p1[1], p2[1]]):
            cv2.line(base, p1, p2, (0, 100 + i * 10, 255 - i * 20), 3)
            cv2.circle(base, p2, 4, (255, 255, 255), -1)

    output_frames.append(base)

# write to video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(save_video_path, fourcc, 15, (width, height))

for f in output_frames:
    bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
    out.write(bgr)

out.release()

print(f"Saved video: {save_video_path}")
