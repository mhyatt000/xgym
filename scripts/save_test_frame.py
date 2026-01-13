import numpy as np
import cv2

#path to episode for testing
ep_path = "/home/jgoldberg/bela/sweep_mano/ep7_20250523_113029.npz"
save_path = "test_frame.png"

# Load episode
data = np.load(ep_path, allow_pickle=True)

# Get first frame
frame = data["low"][10]  # use [i] for a different frame

# Save as PNG (converts from BGR to RGB if needed)
cv2.imwrite(save_path, frame)

print(f"Saved frame to {save_path}")

