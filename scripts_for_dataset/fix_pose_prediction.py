from pathlib import Path

import numpy as np
import torch
from scipy.signal import savgol_filter
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from rotation import matrix_to_axis_angle

# %%
root = Path("/data/datasets/face_video/HDTF_TFHP")
coef_root = root / "FLAME_coeffs"
pose_root = root / "6DRepNet"
coef_list = sorted(coef_root.rglob("*.npz"))


# %%
def fix_pose(coef_f):
    coef = dict(np.load(coef_f))
    pose_f = pose_root / coef_f.relative_to(coef_root).with_suffix(".npy")
    pose = torch.tensor(np.load(pose_f), device="cuda")

    length = min(coef["pose"].shape[0], pose.shape[0])

    # replace the pose in coef
    pose_aa = matrix_to_axis_angle(pose[:length]).cpu().numpy()
    coef["pose"][:length, :3] = savgol_filter(pose_aa, 7, 2, axis=0)

    for k in coef:
        coef[k] = coef[k][:length]

    np.savez(coef_f, **coef)


# %%
for coef_f in tqdm(coef_list):
    fix_pose(coef_f)

# all_keys = process_map(fix_pose, coef_list)
