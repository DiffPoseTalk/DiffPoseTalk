from pathlib import Path

import numpy as np
import torch
from scipy.signal import savgol_filter
from tqdm import tqdm

import rotation as rot

# %%
root = Path("/data/datasets/face_video/HDTF_TFHP")
spectre_root = root / "SPECTRE_coeffs"
mica_root = root / "EMICA_coeffs"
out_root = root / "FLAME_coeffs"
coef_list = sorted(spectre_root.rglob("*.npz"))


# %%
def combine(coef_f):
    spectre = dict(np.load(coef_f))
    mica_f = mica_root / coef_f.relative_to(spectre_root)
    out_f = out_root / coef_f.relative_to(spectre_root)

    if mica_f.exists():
        # transfer shape
        mica = dict(np.load(mica_f))
        length = spectre["pose"].shape[0]
        mica_shape = mica["shape"][:length, :100]
        if mica_shape.shape[0] < length:
            mica_shape = np.concatenate(
                [
                    mica_shape,
                    mica_shape[-1:].repeat(length - mica_shape.shape[0], axis=0),
                ],
                axis=0,
            )
        spectre["shape"] = mica_shape

    # optimize jaw pose
    mouth = torch.tensor(spectre["pose"][:, 3:])
    angles = rot.matrix_to_euler_angles(rot.axis_angle_to_matrix(mouth), "XYZ")
    angles[:, 1:] = 0
    mouth = rot.matrix_to_axis_angle(rot.euler_angles_to_matrix(angles, "XYZ"))
    spectre["pose"][:, 3:] = savgol_filter(mouth.cpu().numpy(), 5, 2, axis=0)

    # smooth exp
    spectre["exp"] = savgol_filter(spectre["exp"], 5, 2, axis=0)

    out_f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_f, **spectre)


# %%
for coef_f in tqdm(coef_list):
    combine(coef_f)
