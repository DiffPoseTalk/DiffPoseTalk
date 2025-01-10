# %%
import io
import math
import pickle
import random
from pathlib import Path
from tempfile import NamedTemporaryFile

import lmdb
import numpy as np
import torchaudio
from tqdm.contrib.concurrent import process_map

# File-like object support in sox_io backend is buggy and deprecated, and will be removed in v2.1.
# https://github.com/pytorch/audio/issues/2950 , https://github.com/pytorch/audio/issues/2356
torchaudio.set_audio_backend("soundfile")

# %%
coef_window = 100
audio_unit = 640

# %%
root = Path("/data/datasets/face_video/HDTF_TFHP")
lmdb_root = root / "lmdb"
coef_root = root / "FLAME_coeffs"
audio_root = root / "audio"
coef_list = sorted(coef_root.rglob("*.npz"))

# %%
db = lmdb.open(str(lmdb_root), map_size=100 * 1024 * 1024 * 1024, readonly=False)
with db.begin(write=True) as txn:
    # write metadata
    metadata = {"seg_len": coef_window}
    txn.put(
        "metadata".encode(), pickle.dumps(metadata, protocol=pickle.HIGHEST_PROTOCOL)
    )


# %%
def write_to_db(coef_f):
    # audio_f = audio_root / coef_f.relative_to(coef_root).with_suffix('.mp3')
    audio_f = audio_root / coef_f.relative_to(coef_root).with_suffix(".flac")
    coef = dict(np.load(coef_f))
    audio = torchaudio.load(audio_f)[0]
    person, video_id = coef_f.with_suffix("").parts[-2:]
    video_id = int(video_id)

    n_frames = min(coef["shape"].shape[0], audio.shape[1] // audio_unit)
    n_segments = math.ceil(n_frames / coef_window)

    keys = []
    with db.begin(write=True) as txn:
        # write metadata
        metadata = {"n_frames": n_frames}
        key = f"{person}/{video_id:03d}"
        keys.append(key)
        txn.put(
            f"{key}/metadata".encode(),
            pickle.dumps(metadata, protocol=pickle.HIGHEST_PROTOCOL),
        )

        # write segmented data
        for i in range(n_segments):
            start_frame = i * coef_window
            end_frame = min((i + 1) * coef_window, n_frames)
            coef_seg = {
                k: v[start_frame:end_frame]
                for k, v in coef.items()
                if k in ["shape", "exp", "pose"]
            }
            audio_seg = audio[:, start_frame * audio_unit : end_frame * audio_unit]
            # with NamedTemporaryFile(suffix='.mp3') as audio_seg_f:
            #     torchaudio.save(audio_seg_f.name, audio_seg, sample_rate=16000, format='mp3')
            #     audio_data = open(audio_seg_f.name, 'rb').read()

            with NamedTemporaryFile(suffix=".flac") as audio_seg_f:
                torchaudio.save(
                    audio_seg_f.name,
                    audio_seg,
                    sample_rate=16000,
                    format="flac",
                    bits_per_sample=16,
                )
                audio_data = open(audio_seg_f.name, "rb").read()
            # buffer = io.BytesIO()
            # torchaudio.save(buffer, audio_seg, sample_rate=16000, format='flac', bits_per_sample=16)
            # audio_data = buffer.read()
            # buffer.close()

            data = {"audio": audio_data, "coef": coef_seg}
            txn.put(
                f"{key}/{i:03d}".encode(),
                pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL),
            )

    return keys


# %%
all_keys = process_map(write_to_db, coef_list)
all_keys = [key for keys in all_keys for key in keys]
with open(lmdb_root / "keys.txt", "w") as f:
    f.write("\n".join(all_keys))
