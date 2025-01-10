import argparse
import pickle
import sys
from pathlib import Path

import lmdb
import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm

from rotation import axis_angle_to_rotation_6d


class LmdbCoefDataset(data.Dataset):
    def __init__(self, lmdb_dir, split_file, n_motions=100, crop_strategy="random"):
        self.split_file = split_file
        self.lmdb_dir = Path(lmdb_dir)

        self.n_motions = n_motions
        self.crop_strategy = crop_strategy

        # Read split file
        self.entries = []
        with open(self.split_file, "r") as f:
            for line in f:
                self.entries.append(line.strip())

        # Load lmdb
        self.lmdb_env = lmdb.open(
            str(self.lmdb_dir),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.lmdb_env.begin(write=False) as txn:
            self.clip_len = pickle.loads(txn.get("metadata".encode()))["seg_len"]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        # Read audio and coef
        with self.lmdb_env.begin(write=False) as txn:
            meta_key = f"{self.entries[index]}/metadata".encode()
            metadata = pickle.loads(txn.get(meta_key))
            seq_len = metadata["n_frames"]

        # Crop the audio and coef
        if self.crop_strategy == "random":
            start_frame = np.random.randint(0, seq_len - self.n_motions + 1)
        elif self.crop_strategy == "begin":
            start_frame = 0
        elif self.crop_strategy == "end":
            start_frame = seq_len - self.n_motions
        else:
            raise ValueError(f"Unknown crop strategy: {self.crop_strategy}")

        coef_keys = ["shape", "exp", "pose"]
        coef_dict = {k: [] for k in coef_keys}
        start_clip = start_frame // self.clip_len
        end_clip = (start_frame + self.n_motions - 1) // self.clip_len
        with self.lmdb_env.begin(write=False) as txn:
            for clip_idx in range(start_clip, end_clip + 1):
                key = f"{self.entries[index]}/{clip_idx:03d}".encode()
                start_idx = max(start_frame - clip_idx * self.clip_len, 0)
                end_idx = min(
                    start_frame + self.n_motions - clip_idx * self.clip_len,
                    self.clip_len,
                )

                entry = pickle.loads(txn.get(key))
                for coef_key in coef_keys:
                    coef_dict[coef_key].append(
                        entry["coef"][coef_key][start_idx:end_idx]
                    )

        coef_dict = {
            k: torch.tensor(np.concatenate(coef_dict[k], axis=0)) for k in coef_keys
        }
        return coef_dict


def main(args):
    dataset = LmdbCoefDataset(
        args.data_root, args.data_root / f"{args.mode}.txt", crop_strategy="random"
    )
    data_loader = data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    keys = ["shape", "exp", "pose"]
    stats_per_entry = {k: {"mean": [], "std": [], "max": [], "min": []} for k in keys}

    for round in tqdm(range(1000)):
        for coef_dict in data_loader:
            for k in keys:
                coef_k = coef_dict[k].numpy()
                stats_per_entry[k]["mean"].append(coef_k.mean(axis=1).mean(axis=0))
                stats_per_entry[k]["std"].append(coef_k.std(axis=1).mean(axis=0))
                stats_per_entry[k]["max"].append(coef_k.max(axis=1).max(axis=0))
                stats_per_entry[k]["min"].append(coef_k.min(axis=1).min(axis=0))

    stats_dict = {}
    for k in keys:
        stats_dict[f"{k}_mean"] = np.stack(stats_per_entry[k]["mean"]).mean(axis=0)
        stats_dict[f"{k}_std"] = np.stack(stats_per_entry[k]["std"]).mean(axis=0)
        stats_dict[f"{k}_max"] = np.stack(stats_per_entry[k]["max"]).max(axis=0)
        stats_dict[f"{k}_min"] = np.stack(stats_per_entry[k]["min"]).min(axis=0)

    np.savez(args.data_root / f"stats_{args.mode}.npz", **stats_dict)


if __name__ == "__main__":
    __dir__ = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=Path,
        default=__dir__.parent / "datasets/HDTF_TFHP/lmdb",
        help="dataset path",
    )
    parser.add_argument(
        "--mode", type=str, default="train", choices=["train", "val", "test"]
    )
    args = parser.parse_args()
    main(args)
