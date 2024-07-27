import io
import pickle
import random
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import lmdb
import numpy as np
import torch
import torchaudio
from torch.utils import data

__dir__ = Path(__file__).parent
sys.path.append(str(__dir__.parent.absolute()))

# https://github.com/pytorch/audio/issues/2950 , https://github.com/pytorch/audio/issues/2356
torchaudio.set_audio_backend('soundfile')

warnings.filterwarnings('ignore', message='PySoundFile failed. Trying audioread instead.')


class LmdbDataset(data.Dataset):
    def __init__(self, lmdb_dir, split_file, coef_stats_file=None, coef_fps=25, n_motions=100, crop_strategy='random',
                 rot_repr='aa'):
        self.split_file = split_file
        self.lmdb_dir = Path(lmdb_dir)
        if coef_stats_file is not None:
            coef_stats = dict(np.load(coef_stats_file))
            self.coef_stats = {x: torch.tensor(coef_stats[x]) for x in coef_stats}
        else:
            self.coef_stats = None
            print('Warning: No stats file found. Coef will not be normalized.')

        self.coef_fps = coef_fps
        self.audio_unit = 16000. / self.coef_fps  # num of samples per frame
        self.n_motions = n_motions
        self.n_audio_samples = round(self.audio_unit * self.n_motions)
        self.coef_total_len = self.n_motions * 2
        self.audio_total_len = round(self.audio_unit * self.coef_total_len)

        self.crop_strategy = crop_strategy
        self.rot_representation = rot_repr

        # Read split file
        self.entries = []
        with open(self.split_file, 'r') as f:
            for line in f:
                self.entries.append(line.strip())

        # Load lmdb
        self.lmdb_env = lmdb.open(str(self.lmdb_dir), readonly=True, lock=False, readahead=False, meminit=False)
        with self.lmdb_env.begin(write=False) as txn:
            self.clip_len = pickle.loads(txn.get('metadata'.encode()))['seg_len']
            self.audio_clip_len = round(self.audio_unit * self.clip_len)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        # Read audio and coef
        with self.lmdb_env.begin(write=False) as txn:
            meta_key = f'{self.entries[index]}/metadata'.encode()
            metadata = pickle.loads(txn.get(meta_key))
            seq_len = metadata['n_frames']

        # Crop the audio and coef
        if self.crop_strategy == 'random':
            start_frame = np.random.randint(0, seq_len - self.coef_total_len + 1)
        elif self.crop_strategy == 'begin':
            start_frame = 0
        elif self.crop_strategy == 'end':
            start_frame = seq_len - self.coef_total_len
        else:
            raise ValueError(f'Unknown crop strategy: {self.crop_strategy}')

        coef_keys = ['shape', 'exp', 'pose']
        coef_dict = {k: [] for k in coef_keys}
        audio = []
        start_clip = start_frame // self.clip_len
        end_clip = (start_frame + self.coef_total_len - 1) // self.clip_len + 1
        with self.lmdb_env.begin(write=False) as txn:
            for clip_idx in range(start_clip, end_clip):
                key = f'{self.entries[index]}/{clip_idx:03d}'.encode()
                start_idx = max(start_frame - clip_idx * self.clip_len, 0)
                end_idx = min(start_frame + self.coef_total_len - clip_idx * self.clip_len, self.clip_len)

                entry = pickle.loads(txn.get(key))
                for coef_key in coef_keys:
                    coef_dict[coef_key].append(entry['coef'][coef_key][start_idx:end_idx])

                audio_data = entry['audio']
                audio_clip, sr = torchaudio.load(io.BytesIO(audio_data))
                assert sr == 16000, f'Invalid sampling rate: {sr}'
                audio_clip = audio_clip.squeeze()
                audio.append(audio_clip[round(start_idx * self.audio_unit):round(end_idx * self.audio_unit)])

        coef_dict = {k: torch.tensor(np.concatenate(coef_dict[k], axis=0)) for k in coef_keys}
        assert coef_dict['exp'].shape[0] == self.coef_total_len, f'Invalid coef length: {coef_dict["exp"].shape[0]}'
        audio = torch.cat(audio, dim=0)
        assert audio.shape[0] == self.coef_total_len * self.audio_unit, f'Invalid audio length: {audio.shape[0]}'
        audio_mean = audio.mean()
        audio_std = audio.std()
        audio = (audio - audio_mean) / (audio_std + 1e-5)

        if self.rot_representation == 'aa':
            keys = ['shape', 'exp', 'pose']
        else:
            raise ValueError(f'Unknown rotation representation: {self.rot_representation}')

        # normalize coef if applicable
        if self.coef_stats is not None:
            coef_dict = {k: (coef_dict[k] - self.coef_stats[f'{k}_mean']) / (self.coef_stats[f'{k}_std'] + 1e-9)
                         for k in keys}

        # Extract two consecutive audio/coef clips
        audio_pair = [audio[:self.n_audio_samples].clone(), audio[-self.n_audio_samples:].clone()]
        coef_pair = [{k: coef_dict[k][:self.n_motions].clone() for k in keys},
                     {k: coef_dict[k][-self.n_motions:].clone() for k in keys}]

        return audio_pair, coef_pair, (audio_mean, audio_std)


class LmdbDatasetForSE(data.Dataset):
    def __init__(self, lmdb_dir, split_file, coef_stats_file=None, coef_fps=25, n_motions=100, crop_strategy='random',
                 rot_repr='aa', no_head_pose=False):
        self.split_file = split_file
        self.lmdb_dir = Path(lmdb_dir)
        if coef_stats_file is not None:
            coef_stats = dict(np.load(coef_stats_file))
            self.coef_stats = {x: torch.tensor(coef_stats[x]) for x in coef_stats}
        else:
            self.coef_stats = None
            print('Warning: No stats file found. Coef will not be normalized.')

        self.coef_fps = coef_fps
        self.audio_unit = 16000. / self.coef_fps  # num of samples per frame
        self.n_motions = n_motions
        self.n_audio_samples = round(self.audio_unit * self.n_motions)
        self.coef_total_len = int(self.n_motions * 2.1)
        self.audio_total_len = round(self.audio_unit * self.coef_total_len)

        self.crop_strategy = crop_strategy
        self.rot_representation = rot_repr
        self.no_head_pose = no_head_pose

        # Read split file
        self.entries = defaultdict(list)
        with open(self.split_file, 'r') as f:
            for line in f:
                person_id = line.strip().split('/')[0]
                self.entries[person_id].append(line.strip().split()[0])
        self.person_ids = list(self.entries.keys())

        # Load lmdb
        self.lmdb_env = lmdb.open(str(self.lmdb_dir), readonly=True, lock=False, readahead=False, meminit=False)
        with self.lmdb_env.begin(write=False) as txn:
            self.clip_len = pickle.loads(txn.get('metadata'.encode()))['seg_len']

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        # Read coef
        with self.lmdb_env.begin(write=False) as txn:
            key = random.choice(self.entries[self.person_ids[index]])
            meta_key = f'{key}/metadata'.encode()
            metadata = pickle.loads(txn.get(meta_key))
            seq_len = metadata['n_frames']

        # Crop the audio and coef
        if self.crop_strategy == 'random':
            start_frame = np.random.randint(0, seq_len - self.coef_total_len + 1)
        elif self.crop_strategy == 'begin':
            start_frame = 0
        elif self.crop_strategy == 'end':
            start_frame = seq_len - self.coef_total_len
        else:
            raise ValueError(f'Unknown crop strategy: {self.crop_strategy}')

        coef_keys = ['exp', 'pose']
        coef_dict = {k: [] for k in coef_keys}
        start_clip = start_frame // self.clip_len
        end_clip = (start_frame + self.coef_total_len - 1) // self.clip_len + 1
        with self.lmdb_env.begin(write=False) as txn:
            for clip_idx in range(start_clip, end_clip):
                clip_key = f'{key}/{clip_idx:03d}'.encode()
                start_idx = max(start_frame - clip_idx * self.clip_len, 0)
                end_idx = min(start_frame + self.coef_total_len - clip_idx * self.clip_len, self.clip_len)

                entry = pickle.loads(txn.get(clip_key))
                for coef_key in coef_keys:
                    coef_dict[coef_key].append(entry['coef'][coef_key][start_idx:end_idx])

        coef_dict = {k: torch.tensor(np.concatenate(coef_dict[k], axis=0)) for k in coef_keys}
        assert coef_dict['exp'].shape[0] == self.coef_total_len, f'Invalid coef length: {coef_dict["exp"].shape[0]}'

        if self.rot_representation == 'aa':
            coef_keys = ['exp', 'pose']
        else:
            raise ValueError(f'Unknown rotation representation: {self.rot_representation}')

        # normalize coef if applicable
        if self.coef_stats is not None:
            coef_dict = {k: (coef_dict[k] - self.coef_stats[f'{k}_mean']) / (self.coef_stats[f'{k}_std'] + 1e-9)
                         for k in coef_keys}

        if self.no_head_pose:
            if self.rot_representation == 'aa':
                mouth_pose_coef = coef_dict['pose'][:, 3:]
            else:
                raise ValueError(f'Unknown rotation representation: {self.rot_representation}')
            motion_coef = torch.cat([coef_dict['exp'], mouth_pose_coef], dim=-1)
        else:
            motion_coef = torch.cat([coef_dict[k] for k in coef_keys], dim=-1)

        if self.rot_representation == 'aa':
            # Remove mouth rotation around y, z axis
            motion_coef = motion_coef[:, :-2]

        # Extract two consecutive coef clips
        coef_pair = [motion_coef[:self.n_motions].clone(), motion_coef[-self.n_motions:].clone()]

        return coef_pair
