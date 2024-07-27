import argparse
from pathlib import Path

import numpy as np
import torch

from models import StyleEncoder
from utils import get_model_path


class StyleExtractor:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = device

        model_data = torch.load(checkpoint_path, map_location=device)
        self.model_args = model_data['args']
        self.model = StyleEncoder(self.model_args).to(device)
        self.model.encoder.load_state_dict(model_data['encoder'], strict=False)
        self.model.eval()

        if self.model_args.stats_file is not None:
            stats_file: Path = self.model_args.stats_file
            if stats_file is not None and not stats_file.is_absolute():
                stats_file = self.model_args.data_root / stats_file
            coef_stats = dict(np.load(stats_file))
            self.coef_stats = {k: torch.from_numpy(v).to(device) for k, v in coef_stats.items()}
        else:
            self.coef_stats = None

        self.n_motions = self.model_args.n_motions
        self.audio_unit = 16000 / self.model_args.fps
        self.rot_repr = self.model_args.rot_repr
        self.no_head_pose = self.model_args.no_head_pose

    @torch.no_grad()
    def extract(self, coef_file, start_frame=0):
        end_frame = start_frame + self.n_motions

        coef = dict(np.load(coef_file))
        coef = {k: torch.from_numpy(coef[k][start_frame:end_frame]).float().to(self.device) for k in ['exp', 'pose']}
        if self.rot_repr == 'aa':
            coef_keys = ['exp', 'pose']
        else:
            raise ValueError(f'Unknown rotation representation: {self.rot_repr}')

        # normalize coef if applicable
        if self.coef_stats is not None:
            coef = {k: (coef[k] - self.coef_stats[f'{k}_mean']) / self.coef_stats[f'{k}_std'] for k in coef_keys}

        if self.no_head_pose:
            if self.rot_repr == 'aa':
                mouth_pose_coef = coef['pose'][:, 3:]
            else:
                raise ValueError(f'Unknown rotation representation: {self.rot_repr}')
            motion_coef = torch.cat([coef['exp'], mouth_pose_coef], dim=-1)
        else:
            motion_coef = torch.cat([coef[k] for k in coef_keys], dim=-1)

        if self.model_args.rot_repr == 'aa':
            # Remove mouth rotation over y, z axis
            motion_coef = motion_coef[:, :-2]

        style_feat = self.model(motion_coef.unsqueeze(0))
        style_feat = style_feat[0].detach().cpu().numpy()

        return style_feat


def main(args):
    checkpoint_path, exp_name = get_model_path(args.exp_name, args.iter, 'SE')
    extractor = StyleExtractor(checkpoint_path, device=args.device)
    output_dir = Path('demo/input/style') / exp_name / f'iter_{args.iter:07}'

    style_feat = extractor.extract(args.coef, args.start_frame)

    output_file: Path = args.output
    if not output_file.is_absolute():
        output_file = output_dir / output_file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_file, style_feat)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--exp_name', type=str, default='HDTF_TFHP', help='experiment name')
    parser.add_argument('--iter', type=int, default=30000, help='number of iterations')
    parser.add_argument('--device', type=str, default='cuda', help='device')

    # data
    parser.add_argument('--coef', '-c', type=Path, help='path to FLAME coefficients')
    parser.add_argument('--start_frame', '-s', type=int, default=0, help='starting frame')
    parser.add_argument('--output', '-o', type=Path, required=True, help='output path')

    args = parser.parse_args()
    main(args)
