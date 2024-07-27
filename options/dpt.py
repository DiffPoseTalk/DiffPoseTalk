import argparse
from pathlib import Path


def add_model_options(parser: argparse.ArgumentParser):
    parser.add_argument('--target', type=str, default='sample', choices=['sample', 'noise'])
    parser.add_argument('--guiding_conditions', type=str, default='audio,style')
    parser.add_argument('--cfg_mode', type=str, default='incremental', choices=['incremental', 'independent'])
    parser.add_argument('--n_diff_steps', type=int, default=500, help='number of diffusion steps')
    parser.add_argument('--diff_schedule', type=str, default='cosine',
                        choices=['linear', 'cosine', 'quadratic', 'sigmoid'])
    parser.add_argument('--rot_repr', type=str, default='aa', choices=['aa'])
    parser.add_argument('--no_head_pose', action='store_true', help='do not predict head pose')
    parser.add_argument('--style_enc_ckpt', type=Path)
    parser.add_argument('--d_style', type=int, default=128, help='dimension of the style feature')

    # transformer
    parser.add_argument('--audio_model', type=str, default='hubert', choices=['wav2vec2', 'hubert'])
    parser.add_argument('--architecture', type=str, default='decoder', choices=['decoder'])
    parser.add_argument('--align_mask_width', type=int, default=1,
                        help='width of the alignment mask, non-positive for no mask')
    parser.add_argument('--no_use_learnable_pe', action='store_true', help='do not use learnable positional encoding')
    parser.add_argument('--use_indicator', action='store_true', help='use indicator for padded frames')
    parser.add_argument('--feature_dim', type=int, default=512, help='dimension of the hidden feature')
    parser.add_argument('--n_heads', type=int, default=8, help='number of attention heads')
    parser.add_argument('--n_layers', type=int, default=8, help='number of encoder/decoder layers')
    parser.add_argument('--mlp_ratio', type=int, default=4, help='ratio of the hidden dimension of the MLP')

    # sequence
    parser.add_argument('--n_motions', type=int, default=100, help='number of motions in a sequence')
    parser.add_argument('--n_prev_motions', type=int, default=10, help='number of pre-motions in a sequence')
    parser.add_argument('--fps', type=int, default=25, help='frame per second')
    parser.add_argument('--pad_mode', type=str, default='zero', choices=['zero', 'replicate'])


def add_data_options(parser: argparse.ArgumentParser):
    parser.add_argument('--data_root', type=Path, default=Path('datasets/HDTF_TFHP/lmdb'),
                        help='dataset path')
    parser.add_argument('--stats_file', type=Path, default=Path('stats_train.npz'))
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataloader')


def add_training_options(parser: argparse.ArgumentParser):
    parser.add_argument('--exp_name', type=str, default='HDTF_TFHP', help='experiment name')
    parser.add_argument('--max_iter', type=int, default=200000, help='max number of iterations')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient accumulation')
    parser.add_argument('--scheduler', type=str, default='None', choices=['None', 'Warmup', 'WarmupThenDecay'])

    parser.add_argument('--criterion', type=str, default='l2', choices=['l1', 'l2'])
    parser.add_argument('--l_vert', type=float, default=2e6, help='weight of the vertex loss')
    parser.add_argument('--l_vel', type=float, default=1e7, help='weight of the velocity loss')
    parser.add_argument('--l_smooth', type=float, default=1e5,
                        help='weight of the vertex acceleration regularization')
    parser.add_argument('--l_head_angle', type=float, default=0.05, help='weight of the head angle loss')
    parser.add_argument('--l_head_vel', type=float, default=5, help='weight of the head angular velocity loss')
    parser.add_argument('--l_head_smooth', type=float, default=0.5,
                        help='weight of the head angular acceleration regularization')
    parser.add_argument('--l_head_trans', type=float, default=0.5,
                        help='weight of the head constraint during window transition')
    parser.add_argument('--no_constrain_prev', action='store_true',
                        help='do not constrain the generated previous motions')

    parser.add_argument('--use_context_audio_feat', action='store_true')

    parser.add_argument('--trunc_prob1', type=float, default=0.3, help='truncation probability for the first sample')
    parser.add_argument('--trunc_prob2', type=float, default=0.4, help='truncation probability for the second sample')

    parser.add_argument('--save_iter', type=int, default=10000, help='save model every x iterations')
    parser.add_argument('--val_iter', type=int, default=5000, help='validate every x iterations')
    parser.add_argument('--log_iter', type=int, default=50, help='log to tensorboard every x iterations')
    parser.add_argument('--log_smooth_win', type=int, default=50, help='smooth window for logging')


def add_additional_options(parser: argparse.ArgumentParser):
    args = parser.parse_known_args()[0]

    if 'Warmup' in args.scheduler:
        parser.add_argument('--warm_iter', type=int, default=5000)
        if args.scheduler == 'WarmupThenDecay':
            parser.add_argument('--cos_max_iter', type=int, default=120000)
            parser.add_argument('--min_lr_ratio', type=float, default=0.02)
