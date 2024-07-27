import argparse
from pathlib import Path


def add_model_options(parser: argparse.ArgumentParser):
    parser.add_argument('--rot_repr', type=str, default='aa', choices=['aa'])
    parser.add_argument('--no_head_pose', action='store_true', help='do not use head pose')

    # transformer
    parser.add_argument('--feature_dim', type=int, default=128, help='dimension of the hidden feature')
    parser.add_argument('--n_heads', type=int, default=4, help='number of attention heads')
    parser.add_argument('--n_layers', type=int, default=4, help='number of encoder/decoder layers')
    parser.add_argument('--mlp_ratio', type=int, default=4, help='ratio of the hidden dimension of the MLP')

    # sequence
    parser.add_argument('--n_motions', type=int, default=100, help='number of motions in a sequence')
    parser.add_argument('--fps', type=int, default=25, help='frame per second')


def add_data_options(parser: argparse.ArgumentParser):
    parser.add_argument('--data_root', type=Path, default=Path('datasets/HDTF_TFHP/lmdb'),
                        help='dataset path')
    parser.add_argument('--stats_file', type=Path, default=Path('stats_train.npz'))
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataloader')


def add_training_options(parser: argparse.ArgumentParser):
    parser.add_argument('--exp_name', type=str, default='HDTF_TFHP', help='experiment name')
    parser.add_argument('--max_iter', type=int, default=100000, help='max number of iterations')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--temperature', type=float, default=0.1, help='temperature for contrastive loss')

    parser.add_argument('--save_iter', type=int, default=2000, help='save model every x iterations')
    parser.add_argument('--val_iter', type=int, default=2000, help='validate every x iterations')
    parser.add_argument('--log_iter', type=int, default=50, help='log to tensorboard every x iterations')
    parser.add_argument('--log_smooth_win', type=int, default=50, help='smooth window for logging')
