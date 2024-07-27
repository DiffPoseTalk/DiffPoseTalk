import argparse
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from colorama import Fore, Back, Style
from tensorboardX import SummaryWriter
from torch.utils import data
from tqdm import tqdm

import options.se as options
import utils
from data import LmdbDatasetForSE, infinite_data_loader
from models import StyleEncoder


def train(args, model: StyleEncoder, train_loader, val_loader, optimizer, save_dir,
          writer=None):
    device = model.device
    save_dir.mkdir(parents=True, exist_ok=True)

    model.encoder.train()
    data_loader = infinite_data_loader(train_loader)

    loss_log = deque(maxlen=args.log_smooth_win)
    pbar = tqdm(range(args.max_iter + 1), dynamic_ncols=True)
    optimizer.zero_grad()
    for it in pbar:
        # Load data
        coef_pair = next(data_loader)
        coef_pair = [coef.to(device) for coef in coef_pair]

        # Forward
        feat_a = model(coef_pair[0])
        feat_b = model(coef_pair[1])

        loss = utils.nt_xent_loss(feat_a, feat_b, args.temperature)

        # Backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Logging
        loss_log.append(loss.item())
        description = f'Train loss: [{np.mean(loss_log):.3e}]'
        pbar.set_description(description)

        if it % args.log_iter == 0 and writer is not None:
            # write to tensorboard
            writer.add_scalar('train/loss', np.mean(loss_log), it)

        # Validation
        if (it % args.val_iter == 0 and it != 0) or it == args.max_iter:
            test(args, model, val_loader, it, 200, 'val', writer)

        # save model
        if (it % args.save_iter == 0 and it != 0) or it == args.max_iter:
            torch.save({
                'args': args,
                'encoder': model.encoder.state_dict(),
                'iter': it,
            }, save_dir / f'iter_{it:07}.pt')


@torch.no_grad()
def test(args, model: StyleEncoder, test_loader, current_iter, n_rounds=10, mode='val',
         writer=None):
    is_training = model.encoder.training
    device = model.device
    model.encoder.eval()

    loss_log = []
    for test_round in range(n_rounds):
        for coef_pair in test_loader:
            # Load data
            coef_pair = [coef.to(device) for coef in coef_pair]

            # Forward
            feat_a = model(coef_pair[0])
            feat_b = model(coef_pair[1])
            loss = utils.nt_xent_loss(feat_a, feat_b, args.temperature)

            # Logging
            loss_log.append(loss.item())

    description = f'(Iter {current_iter:>6}) {mode} loss: [{np.mean(loss_log):.3e}]'
    print(description)

    if writer is not None:
        # write to tensorboard
        writer.add_scalar(f'{mode}/loss', np.mean(loss_log), current_iter)

    if is_training:
        model.encoder.train()


def main(args, option_text=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load data
    data_root = Path(args.data_root)
    coef_stats_file: Path = args.stats_file
    if not coef_stats_file.is_absolute():
        coef_stats_file = data_root / coef_stats_file

    if args.mode == 'train':
        # Build model
        model = StyleEncoder(args).to(device)

        # Dataset
        train_dataset = LmdbDatasetForSE(data_root, args.data_root / 'train.txt', coef_stats_file, args.fps,
                                         args.n_motions,
                                         rot_repr=args.rot_repr, no_head_pose=args.no_head_pose)
        val_dataset = LmdbDatasetForSE(data_root, args.data_root / 'val.txt', coef_stats_file, args.fps, args.n_motions,
                                       rot_repr=args.rot_repr, no_head_pose=args.no_head_pose)
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                       num_workers=args.num_workers, pin_memory=True, drop_last=True,
                                       persistent_workers=True)
        val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                     num_workers=args.num_workers, drop_last=True)

        # Logging
        exp_dir = Path('experiments/SE') / f'{args.exp_name}-{datetime.now().strftime("%y%m%d_%H%M%S")}'
        log_dir = exp_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(log_dir))
        if option_text is not None:
            with open(log_dir / 'options.log', 'w') as f:
                f.write(option_text)
            writer.add_text('options', option_text)

        print(Back.RED + Fore.YELLOW + Style.BRIGHT + exp_dir.name + Style.RESET_ALL)
        print('model parameters: ', utils.count_parameters(model))

        # Train the model
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        train(args, model, train_loader, val_loader, optimizer, exp_dir / 'checkpoints', writer)
    else:
        # Build model
        checkpoint_path, _ = utils.get_model_path(args.exp_name, args.iter, 'SE')
        model_data = torch.load(checkpoint_path, map_location=device)
        model_args = model_data['args']
        model = StyleEncoder(model_args).to(device)
        model.encoder.load_state_dict(model_data['encoder'], strict=False)
        model.eval()

        # Dataset
        test_dataset = LmdbDatasetForSE(data_root, args.data_root / 'test.txt', coef_stats_file, args.fps,
                                        args.n_motions,
                                        rot_repr=args.rot_repr, no_head_pose=args.no_head_pose)
        test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers)

        # Test the model
        test(args, model, test_loader, args.iter, 100, 'test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--iter', type=int, default=100000, help='iteration to test')

    # Model
    options.add_model_options(parser)

    # Dataset
    options.add_data_options(parser)

    # Training
    options.add_training_options(parser)

    args = parser.parse_args()
    if args.mode == 'train':
        option_text = utils.get_option_text(args, parser)
    else:
        option_text = None

    main(args, option_text)
