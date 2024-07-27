import argparse
from collections import deque, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
from colorama import Fore, Back, Style
from tensorboardX import SummaryWriter
from torch.utils import data
from tqdm import tqdm

import options.dpt as options
import utils
from data import LmdbDataset, infinite_data_loader
from models import DiffTalkingHead, StyleEncoder
from models.flame import FLAME, FLAMEConfig


def train(args, model: DiffTalkingHead, style_enc: Optional[StyleEncoder], train_loader, val_loader, optimizer,
          save_dir, scheduler=None, writer=None, flame=None):
    device = model.device
    save_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    data_loader = infinite_data_loader(train_loader)

    coef_stats = train_loader.dataset.coef_stats
    if coef_stats is not None:
        coef_stats = {x: coef_stats[x].to(device) for x in coef_stats}
    audio_unit = train_loader.dataset.audio_unit
    predict_head_pose = not args.no_head_pose

    loss_log = defaultdict(lambda: deque(maxlen=args.log_smooth_win))
    pbar = tqdm(range(args.max_iter + 1), dynamic_ncols=True)
    optimizer.zero_grad()
    for it in pbar:
        # Load data
        audio_pair, coef_pair, audio_stats = next(data_loader)
        audio_pair = [audio.to(device) for audio in audio_pair]
        coef_pair = [{x: coef_pair[i][x].to(device) for x in coef_pair[i]} for i in range(2)]
        motion_coef_pair = [
            utils.get_motion_coef(coef_pair[i], args.rot_repr, predict_head_pose) for i in range(2)
        ]  # (N, L, 50+x)

        # Use the shape coefficients from the first frame of the first clip as the condition
        if coef_pair[0]['shape'].ndim == 2:  # (N, 100)
            shape_coef = coef_pair[0]['shape'].clone().to(device)
        else:  # (N, L, 100)
            shape_coef = coef_pair[0]['shape'][:, 0].clone().to(device)

        # Extract style features
        if style_enc is not None:
            with torch.no_grad():
                style_pair = [style_enc(motion_coef_pair[i]) for i in range(2)]

        if args.use_context_audio_feat:
            # Extract audio features
            audio_feat = model.extract_audio_feature(torch.cat(audio_pair, dim=1), args.n_motions * 2)  # (N, 2L, :)

        loss_noise = 0
        loss_vert = 0
        loss_vel = torch.tensor(0, device=device)
        loss_smooth = torch.tensor(0, device=device)
        loss_head_angle = 0
        loss_head_vel = torch.tensor(0, device=device)
        loss_head_smooth = torch.tensor(0, device=device)
        loss_head_trans = 0
        for i in range(2):
            audio = audio_pair[i]  # (N, L_a)
            motion_coef = motion_coef_pair[i]  # (N, L, 50+x)
            style = style_pair[1 - i] if style_enc is not None else None
            batch_size = audio.shape[0]

            # truncate input audio and motion according to trunc_prob
            if (i == 0 and np.random.rand() < args.trunc_prob1) or (i != 0 and np.random.rand() < args.trunc_prob2):
                audio_in, motion_coef_in, end_idx = utils.truncate_motion_coef_and_audio(
                    audio, motion_coef, args.n_motions, audio_unit, args.pad_mode)
                if args.use_context_audio_feat and i != 0:
                    # use contextualized audio feature for the second clip
                    audio_in = model.extract_audio_feature(torch.cat([audio_pair[i - 1], audio_in], dim=1),
                                                           args.n_motions * 2)[:, -args.n_motions:]
            else:
                if args.use_context_audio_feat:
                    audio_in = audio_feat[:, i * args.n_motions:(i + 1) * args.n_motions]
                else:
                    audio_in = audio
                motion_coef_in, end_idx = motion_coef, None

            if args.use_indicator:
                if end_idx is not None:
                    indicator = torch.arange(args.n_motions, device=device).expand(batch_size, -1) < end_idx.unsqueeze(
                        1)
                else:
                    indicator = torch.ones(batch_size, args.n_motions, device=device)
            else:
                indicator = None

            # Inference
            if i == 0:
                noise, target, prev_motion_coef, prev_audio_feat = model(
                    motion_coef_in, audio_in, shape_coef, style, indicator=indicator)
                if end_idx is not None:  # was truncated, needs to use the complete feature
                    prev_motion_coef = motion_coef[:, -args.n_prev_motions:]
                    if args.use_context_audio_feat:
                        prev_audio_feat = audio_feat[:, args.n_motions - args.n_prev_motions:args.n_motions].detach()
                    else:
                        with torch.no_grad():
                            prev_audio_feat = model.extract_audio_feature(audio)[:, -args.n_prev_motions:]
                else:
                    prev_motion_coef = prev_motion_coef[:, -args.n_prev_motions:]
                    prev_audio_feat = prev_audio_feat[:, -args.n_prev_motions:]
            else:
                noise, target, _, _ = model(motion_coef_in, audio_in, shape_coef, style,
                                            prev_motion_coef, prev_audio_feat, indicator=indicator)

            loss_n, loss_v, loss_c, loss_s, loss_ha, loss_hc, loss_hs, loss_ht = utils.compute_loss(
                args, i == 0, shape_coef, motion_coef_in, noise, target, prev_motion_coef, coef_stats, flame, end_idx)
            loss_noise = loss_noise + loss_n / 2
            if args.target == 'sample' and args.l_vert > 0:
                loss_vert = loss_vert + loss_v / 2
            if args.target == 'sample' and args.l_vel > 0 and loss_c is not None:
                loss_vel = loss_vel + loss_c / 2
            if args.target == 'sample' and args.l_smooth > 0 and loss_s is not None:
                loss_smooth = loss_smooth + loss_s / 2
            if args.target == 'sample' and predict_head_pose and args.l_head_angle > 0:
                loss_head_angle = loss_head_angle + loss_ha / 2
            if args.target == 'sample' and predict_head_pose and args.l_head_vel > 0 and loss_hc is not None:
                loss_head_vel = loss_head_vel + loss_hc / 2
            if args.target == 'sample' and predict_head_pose and args.l_head_smooth > 0 and loss_hs is not None:
                loss_head_smooth = loss_head_smooth + loss_hs / 2
            if args.target == 'sample' and predict_head_pose and args.l_head_trans > 0 and loss_ht is not None:
                # no need to divide by 2 because it only applies to the second clip
                loss_head_trans = loss_head_trans + loss_ht

        loss_log['noise'].append(loss_noise.item())
        loss = loss_noise
        if args.target == 'sample' and args.l_vert > 0:
            loss_log['vert'].append(loss_vert.item())
            loss = loss + args.l_vert * loss_vert
        if args.target == 'sample' and args.l_vel > 0:
            loss_log['vel'].append(loss_vel.item())
            loss = loss + args.l_vel * loss_vel
        if args.target == 'sample' and args.l_smooth > 0:
            loss_log['smooth'].append(loss_smooth.item())
            loss = loss + args.l_smooth * loss_smooth
        if args.target == 'sample' and predict_head_pose and args.l_head_angle > 0:
            loss_log['head_angle'].append(loss_head_angle.item())
            loss = loss + args.l_head_angle * loss_head_angle
        if args.target == 'sample' and predict_head_pose and args.l_head_vel > 0:
            loss_log['head_vel'].append(loss_head_vel.item())
            loss = loss + args.l_head_vel * loss_head_vel
        if args.target == 'sample' and predict_head_pose and args.l_head_smooth > 0:
            loss_log['head_smooth'].append(loss_head_smooth.item())
            loss = loss + args.l_head_smooth * loss_head_smooth
        if args.target == 'sample' and predict_head_pose and args.l_head_trans > 0:
            loss_log['head_trans'].append(loss_head_trans.item())
            loss = loss + args.l_head_trans * loss_head_trans
        loss.backward()

        if it % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Logging
        loss_log['loss'].append(loss.item())
        description = f'Train loss: [N: {np.mean(loss_log["noise"]):.3e}'
        if args.target == 'sample' and args.l_vert > 0:
            description += f', V: {np.mean(loss_log["vert"]):.3e}'
        if args.target == 'sample' and args.l_vel > 0:
            description += f', C: {np.mean(loss_log["vel"]):.3e}'
        if args.target == 'sample' and args.l_smooth > 0:
            description += f', S: {np.mean(loss_log["smooth"]):.3e}'
        if args.target == 'sample' and predict_head_pose and args.l_head_angle > 0:
            description += f', HA: {np.mean(loss_log["head_angle"]):.3e}'
        if args.target == 'sample' and predict_head_pose and args.l_head_vel > 0:
            description += f', HC: {np.mean(loss_log["head_vel"]):.3e}'
        if args.target == 'sample' and predict_head_pose and args.l_head_smooth > 0:
            description += f', HS: {np.mean(loss_log["head_smooth"]):.3e}'
        if args.target == 'sample' and predict_head_pose and args.l_head_trans > 0:
            description += f', HT: {np.mean(loss_log["head_trans"]):.3e}'
        description += ']'
        pbar.set_description(description)

        if it % args.log_iter == 0 and writer is not None:
            # write to tensorboard
            writer.add_scalar('train/loss', np.mean(loss_log['loss']), it)
            writer.add_scalar('train/noise', np.mean(loss_log['noise']), it)
            if args.target == 'sample' and args.l_vert > 0:
                writer.add_scalar('train/vert', np.mean(loss_log['vert']), it)
            if args.target == 'sample' and args.l_vel > 0:
                writer.add_scalar('train/vel', np.mean(loss_log['vel']), it)
            if args.target == 'sample' and args.l_smooth > 0:
                writer.add_scalar('train/smooth', np.mean(loss_log['smooth']), it)
            if args.target == 'sample' and predict_head_pose and args.l_head_angle > 0:
                writer.add_scalar('train/head_angle', np.mean(loss_log['head_angle']), it)
            if args.target == 'sample' and predict_head_pose and args.l_head_vel > 0:
                writer.add_scalar('train/head_vel', np.mean(loss_log['head_vel']), it)
            if args.target == 'sample' and predict_head_pose and args.l_head_smooth > 0:
                writer.add_scalar('train/head_smooth', np.mean(loss_log['head_smooth']), it)
            if args.target == 'sample' and predict_head_pose and args.l_head_trans > 0:
                writer.add_scalar('train/head_trans', np.mean(loss_log['head_trans']), it)
            writer.add_scalar('opt/lr', optimizer.param_groups[0]['lr'], it)

        # update learning rate
        if scheduler is not None:
            if args.scheduler != 'WarmupThenDecay' or (args.scheduler == 'WarmupThenDecay' and it < args.cos_max_iter):
                scheduler.step()

        # save model
        if (it % args.save_iter == 0 and it != 0) or it == args.max_iter:
            torch.save({
                'args': args,
                'model': model.state_dict(),
                'iter': it,
            }, save_dir / f'iter_{it:07}.pt')

        # validation
        if (it % args.val_iter == 0 and it != 0) or it == args.max_iter:
            test(args, model, style_enc, val_loader, it, 200, 'val', writer, flame)


@torch.no_grad()
def test(args, model: DiffTalkingHead, style_enc: Optional[StyleEncoder], test_loader, current_iter, n_rounds=10,
         mode='val', writer=None, flame=None):
    is_training = model.training
    device = model.device
    model.eval()

    coef_stats = test_loader.dataset.coef_stats
    if coef_stats is not None:
        coef_stats = {x: coef_stats[x].to(device) for x in coef_stats}
    audio_unit = test_loader.dataset.audio_unit
    predict_head_pose = not args.no_head_pose

    loss_log = defaultdict(list)
    for test_round in range(n_rounds):
        for audio_pair, coef_pair, audio_stats in test_loader:
            audio_pair = [audio.to(device) for audio in audio_pair]
            coef_pair = [{x: coef_pair[i][x].to(device) for x in coef_pair[i]} for i in range(2)]
            motion_coef_pair = [
                utils.get_motion_coef(coef_pair[i], args.rot_repr, predict_head_pose) for i in range(2)
            ]  # (N, L, 50+x)

            # Use the shape coefficients from the first frame of the first clip as the condition
            if coef_pair[0]['shape'].ndim == 2:  # (N, 100)
                shape_coef = coef_pair[0]['shape'].clone().to(device)
            else:  # (N, L, 100)
                shape_coef = coef_pair[0]['shape'][:, 0].clone().to(device)

            # Extract style features
            if style_enc is not None:
                with torch.no_grad():
                    style_pair = [style_enc(motion_coef_pair[i]) for i in range(2)]

            if args.use_context_audio_feat:
                # Extract audio features
                audio_feat = model.extract_audio_feature(torch.cat(audio_pair, dim=1), args.n_motions * 2)  # (N, 2L, :)

            loss_noise = 0
            loss_vert = 0
            loss_vel = torch.tensor(0, device=device)
            loss_smooth = torch.tensor(0, device=device)
            loss_head_angle = 0
            loss_head_vel = torch.tensor(0, device=device)
            loss_head_smooth = torch.tensor(0, device=device)
            loss_head_trans = 0
            for i in range(2):
                audio = audio_pair[i]  # (N, L_a)
                motion_coef = motion_coef_pair[i]  # (N, L, 50+x)
                style = style_pair[1 - i] if style_enc is not None else None
                batch_size = audio.shape[0]

                # truncate input audio and motion according to trunc_prob
                if (i == 0 and np.random.rand() < args.trunc_prob1) or (i != 0 and np.random.rand() < args.trunc_prob2):
                    audio_in, motion_coef_in, end_idx = utils.truncate_motion_coef_and_audio(
                        audio, motion_coef, args.n_motions, audio_unit, args.pad_mode)
                    if args.use_context_audio_feat and i != 0:
                        # use contextualized audio feature for the second clip
                        audio_in = model.extract_audio_feature(torch.cat([audio_pair[i - 1], audio_in], dim=1),
                                                               args.n_motions * 2)[:, -args.n_motions:]

                else:
                    if args.use_context_audio_feat:
                        audio_in = audio_feat[:, i * args.n_motions:(i + 1) * args.n_motions]
                    else:
                        audio_in = audio
                    motion_coef_in, end_idx = motion_coef, None

                if args.use_indicator:
                    if end_idx is not None:
                        indicator = torch.arange(args.n_motions, device=device).expand(batch_size,
                                                                                       -1) < end_idx.unsqueeze(1)
                    else:
                        indicator = torch.ones(batch_size, args.n_motions, device=device)
                else:
                    indicator = None

                # Inference
                if i == 0:
                    noise, target, prev_motion_coef, prev_audio_feat = model(
                        motion_coef_in, audio_in, shape_coef, style, indicator=indicator)
                    if end_idx is not None:  # was truncated, needs to use the complete feature
                        prev_motion_coef = motion_coef[:, -args.n_prev_motions:]
                        if args.use_context_audio_feat:
                            prev_audio_feat = audio_feat[:, args.n_motions - args.n_prev_motions:args.n_motions]
                        else:
                            with torch.no_grad():
                                prev_audio_feat = model.extract_audio_feature(audio)[:, -args.n_prev_motions:]
                    else:
                        prev_motion_coef = prev_motion_coef[:, -args.n_prev_motions:]
                        prev_audio_feat = prev_audio_feat[:, -args.n_prev_motions:]
                else:
                    noise, target, _, _ = model(motion_coef_in, audio, shape_coef, style,
                                                prev_motion_coef, prev_audio_feat, indicator=indicator)

                loss_n, loss_v, loss_c, loss_s, loss_ha, loss_hc, loss_hs, loss_ht = utils.compute_loss(
                    args, i == 0, shape_coef, motion_coef_in, noise, target, prev_motion_coef, coef_stats, flame,
                    end_idx
                )
                loss_noise = loss_noise + loss_n / 2
                if args.target == 'sample' and args.l_vert > 0:
                    loss_vert = loss_vert + loss_v / 2
                if args.target == 'sample' and args.l_vel > 0 and loss_c is not None:
                    loss_vel = loss_vel + loss_c / 2
                if args.target == 'sample' and args.l_smooth > 0 and loss_s is not None:
                    loss_smooth = loss_smooth + loss_s / 2
                if args.target == 'sample' and predict_head_pose and args.l_head_angle > 0:
                    loss_head_angle = loss_head_angle + loss_ha / 2
                if args.target == 'sample' and predict_head_pose and args.l_head_vel > 0 and loss_hc is not None:
                    loss_head_vel = loss_head_vel + loss_hc / 2
                if args.target == 'sample' and predict_head_pose and args.l_head_smooth > 0 and loss_hs is not None:
                    loss_head_smooth = loss_head_smooth + loss_hs / 2
                if args.target == 'sample' and predict_head_pose and args.l_head_trans > 0 and loss_ht is not None:
                    # no need to divide by 2 because it only applies to the second clip
                    loss_head_trans = loss_head_trans + loss_ht

            loss_log['noise'].append(loss_noise.item())
            loss = loss_noise
            if args.target == 'sample' and args.l_vert > 0:
                loss_log['vert'].append(loss_vert.item())
                loss = loss + args.l_vert * loss_vert
            if args.target == 'sample' and args.l_vel > 0:
                loss_log['vel'].append(loss_vel.item())
                loss = loss + args.l_vel * loss_vel
            if args.target == 'sample' and args.l_smooth > 0:
                loss_log['smooth'].append(loss_smooth.item())
                loss = loss + args.l_smooth * loss_smooth
            if args.target == 'sample' and predict_head_pose and args.l_head_angle > 0:
                loss_log['head_angle'].append(loss_head_angle.item())
                loss = loss + args.l_head_angle * loss_head_angle
            if args.target == 'sample' and predict_head_pose and args.l_head_vel > 0:
                loss_log['head_vel'].append(loss_head_vel.item())
                loss = loss + args.l_head_vel * loss_head_vel
            if args.target == 'sample' and predict_head_pose and args.l_head_smooth > 0:
                loss_log['head_smooth'].append(loss_head_smooth.item())
                loss = loss + args.l_head_smooth * loss_head_smooth
            if args.target == 'sample' and predict_head_pose and args.l_head_trans > 0:
                loss_log['head_trans'].append(loss_head_trans.item())
                loss = loss + args.l_head_trans * loss_head_trans
            loss_log['loss'].append(loss.item())

    description = f'(Iter {current_iter:>6}) {mode} loss: [N: {np.mean(loss_log["noise"]):.3e}'
    if args.target == 'sample' and args.l_vert > 0:
        description += f', V: {np.mean(loss_log["vert"]):.3e}'
    if args.target == 'sample' and args.l_vel > 0:
        description += f', C: {np.mean(loss_log["vel"]):.3e}'
    if args.target == 'sample' and args.l_smooth > 0:
        description += f', S: {np.mean(loss_log["smooth"]):.3e}'
    if args.target == 'sample' and predict_head_pose and args.l_head_angle > 0:
        description += f', HA: {np.mean(loss_log["head_angle"]):.3e}'
    if args.target == 'sample' and predict_head_pose and args.l_head_vel > 0:
        description += f', HC: {np.mean(loss_log["head_vel"]):.3e}'
    if args.target == 'sample' and predict_head_pose and args.l_head_smooth > 0:
        description += f', HS: {np.mean(loss_log["head_smooth"]):.3e}'
    if args.target == 'sample' and predict_head_pose and args.l_head_trans > 0:
        description += f', HT: {np.mean(loss_log["head_trans"]):.3e}'
    description += ']'
    print(description)

    if writer is not None:
        # write to tensorboard
        writer.add_scalar(f'{mode}/loss', np.mean(loss_log['loss']), current_iter)
        writer.add_scalar(f'{mode}/noise', np.mean(loss_log['noise']), current_iter)
        if args.target == 'sample' and args.l_vert > 0:
            writer.add_scalar(f'{mode}/vert', np.mean(loss_log['vert']), current_iter)
        if args.target == 'sample' and args.l_vel > 0:
            writer.add_scalar(f'{mode}/vel', np.mean(loss_log['vel']), current_iter)
        if args.target == 'sample' and args.l_smooth > 0:
            writer.add_scalar(f'{mode}/smooth', np.mean(loss_log['smooth']), current_iter)
        if args.target == 'sample' and predict_head_pose and args.l_head_angle > 0:
            writer.add_scalar(f'{mode}/head_angle', np.mean(loss_log['head_angle']), current_iter)
        if args.target == 'sample' and predict_head_pose and args.l_head_vel > 0:
            writer.add_scalar(f'{mode}/head_vel', np.mean(loss_log['head_vel']), current_iter)
        if args.target == 'sample' and predict_head_pose and args.l_head_smooth > 0:
            writer.add_scalar(f'{mode}/head_smooth', np.mean(loss_log['head_smooth']), current_iter)
        if args.target == 'sample' and predict_head_pose and args.l_head_trans > 0:
            writer.add_scalar(f'{mode}/head_trans', np.mean(loss_log['head_trans']), current_iter)

    if is_training:
        model.train()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(args, option_text=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load data
    data_root = args.data_root
    coef_stats_file: Path = args.stats_file
    if not coef_stats_file.is_absolute():
        coef_stats_file = data_root / coef_stats_file

    # Loss
    if args.l_vert > 0 or args.l_vel > 0:
        flame = FLAME(FLAMEConfig).to(device)
    else:
        flame = None

    if args.mode == 'train':
        # Style Encoder
        if args.style_enc_ckpt:
            # Build model
            enc_model_data = torch.load(args.style_enc_ckpt, map_location=device)
            enc_model_args = utils.NullableArgs(enc_model_data['args'])
            style_enc = StyleEncoder(enc_model_args).to(device)
            style_enc.encoder.load_state_dict(enc_model_data['encoder'], strict=False)
            style_enc.eval()
        else:
            style_enc = None

        # Build model
        model = DiffTalkingHead(args, device=device)

        # Dataset
        train_dataset = LmdbDataset(data_root, data_root / 'train.txt', coef_stats_file, args.fps, args.n_motions,
                                    rot_repr=args.rot_repr)
        val_dataset = LmdbDataset(data_root, data_root / 'val.txt', coef_stats_file, args.fps, args.n_motions,
                                  rot_repr=args.rot_repr)
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                       num_workers=args.num_workers, pin_memory=True)
        val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_workers)

        # Logging
        exp_dir = Path('experiments/DPT') / f'{args.exp_name}-{datetime.now().strftime("%y%m%d_%H%M%S")}'
        log_dir = exp_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(log_dir))
        if option_text is not None:
            with open(log_dir / 'options.log', 'w') as f:
                f.write(option_text)
            writer.add_text('options', option_text)

        print(Back.RED + Fore.YELLOW + Style.BRIGHT + exp_dir.name + Style.RESET_ALL)
        print('model parameters: ', count_parameters(model))

        # Train the model
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        # scheduler
        if args.scheduler == 'Warmup':
            from scheduler import GradualWarmupScheduler
            scheduler = GradualWarmupScheduler(optimizer, 1, args.warm_iter)
        elif args.scheduler == 'WarmupThenDecay':
            from scheduler import GradualWarmupScheduler
            after_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.cos_max_iter - args.warm_iter,
                                                                   args.lr * args.min_lr_ratio)
            scheduler = GradualWarmupScheduler(optimizer, 1, args.warm_iter, after_scheduler)
        else:
            scheduler = None

        train(args, model, style_enc, train_loader, val_loader, optimizer, exp_dir / 'checkpoints', scheduler, writer,
              flame)
    else:
        # Load model
        checkpoint_path, exp_name = utils.get_model_path(args.exp_name, args.iter)
        model_data = torch.load(checkpoint_path, map_location=device)
        model_args = utils.NullableArgs(model_data['args'])

        # Style Encoder
        if model_args.style_enc_ckpt:
            # Build model
            enc_model_data = torch.load(model_args.style_enc_ckpt, map_location=device)
            enc_model_args = utils.NullableArgs(enc_model_data['args'])
            style_enc = StyleEncoder(enc_model_args).to(device)
            style_enc.encoder.load_state_dict(enc_model_data['encoder'], strict=False)
            style_enc.eval()
        else:
            style_enc = None

        # Build model
        model = DiffTalkingHead(model_args, device=device)
        model.load_state_dict(model_data['model'])
        model.eval()

        # Dataset
        test_dataset = LmdbDataset(data_root, data_root / 'test.txt', coef_stats_file, args.fps, args.n_motions,
                                   rot_repr=args.rot_repr)
        test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.num_workers)

        # Test the model
        test(model_args, model, style_enc, test_loader, args.iter, 200, 'test', None, flame)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DiffTalkingHead: Speech-Driven 3D Facial Animation')

    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--iter', type=int, default=100000, help='iteration to test')

    # Dataset
    options.add_data_options(parser)

    # Model
    options.add_model_options(parser)

    # Training
    options.add_training_options(parser)

    # Additional options depending on previous options
    options.add_additional_options(parser)

    args = parser.parse_args()
    if args.mode == 'train':
        option_text = utils.get_option_text(args, parser)
    else:
        option_text = None

    main(args, option_text)
