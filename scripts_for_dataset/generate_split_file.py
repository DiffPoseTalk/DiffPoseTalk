import argparse
import random
from collections import defaultdict
from pathlib import Path


def main(key_file, output_dir, seed=None):
    output_dir = Path(output_dir)

    with open(key_file, 'r') as f:
        all_keys = [line.strip() for line in f]

    key_dict = defaultdict(list)
    for key in all_keys:
        person, video_id = key.split('/')
        key_dict[person].append(video_id)

    all_keys = list(key_dict.keys())

    if seed is not None:
        random.seed(seed)
    random.shuffle(all_keys)
    # train_keys = sorted(all_keys[:int(len(all_keys) * 0.8)])
    # val_keys = sorted(all_keys[int(len(all_keys) * 0.8): int(len(all_keys) * 0.9)])
    # test_keys = sorted(all_keys[int(len(all_keys) * 0.9):])
    train_keys = sorted(all_keys[:-128])
    val_keys = sorted(all_keys[-128:-64])
    test_keys = sorted(all_keys[-64:])

    with open(output_dir / 'train.txt', 'w') as f:
        for key in train_keys:
            for video_id in key_dict[key]:
                f.write(f'{key}/{video_id}\n')

    with open(output_dir / 'val.txt', 'w') as f:
        for key in val_keys:
            for video_id in key_dict[key]:
                f.write(f'{key}/{video_id}\n')

    with open(output_dir / 'test.txt', 'w') as f:
        for key in test_keys:
            for video_id in key_dict[key]:
                f.write(f'{key}/{video_id}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('key_file')
    parser.add_argument('output_dir')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()
    main(args.key_file, args.output_dir, args.seed)
