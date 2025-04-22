#!/usr/bin/env python3
"""
split_npz.py

move or copy .npz files from a source folder into train/ and test/ subfolders
with a random split.

usage:
    python split_npz.py /path/to/src /path/to/train /path/to/test \
        --train_frac 0.8 [--copy] [--seed 42]
"""

import os
import shutil
import argparse
import random
import sys

def parse_args():
    parser = argparse.ArgumentParser(
        description="randomly split .npz files into train and test folders (move by default)"
    )
    parser.add_argument("src_dir",
                        help="source directory containing .npz files")
    parser.add_argument("train_dir",
                        help="destination directory for training files")
    parser.add_argument("test_dir",
                        help="destination directory for testing files")
    parser.add_argument("--train_frac", "-f",
                        type=float,
                        default=0.8,
                        help="fraction of files to assign to train (default: 0.8)")
    parser.add_argument("--copy", "-c",
                        action="store_true",
                        help="copy files instead of moving")
    parser.add_argument("--seed", "-s",
                        type=int,
                        default=None,
                        help="random seed for reproducibility")
    return parser.parse_args()

def ensure_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        print(f"error creating directory {path}: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    args = parse_args()

    # set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)

    # ensure destination dirs exist
    ensure_dir(args.train_dir)
    ensure_dir(args.test_dir)

    # collect .npz and .pt files
    all_files = [f for f in os.listdir(args.src_dir) if f.lower().endswith(".npz") or f.lower().endswith(".pt")]
    if not all_files:
        print("no .npz or .pt files found in source directory.", file=sys.stderr)
        sys.exit(1)

    # shuffle and split
    random.shuffle(all_files)
    n_train = int(len(all_files) * args.train_frac)
    train_files = all_files[:n_train]
    test_files = all_files[n_train:]

    op = shutil.copy2 if args.copy else shutil.move

    # perform operation
    for fname in train_files:
        src = os.path.join(args.src_dir, fname)
        dst = os.path.join(args.train_dir, fname)
        op(src, dst)
    for fname in test_files:
        src = os.path.join(args.src_dir, fname)
        dst = os.path.join(args.test_dir, fname)
        op(src, dst)

    # summary
    action = "copied" if args.copy else "moved"
    print(f"{len(train_files)} files {action} to {args.train_dir}")
    print(f"{len(test_files)} files {action} to {args.test_dir}")

if __name__ == "__main__":
    main()
