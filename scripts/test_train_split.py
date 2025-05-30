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
    parser.add_argument("--src_dir",
                        help="source directory containing .npz files")
    parser.add_argument("--train_dir",
                        help="destination directory for training files")
    parser.add_argument("--val_dir",
                        help="destination directory for validation files")
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
    ensure_dir(args.val_dir)

    # collect .npz and .pt files recursively
    all_files = []
    for root, _, files in os.walk(args.src_dir):
        for f in files:
            if f.lower().endswith(".npz") or f.lower().endswith(".pt"):
                all_files.append(os.path.relpath(os.path.join(root, f), args.src_dir))
    if not all_files:
        print("no .npz or .pt files found in source directory.", file=sys.stderr)
        sys.exit(1)

    # shuffle and split
    random.shuffle(all_files)
    n_train = int(len(all_files) * args.train_frac)
    train_files = all_files[:n_train]
    val_files = all_files[n_train:]

    op = shutil.copy2 if args.copy else shutil.move

    # perform operation
    for rel_path in train_files:
        src = os.path.join(args.src_dir, rel_path)
        dst = os.path.join(args.train_dir, rel_path)
        ensure_dir(os.path.dirname(dst))
        op(src, dst)
    for rel_path in val_files:
        src = os.path.join(args.src_dir, rel_path)
        dst = os.path.join(args.val_dir, rel_path)
        ensure_dir(os.path.dirname(dst))
        op(src, dst)

    # summary
    action = "copied" if args.copy else "moved"
    print(f"{len(train_files)} files {action} to {args.train_dir}")
    print(f"{len(val_files)} files {action} to {args.val_dir}")

if __name__ == "__main__":
    main()
