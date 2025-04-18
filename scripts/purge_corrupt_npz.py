#!/usr/bin/env python3
"""
purge_corrupt_npz.py  –  delete unreadable .npz spectrograms

usage examples
--------------
# nuke bad files under two data roots
python purge_corrupt_npz.py /path/to/train_specs /path/to/val_specs

# dry‑run (just list what *would* be removed)
python purge_corrupt_npz.py data_root --dry

# feed explicit files
python purge_corrupt_npz.py *.npz
"""
import argparse, sys, numpy as np
from pathlib import Path
import zipfile, os, itertools, multiprocessing as mp

BAD = (zipfile.BadZipFile, ValueError, OSError)

def is_corrupt(p: Path) -> bool:
    try:
        np.load(p, mmap_mode='r')  # attempt lazy load
        return False
    except BAD:
        return True

def find_npz(paths):
    for path in paths:
        p = Path(path)
        if p.is_dir():
            yield from p.rglob("*.npz")
        elif p.suffix.lower() == ".npz":
            yield p

def worker(path):
    return path if is_corrupt(path) else None

def main():
    ap = argparse.ArgumentParser(description="delete corrupt .npz files")
    ap.add_argument("paths", nargs="+",
                    help="directory or file paths to scan")
    ap.add_argument("--dry", action="store_true",
                    help="only list corrupt files, do not delete")
    ap.add_argument("--jobs", type=int, default=mp.cpu_count(),
                    help="parallel workers (default: all cores)")
    args = ap.parse_args()

    files = list(find_npz(args.paths))
    if not files:
        print("no .npz files found", file=sys.stderr)
        return

    with mp.Pool(args.jobs) as pool:
        corrupt = [p for p in pool.imap_unordered(worker, files) if p]

    if not corrupt:
        print("all clear – no corrupt archives detected ✨")
        return

    if args.dry:
        print("corrupt .npz files:")
        for p in corrupt:
            print("  ", p)
        print(f"total: {len(corrupt)} (dry‑run, nothing deleted)")
    else:
        for p in corrupt:
            try:
                os.remove(p)
            except OSError as e:
                print(f"failed to remove {p}: {e}", file=sys.stderr)
        print(f"removed {len(corrupt)} corrupt .npz files")

if __name__ == "__main__":
    main()
