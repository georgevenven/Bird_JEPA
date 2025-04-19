#!/usr/bin/env python3
import argparse, numpy as np, torch, gzip
from pathlib import Path
import os

def main():
    p=argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("--dtype", choices=["fp32","fp16"], default="fp16")
    args=p.parse_args()
    paths=list(Path(args.src).rglob("*.npz"))
    for npz in paths:
        try:
            m = np.load(npz, mmap_mode='r')['arr_0']
            t = torch.from_numpy(m).to(torch.float16 if args.dtype=="fp16" else torch.float32)
            out = npz.with_suffix('.pt')
            torch.save(t, out)
            os.remove(npz)
        except Exception as e:
            print("skip", npz, e)

if __name__ == "__main__":
    main() 