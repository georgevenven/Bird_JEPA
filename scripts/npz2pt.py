#!/usr/bin/env python3
import argparse, numpy as np, torch, os
from pathlib import Path

p = argparse.ArgumentParser(description="npz → pt (in‑place, deletes npz)")
p.add_argument("root", help="root directory full of .npz files")
p.add_argument("--dtype", choices=["fp32","fp16"], default="fp16")
args = p.parse_args()

paths = list(Path(args.root).rglob("*.npz"))
for f in paths:
    try:
        with np.load(f, mmap_mode="r") as z:
            arr = z["S"] if "S" in z else z["s"]       # spectro key
        t = torch.from_numpy(arr).to(
            torch.float16 if args.dtype=="fp16" else torch.float32)
        out = f.with_suffix(".pt")                     # same folder, new ext
        torch.save(t, out)
        os.remove(f)                                   # kill original .npz
    except Exception as e:
        print("skip", f, e)
