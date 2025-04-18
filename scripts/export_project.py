#!/usr/bin/env python
import argparse, zipfile, datetime as dt
from pathlib import Path

# tweak these or pass via CLI
RUN_DIR = Path("/home/george-vengrovski/Documents/projects/Bird_JEPA/runs/finetuned")
SRC_DIR = Path("/home/george-vengrovski/Documents/projects/Bird_JEPA/src")
OUT_DIR = Path("/home/george-vengrovski/Documents/projects/Bird_JEPA/zips")

def make_zip(run_dir: Path, src_dir: Path, out_dir: Path):
    if not run_dir.exists():  raise FileNotFoundError(run_dir)
    if not src_dir.exists():  raise FileNotFoundError(src_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts   = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{run_dir.name}_{ts}.zip"
    zpath = out_dir / name
    print("creating", zpath)

    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        # dump the finetune run
        for p in run_dir.rglob("*"):
            z.write(p, p.relative_to(run_dir.parent))
        # dump project src
        for p in src_dir.rglob("*"):
            z.write(p, p.relative_to(src_dir.parent))
    print("done.")
    return zpath

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=RUN_DIR)
    ap.add_argument("--src_dir", default=SRC_DIR)
    ap.add_argument("--out_dir", default=OUT_DIR)
    a = ap.parse_args()
    make_zip(Path(a.run_dir), Path(a.src_dir), Path(a.out_dir))
