#!/usr/bin/env python3
# download_xc.py  –  bulk Xeno‑canto fetcher + metadata archiver
# --------------------------------------------------------------------
import argparse, csv, json, os, random, sys, time
from pathlib import Path
from typing import Dict, Set

import psutil           # pip install psutil
import requests
from tqdm import tqdm

API_URL = "https://www.xeno-canto.org/api/2/recordings"
SPACE_GB_LIMIT = 1_000    # stop when < 1 TB left (≈ 1000 GB)

# --------------------------------------------------------------------
def freespace_gb(path: Path) -> float:
    """Return free disk space in GB for the volume hosting *path*."""
    return psutil.disk_usage(path).free / 1_073_741_824   # 1024**3


def fetch_page(page: int, query: str) -> Dict:
    r = requests.get(API_URL, params={"query": query, "page": page}, timeout=30)
    r.raise_for_status()
    return r.json()


def write_jsonl(out: Path, rec: Dict) -> None:
    with out.open("a", encoding="utf‑8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def append_csv(out: Path, row: Dict, fieldnames) -> None:
    new_file = not out.exists()
    with out.open("a", newline="", encoding="utf‑8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            w.writeheader()
        w.writerow(row)


# --------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description="Download random Xeno‑canto recordings and store metadata")
    p.add_argument("--out_dir", required=True,
                   help="Root folder for MP3s + metadata files")
    p.add_argument("--num_recordings", type=int, default=None,
                   help="Limit (random sample). Omit → grab all.")
    p.add_argument("--quality", default="A,B,C",
                   help='Comma‑separated list of grades to fetch (e.g. "A,B,C"). '
                        'Leave empty to disable the quality filter.')
    p.add_argument("--min_free_gb", type=int, default=SPACE_GB_LIMIT,
                   help="Stop when free space drops below this (GB).")
    args = p.parse_args()

    out_root = Path(args.out_dir).expanduser()
    mp3_dir  = out_root / "mp3"
    mp3_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_root / "recordings.jsonl"
    csv_path   = out_root / "meta_index.csv"

    # ----------------------------------------------------------------
    # resume support – read already downloaded IDs
    done: Set[str] = set()
    if csv_path.exists():
        with csv_path.open() as f:
            for line in csv.reader(f):
                if line and line[0].isdigit():
                    done.add(line[0])

    print(f"[resume] {len(done)} recordings already present.")

    # ----------------------------------------------------------------
    grades = [g.strip().upper() for g in args.quality.split(",") if g.strip()]
    if not grades:            # user gave "", so no quality filter at all
        grades = [""]

    # pre‑flight step: ask the API once per grade to know page counts
    grade_info = {}
    for g in grades:
        # ── build API query string ───────────────────────────
        if g.upper() in {"", "*", "ANY"}:
            q = "*"                # unrestricted, no 'q:' prefix
        else:
            q = f"q:{g}"           # quality filter
        meta = fetch_page(1, q)
        grade_info[g] = {
            "pages":  int(meta["numPages"]),
            "total":  int(meta["numRecordings"]),
        }
        print(f"[api] grade '{g or 'ALL'}' → {meta['numRecordings']} recs "
              f"across {meta['numPages']} pages")

    target = args.num_recordings or sum(v["total"] for v in grade_info.values())
    fieldnames = (
        "id", "gen", "sp", "en", "length", "q", "file", "file_local",
        "sampling_rate", "bitrate"
    )

    pbar = tqdm(total=target, desc="downloaded", initial=len(done))
    tries = 0
    MAX_TRIES = sum(v["pages"] for v in grade_info.values()) * 3

    while len(done) < target and tries < MAX_TRIES:
        if freespace_gb(out_root) < args.min_free_gb:
            print(f"\n[stop] free space < {args.min_free_gb}\u00a0GB – aborting.")
            break

        tries += 1
        # pick a random grade bucket first, then a random page inside it
        g = random.choice(grades)
        pages_g = grade_info[g]["pages"]
        page = random.randint(1, pages_g)
        # ── build API query string ───────────────────────────
        if g.upper() in {"", "*", "ANY"}:
            query = "*"                # unrestricted, no 'q:' prefix
        else:
            query = f"q:{g}"           # quality filter
        try:
            data = fetch_page(page, query)
            for rec in data["recordings"]:
                rid = rec["id"]
                if rid in done:
                    continue
                mp3_url = rec["file"]
                if not mp3_url:
                    continue

                local_path = mp3_dir / f"{rid}.mp3"
                # ---------- download ---------------------------------
                try:
                    with requests.get(mp3_url, stream=True, timeout=30) as r:
                        r.raise_for_status()
                        with local_path.open("wb") as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
                except Exception as e:
                    print(f"[warn] failed {rid}: {e}")
                    continue

                # ---------- save metadata ----------------------------
                write_jsonl(jsonl_path, rec)
                meta_row = {
                    "id": rid,
                    "gen": rec.get("gen"),
                    "sp": rec.get("sp"),
                    "en": rec.get("en"),
                    "length": rec.get("length"),
                    "q": rec.get("q"),
                    "file": mp3_url,
                    "file_local": str(local_path),
                    "sampling_rate": rec.get("smp", ""),
                    "bitrate": rec.get("bitrate", ""),
                }
                append_csv(csv_path, meta_row, fieldnames)
                done.add(rid)
                pbar.update(1)
                if len(done) >= target:
                    break

        except Exception as e:
            print(f"[warn] error fetching page {page}: {e}")
            time.sleep(2)

    pbar.close()
    print(f"\nFinished: {len(done):,} recordings archived at {mp3_dir}")


if __name__ == "__main__":
    main()
