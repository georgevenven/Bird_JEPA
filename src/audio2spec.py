# ──────────────────────────────────────────────────────────────────────────────
# audio2spec.py  ‑  simple .wav /.mp3/.ogg ➜ spectrogram (.npz) converter
# ──────────────────────────────────────────────────────────────────────────────
import os, json, time, gc, argparse, logging, random, psutil
import multiprocessing as mp
from pathlib import Path

import numpy as np
import librosa
import librosa.display                       # noqa: F401  (kept for future plots)
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════════════════════
# helper: STFT → log‑magnitude
# ══════════════════════════════════════════════════════════════════════════════
def compute_spectrogram(
    wav: np.ndarray,
    sr: int,
    n_fft: int,
    hop: int,
    *,
    mel: bool,
    n_mels: int
) -> np.ndarray:
    """
    Returns log‑magnitude spectrogram in **dB**.
    • linear STFT  → shape (n_fft//2 + 1, T)   (default 513 × T for n_fft=1024)  
    • mel filter‑bank → shape (n_mels, T)
    """
    if mel:
        S = librosa.feature.melspectrogram(
            y=wav.astype(float),
            sr=sr,
            n_fft=n_fft,
            hop_length=hop,
            power=2.0,         # power‑spectrogram
            n_mels=n_mels,
            fmin=20,
            fmax=sr // 2,
        )
    else:
        S = np.abs(
            librosa.stft(
                wav.astype(float),
                n_fft=n_fft,
                hop_length=hop,
                window="hann",
            )
        ) ** 2

    S_db = librosa.power_to_db(S, ref=np.max).astype(np.float32)
    return S_db


# ══════════════════════════════════════════════════════════════════════════════
# main worker class
# ══════════════════════════════════════════════════════════════════════════════
class WavToSpec:
    """
    Convert a directory (or explicit list) of audio files to .npz spectrograms.
    Keys inside the .npz **match what BirdSpectrogramDataset expects**:
        s       -> float32 (F,T)   log spectrogram
        labels  -> int32   (T,)    all zeros (placeholder)
    """

    def __init__(
        self,
        src_dir: str | None,
        dst_dir: str,
        *,
        file_list: str | None = None,
        step_size: int = 160,
        n_fft: int = 1024,
        sr: int = 32_000,
        take_n_random: int | None = None,
        single_threaded: bool = True,
        min_len_ms: int = 25,
        min_timebins: int = 25,
        fmt: str = "pt",
        mel: bool = True,
        n_mels: int = 128,
        json_path: str | None = None,
    ) -> None:
        self.src_dir = Path(src_dir) if src_dir is not None else None
        self.dst_dir = Path(dst_dir)
        self.dst_dir.mkdir(parents=True, exist_ok=True)

        self.file_list = Path(file_list) if file_list else None
        self.step = step_size
        self.n_fft = n_fft
        self.sr = sr
        self.take_n_random = take_n_random
        self.single = single_threaded
        self.min_len_ms = min_len_ms
        self.min_timebins = min_timebins
        self.fmt = fmt
        self.use_mel = mel
        self.n_mels = n_mels

        self._setup_logging()
        mgr = mp.Manager()
        self.skipped = mgr.Value("i", 0)  # share across workers, manager-backed

        self.audio_files = self._gather_files()

        # Build label map if json_path is provided
        if json_path is not None:
            self.lab_map = {}
            p = Path(json_path)
            json_files = [p] if p.is_file() else list(p.glob("*.json"))

            for jfp in json_files:
                text = jfp.read_text()
                # Allow [ {...}, {...} ]    OR    {"filename": ...}    OR   NDJSON
                to_parse = text.strip()
                items = []
                if to_parse.startswith('['):                       # big list
                    items = json.loads(to_parse)
                elif to_parse.startswith('{'):                     # single object
                    items = [json.loads(to_parse)]
                else:                                              # NDJSON
                    items = [json.loads(line) for line in to_parse.splitlines() if line.strip()]

                for jo in items:
                    fname = jo["filename"]
                    hop_ms = 1e3 * self.step / self.sr
                    tmp = []
                    for lab, spans in jo.get("syllable_labels", {}).items():
                        for on, off in spans:
                            tb_on  = int(round(on  * 1e3 / hop_ms))
                            tb_off = int(round(off * 1e3 / hop_ms))
                            tmp.append((int(lab), tb_on, tb_off))
                    self.lab_map[Path(fname).stem] = tmp            # key on *stem*
        else:
            self.lab_map = {}

    def __getstate__(self):
        st = self.__dict__.copy()
        st.pop("skipped", None)  # mp.Value isn't picklable
        return st

    # ──────────────────────────────────────────────────────────────────────
    # misc
    # ──────────────────────────────────────────────────────────────────────
    def _setup_logging(self) -> None:
        logging.basicConfig(
            filename="error_log.log",
            level=logging.ERROR,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def _gather_files(self) -> list[Path]:
        if self.file_list:
            files = [Path(line.strip()) for line in self.file_list.read_text().splitlines() if line.strip()]
        else:
            exts = (".wav", ".mp3", ".ogg")
            files = [
                Path(root) / f
                for root, _, fs in os.walk(self.src_dir)
                for f in fs if f.lower().endswith(exts)
            ]

        if not files:
            print("no audio files matched ‑ nothing to do.")
            return []

        if self.take_n_random and self.take_n_random < len(files):
            files = random.sample(files, self.take_n_random)

        return files

    # ──────────────────────────────────────────────────────────────────────
    # public entry
    # ──────────────────────────────────────────────────────────────────────
    def run(self) -> None:
        if not self.audio_files:
            return                       # exit 0, no fuss
        pbar = tqdm(total=len(self.audio_files), desc="processing files")

        if self.single:
            # ───── single threaded ─────
            for fp in self.audio_files:
                self._safe_process(fp)
                pbar.update()

        else:
            # ───── multi‑process pool ─────
            ctx_name = "fork" if mp.get_start_method(allow_none=True) != "spawn" else "spawn"
            ctx = mp.get_context(ctx_name)
            cpu = ctx.cpu_count()
            mgr = mp.Manager()
            failures = mgr.list()
            with ctx.Pool(processes=cpu, maxtasksperchild=100) as pool:

                def _done(_):
                    pbar.update()

                def _err(e, fp=None):
                    logging.error(f"[pool] {fp}: {e}")
                    failures.append(str(fp))
                    self.skipped.value += 1
                    pbar.update()

                for fp in self.audio_files:
                    # simple memory guard
                    while psutil.virtual_memory().available < 1.2 * 1024**3:  # 1.2 GB
                        time.sleep(1)

                    pool.apply_async(
                        self._safe_process,
                        args=(fp,),
                        callback=_done,
                        error_callback=lambda e, fp=fp: _err(e, fp)
                    )

                pool.close()
                pool.join()

            # Retry failed files single-threaded
            if failures:
                print("retrying failed files single‑thread…")
                for fp in failures:
                    self._safe_process(Path(fp))

        pbar.close()
        print(f"Total processed: {len(self.audio_files) - self.skipped.value}")
        print(f"Total skipped  : {self.skipped.value}")

    # ──────────────────────────────────────────────────────────────────────
    # helpers
    # ──────────────────────────────────────────────────────────────────────
    def _safe_process(self, fp: Path) -> None:
        try:
            self._process_file(fp)
        except Exception as e:
            logging.error(f"{fp}: {e}")
            self.skipped.value += 1

    def _process_file(self, fp: Path) -> None:
        # ─── read ────────────────────────────────────────────────────
        wav, sr_native = librosa.load(fp, sr=self.sr, mono=True)
        if len(wav) / sr_native * 1_000 < self.min_len_ms:
            self.skipped.value += 1
            return

        # ─── spectrogram ─────────────────────────────────────────────
        S = compute_spectrogram(
                wav, self.sr, self.n_fft, self.step,
                mel=self.use_mel, n_mels=self.n_mels)

        if S.shape[1] < self.min_timebins:
            self.skipped.value += 1
            return

        labels = np.zeros(S.shape[1], dtype=np.int32)
        for lab, tb_on, tb_off in self.lab_map.get(fp.stem, []):
            labels[tb_on:tb_off] = lab

        if self.fmt == "pt":
            import torch
            out = self.dst_dir / (fp.stem + ".pt")
            torch.save({"s": torch.as_tensor(S).to(torch.float16),
                        "labels": torch.as_tensor(labels)}, out)
        else:  # npz (uncompressed)
            out = self.dst_dir / (fp.stem + ".npz")
            np.savez(out, s=S.astype(np.float32), labels=labels)

        # free memory fast in workers
        del wav, S, labels
        gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
def cli() -> None:
    p = argparse.ArgumentParser(
        description="Convert audio → log‑spectrogram .npz (no JSON, no filtering).")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--src_dir",   type=str,
                     help="Root folder with wav/mp3/ogg files (searched recursively).")
    grp.add_argument("--file_list", type=str,
                     help="Text file with absolute/relative paths, one per line.")
    p.add_argument("--dst_dir",  type=str, required=True,
                   help="Where outputs go.")
    p.add_argument("--format", choices=["pt","npz"], default="pt",
                   help="output format (default: pt, fp16)")

    p.add_argument("--step_size", type=int, default=625,
                   help="STFT hop length (samples at 32 kHz).")
    p.add_argument("--nfft",      type=int, default=1024,
                   help="FFT size.")
    p.add_argument("--take_n_random", type=int, default=None,
                   help="Pick N random files instead of the full set.")
    p.add_argument("--single_threaded",
                   choices=["true", "false", "1", "0", "yes", "no"],
                   default="true",
                   help="Force single‑thread. Default true.")
    mel_grp = p.add_mutually_exclusive_group()
    mel_grp.add_argument("--mel", action="store_true",
                         help="Output log‑mel (default).")
    mel_grp.add_argument("--linear", action="store_true",
                         help="Output linear‑frequency STFT bins.")
    p.add_argument("--n_mels", type=int, default=128,
                   help="Number of mel bands (default: 128)")
    p.add_argument("--json_path", type=str, default=None,
                   help="Directory containing label JSON files (optional)")
    args = p.parse_args()

    single = args.single_threaded.lower() in {"true", "1", "yes"}

    converter = WavToSpec(
        src_dir=args.src_dir,
        dst_dir=args.dst_dir,
        file_list=args.file_list,
        step_size=args.step_size,
        n_fft=args.nfft,
        take_n_random=args.take_n_random,
        single_threaded=single,
        fmt=args.format,
        mel=not args.linear,
        n_mels=args.n_mels,
        json_path=args.json_path,
    )
    converter.run()


if __name__ == "__main__":
    cli()