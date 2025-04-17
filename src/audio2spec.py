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
    hop: int
) -> np.ndarray:
    """
    Returns a (freq, time) float32 log‑magnitude spectrogram in dB.
    """
    S = librosa.stft(wav.astype(float), n_fft=n_fft, hop_length=hop, window="hann")
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max).astype(np.float32)
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

        self._setup_logging()
        self.skipped = mp.Value("i", 0)          # share across workers

        self.audio_files = self._gather_files()

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
            raise RuntimeError("No audio files found.")

        if self.take_n_random and self.take_n_random < len(files):
            files = random.sample(files, self.take_n_random)

        return files

    # ──────────────────────────────────────────────────────────────────────
    # public entry
    # ──────────────────────────────────────────────────────────────────────
    def run(self) -> None:
        pbar = tqdm(total=len(self.audio_files), desc="Processing files")

        if self.single:
            # ───── single threaded ─────
            for fp in self.audio_files:
                self._safe_process(fp)
                pbar.update()

        else:
            # ───── multi‑process pool ─────
            cpu = mp.cpu_count()
            with mp.get_context("spawn").Pool(processes=cpu, maxtasksperchild=100) as pool:

                def _done(_):
                    pbar.update()

                def _err(e):
                    logging.error(f"[pool] {e}")
                    self.skipped.value += 1
                    pbar.update()

                for fp in self.audio_files:
                    # simple memory guard
                    while psutil.virtual_memory().available < 1.2 * 1024**3:  # 1.2 GB
                        time.sleep(1)

                    pool.apply_async(
                        self._safe_process,
                        args=(fp,),
                        callback=_done,
                        error_callback=_err
                    )

                pool.close()
                pool.join()

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
        S = compute_spectrogram(wav, self.sr, self.n_fft, self.step)

        if S.shape[1] < self.min_timebins:
            self.skipped.value += 1
            return

        labels = np.zeros(S.shape[1], dtype=np.int32)

        out_name = fp.stem + ".npz"
        np.savez(self.dst_dir / out_name, s=S, labels=labels)

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
                   help="Where .npz outputs go.")

    p.add_argument("--step_size", type=int, default=160,
                   help="STFT hop length (samples at 32 kHz).")
    p.add_argument("--nfft",      type=int, default=1024,
                   help="FFT size.")
    p.add_argument("--take_n_random", type=int, default=None,
                   help="Pick N random files instead of the full set.")
    p.add_argument("--single_threaded",
                   choices=["true", "false", "1", "0", "yes", "no"],
                   default="true",
                   help="Force single‑thread. Default true.")
    args = p.parse_args()

    single = args.single_threaded.lower() in {"true", "1", "yes"}

    converter = WavToSpec(
        src_dir=args.src_dir,
        dst_dir=args.dst_dir,
        file_list=args.file_list,
        step_size=args.step_size,
        n_fft=args.nfft,
        take_n_random=args.take_n_random,
        single_threaded=single
    )
    converter.run()


if __name__ == "__main__":
    cli()