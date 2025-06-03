# ──────────────────────────────────────────────────────────────────────────────
# main.py
# top‑level launcher for BirdJEPA experiments
# call pattern:
#    python main.py pretrain --train_dir ...  (for contrastive / masked pre‑train)
#    python main.py finetune --train_spec_dir ... --val_spec_dir ...
#    python main.py infer    --val_spec_dir ... --output_dir ...
# Everything after the sub‑command is *verbatim* passed to that script.
# ──────────────────────────────────────────────────────────────────────────────
import argparse, importlib, runpy, sys, pathlib, subprocess, os

_ROOT = pathlib.Path(__file__).resolve().parent

# map sub‑commands → module / script
ENTRY_POINTS = {
    "pretrain": _ROOT / "src" / "pretrain.py",   # wraps trainer.ModelTrainer
    "finetune": _ROOT / "src" / "finetune.py",   # new fine‑tune + infer combo
    "infer"   : _ROOT / "src" / "finetune.py",   # run with --mode infer
}

def _exec(path: pathlib.Path, argv: list[str]) -> None:
    """
    Runs *another* python file in a fresh interpreter so its arg‑parser
    behaves exactly as if you called it from a shell script.
    """
    cmd = [sys.executable, str(path), *argv]
    os.execv(sys.executable, cmd)    # replaces current process

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BirdJEPA experiment launcher",
        usage=(
            "python main.py {pretrain,finetune,infer} [<script‑specific flags>]\n\n"
            "Examples\n"
            "  python main.py pretrain  --run_dir jepa_base --steps 500k --batch_size 64\n"
            "  python main.py finetune  --train_spec_dir data/train_specs "
            "                           --val_spec_dir data/val_specs\n"
            "  python main.py infer     --val_spec_dir data/val_specs "
            "                           --output_dir runs/ft_run"
        ),
        add_help=False)                                # defer -h to children
    parser.add_argument("mode", choices=ENTRY_POINTS.keys(),
                        help="Which stage to run.")
    args, remainder = parser.parse_known_args()

    target = ENTRY_POINTS[args.mode]

    # For the dedicated `infer` command we just re‑use finetune.py with --mode infer
    if args.mode == "infer":
        remainder = ["--mode", "infer", *remainder]

    _exec(target, remainder)

if __name__ == "__main__":
    main()