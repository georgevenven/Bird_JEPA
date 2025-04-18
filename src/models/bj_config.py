# src/models/bj_config.py
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class BJConfig:
    # ───────── conv stem ─────────
    n_mels:    int            = 513                 # input freq bins
    conv_ch:   List[int]      = field(default_factory=lambda: [32, 32, 32])
    conv_k:    List[int]      = field(default_factory=lambda: [5, 3, 3])
    conv_str:  List[Tuple[int,int]] = field(
                     default_factory=lambda: [(2,1), (2,2), (2,2)])
    # freq stride = 2·2·2 = 8   ;   time stride = 1·2·2 = 4

    # ───────── transformer ───────
    pattern:   str            = "local64,global128,local64,global128"
    d_model:   int            = 96
    n_heads:   int            = 4          # d_model must be divisible by n_heads
    ff_mult:   int            = 4          # MLP hidden = ff_mult·d_model

    # ───────── predictor (unused in finetune) ─────
    pred_dim:  int            = 96
    pred_layers:int           = 1

    # ───────── pre‑train masking defaults (unchanged) ─────
    mask_t:    int            = 50
    mask_f:    int            = 4
    keep_p:    float          = 0.25
