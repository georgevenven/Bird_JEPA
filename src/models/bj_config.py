from dataclasses import dataclass, field

@dataclass
class BJConfig:
    # ------- conv stem ------------------------------
    conv_ch:   list[int]   = field(default_factory=lambda: [32, 32, 32])
    conv_k:    list[int]   = field(default_factory=lambda: [5, 3, 3])
    conv_str:  list[tuple] = field(default_factory=lambda: [(2, 1), (2, 2), (2, 2)])

    # ------- encoder -------------------------------
    layers:       int = 2        # GL blocks total
    d_model:      int = 48
    n_heads:      int = 4
    ff_mult:      int = 4
    global_every: int = 2        # every n‑th block is global
    n_global:     int = 20       # #tokens granted global attention

    # ------- predictor -----------------------------
    pred_layers: int = 1
    pred_dim:    int = 48

    # ------- masking (pre‑train only) --------------
    mask_t:  int   = 50          # 50 frames = 250 ms if 5 ms hop
    mask_f:  int   = 4
    keep_p:  float = 0.25

    # ------- encoder pattern -----------------------------
    pattern: str = "local50,global100,local50,global100"