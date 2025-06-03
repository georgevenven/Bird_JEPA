# src/models/birdjepa.py
# Refactored for Masked Autoencoder (MAE) pretraining on spectrogram stem features.
# Removes Teacher/JEPA components, adds a Decoder for reconstruction.
# local / global sparse‑aware blocks.  < 300 LoC

import math, torch, torch.nn as nn, warnings, copy
import torch.nn.functional as F
from dataclasses import dataclass, field
from .rope import rope, rope_2d
from .utils_misc import count_params
import random # For potential future masking strategies if needed
from typing import List, Tuple, Optional

print(">> birdjepa loaded from", __file__)

# ──────────────────────────────────────
#  config
# ──────────────────────────────────────
@dataclass
class BJConfig:
    # stem
    n_mels:  int       = 128
    # four convs:  F halved on first two,  T halved on *all* four
    conv_ch:  list[int] = field(default_factory=lambda: [32, 64, 128, 256])
    conv_k:   list[int] = field(default_factory=lambda: [5, 3, 3, 3])
    conv_str: list[tuple]=field(default_factory=lambda: [(2,1), (2,1), (2,1), (2,1)])

    # encoder
    layers: int = 8
    d_model: int = 192 # Encoder dimension
    n_heads: int = 6
    ff_mult: int = 4
    context_length: int = 256 # Add context_length (provide a default if desired)

    # decoder
    decoder_d_model: int = None # If None, use encoder d_model
    decoder_layers: int = 4
    decoder_n_heads: int = 4
    decoder_ff_mult: int = 4

    # patch size
    patch_size_f: int = 16 # Freq patch size
    patch_size_t: int = 16 # Time patch size - DEPRECATED (calculated from context_length)

# ──────────────────────────────────────
#  stem: conv stack, tracks Fp
# ──────────────────────────────────────
class Stem(nn.Sequential):
    def __init__(self, cfg: BJConfig):
        layers = []
        C_in = 1
        Fp = cfg.n_mels
        print(f"Initial input shape: (B, {C_in}, {Fp}, T)")
        for ch, k, (s_f, s_t) in zip(cfg.conv_ch, cfg.conv_k, cfg.conv_str):
            layers += [nn.Conv2d(C_in, ch, k, stride=(s_f, s_t), padding=k//2), nn.GELU()]
            print(f"After conv {len(layers)//2}: (B, {ch}, {Fp//s_f}, T//{s_t})")
            C_in = ch
            Fp //= s_f
        super().__init__(*layers)
        cfg.Fp = Fp  # remember final freq bins
        print(f"Final output shape: (B, {C_in}, {Fp}, T//{2**len(cfg.conv_str)})")

# ──────────────────────────────────────
#  SDPA-based attention blocks
# ──────────────────────────────────────
class SDPABlock(nn.Module):
    def __init__(self, d, h, ff_mult):
        super().__init__()
        self.n_heads = h
        self.d_head = d // h
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, d*ff_mult),
            nn.GELU(),
            nn.Linear(d*ff_mult, d)
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, grid_shape):
        B, T, D = x.shape
        x_norm = self.norm1(x)
        x_norm = rope_2d(x_norm, grid_shape)
        q = self.q_proj(x_norm).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x_norm).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        h = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        h = h.transpose(1, 2).contiguous().view(B, T, D)
        h = self.out_proj(h)
        x = x + h
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x

# ──────────────────────────────────────
#  encoder builder from pattern string
# ──────────────────────────────────────
def build_encoder(cfg: BJConfig):
    return nn.ModuleList([SDPABlock(cfg.d_model, cfg.n_heads, cfg.ff_mult)
                         for _ in range(cfg.layers)])

# ──────────────────────────────────────
#   Encoder
# ──────────────────────────────────────
class BirdJEPA(nn.Module):
    def __init__(self, cfg: BJConfig):
        super().__init__()

        # ----- conv stem: (B,1,F,T) -> (B,C,F',T') ------------------------
        stem_layers = []
        current_channels = 1
        for out_ch, k, s in zip(cfg.conv_ch, cfg.conv_k, cfg.conv_str):
            conv = nn.Conv2d(current_channels, out_ch, k, stride=s, padding=k//2)
            norm = nn.BatchNorm2d(out_ch)
            act = nn.GELU()
            stem_layers.extend([conv, norm, act])
            current_channels = out_ch
        self.stem = nn.Sequential(*stem_layers)
        self.stem_channels = current_channels # Store C

        # ----- figure out flattened dim dynamically -----------------------
        with torch.no_grad():
            dummy = torch.zeros(1, 1, cfg.n_mels, 256)     # 5 s clip → 256 frames
            C, Fp, Tp = self.stem(dummy).shape[1:]
        self.Fp = Fp
        self.Tp = Tp
        self.proj = nn.Linear(C, cfg.d_model, bias=False)  # per-patch linear

        # ----- transformer encoder ----------------------------------------
        self.core = build_encoder(cfg)         # ModuleList
        self._print_par_count()
        # ------------------------------------------------------------------

    def forward(self, spec):                 # spec (B,1,F,T)
        """ Encodes the full spectrogram, returns all token embeddings and stem output"""
        z = self.stem(spec)
        B, C, Fp, Tp = z.shape
        assert Fp == self.Fp and Tp == self.Tp, "Grid shape mismatch"

        z_flat = z.permute(0, 2, 3, 1).reshape(B, Fp * Tp, C) # Target for reconstruction
        x = self.proj(z_flat) # (B, Fp*Tp, D)

        # Apply encoder blocks to all tokens
        encoded_all = x
        for blk in self.core:                  # ← iterate manually
            encoded_all = blk(encoded_all, (Fp, Tp)) # RoPE applied inside blocks

        return encoded_all, self.Fp, self.Tp # Return all encoded, Fp, Tp as scalars

    def _print_par_count(self):
        m = count_params(self)
        print(f"[Encoder] {m:.2f}M parameters")

class SpectrogramDecoder(nn.Module):
    """
    Takes encoded visible tokens, uses mask tokens for masked positions,
    and reconstructs the *pixel values* for the masked patches.
    """
    def __init__(self, cfg: BJConfig, patch_size_f: int, patch_size_t: int, in_chans: int = 1):
        """
        Args:
            cfg: Configuration object (BJConfig).
            patch_size_f: Height of the original patch in pixels (frequency).
            patch_size_t: Width of the original patch in pixels (time).
            in_chans: Number of input channels (e.g., 1 for grayscale spectrogram).
        """
        super().__init__()
        self.decoder_d_model = cfg.decoder_d_model or cfg.d_model
        self.decoder_layers = cfg.decoder_layers
        self.decoder_n_heads = cfg.decoder_n_heads
        self.decoder_ff_mult = cfg.decoder_ff_mult
        self.d_model = cfg.d_model # Encoder output dim

        # Calculate the number of pixels per patch to predict
        self.patch_size_f = patch_size_f
        self.patch_size_t = patch_size_t
        self.in_chans = in_chans
        self.num_pixels_per_patch = self.in_chans * self.patch_size_f * self.patch_size_t

        # Project encoder output to decoder dimension if different
        self.enc_to_dec_proj = nn.Linear(self.d_model, self.decoder_d_model) if self.d_model != self.decoder_d_model else nn.Identity()

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.decoder_d_model))
        # Initialize mask token parameter - important for MAE stability
        torch.nn.init.normal_(self.mask_token, std=.02)
        # self._mask_token_init_done = False # Flag removed, init done here

        # Decoder blocks (using standard Transformer blocks with RoPE)
        decoder_block = SDPABlock(
            d=self.decoder_d_model,
            h=self.decoder_n_heads,
            ff_mult=self.decoder_ff_mult
        )
        self.decoder_blocks = nn.ModuleList([copy.deepcopy(decoder_block) for _ in range(self.decoder_layers)])

        # Final normalization and projection layer to predict pixels
        self.norm_out = nn.LayerNorm(self.decoder_d_model)
        # Head projects from decoder dimension to the flattened pixel count of a patch
        self.pixel_head = nn.Linear(self.decoder_d_model, self.num_pixels_per_patch)
        # Initialize the head layer - often beneficial
        # Example: Xavier uniform initialization
        torch.nn.init.xavier_uniform_(self.pixel_head.weight)
        torch.nn.init.zeros_(self.pixel_head.bias)


        self._print_par_count()

    def forward(self, encoded_vis, mask_tok, grid_shape):
        """
        Args:
            encoded_vis : (B, num_vis, D_enc) - Encoded visible tokens from encoder.
            mask_tok    : (B, Fp*Tp) bool     - True where token IS masked.
            grid_shape  : (Fp, Tp)            - Token grid dimensions.

        Returns:
            predictions for masked tokens : (total_masked, num_pixels_per_patch)
                                            Predicted pixel values (flattened) for each masked patch.
        """
        B, T_all = mask_tok.shape[0], mask_tok.shape[1]
        Fp, Tp = grid_shape
        assert T_all == Fp * Tp, "Mask shape mismatch"
        device = encoded_vis.device
        D_dec = self.decoder_d_model

        # Project visible tokens to decoder dimension
        encoded_vis = self.enc_to_dec_proj(encoded_vis) # (B, num_vis, D_dec)

        # ----- Prepare full sequence for decoder input -------------
        # RoPE positional embeddings will be applied inside the decoder blocks.
        # Create placeholder for the full sequence length.
        x_full = torch.zeros(B, T_all, D_dec, device=device, dtype=encoded_vis.dtype)
        vis_mask = ~mask_tok # Boolean mask: True where visible

        # Scatter visible tokens and mask tokens into the full sequence tensor
        # Need indices to handle potentially varying numbers of visible tokens per sample
        num_vis_per_sample = vis_mask.sum(dim=1)

        # Iterate through batch to correctly place visible tokens and mask tokens
        for b in range(B):
            num_vis = num_vis_per_sample[b].item()
            idx_vis = torch.nonzero(vis_mask[b]).squeeze(-1)
            idx_mask = torch.nonzero(mask_tok[b]).squeeze(-1)

            # Place encoded visible tokens
            if num_vis > 0: # Handle cases where a sample might be fully masked (unlikely but possible)
                 # Ensure slicing matches the actual number of visible tokens for this sample
                x_full[b, idx_vis] = encoded_vis[b, :num_vis]

            # Place mask tokens (cast to correct dtype if needed, e.g., for AMP)
            if idx_mask.numel() > 0: # Check if there are any masked tokens
                x_full[b, idx_mask] = self.mask_token.to(x_full.dtype)

        # ----- Decoder forward pass -------------------------------
        # Pass the full sequence through the decoder blocks
        decoded_all = x_full
        for blk in self.decoder_blocks:
            # Pass grid_shape for RoPE calculation inside the block
            decoded_all = blk(decoded_all, grid_shape)

        # ----- Project features of MASKED tokens to pixels --------
        # Apply final normalization
        decoded_all_norm = self.norm_out(decoded_all)

        # Select only the feature vectors corresponding to the MASKED tokens
        decoded_masked_norm = decoded_all_norm[mask_tok] # Shape: (total_masked, D_dec)

        # Predict pixel values using the final head
        pred_pixels_flat = self.pixel_head(decoded_masked_norm) # Shape: (total_masked, num_pixels_per_patch)

        return pred_pixels_flat

    def _print_par_count(self):
        m = count_params(self)
        print(f"[SpectrogramDecoder (Pixel)] {m:.2f}M parameters")

class Predictor(nn.Module):
    pass 

class DecoderPredictor(nn.Module):
    pass 