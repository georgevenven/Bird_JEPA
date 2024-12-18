# birdjepa.py
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

def sinusoidal_positional_encoding(seq_len, dim):
    # standard positional encoding for seq_len x dim
    pe = torch.zeros(seq_len, dim)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class ViT(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, depth=1, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        # input_dim = D (freq bins)
        # input to forward: (B,D,T)
        # we transpose to (B,T,D) and project to hidden_dim
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(hidden_dim),
                nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True),
                nn.LayerNorm(hidden_dim),
                nn.Sequential(
                    nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
                    nn.Dropout(dropout),
                )
            ]))
        self.pos_embedding = None

    def forward(self, x):
        # x: (B,D,T)
        # transpose to (B,T,D)
        x = x.transpose(1,2)  # (B,T,D)
        B,T,D = x.shape

        if (self.pos_embedding is None) or (self.pos_embedding.shape[1] != T+1):
            pe = sinusoidal_positional_encoding(T+1, self.hidden_dim).to(x.device)
            self.pos_embedding = pe.unsqueeze(0)  # (1,T+1,hidden_dim)

        # project freq dimension to hidden_dim
        x = self.proj(x)  # (B,T,hidden_dim)

        # prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1) # (B,1,H)
        x = torch.cat((cls_tokens, x), dim=1) # (B,T+1,H)
        # add positional embeddings
        x = x + self.pos_embedding
        x = self.dropout(x)

        attention_outputs = []
        for (ln1, attn, ln2, mlp) in self.layers:
            x2 = ln1(x)
            attn_output, _ = attn(x2, x2, x2, need_weights=False)
            x = x + attn_output
            attention_outputs.append(attn_output)
            x2 = ln2(x)
            x = x + mlp(x2)

        # remove CLS token
        return x[:, 1:], attention_outputs  # (B,T,H), [list of attn]

class Predictor(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return self.net(x)

class BirdJEPA(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=256):
        super().__init__()
        self.context_encoder = ViT(input_dim=input_dim, hidden_dim=hidden_dim)
        self.target_encoder = ViT(input_dim=input_dim, hidden_dim=hidden_dim)
        
        # Initialize target encoder with same parameters as context encoder
        for param_q, param_k in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.copy_(param_q.data)
        
        # Freeze target encoder parameters after copying
        for p in self.target_encoder.parameters():
            p.requires_grad = False
            
        self.predictor = Predictor(dim=hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.ema_m = 0.90
        # No initial update_ema() call since parameters are already identical

    @torch.no_grad()
    def update_ema(self):
        for param_q, param_k in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data = param_k.data * self.ema_m + param_q.data * (1. - self.ema_m)

    def forward(self, context_spectrogram, target_spectrogram, use_no_mask=False):
        # context_spectrogram, target_spectrogram: (B,D,T)
        # masked: -1 in context
        mask = (context_spectrogram != -1.0).float()  # (B,D,T)
        context_clean = torch.where(context_spectrogram == -1.0,
                                    torch.zeros_like(context_spectrogram),
                                    context_spectrogram)

        if use_no_mask:
            context_repr, _ = self.context_encoder(context_clean)
        else:
            context_repr, _ = self.context_encoder(context_clean * mask)

        with torch.no_grad():
            target_repr, _ = self.target_encoder(target_spectrogram)

        pred = self.predictor(context_repr)  # (B,T,H)
        return pred, target_repr

    def training_step(self, context_spectrogram, target_spectrogram):
        pred, target = self.forward(context_spectrogram, target_spectrogram)
        # identify masked positions from target_spectrogram
        # target_spectrogram has original values at masked positions
        # so masked tokens: where any freq bin !=0
        masked_positions = (target_spectrogram != 0).any(dim=1, keepdim=True)  # (B,1,T)
        loss = ((pred - target)**2 * masked_positions.transpose(1,2)).mean()
        return loss

    def train_forward(self, spec):
        # spec: (B,D,T) with -1.0 at masked positions
        mask = (spec == -1.0)
        
        # The target_spectrogram and context_spectrogram are already correctly masked from the data loader
        context_spectrogram = torch.where(spec == -1.0, torch.zeros_like(spec), spec)
        target_spectrogram = spec  # Use the input directly since it's already masked correctly

        # encode context
        context_repr, intermediate_outputs = self.context_encoder(context_spectrogram)
        pred = self.predictor(context_repr)       
        decoded_pred = self.decoder(pred)         
        decoded_pred = decoded_pred.transpose(1,2)
        return decoded_pred, mask, target_spectrogram, {"layer_outputs": torch.stack(intermediate_outputs, dim=0)}

    def mse_loss(self, predictions, spec, mask, intermediate_layers=None, vocalization=None):
        # predictions, spec, mask: (B,D,T)
        masked_loss = ((predictions - spec)**2 * mask.float()).mean()
        with torch.no_grad():
            masked_seq_acc = masked_loss
            unmasked_seq_acc = ((predictions - spec)**2 * (~mask).float()).mean()
        return masked_loss, masked_seq_acc, unmasked_seq_acc
