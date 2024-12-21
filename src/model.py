# birdjepa.py
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from custom_transformer import CustomEncoderBlock

class ViT(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, depth=3, num_heads=4, mlp_ratio=4.0, dropout=0.1, max_len=512):
        super().__init__()
        # input_dim = D (freq bins)
        # input to forward: (B,D,T)
        # we transpose to (B,T,D) and project to hidden_dim
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            CustomEncoderBlock(
                d_model=hidden_dim,
                num_heads=num_heads,
                ffn_dim=int(hidden_dim * mlp_ratio),
                dropout=dropout,
                pos_enc_type="relative",
                length=max_len + 1  # +1 for CLS token
            ) for _ in range(depth)
        ])

    def forward(self, x):
        # x: (B,D,T)
        # transpose to (B,T,D)
        x = x.transpose(1,2)  # (B,T,D)
        B,T,D = x.shape

        # project freq dimension to hidden_dim
        x = self.proj(x)  # (B,T,hidden_dim)

        # prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1) # (B,1,H)
        x = torch.cat((cls_tokens, x), dim=1) # (B,T+1,H)
        
        x = self.dropout(x)

        layer_outputs = []
        for block in self.layers:
            block_output = block(x)
            x = block_output['feed_forward_output']
            layer_outputs.append(x[:, 1:])  # Store output excluding CLS token

        return x[:, 1:], layer_outputs  # (B,T,H), [list of layer outputs]

class Predictor(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return self.net(x)

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class PredictorViT(nn.Module):
    def __init__(self, 
                 hidden_dim=384,  
                 depth=6,         
                 num_heads=6, 
                 mlp_ratio=4.0, 
                 dropout=0.1,
                 max_len=512):   
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.mask_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        self.layers = nn.ModuleList([
            CustomEncoderBlock(
                d_model=hidden_dim,
                num_heads=num_heads,
                ffn_dim=int(hidden_dim * mlp_ratio),
                dropout=dropout,
                pos_enc_type="relative",
                length=max_len
            ) for _ in range(depth)
        ])

    def forward(self, x, mask):
        # x: (B,T,H) context embeddings
        # mask: (B,T) boolean mask indicating masked positions
        B, T, H = x.shape
        assert x.shape == (B, T, H), f"Input shape wrong: {x.shape}"
        assert mask.shape == (B, T), f"Mask shape wrong: {mask.shape}"
        
        # Expand mask tokens
        mask_tokens = self.mask_token.expand(B, T, H)  # (B,T,H)
        
        # Keep mask as (B,T), don't add extra dimension
        x = torch.where(mask.unsqueeze(-1), mask_tokens, x)
        assert x.shape == (B, T, H), f"After masking shape wrong: {x.shape}"
        
        # Apply transformer layers
        for block in self.layers:
            block_output = block(x)
            x = block_output['feed_forward_output']

        return x  # (B,T,H)

class BirdJEPA(nn.Module):
    def __init__(self, 
                 input_dim=513,
                 hidden_dim=256,
                 num_layers=4,
                 num_heads=8,
                 dropout=0.1,
                 mlp_dim=1024,
                 pred_hidden_dim=384,
                 pred_num_layers=6,
                 pred_num_heads=6,
                 pred_mlp_ratio=4.0,
                 max_seq_len=512):
        super().__init__()
        
        # Store configuration
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.mlp_dim = mlp_dim
        
        # Initialize encoders with max_seq_len parameter
        self.context_encoder = ViT(input_dim=input_dim, 
                                 hidden_dim=hidden_dim, 
                                 depth=num_layers, 
                                 num_heads=num_heads, 
                                 dropout=dropout, 
                                 mlp_ratio=mlp_dim/hidden_dim,
                                 max_len=max_seq_len)
        
        self.target_encoder = ViT(input_dim=input_dim, 
                                hidden_dim=hidden_dim,
                                depth=num_layers, 
                                num_heads=num_heads,
                                dropout=dropout, 
                                mlp_ratio=mlp_dim/hidden_dim,
                                max_len=max_seq_len)

        # Predictor ViT with configurable parameters
        self.predictor = PredictorViT(
            hidden_dim=pred_hidden_dim,
            depth=pred_num_layers,
            num_heads=pred_num_heads,
            mlp_ratio=pred_mlp_ratio,
            dropout=dropout,
            max_len=max_seq_len
        )

        # Decoder now matches predictor's dimension
        self.decoder = nn.Linear(pred_hidden_dim, input_dim)
        
        # Initialize EMA for target encoder
        self.ema_updater = EMA(0.999)
        self.ema_m = 0.999
        
        # Apply EMA to target encoder parameters
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_ema(self):
        """Update target encoder parameters using EMA"""
        for param_q, param_k in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data = self.ema_updater.update_average(param_k.data, param_q.data)

    def forward(self, context_spectrogram, target_spectrogram, use_no_mask=False):
        # Get mask from context spectrogram (True where masked)
        mask = (context_spectrogram == -1.0).any(dim=1)  # (B,T)
        
        # Clean context input by zeroing masked positions
        context_clean = torch.where(context_spectrogram == -1.0, 
                                  torch.zeros_like(context_spectrogram), 
                                  context_spectrogram)
        
        # Encode context
        context_repr, _ = self.context_encoder(context_clean)
        
        # Encode target with frozen encoder
        with torch.no_grad():
            target_repr, _ = self.target_encoder(target_spectrogram)
        
        # Pass mask to predictor
        pred = self.predictor(context_repr, mask=mask)  # (B,T,H)
        return pred, target_repr

    def training_step(self, context_spectrogram, target_spectrogram, mask):
        """
        Args:
            context_spectrogram: Input spectrogram with noise at masked positions (B,D,T)
            target_spectrogram: Target spectrogram (B,D,T)
            mask: Boolean tensor indicating masked positions (B,T)
        """
        # Encode context (with noise at masked positions)
        context_repr, _ = self.context_encoder(context_spectrogram)
        
        # Encode target with frozen encoder
        with torch.no_grad():
            target_repr, _ = self.target_encoder(target_spectrogram)
        
        # Pass mask to predictor
        pred = self.predictor(context_repr, mask=mask)
        
        # Compute loss only on masked positions
        loss = ((pred - target_repr)**2 * mask.unsqueeze(-1)).mean()
        return loss

    def train_forward(self, context_spectrogram, target_spectrogram):
        # context_spectrogram: (B,D,T) with -1.0 at masked positions
        # target_spectrogram: (B,D,T) with original values at masked positions
        mask = (context_spectrogram == -1.0)
        
        # Zero out masked positions in context
        context_clean = torch.where(context_spectrogram == -1.0, 
                                  torch.zeros_like(context_spectrogram), 
                                  context_spectrogram)

        # encode context
        context_repr, intermediate_outputs = self.context_encoder(context_clean)
        
        # encode target with frozen target encoder
        with torch.no_grad():
            target_repr, target_outputs = self.target_encoder(target_spectrogram)
        
        # predict and decode
        pred = self.predictor(context_repr)       
        decoded_pred = self.decoder(pred)         
        decoded_pred = decoded_pred.transpose(1,2)
        
        return decoded_pred, mask, target_spectrogram, {
            "layer_outputs": torch.stack(intermediate_outputs, dim=0),
            "target_outputs": torch.stack(target_outputs, dim=0)
        }

    def mse_loss(self, predictions, spec, mask, intermediate_layers=None, vocalization=None):
        # predictions, spec, mask: (B,D,T)
        masked_loss = ((predictions - spec)**2 * mask.float()).mean()
        with torch.no_grad():
            masked_seq_acc = masked_loss
            unmasked_seq_acc = ((predictions - spec)**2 * (~mask).float()).mean()
        return masked_loss, masked_seq_acc, unmasked_seq_acc

    def inference_forward(self, x):
        """
        Run the model in inference mode without masking.
        Args:
            x: Input tensor of shape (B,1,T,F) from analysis code
        Returns:
            tuple: (context_repr, layers) where:
                - context_repr: Final representation (B,T,H)
                - layers: List of dicts containing intermediate attention outputs
        """
        # Remove channel dimension and transpose to (B,F,T)
        x = x.squeeze(1)  # (B,T,F)
        x = x.transpose(1,2)  # (B,F,T)

        # Encode without masking
        context_repr, intermediate_outputs = self.context_encoder(x)

        # Format intermediate outputs as list of dicts
        layers = [{"attention_output": out} for out in intermediate_outputs]

        return context_repr, layers
