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
        self.input_dim = input_dim
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
        x = x.transpose(1, 2)  # (B,T,D)
        B, T, D = x.shape
        
        # Verify input dimensions
        assert D == self.input_dim, f"Expected input dimension {self.input_dim}, got {D}"
        
        # project freq dimension to hidden_dim
        x = self.proj(x)  # (B,T,hidden_dim)

        # prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B,1,H)
        x = torch.cat((cls_tokens, x), dim=1)  # (B,T+1,H)
        
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
                 mlp_dim=1024,
                 dropout=0.1,
                 max_len=512):   
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len

        # Add input projection to match dimensions
        self.input_proj = nn.Linear(hidden_dim, hidden_dim)

        # Make sure max_len matches the input sequence length
        self.layers = nn.ModuleList([
            CustomEncoderBlock(
                d_model=hidden_dim,
                num_heads=num_heads,
                ffn_dim=mlp_dim,
                dropout=dropout,
                pos_enc_type="relative",
                length=max_len
            ) for _ in range(depth)
        ])

    def forward(self, x):
        B, T, H = x.shape
        assert H == self.hidden_dim, f"Expected hidden dim {self.hidden_dim}, got {H}"
        assert T <= self.max_len, f"Input sequence length {T} exceeds maximum length {self.max_len}"
        
        # Project input
        x = self.input_proj(x)
        
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
                 pred_num_heads=4,
                 pred_mlp_dim=1024,
                 max_seq_len=512,
                 zero_predictor_input=False):
        super().__init__()
        
        # Store configuration
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.mlp_dim = mlp_dim
        self.zero_predictor_input = zero_predictor_input
        
        # If pred_num_heads not specified, use same as encoder
        if pred_num_heads is None:
            pred_num_heads = num_heads
        
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

        # Update predictor to use specified number of heads
        self.predictor = PredictorViT(
            hidden_dim=hidden_dim,
            depth=pred_num_layers,
            num_heads=pred_num_heads,
            mlp_dim=pred_mlp_dim,
            dropout=dropout,
            max_len=max_seq_len
        )

        # Fix decoder to handle the sequence dimension correctly
        self.decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),  # Add LayerNorm
            nn.Linear(hidden_dim, input_dim),
            nn.GELU()
        )
        
        # Initialize EMA with beta=0.5
        self.ema_updater = EMA(0.95)
        self.ema_m = 0.95
        
        # Debug print to verify trainable parameters
        context_trainable = sum(p.requires_grad for p in self.context_encoder.parameters())
        decoder_trainable = sum(p.requires_grad for p in self.decoder.parameters())
        print(f"Context encoder has {context_trainable} trainable parameters")
        print(f"Decoder has {decoder_trainable} trainable parameters")

    @torch.no_grad()
    def update_ema(self):
        """Update target encoder parameters using EMA"""
        for param_q, param_k in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data = self.ema_updater.update_average(param_k.data, param_q.data)

    def forward(self, context_spectrogram, target_spectrogram, use_no_mask=False):
        # Get mask from context spectrogram (True where masked)
        mask = (context_spectrogram == 0.0).any(dim=1)  # Changed from -1.0 to 0.0
        
        # Clean context input by zeroing masked positions (no change needed here since already zeros)
        context_clean = context_spectrogram  # Simplified since input is already properly masked
        
        # Encode context
        context_repr, _ = self.context_encoder(context_clean)
        
        # Add new logic to zero out masked embeddings
        if self.zero_predictor_input:
            mask_3d = mask.unsqueeze(-1).expand_as(context_repr)
            context_repr = context_repr.clone()
            context_repr[mask_3d] = 0
        
        # Encode target with frozen encoder
        with torch.no_grad():
            target_repr, _ = self.target_encoder(target_spectrogram)
        
        # Pass mask to predictor
        pred = self.predictor(context_repr)  # (B,T,H)
        
        # Update decoder usage
        decoded = self.decoder(pred)  # (B,T,D)
        decoded = decoded.transpose(1, 2)  # (B,D,T)
        
        return decoded, target_repr

    def compute_latent_loss(self, context_spectrogram, target_spectrogram, mask, is_eval_step=False):
        """Computes loss in embedding space between predictor output and target encoding"""
        # Get context representation and predict target
        context_repr, _ = self.context_encoder(context_spectrogram)  # (B,T,H)
        
        # Add new logic to zero out masked embeddings
        if self.zero_predictor_input:
            mask_3d = mask.unsqueeze(-1).expand_as(context_repr)
            context_repr = context_repr.clone()
            context_repr[mask_3d] = 0
        
        pred = self.predictor(context_repr)  # (B,T,H)
        
        # Get target representation (with no grad since it's the target)
        with torch.no_grad():
            target_repr, _ = self.target_encoder(target_spectrogram)  # (B,T,H)
        
        # Fix mask dimension for loss computation
        mask = mask.unsqueeze(-1)  # (B,T,1) to broadcast across hidden dim
        
        # Calculate squared differences
        diff = ((pred - target_repr)**2 * mask)  # (B,T,H)
        
        # Compute both sum-based and average-based losses for debugging
        with torch.no_grad():
            total_loss = diff.sum()
            num_masked = mask.sum()
            avg_loss = total_loss / (num_masked + 1e-8)
            if is_eval_step:
                print(f"debug>>> sum_loss={total_loss.item():.4f}, "
                      f"avg_loss={avg_loss.item():.4f}, "
                      f"masked_positions={num_masked.item()}")
        
        # Use average-based loss for training
        loss = diff.sum() / (mask.sum() + 1e-8)
        
        return loss, diff, pred, target_repr, context_repr

    def training_step(self, context_spectrogram, target_spectrogram, mask):
        return self.compute_latent_loss(context_spectrogram, target_spectrogram, mask)

    def train_forward(self, context_spectrogram, target_spectrogram):
        # context_spectrogram: (B,D,T) already masked from dataloader
        # target_spectrogram: (B,D,T) already contains original values
        
        # encode context (using already masked input)
        context_repr, intermediate_outputs = self.context_encoder(context_spectrogram)
        
        # encode target with frozen target encoder
        with torch.no_grad():
            target_repr, target_outputs = self.target_encoder(target_spectrogram)
        
        # predict and decode
        pred = self.predictor(context_repr)       # (B,T,H)
        decoded_pred = self.decoder(pred)         # (B,T,D)
        decoded_pred = decoded_pred.transpose(1,2)  # (B,D,T)
        
        return decoded_pred, mask, target_spectrogram, {
            "layer_outputs": torch.stack(intermediate_outputs, dim=0),
            "target_outputs": torch.stack(target_outputs, dim=0)
        }

    # def mse_loss(self, predictions, spec, mask, intermediate_layers=None, vocalization=None):
    #     """Computes loss in spectrogram space for training decoder"""
    #     # Add frequency dimension to mask
    #     mask = mask.unsqueeze(1)  # (B,1,T)
        
    #     # predictions, spec: (B,D,T), mask: (B,1,T)
    #     masked_loss = ((predictions - spec)**2 * mask.float()).mean()
    #     with torch.no_grad():
    #         masked_seq_acc = masked_loss
    #         unmasked_seq_acc = ((predictions - spec)**2 * (~mask).float()).mean()
    #     return masked_loss, masked_seq_acc, unmasked_seq_acc

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
