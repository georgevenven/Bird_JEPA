# birdjepa.py
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from custom_transformer import CustomEncoderBlock

class ConvolutionalFeatureExtractor(nn.Module):
    """
    This module applies 4 convolutional + pooling layers
    across the frequency dimension only, preserving the time dimension T.
    """
    def __init__(self):
        super().__init__()
        # Each conv uses kernel_size=(5,5) with padding=2, stride=1
        # Each pool uses kernel_size=(2,1), stride=(2,1) => halves freq, keeps time
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5,5), stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5,5), stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5,5), stride=1, padding=2)
        self.pool3 = nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5,5), stride=1, padding=2)
        self.pool4 = nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))

    def forward(self, x):
        """
        x: (B, 1, D, T)
          - B is batch size
          - 1 is 'channel' dimension for spectrogram
          - D is freq bins
          - T is time

        Returns:
          out: (B, C, D', T) after 4 conv+pool layers
        """
        # conv1 + pool1
        x = F.gelu(self.conv1(x))
        x = self.pool1(x)
        # conv2 + pool2
        x = F.gelu(self.conv2(x))
        x = self.pool2(x)
        # conv3 + pool3
        x = F.gelu(self.conv3(x))
        x = self.pool3(x)
        # conv4 + pool4
        x = F.gelu(self.conv4(x))
        x = self.pool4(x)

        return x  # shape (B, 64, new_freq, T)

class ViT(nn.Module):
    """
    A Vision-Transformer-like encoder, with four initial convolutional layers
    that reduce frequency dimension but preserve time dimension T.
    After the conv stack, we flatten the freq dimension and project
    to hidden_dim, then apply standard transformer blocks.
    """
    def __init__(
        self,
        input_dim,      # This is freq bins, but we won't directly use it because conv blocks handle that.
        hidden_dim=64,
        depth=3,
        num_heads=4,
        mlp_ratio=4.0,
        dropout=0.1,
        max_len=512
    ):
        super().__init__()

        # --- 4-layer CNN feature extractor (freq-downsample, preserve T) ---
        self.feature_extractor = ConvolutionalFeatureExtractor()
        
        # We'll figure out how many channels and freq remain after CNN
        # Let's assume the final channel count is 64, but freq is (input_dim // 2^4)
        # or something similar. Instead of computing dynamically, we can do a small hack:
        #   - We'll pass a dummy tensor through the conv stack in __init__ to get the shape.

        dummy = torch.zeros(1, 1, input_dim, max_len)  # (B=1,1,D,T)
        conv_out = self.feature_extractor(dummy)
        _, final_channels, final_freq, _ = conv_out.shape
        conv_output_size = final_channels * final_freq  # This is the dimension we flatten to

        # Project from conv_output_size -> hidden_dim
        self.input_proj = nn.Linear(conv_output_size, hidden_dim)

        self.hidden_dim = hidden_dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.dropout = nn.Dropout(dropout)

        # Create the transformer blocks
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
        """
        x: (B, 1, D, T) or (B, D, T)
          - B is batch size
          - 1 is 'channel' dimension for spectrogram (optional)
          - D is freq bins
          - T is time
        """
        # Check input dimensionality and add channel dimension if needed
        if x.dim() == 3:  # If (B, D, T)
            x = x.unsqueeze(1)  # Add channel dimension -> (B, 1, D, T)
        
        B, _, D, T = x.shape

        # 1) CNN feature extractor -> (B, 64, D', T)
        feats = self.feature_extractor(x)  # shape: (B, final_channels, new_freq, T)

        # 2) Flatten freq dimension
        #    shape => (B, final_channels*new_freq, T)
        B, C, D_prime, T = feats.shape
        feats = feats.view(B, C * D_prime, T)  # (B, C*D', T)

        # 3) Transpose to (B, T, conv_output_size) for a standard BxTxF
        feats = feats.transpose(1, 2)  # (B, T, C*D')

        # 4) Linear projection to hidden_dim
        feats = self.input_proj(feats)  # (B, T, hidden_dim)

        # 5) Prepend CLS token => shape (B, T+1, hidden_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B,1,H)
        x = torch.cat((cls_tokens, feats), dim=1)

        x = self.dropout(x)

        layer_outputs = []
        for block in self.layers:
            block_output = block(x)
            x = block_output['feed_forward_output']
            # Collect the per-layer output *excluding* CLS (like the previous code)
            layer_outputs.append(x[:, 1:])  # shape (B, T, hidden_dim)

        # Final output: (B, T, hidden_dim) ignoring the prepended CLS token
        return x[:, 1:], layer_outputs

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
        """
        context_spectrogram: (B,1,D,T)
        target_spectrogram:  (B,1,D,T)
        """
        # Determine which positions are masked
        # (Here we assume masked positions in freq or amplitude are zeroed out, so this is a quick check.)
        mask = (context_spectrogram == 0.0).any(dim=1)  # (B, D, T) => (B, T) after .any(dim=1)

        context_clean = context_spectrogram

        # 1) Encode context
        context_repr, _ = self.context_encoder(context_clean)  # (B, T, hidden_dim)

        if self.zero_predictor_input:
            mask_3d = mask.unsqueeze(-1).expand_as(context_repr)
            context_repr = context_repr.clone()
            context_repr[mask_3d] = 0

        # 2) Encode target with frozen encoder
        with torch.no_grad():
            target_repr, _ = self.target_encoder(target_spectrogram)

        # 3) Predictor
        pred = self.predictor(context_repr)  # (B, T, hidden_dim)

        # 4) Decoder
        decoded = self.decoder(pred)  # (B, T, input_dim)
        decoded = decoded.transpose(1, 2)  # (B, input_dim, T)

        return decoded, target_repr

    def compute_latent_loss(self, context_spectrogram, target_spectrogram, mask, is_eval_step=False):
        """Computes loss in embedding space between predictor output and target encoding"""
        # Ensure inputs have the channel dimension for the CNN part
        if context_spectrogram.dim() == 3:  # If (B,D,T)
            context_spectrogram = context_spectrogram.unsqueeze(1)  # (B,1,D,T)
        
        if target_spectrogram.dim() == 3:  # If (B,D,T)
            target_spectrogram = target_spectrogram.unsqueeze(1)  # (B,1,D,T)
        
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
