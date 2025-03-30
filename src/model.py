import torch
import torch.nn as nn
import torch.nn.functional as F
import math

################################################################################
# LOCAL AND GLOBAL ATTENTION IMPLEMENTATIONS
################################################################################

class LocalAttentionBlock(nn.Module):
    """
    a transformer encoder block that restricts attention to a sliding local window.
    """
    def __init__(self, d_model, num_heads, window_size, mlp_dim, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.window_size = window_size
        
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, debug=False):
        # x shape: (B, T, d_model)
        if debug:
            print(f"[LocalAttentionBlock] input: {x.shape}")
        
        # create local attention mask (band mask) for each query position
        # so each token attends to [i-window_size : i+window_size]
        B, T, _ = x.shape
        attn_mask = torch.ones((T, T), device=x.device).bool()
        for i in range(T):
            start = max(i - self.window_size, 0)
            end = min(i + self.window_size + 1, T)
            attn_mask[i, start:end] = False  # False => allowed to attend

        # self-attention
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(
            x_norm,
            x_norm,
            x_norm,
            attn_mask=attn_mask,  # shape (T, T)
        )
        x = x + attn_output

        # feed-forward
        x_norm = self.norm2(x)
        mlp_output = self.mlp(x_norm)
        x = x + mlp_output
        
        if debug:
            print(f"[LocalAttentionBlock] output: {x.shape}")
        return x


class GlobalAttentionBlock(nn.Module):
    """
    a transformer encoder block that provides global attention every stride steps.
    specifically, tokens at indices multiple of 'global_stride' can attend to all tokens,
    while others attend only to themselves (or a smaller subset).
    this is just one example pattern of "global" connections.
    """
    def __init__(self, d_model, num_heads, global_stride, mlp_dim, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.global_stride = global_stride

        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, debug=False):
        # x shape: (B, T, d_model)
        if debug:
            print(f"[GlobalAttentionBlock] input: {x.shape}")
        
        B, T, _ = x.shape
        # create a mask that allows tokens at multiples of stride
        # to attend to all tokens, while others attend only to themselves
        attn_mask = torch.ones((T, T), device=x.device).bool()
        for i in range(T):
            if i % self.global_stride == 0:
                attn_mask[i, :] = False  # can attend anywhere
            else:
                # only allow self-attention
                attn_mask[i, :] = True
                attn_mask[i, i] = False

        x_norm = self.norm1(x)
        attn_output, _ = self.attn(
            x_norm,
            x_norm,
            x_norm,
            attn_mask=attn_mask
        )
        x = x + attn_output

        x_norm = self.norm2(x)
        mlp_output = self.mlp(x_norm)
        x = x + mlp_output

        if debug:
            print(f"[GlobalAttentionBlock] output: {x.shape}")
        return x

################################################################################
# POSITIONAL ENCODING (SINE, WITHOUT FIXED MAX LENGTH LIMIT)
################################################################################

class SinePositionalEncoding(nn.Module):
    """
    standard sine positional encoding for variable-length input.
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, x):
        # x shape: (B, T, d_model)
        B, T, D = x.shape
        device = x.device

        # create the positional enc
        # shape (T, D)
        pe = torch.zeros(T, D, device=device)
        position = torch.arange(0, T, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, D, 2, device=device) * -(math.log(10000.0) / D))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # add
        x = x + pe.unsqueeze(0)
        return x

################################################################################
# CONVOLUTIONAL FEATURE EXTRACTOR
################################################################################

class BirdCLEF_ConvolutionalFeatureExtractor(nn.Module):
    """
    4-layer convolution stack with progressively larger kernels and out_channels,
    reducing frequency dimension (height) while preserving time dimension.
    """
    def __init__(self, in_channels=1, debug=False):
        super().__init__()
        self.debug = debug

        # conv1: smaller kernel
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=(3,3), stride=(2,1), padding=(1,1))
        self.bn1 = nn.BatchNorm2d(32)

        # conv2: bigger kernel
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5,5), stride=(2,1), padding=(2,2))
        self.bn2 = nn.BatchNorm2d(64)

        # conv3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(7,7), stride=(2,1), padding=(3,3))
        self.bn3 = nn.BatchNorm2d(128)

        # conv4
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(7,7), stride=(2,1), padding=(3,3))
        self.bn4 = nn.BatchNorm2d(128)

    def forward(self, x):
        # x shape: (B, 1, F, T)
        if self.debug:
            print(f"[ConvExtractor] input: {x.shape}")

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.gelu(x)
        if self.debug:
            print(f"[ConvExtractor] after conv1: {x.shape}")

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.gelu(x)
        if self.debug:
            print(f"[ConvExtractor] after conv2: {x.shape}")

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.gelu(x)
        if self.debug:
            print(f"[ConvExtractor] after conv3: {x.shape}")

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.gelu(x)
        if self.debug:
            print(f"[ConvExtractor] after conv4: {x.shape}")

        # return shape: (B, out_channels=128, freq_reduced, T)
        return x

################################################################################
# ENCODER WITH LOCAL THEN GLOBAL ATTENTION
################################################################################

class BirdCLEFEncoder(nn.Module):
    """
    applies the convolutional feature extractor, flattens the freq dimension,
    projects to hidden_dim, then applies a series of local attention blocks
    followed by global attention blocks, with sine positional encoding.
    """
    def __init__(
        self,
        input_dim,        # freq bins
        hidden_dim=256,
        num_local_blocks=2,
        local_window_sizes=[8, 16],   # example window sizes
        num_global_blocks=2,
        global_stride=16,
        mlp_dim=1024,
        dropout=0.1,
        max_len=512,
        debug=False
    ):
        super().__init__()
        self.debug = debug

        # feature extractor
        self.feature_extractor = BirdCLEF_ConvolutionalFeatureExtractor(in_channels=1, debug=debug)

        # pass a dummy input to figure out the flattened dimension
        dummy = torch.zeros(1, 1, input_dim, max_len)
        with torch.no_grad():
            test_out = self.feature_extractor(dummy)
        _, c_out, freq_out, _ = test_out.shape
        self.flattened_dim = c_out * freq_out

        # linear projection to transformer dimension
        self.input_proj = nn.Linear(self.flattened_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # positional encoding
        self.pos_enc = SinePositionalEncoding(hidden_dim)

        # local blocks
        self.local_blocks = nn.ModuleList([
            LocalAttentionBlock(
                d_model=hidden_dim,
                num_heads=8,  # or param
                window_size=local_window_sizes[i] if i < len(local_window_sizes) else local_window_sizes[-1],
                mlp_dim=mlp_dim,
                dropout=dropout
            ) for i in range(num_local_blocks)
        ])

        # global blocks
        self.global_blocks = nn.ModuleList([
            GlobalAttentionBlock(
                d_model=hidden_dim,
                num_heads=8,  # or param
                global_stride=global_stride,
                mlp_dim=mlp_dim,
                dropout=dropout
            ) for _ in range(num_global_blocks)
        ])

    def forward(self, x):
        """
        x shape: (B,1,D,T) or (B,D,T)
        returns final hidden representation of shape (B, T, hidden_dim)
        plus a list of intermediate outputs if needed.
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)

        if self.debug:
            print(f"[BirdCLEFEncoder] input: {x.shape}")

        # pass through convolutional feature extractor
        x = self.feature_extractor(x)
        # shape => (B, 128, freq_reduced, T)

        B, C, Freq, T = x.shape
        # flatten freq dimension
        x = x.view(B, C * Freq, T)  # (B, flattened_freq, T)
        # transpose to (B, T, flattened_freq)
        x = x.transpose(1, 2)       # (B, T, flattened_freq)

        # project to hidden_dim
        x = self.input_proj(x)      # (B, T, hidden_dim)
        x = self.dropout(x)

        # add positional encoding
        x = self.pos_enc(x)

        # local attention blocks
        for block in self.local_blocks:
            x = block(x, debug=self.debug)

        # global attention blocks
        for block in self.global_blocks:
            x = block(x, debug=self.debug)

        if self.debug:
            print(f"[BirdCLEFEncoder] final output: {x.shape}")

        return x, []  # returning empty list for intermediate outputs, to keep interface consistent

################################################################################
# PREDICTOR (AS BEFORE)
################################################################################

class Predictor(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return self.net(x)

class PredictorViT(nn.Module):
    """
    retained for interface consistency; you can still use the same approach,
    or just keep a simpler MLP. here we'll do something minimal for compatibility.
    """
    def __init__(self, 
                 hidden_dim=384,  
                 depth=6,         
                 num_heads=6, 
                 mlp_dim=1024,
                 dropout=0.1,
                 max_len=512,
                 debug=False):   
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len
        self.debug = debug

        # simple linear to unify dimensions
        self.input_proj = nn.Linear(hidden_dim, hidden_dim)

        # for demonstration, just do repeated linear-lnorm pairs
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, mlp_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_dim, hidden_dim),
                nn.Dropout(dropout)
            )
            for _ in range(depth)
        ])

    def forward(self, x):
        # x shape: (B, T, H)
        if self.debug:
            print(f"[PredictorViT] input: {x.shape}")
        x = self.input_proj(x)
        for layer in self.layers:
            ln_out = layer[0](x)  # layer norm
            ff_out = layer[1:](ln_out)
            x = x + ff_out
        return x

################################################################################
# EMA
################################################################################

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

################################################################################
# THE MAIN BirdJEPA MODULE
################################################################################

class BirdJEPA(nn.Module):
    def __init__(self, 
                 input_dim=513,
                 hidden_dim=256,
                 num_layers=4,     # not strictly used the same way as before
                 num_heads=8,     # optional usage
                 dropout=0.1,
                 mlp_dim=1024,
                 pred_hidden_dim=384,
                 pred_num_layers=6,
                 pred_num_heads=4,
                 pred_mlp_dim=1024,
                 max_seq_len=512,
                 zero_predictor_input=False,
                 debug=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.mlp_dim = mlp_dim
        self.zero_predictor_input = zero_predictor_input
        self.debug = debug

        # we preserve the same 'context_encoder' and 'target_encoder' naming
        # but we'll create them with our new local+global structure
        self.context_encoder = BirdCLEFEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_local_blocks=2,      # can be tuned
            local_window_sizes=[8,16],
            num_global_blocks=2,
            global_stride=16,
            mlp_dim=mlp_dim,
            dropout=dropout,
            max_len=max_seq_len,
            debug=debug
        )
        self.target_encoder = BirdCLEFEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_local_blocks=2,
            local_window_sizes=[8,16],
            num_global_blocks=2,
            global_stride=16,
            mlp_dim=mlp_dim,
            dropout=dropout,
            max_len=max_seq_len,
            debug=False  # typically we keep target encoder quiet
        )

        # predictor
        self.predictor = PredictorViT(
            hidden_dim=hidden_dim,
            depth=pred_num_layers,
            num_heads=pred_num_heads,
            mlp_dim=pred_mlp_dim,
            dropout=dropout,
            max_len=max_seq_len,
            debug=debug
        )

        # decoder
        self.decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, input_dim),
            nn.GELU()
        )

        # EMA
        self.ema_updater = EMA(0.95)
        self.ema_m = 0.95

        if self.debug:
            # debug prints: count trainable params
            context_trainable = sum(p.numel() for p in self.context_encoder.parameters() if p.requires_grad)
            decoder_trainable = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
            print(f"[BirdJEPA] context encoder trainable params: {context_trainable}")
            print(f"[BirdJEPA] decoder trainable params: {decoder_trainable}")

    @torch.no_grad()
    def update_ema(self):
        for param_q, param_k in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data = self.ema_updater.update_average(param_k.data, param_q.data)

    def forward(self, context_spectrogram, target_spectrogram, use_no_mask=False):
        """
        context_spectrogram: (B,1,D,T)
        target_spectrogram:  (B,1,D,T)
        """
        if self.debug:
            print("[BirdJEPA] forward called.")

        # quick mask check if needed (but here we won't do anything special)
        mask = (context_spectrogram == 0.0).any(dim=1)

        # encode context
        context_repr, _ = self.context_encoder(context_spectrogram)

        if self.zero_predictor_input:
            mask_3d = mask.unsqueeze(-1).expand_as(context_repr)
            context_repr = context_repr.clone()
            context_repr[mask_3d] = 0

        # encode target w/ no grad
        with torch.no_grad():
            target_repr, _ = self.target_encoder(target_spectrogram)

        # predictor
        pred = self.predictor(context_repr)

        # decode
        decoded = self.decoder(pred)  # (B,T,input_dim)
        decoded = decoded.transpose(1, 2)  # (B,input_dim,T)

        return decoded, target_repr

    def compute_latent_loss(self, context_spectrogram, target_spectrogram, mask, is_eval_step=False):
        if self.debug:
            print("[BirdJEPA] compute_latent_loss called.")

        # ensure shape
        if context_spectrogram.dim() == 3:
            context_spectrogram = context_spectrogram.unsqueeze(1)
        if target_spectrogram.dim() == 3:
            target_spectrogram = target_spectrogram.unsqueeze(1)

        # encode context
        context_repr, _ = self.context_encoder(context_spectrogram)

        if self.zero_predictor_input:
            mask_3d = mask.unsqueeze(-1).expand_as(context_repr)
            context_repr = context_repr.clone()
            context_repr[mask_3d] = 0

        pred = self.predictor(context_repr)

        # encode target
        with torch.no_grad():
            target_repr, _ = self.target_encoder(target_spectrogram)

        # compute mse in latent space
        mask = mask.unsqueeze(-1)  # (B,T,1)
        diff = (pred - target_repr) ** 2 * mask
        total_loss = diff.sum()
        num_masked = mask.sum()
        avg_loss = total_loss / (num_masked + 1e-8)
        if is_eval_step and self.debug:
            print(f"[BirdJEPA] sum_loss={total_loss.item():.4f}, avg_loss={avg_loss.item():.4f}")

        loss = avg_loss
        return loss, diff, pred, target_repr, context_repr

    def training_step(self, context_spectrogram, target_spectrogram, mask):
        return self.compute_latent_loss(context_spectrogram, target_spectrogram, mask)

    def train_forward(self, context_spectrogram, target_spectrogram):
        """
        context_spectrogram: (B,D,T) masked
        target_spectrogram: (B,D,T) unmasked
        """
        if self.debug:
            print("[BirdJEPA] train_forward called.")
        # add channel dimension
        if context_spectrogram.dim() == 3:
            context_spectrogram = context_spectrogram.unsqueeze(1)
        if target_spectrogram.dim() == 3:
            target_spectrogram = target_spectrogram.unsqueeze(1)

        # encode context
        context_repr, inter_ctx = self.context_encoder(context_spectrogram)

        # encode target
        with torch.no_grad():
            target_repr, inter_tgt = self.target_encoder(target_spectrogram)

        # predictor + decode
        pred = self.predictor(context_repr)
        decoded_pred = self.decoder(pred)    # (B,T,D)
        decoded_pred = decoded_pred.transpose(1,2)  # (B,D,T)

        return decoded_pred, None, target_spectrogram, {
            "layer_outputs": torch.stack([]),
            "target_outputs": torch.stack([])
        }

    def inference_forward(self, x):
        """
        run model in inference mode (no masking).
        x: (B,1,T,F) from analysis code => we will reorder to (B,F,T)
        returns (context_repr, layers)
        """
        if self.debug:
            print("[BirdJEPA] inference_forward called.")
        # reorder
        x = x.squeeze(1)         # (B,T,F)
        x = x.transpose(1,2)     # (B,F,T)
        # encode
        context_repr, intermediate_outputs = self.context_encoder(x.unsqueeze(1))
        # format intermediate outputs (placeholder)
        layers = []
        return context_repr, layers