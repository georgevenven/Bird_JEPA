import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.cuda
import sys

################################################################################
# LOCAL AND GLOBAL ATTENTION IMPLEMENTATIONS
################################################################################

class LocalAttentionBlock(nn.Module):
    """
    a transformer encoder block that restricts attention to a sliding local window.
    precomputes an attention mask up to max_seq_len, then slices it at runtime.
    """
    def __init__(self, d_model, num_heads, window_size, mlp_dim, dropout=0.1, max_seq_len=512):
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

        # precompute the local attention mask up to max_seq_len
        attn_mask = torch.ones((max_seq_len, max_seq_len), dtype=torch.bool)
        for i in range(max_seq_len):
            start = max(i - self.window_size, 0)
            end = min(i + self.window_size + 1, max_seq_len)
            attn_mask[i, start:end] = False
        self.register_buffer("precomputed_attn_mask", attn_mask, persistent=False)

    def forward(self, x, debug=False):
        # x shape: (B, T, d_model)
        B, T, _ = x.shape

        # slice the precomputed mask
        attn_mask = self.precomputed_attn_mask[:T, :T]
        attn_mask = attn_mask.to(x.device)

        x_norm = self.norm1(x)
        attn_output, _ = self.attn(
            x_norm,
            x_norm,
            x_norm,
            attn_mask=attn_mask,
        )
        x = x + attn_output

        x_norm = self.norm2(x)
        mlp_output = self.mlp(x_norm)
        x = x + mlp_output
            
        return x


class GlobalAttentionBlock(nn.Module):
    """
    a transformer encoder block that provides global attention every stride steps.
    specifically, tokens at indices multiple of 'global_stride' can attend to all tokens,
    while others attend only to themselves.
    precomputes an attention mask up to max_seq_len, then slices it at runtime.
    """
    def __init__(self, d_model, num_heads, global_stride, mlp_dim, dropout=0.1, max_seq_len=512):
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

        # precompute the global attention mask up to max_seq_len
        attn_mask = torch.ones((max_seq_len, max_seq_len), dtype=torch.bool)
        for i in range(max_seq_len):
            if i % self.global_stride == 0:
                attn_mask[i, :] = False  # can attend anywhere
            else:
                attn_mask[i, :] = True
                attn_mask[i, i] = False
        self.register_buffer("precomputed_attn_mask", attn_mask, persistent=False)

    def forward(self, x, debug=False):
        # x shape: (B, T, d_model)
        B, T, _ = x.shape

        # slice the precomputed mask
        attn_mask = self.precomputed_attn_mask[:T, :T]
        attn_mask = attn_mask.to(x.device)

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

        pe = torch.zeros(T, D, device=device)
        position = torch.arange(0, T, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, D, 2, device=device) * -(math.log(10000.0) / D))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

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

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=(3,3), stride=(2,1), padding=(1,1))
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5,5), stride=(2,1), padding=(2,2))
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(7,7), stride=(2,1), padding=(3,3))
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=(7,7), stride=(2,1), padding=(3,3))
        self.bn4 = nn.BatchNorm2d(128)

    def forward(self, x):
        # x shape: (B, 1, F, T)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.gelu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.gelu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.gelu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.gelu(x)
        
        return x

################################################################################
# ENCODER WITH LOCAL THEN GLOBAL ATTENTION
################################################################################

class BirdCLEFEncoder(nn.Module):
    """
    applies the convolutional feature extractor, flattens the freq dimension,
    projects to hidden_dim, then applies a series of blocks as specified
    by the blocks_config parameter.
    """
    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        blocks_config=None,
        mlp_dim=1024,
        dropout=0.1,
        max_len=512,
        debug=False
    ):
        super().__init__()
        self.debug = debug

        self.feature_extractor = BirdCLEF_ConvolutionalFeatureExtractor(in_channels=1, debug=debug)

        # dimension calculation
        dummy = torch.zeros(1, 1, input_dim, max_len)
        with torch.no_grad():
            test_out = self.feature_extractor(dummy)
        _, c_out, freq_out, _ = test_out.shape
        self.flattened_dim = c_out * freq_out

        self.input_proj = nn.Linear(self.flattened_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.pos_enc = SinePositionalEncoding(hidden_dim)

        self.attention_blocks = nn.ModuleList()
        
        if not blocks_config:
            blocks_config = [
                {"type": "local", "window_size": 8},
                {"type": "global", "stride": 100}
            ]
            
        for block in blocks_config:
            if block["type"] == "local":
                self.attention_blocks.append(
                    LocalAttentionBlock(
                        d_model=hidden_dim,
                        num_heads=8,
                        window_size=block["window_size"],
                        mlp_dim=mlp_dim,
                        dropout=dropout,
                        max_seq_len=max_len
                    )
                )
            elif block["type"] == "global":
                self.attention_blocks.append(
                    GlobalAttentionBlock(
                        d_model=hidden_dim,
                        num_heads=8,
                        global_stride=block["stride"],
                        mlp_dim=mlp_dim,
                        dropout=dropout,
                        max_seq_len=max_len
                    )
                )

    def forward(self, x, layer_index=None, dict_key=None):
        """
        x shape: (B,1,D,T) or (B,D,T)
        returns final hidden representation of shape (B, T, hidden_dim)
        plus a list of intermediate outputs if needed.
        
        If layer_index and dict_key are provided, returns only the output
        from the specified layer for inference optimization.
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.feature_extractor(x)
        B, C, Freq, T = x.shape
        x = x.view(B, C * Freq, T)
        x = x.transpose(1, 2)

        x = self.input_proj(x)
        x = self.dropout(x)

        x = self.pos_enc(x)
        
        initial_embedding = x
        
        if layer_index == 0 and dict_key is not None:
            return {dict_key: initial_embedding}, []
        
        intermediate_outputs = []
        intermediate_outputs.append({"attention_output": initial_embedding})
        
        for i, block in enumerate(self.attention_blocks):
            x = block(x, debug=self.debug)
            intermediate_outputs.append({"attention_output": x})
            
            if layer_index is not None and dict_key is not None and i + 1 == layer_index:
                return {dict_key: x}, []
        
        if layer_index == -1 and dict_key is not None:
            return {dict_key: x}, []
            
        if layer_index is None or dict_key is None:
            return x, intermediate_outputs[1:]
        else:
            print(f"Warning: layer_index {layer_index} is out of range. Using the last layer instead.")
            return {dict_key: x}, []

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

        self.input_proj = nn.Linear(hidden_dim, hidden_dim)

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
        x = self.input_proj(x)
            
        for layer in self.layers:
            ln_out = layer[0](x)
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
                 num_layers=4,
                 num_heads=8,
                 dropout=0.1,
                 mlp_dim=1024,
                 pred_hidden_dim=384,
                 pred_num_layers=6,
                 pred_num_heads=4,
                 pred_mlp_dim=1024,
                 max_seq_len=512,
                 zero_predictor_input=False,
                 debug=False,
                 blocks_config=None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.mlp_dim = mlp_dim
        self.zero_predictor_input = zero_predictor_input
        self.debug = debug

        self.context_encoder = BirdCLEFEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            blocks_config=blocks_config,
            mlp_dim=mlp_dim,
            dropout=dropout,
            max_len=max_seq_len,
            debug=debug
        )
        self.target_encoder = BirdCLEFEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            blocks_config=blocks_config,
            mlp_dim=mlp_dim,
            dropout=dropout,
            max_len=max_seq_len,
            debug=False
        )

        self.predictor = PredictorViT(
            hidden_dim=hidden_dim,
            depth=pred_num_layers,
            num_heads=pred_num_heads,
            mlp_dim=pred_mlp_dim,
            dropout=dropout,
            max_len=max_seq_len,
            debug=debug
        )

        self.decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, input_dim),
            nn.GELU()
        )

        self.ema_updater = EMA(0.95)
        self.ema_m = 0.95

        if self.debug:
            context_trainable = sum(p.numel() for p in self.context_encoder.parameters() if p.requires_grad)
            decoder_trainable = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
            print(f"[BirdJEPA] context encoder trainable params: {context_trainable}")
            print(f"[BirdJEPA] decoder trainable params: {decoder_trainable}")

    @torch.no_grad()
    def update_ema(self):
        for param_q, param_k in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data = self.ema_updater.update_average(param_k.data, param_q.data)

    def forward(self, context_spectrogram, target_spectrogram, use_no_mask=False):
        context_repr, _ = self.context_encoder(context_spectrogram)

        # if user wants to zero out
        # (the original code references a mask, but doesn't define it in forward,
        # so we just keep the check for consistency)
        if self.zero_predictor_input and use_no_mask:
            pass  # do nothing here, no mask is defined in original code

        with torch.no_grad():
            target_repr, _ = self.target_encoder(target_spectrogram)

        pred = self.predictor(context_repr)

        decoded = self.decoder(pred)  # (B,T,input_dim)
        decoded = decoded.transpose(1, 2)  # (B,input_dim,T)
        
        return decoded, target_repr

    def compute_latent_loss(self, context_spectrogram, target_spectrogram, mask, is_eval_step=False):
        if context_spectrogram.dim() == 3:
            context_spectrogram = context_spectrogram.unsqueeze(1)
        if target_spectrogram.dim() == 3:
            target_spectrogram = target_spectrogram.unsqueeze(1)

        context_repr, _ = self.context_encoder(context_spectrogram)

        if self.zero_predictor_input:
            mask_3d = mask.unsqueeze(-1).expand_as(context_repr)
            context_repr[mask_3d] = 0

        with torch.no_grad():
            target_repr, _ = self.target_encoder(target_spectrogram)

        pred = self.predictor(context_repr)

        mask = mask.unsqueeze(-1)
        diff = (pred - target_repr) ** 2 * mask
        total_loss = diff.sum()
        num_masked = mask.sum()
        avg_loss = total_loss / (num_masked + 1e-8)
        if self.debug or is_eval_step:
            msg = f"[BirdJEPA] sum_loss={total_loss.item():.4f}, avg_loss={avg_loss.item():.4f}, num_masked={num_masked.item()}"
            if hasattr(sys.stdout, 'log_file'):
                sys.stdout.log_file.write(msg + "\n")
                sys.stdout.log_file.flush()
            else:
                print(msg)

        return avg_loss, diff, pred, target_repr, context_repr

    def training_step(self, context_spectrogram, target_spectrogram, mask):
        return self.compute_latent_loss(context_spectrogram, target_spectrogram, mask)

    def train_forward(self, context_spectrogram, target_spectrogram):
        if context_spectrogram.dim() == 3:
            context_spectrogram = context_spectrogram.unsqueeze(1)
        if target_spectrogram.dim() == 3:
            target_spectrogram = target_spectrogram.unsqueeze(1)

        context_repr, inter_ctx = self.context_encoder(context_spectrogram)

        with torch.no_grad():
            target_repr, inter_tgt = self.target_encoder(target_spectrogram)

        pred = self.predictor(context_repr)
        decoded_pred = self.decoder(pred)
        decoded_pred = decoded_pred.transpose(1,2)
        
        return decoded_pred, None, target_spectrogram, {
            "layer_outputs": torch.stack([]),
            "target_outputs": torch.stack([])
        }

    def inference_forward(self, x, layer_index=None, dict_key=None):
        """
        run model in inference mode (no masking).
        x: (B,1,T,F) => reorder to (B,F,T)
        returns (context_repr, layers)
        
        If layer_index and dict_key are provided, returns the output from
        the specified layer directly in the second return value.
        """
        x = x.squeeze(1)
        x = x.transpose(1,2)
        
        if dict_key is None:
            dict_key = "attention_output"
            
        context_repr, intermediate_outputs = self.context_encoder(x.unsqueeze(1), layer_index=layer_index, dict_key=dict_key)
        
        if isinstance(context_repr, dict):
            if dict_key in context_repr:
                return None, context_repr
            else:
                if len(context_repr) > 0:
                    first_key = list(context_repr.keys())[0]
                    output_dict = {dict_key: context_repr[first_key]}
                    return None, output_dict
                else:
                    return None, {dict_key: None}
        
        if len(intermediate_outputs) > 0:
            formatted_outputs = []
            for i, layer_output in enumerate(intermediate_outputs):
                if isinstance(layer_output, dict) and dict_key in layer_output:
                    formatted_outputs.append(layer_output)
                else:
                    formatted_outputs.append({dict_key: layer_output})
            return context_repr, formatted_outputs
        else:
            empty_dict = {dict_key: None}
            return context_repr, [empty_dict]

def get_memory_usage():
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        mem_reserved = torch.cuda.memory_reserved() / 1024**2    # MB
        return f"Memory: {mem_allocated:.1f}MB allocated, {mem_reserved:.1f}MB reserved"
    return "CUDA not available"
