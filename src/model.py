import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.cuda
import time
from timing_utils import Timer, timed_operation, timing_stats
import sys

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

    @timed_operation("local_attention_forward")
    def forward(self, x, debug=False):
        # x shape: (B, T, d_model)
        # No individual timers for operations within this method
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

    @timed_operation("global_attention_forward")
    def forward(self, x, debug=False):
        # x shape: (B, T, d_model)
        # No individual timers for operations within this method
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

    @timed_operation("positional_encoding_forward")
    def forward(self, x):
        # x shape: (B, T, d_model)
        # No inner timer
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

    @timed_operation("conv_feature_extractor_forward")
    def forward(self, x):
        # x shape: (B, 1, F, T)
        # No individual timers for operations within this method
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

        # feature extractor
        self.feature_extractor = BirdCLEF_ConvolutionalFeatureExtractor(in_channels=1, debug=debug)

        # pass a dummy input to figure out the flattened dimension
        with Timer("encoder_dimension_calculation", debug=debug):
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

        # Create attention blocks
        self.attention_blocks = nn.ModuleList()
        
        # If no blocks_config is provided, create a simple default configuration
        if not blocks_config:
            blocks_config = [
                {"type": "local", "window_size": 8},
                {"type": "global", "stride": 100}
            ]
            
        # Create blocks according to config
        with Timer("encoder_block_creation", debug=debug):
            for block in blocks_config:
                if block["type"] == "local":
                    self.attention_blocks.append(
                        LocalAttentionBlock(
                            d_model=hidden_dim,
                            num_heads=8,
                            window_size=block["window_size"],
                            mlp_dim=mlp_dim,
                            dropout=dropout
                        )
                    )
                elif block["type"] == "global":
                    self.attention_blocks.append(
                        GlobalAttentionBlock(
                            d_model=hidden_dim,
                            num_heads=8,
                            global_stride=block["stride"],
                            mlp_dim=mlp_dim,
                            dropout=dropout
                        )
                    )

    @timed_operation("encoder_forward")
    def forward(self, x, layer_index=None, dict_key=None):
        """
        x shape: (B,1,D,T) or (B,D,T)
        returns final hidden representation of shape (B, T, hidden_dim)
        plus a list of intermediate outputs if needed.
        
        If layer_index and dict_key are provided, returns only the output
        from the specified layer for inference optimization.
        """
        # No individual timers for operations within this method
        if x.dim() == 3:
            x = x.unsqueeze(1)

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
        
        # Store initial embedding
        initial_embedding = x.clone()
        
        # If we only need initial embedding, return early
        if layer_index == 0 and dict_key is not None:
            return {dict_key: initial_embedding}, []
        
        # Process through attention blocks
        intermediate_outputs = []
        intermediate_outputs.append({"attention_output": initial_embedding})
        
        # Process through attention blocks up to specified layer_index if provided
        for i, block in enumerate(self.attention_blocks):
            x = block(x, debug=self.debug)
            intermediate_outputs.append({"attention_output": x.clone()})
            
            # Return early if we've reached the requested layer
            if layer_index is not None and dict_key is not None and i + 1 == layer_index:
                return {dict_key: x}, []
        
        # Special handling for layer_index = -1 (last layer)
        if layer_index == -1 and dict_key is not None:
            # For layer_index = -1, return the final output directly
            return {dict_key: x}, []
            
        # If layer_index was specified but not found, or no early return requested
        if layer_index is None or dict_key is None:
            return x, intermediate_outputs[1:]  # Normal operation
        else:
            # If an invalid layer_index was requested, return the last layer output
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

    @timed_operation("predictor_forward")
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

    @timed_operation("predictor_vit_forward")
    def forward(self, x):
        # x shape: (B, T, H)
        # No individual timers for operations within this method
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

    @timed_operation("ema_update")
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

        # we preserve the same 'context_encoder' and 'target_encoder' naming
        # but we'll create them with our new block structure
        with Timer("model_initialization", debug=debug):
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
    @timed_operation("model_update_ema")
    def update_ema(self):
        for param_q, param_k in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data = self.ema_updater.update_average(param_k.data, param_q.data)

    @timed_operation("model_forward")
    def forward(self, context_spectrogram, target_spectrogram, use_no_mask=False):
        # No individual timers for operations within this method
        # encode context
        context_repr, inter_ctx = self.context_encoder(context_spectrogram)

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

    @timed_operation("model_compute_latent_loss")
    def compute_latent_loss(self, context_spectrogram, target_spectrogram, mask, is_eval_step=False):
        # No individual timers for operations within this method
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

        # encode target
        with torch.no_grad():
            target_repr, _ = self.target_encoder(target_spectrogram)

        # predictor
        pred = self.predictor(context_repr)

        # compute mse in latent space
        mask = mask.unsqueeze(-1)  # (B,T,1)
        diff = (pred - target_repr) ** 2 * mask
        total_loss = diff.sum()
        num_masked = mask.sum()
        avg_loss = total_loss / (num_masked + 1e-8)
        if self.debug or is_eval_step:
            # Use stdout.log_file.write instead of print to ensure file-only logging
            msg = f"[BirdJEPA] sum_loss={total_loss.item():.4f}, avg_loss={avg_loss.item():.4f}, num_masked={num_masked.item()}"
            if hasattr(sys.stdout, 'log_file'):
                sys.stdout.log_file.write(msg + "\n")
                sys.stdout.log_file.flush()
            else:
                print(msg)

        loss = avg_loss
        return loss, diff, pred, target_repr, context_repr

    @timed_operation("model_training_step")
    def training_step(self, context_spectrogram, target_spectrogram, mask):
        return self.compute_latent_loss(context_spectrogram, target_spectrogram, mask)

    @timed_operation("model_train_forward")
    def train_forward(self, context_spectrogram, target_spectrogram):
        """
        context_spectrogram: (B,D,T) masked
        target_spectrogram: (B,D,T) unmasked
        """
        # No individual timers for operations within this method
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

    @timed_operation("model_inference_forward")
    def inference_forward(self, x, layer_index=None, dict_key=None):
        """
        run model in inference mode (no masking).
        x: (B,1,T,F) from analysis code => we will reorder to (B,F,T)
        returns (context_repr, layers)
        
        If layer_index and dict_key are provided, returns the output from
        the specified layer directly in the second return value.
        """
        # No individual timers for operations within this method
        # reorder
        x = x.squeeze(1)         # (B,T,F)
        x = x.transpose(1,2)     # (B,F,T)
        
        # If dict_key isn't specified, use default value
        if dict_key is None:
            dict_key = "attention_output"
            
        print(f"[MODEL] Starting inference_forward with layer_index={layer_index}, dict_key={dict_key}")
            
        # encode
        context_repr, intermediate_outputs = self.context_encoder(x.unsqueeze(1), layer_index=layer_index, dict_key=dict_key)
        
        print(f"[MODEL] Encoder returned: context_repr type={type(context_repr)}, intermediate_outputs={type(intermediate_outputs)} with {len(intermediate_outputs)} items")
        
        # Check if we got a dict from the encoder (specific layer request)
        if isinstance(context_repr, dict):
            print(f"[MODEL] context_repr is a dict with keys: {list(context_repr.keys())}")
            if dict_key in context_repr:
                print(f"[MODEL] Found dict_key in context_repr, returning directly")
                # Ensure the value is a tensor
                if not isinstance(context_repr[dict_key], torch.Tensor):
                    print(f"[MODEL] Warning: value is not a tensor but {type(context_repr[dict_key])}")
                return None, context_repr
            else:
                # If the key doesn't exist but we have content, add it with the proper key
                if len(context_repr) > 0:
                    first_key = list(context_repr.keys())[0]
                    print(f"[MODEL] dict_key not found, using first key: {first_key}")
                    output_dict = {dict_key: context_repr[first_key]}
                    return None, output_dict
                else:
                    print(f"[MODEL] Empty context_repr dict, returning empty dict")
                    return None, {dict_key: None}
        
        # For normal operation or when we have a list of layer outputs
        if len(intermediate_outputs) > 0:
            print(f"[MODEL] Processing {len(intermediate_outputs)} intermediate outputs")
            # Make sure we have output with the correct key
            formatted_outputs = []
            for i, layer_output in enumerate(intermediate_outputs):
                if isinstance(layer_output, dict) and dict_key in layer_output:
                    print(f"[MODEL] Layer {i} has dict with correct key")
                    formatted_outputs.append(layer_output)
                else:
                    # If it's not a dict or doesn't have the right key, wrap it
                    print(f"[MODEL] Layer {i} needs reformatting, type={type(layer_output)}")
                    if isinstance(layer_output, torch.Tensor):
                        print(f"[MODEL] Layer {i} is tensor with shape {layer_output.shape}")
                    formatted_outputs.append({dict_key: layer_output})
            return context_repr, formatted_outputs
        else:
            # No outputs available
            print(f"[MODEL] No intermediate outputs, returning empty list with dict")
            empty_dict = {dict_key: None}
            return context_repr, [empty_dict]

# Helper function to get memory usage in a readable format
@timed_operation("get_memory_usage")
def get_memory_usage():
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        mem_reserved = torch.cuda.memory_reserved() / 1024**2    # MB
        return f"Memory: {mem_allocated:.1f}MB allocated, {mem_reserved:.1f}MB reserved"
    return "CUDA not available"