import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.cuda

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
            b, c, f, t = x.shape
            print(f"[ConvExtractor] input: {x.shape} (Batch: {b}, Channels: {c}, Freq bins: {f}, Time: {t})")

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.gelu(x)
        if self.debug:
            b, c, f, t = x.shape
            print(f"[ConvExtractor] after conv1: {x.shape} (Batch: {b}, Channels: {c}, Freq bins: {f}, Time: {t})")

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
        
        # Process through attention blocks
        intermediate_outputs = []
        intermediate_outputs.append(x.clone())
        
        for block in self.attention_blocks:
            x = block(x, debug=self.debug)
            intermediate_outputs.append(x.clone())
        
        if self.debug:
            print(f"[BirdCLEFEncoder] final output: {x.shape}")
            print(f"[BirdCLEFEncoder] collected {len(intermediate_outputs) - 1} intermediate outputs")

        return x, intermediate_outputs[1:]  # Now returning meaningful intermediate outputs

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
    def update_ema(self):
        for param_q, param_k in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data = self.ema_updater.update_average(param_k.data, param_q.data)

    def forward(self, context_spectrogram, target_spectrogram, use_no_mask=False):
        """
        context_spectrogram: (B,1,D,T)
        target_spectrogram:  (B,1,D,T)
        """
        if self.debug:
            print("\n[BirdJEPA] forward - Input shapes:")
            b, c, d, t = context_spectrogram.shape
            print(f"  context_spectrogram: {context_spectrogram.shape} (Batch: {b}, Channels: {c}, Freq bins: {d}, Time: {t})")
            b, c, d, t = target_spectrogram.shape
            print(f"  target_spectrogram: {target_spectrogram.shape} (Batch: {b}, Channels: {c}, Freq bins: {d}, Time: {t})")
            print(f"  {get_memory_usage()} at start of forward pass")

        # quick mask check if needed (but here we won't do anything special)
        mask = (context_spectrogram == 0.0).any(dim=1)
        if self.debug:
            b, t = mask.shape
            print(f"  mask shape: {mask.shape} (Batch: {b}, Time: {t})")

        # encode context
        if self.debug:
            print("\n[BirdJEPA] Encoding context...")
            print(f"  {get_memory_usage()} before context encoding")
        context_repr, inter_ctx = self.context_encoder(context_spectrogram)
        if self.debug:
            b, t, d = context_repr.shape
            print(f"  context_repr shape: {context_repr.shape} (Batch: {b}, Time: {t}, Hidden dim: {d})")
            print(f"  {get_memory_usage()} after context encoding")

        if self.zero_predictor_input:
            mask_3d = mask.unsqueeze(-1).expand_as(context_repr)
            context_repr = context_repr.clone()
            context_repr[mask_3d] = 0
            if self.debug:
                print(f"  context_repr after masking: {context_repr.shape}")

        # encode target w/ no grad
        if self.debug:
            print("\n[BirdJEPA] Encoding target...")
            print(f"  {get_memory_usage()} before target encoding")
        with torch.no_grad():
            target_repr, _ = self.target_encoder(target_spectrogram)
        if self.debug:
            print(f"  target_repr shape: {target_repr.shape}")
            print(f"  {get_memory_usage()} after target encoding")

        # predictor
        if self.debug:
            print("\n[BirdJEPA] Running predictor...")
            print(f"  {get_memory_usage()} before predictor")
        pred = self.predictor(context_repr)
        if self.debug:
            print(f"  prediction shape: {pred.shape}")
            print(f"  {get_memory_usage()} after predictor")

        # decode
        if self.debug:
            print("\n[BirdJEPA] Decoding predictions...")
            print(f"  {get_memory_usage()} before decoder")
        decoded = self.decoder(pred)  # (B,T,input_dim)
        if self.debug:
            print(f"  decoded shape: {decoded.shape}")
        decoded = decoded.transpose(1, 2)  # (B,input_dim,T)
        if self.debug:
            print(f"  decoded after transpose: {decoded.shape}")
            print(f"  {get_memory_usage()} after decoder")

        return decoded, target_repr

    def compute_latent_loss(self, context_spectrogram, target_spectrogram, mask, is_eval_step=False):
        if self.debug:
            print("\n[BirdJEPA] compute_latent_loss - Input shapes:")
            if context_spectrogram.dim() == 4:
                b, c, d, t = context_spectrogram.shape
                print(f"  context_spectrogram: {context_spectrogram.shape} (Batch: {b}, Channels: {c}, Freq bins: {d}, Time: {t})")
            else:
                b, d, t = context_spectrogram.shape
                print(f"  context_spectrogram: {context_spectrogram.shape} (Batch: {b}, Freq bins: {d}, Time: {t})")
            
            if target_spectrogram.dim() == 4:
                b, c, d, t = target_spectrogram.shape
                print(f"  target_spectrogram: {target_spectrogram.shape} (Batch: {b}, Channels: {c}, Freq bins: {d}, Time: {t})")
            else:
                b, d, t = target_spectrogram.shape
                print(f"  target_spectrogram: {target_spectrogram.shape} (Batch: {b}, Freq bins: {d}, Time: {t})")
            
            b, t = mask.shape
            print(f"  mask: {mask.shape} (Batch: {b}, Time: {t})")
            print(f"  {get_memory_usage()} at start of compute_latent_loss")

        # ensure shape
        if context_spectrogram.dim() == 3:
            context_spectrogram = context_spectrogram.unsqueeze(1)
            if self.debug:
                print(f"  context_spectrogram after unsqueeze: {context_spectrogram.shape}")
        if target_spectrogram.dim() == 3:
            target_spectrogram = target_spectrogram.unsqueeze(1)
            if self.debug:
                print(f"  target_spectrogram after unsqueeze: {target_spectrogram.shape}")

        # encode context
        if self.debug:
            print("\n[BirdJEPA] Encoding context...")
            print(f"  {get_memory_usage()} before context encoding")
        context_repr, _ = self.context_encoder(context_spectrogram)
        if self.debug:
            print(f"  context_repr after encoding: {context_repr.shape}")
            print(f"  {get_memory_usage()} after context encoding")

        if self.zero_predictor_input:
            mask_3d = mask.unsqueeze(-1).expand_as(context_repr)
            context_repr = context_repr.clone()
            context_repr[mask_3d] = 0
            if self.debug:
                print(f"  context_repr after masking: {context_repr.shape}")

        # encode target
        if self.debug:
            print("\n[BirdJEPA] Encoding target...")
        with torch.no_grad():
            target_repr, _ = self.target_encoder(target_spectrogram)
        if self.debug:
            print(f"  target_repr shape: {target_repr.shape}")

        # predictor
        if self.debug:
            print("\n[BirdJEPA] Running predictor...")
        pred = self.predictor(context_repr)
        if self.debug:
            print(f"  prediction output shape: {pred.shape}")

        # compute mse in latent space
        if self.debug:
            print("\n[BirdJEPA] Computing loss...")
        mask = mask.unsqueeze(-1)  # (B,T,1)
        if self.debug:
            print(f"  expanded mask shape: {mask.shape}")
        diff = (pred - target_repr) ** 2 * mask
        if self.debug:
            print(f"  diff shape: {diff.shape}")
        total_loss = diff.sum()
        num_masked = mask.sum()
        avg_loss = total_loss / (num_masked + 1e-8)
        if self.debug or is_eval_step:
            print(f"[BirdJEPA] sum_loss={total_loss.item():.4f}, avg_loss={avg_loss.item():.4f}, num_masked={num_masked.item()}")

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
            print("\n[BirdJEPA] train_forward - Input shapes:")
            print(f"  context_spectrogram: {context_spectrogram.shape}")
            print(f"  target_spectrogram: {target_spectrogram.shape}")

        # add channel dimension
        if context_spectrogram.dim() == 3:
            context_spectrogram = context_spectrogram.unsqueeze(1)
            if self.debug:
                print(f"  context_spectrogram after unsqueeze: {context_spectrogram.shape}")
        if target_spectrogram.dim() == 3:
            target_spectrogram = target_spectrogram.unsqueeze(1)
            if self.debug:
                print(f"  target_spectrogram after unsqueeze: {target_spectrogram.shape}")

        # encode context
        if self.debug:
            print("\n[BirdJEPA] Encoding context...")
        context_repr, inter_ctx = self.context_encoder(context_spectrogram)
        if self.debug:
            print(f"  context_repr shape: {context_repr.shape}")

        # encode target
        if self.debug:
            print("\n[BirdJEPA] Encoding target...")
        with torch.no_grad():
            target_repr, inter_tgt = self.target_encoder(target_spectrogram)
        if self.debug:
            print(f"  target_repr shape: {target_repr.shape}")

        # predictor + decode
        if self.debug:
            print("\n[BirdJEPA] Running predictor and decoding...")
        pred = self.predictor(context_repr)
        if self.debug:
            print(f"  pred shape: {pred.shape}")
        decoded_pred = self.decoder(pred)    # (B,T,D)
        if self.debug:
            print(f"  decoded_pred shape: {decoded_pred.shape}")
        decoded_pred = decoded_pred.transpose(1,2)  # (B,D,T)
        if self.debug:
            print(f"  decoded_pred after transpose: {decoded_pred.shape}")

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
            print("\n[BirdJEPA] inference_forward - Input shape:")
            print(f"  x: {x.shape}")
        
        # reorder
        x = x.squeeze(1)         # (B,T,F)
        if self.debug:
            print(f"  x after squeeze: {x.shape}")
        x = x.transpose(1,2)     # (B,F,T)
        if self.debug:
            print(f"  x after transpose: {x.shape}")
        
        # encode
        if self.debug:
            print("\n[BirdJEPA] Encoding for inference...")
        context_repr, intermediate_outputs = self.context_encoder(x.unsqueeze(1))
        if self.debug:
            print(f"  context_repr shape: {context_repr.shape}")
            print(f"  Number of intermediate outputs: {len(intermediate_outputs)}")
        
        # format intermediate outputs (placeholder)
        layers = []
        return context_repr, layers

# Helper function to get memory usage in a readable format
def get_memory_usage():
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        mem_reserved = torch.cuda.memory_reserved() / 1024**2    # MB
        return f"Memory: {mem_allocated:.1f}MB allocated, {mem_reserved:.1f}MB reserved"
    return "CUDA not available"