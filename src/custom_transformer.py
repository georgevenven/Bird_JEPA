import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, pos_encodings, mask=None):
    matmul_qk = torch.matmul(Q, K.transpose(-2, -1))

    # only add if pos enc is relative 
    if isinstance(pos_encodings, torch.Tensor) and isinstance(matmul_qk, torch.Tensor):
        matmul_qk += pos_encodings

    d_k = Q.size(-1)
    scaled_attention_logits = matmul_qk / math.sqrt(d_k)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)   

    attention_weights = F.softmax(scaled_attention_logits, dim=-1)

    output = torch.matmul(attention_weights, V)
    return output, attention_weights

class CustomMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, pos_enc_type, max_len=1024):
        super(CustomMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.pos_enc_type = pos_enc_type

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

        if pos_enc_type == "relative":
            self.max_len = max_len
            self.Er = nn.Parameter(torch.randn(max_len, self.depth))
            self.register_buffer("zero_pad", torch.zeros((1, 1, 1, max_len)))

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, Q, K, V, mask):
        batch_size = Q.size(0)

        Q = self.q_linear(Q)
        K = self.k_linear(K)
        V = self.v_linear(V)

        Q_split = self.split_heads(Q, batch_size)
        K_split = self.split_heads(K, batch_size)
        V_split = self.split_heads(V, batch_size)

        if self.pos_enc_type == "relative":
            seq_len = Q.size(1)
            if seq_len > self.max_len:
                raise ValueError("Sequence length exceeds model capacity")

            Er = self.Er[:seq_len, :]
            QEr = torch.matmul(Q_split, Er.transpose(-2, -1))
            Srel = self.skew(QEr)
            output, attention_weights = scaled_dot_product_attention(Q_split, K_split, V_split, Srel, mask)
        else:
            output, attention_weights = scaled_dot_product_attention(Q_split, K_split, V_split, mask)

        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)
        output = self.dense(output)

        return {'output': output, 'attention_weights': attention_weights, 'Q': Q, 'K': K, 'V': V}

    def skew(self, QEr):
        batch_size, num_heads, seq_len, _ = QEr.shape
        zero_pad = torch.zeros((batch_size, num_heads, seq_len, 1), device=QEr.device, dtype=QEr.dtype)
        
        padded_QEr = torch.cat([zero_pad, QEr], dim=-1)
        reshaped = padded_QEr.reshape(batch_size, num_heads, seq_len + 1, seq_len)
        Srel = reshaped[:, :, 1:].contiguous()
        return Srel

class CustomEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, ffn_dim, dropout, pos_enc_type, length):
        super(CustomEncoderBlock, self).__init__()

        self.self_attn = CustomMultiHeadAttention(d_model, num_heads, pos_enc_type, length)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.feed_forward1 = nn.Linear(d_model, ffn_dim)
        self.feed_forward2 = nn.Linear(ffn_dim, d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x_norm = self.layer_norm1(x)
        attn_result = self.self_attn(Q=x_norm, K=x_norm, V=x_norm, mask=mask)
        attention_graph = (attn_result['attention_weights'])

        attn_output = self.dropout(attn_result['output'])
        attn_output += x  # Residual connection

        mlp_input_norm = self.layer_norm2(attn_output)
        ff_output = self.feed_forward1(mlp_input_norm)
        ff_output_gelu = F.gelu(ff_output)
        ff_output = self.feed_forward2(ff_output_gelu)
        ff_output = self.dropout(ff_output)
        ff_output += attn_output

        output_dict = {
            'Q': attn_result['Q'],
            'K': attn_result['K'],
            'V': attn_result['V'],
            'attention_output': attn_output,
            'intermediate_residual_stream': x,
            'feed_forward_output_gelu': ff_output_gelu,
            'feed_forward_output': ff_output
        }

        return output_dict