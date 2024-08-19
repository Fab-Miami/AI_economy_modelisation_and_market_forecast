import torch
import torch.nn as nn
import torch.nn.functional as F

class Informer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=512, n_heads=8, n_layers=3, dropout=0.1, factor=5):
        super(Informer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model

        # Embedding layer for the input
        self.embedding = nn.Linear(input_dim, d_model)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Fully connected output layer
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        x = self.embedding(x)  # Shape: (batch_size, sequence_length, d_model)
        x = x.permute(1, 0, 2)  # Shape: (sequence_length, batch_size, d_model)
        x = self.transformer_encoder(x)  # Shape: (sequence_length, batch_size, d_model)
        x = x[-1, :, :]  # Take the last time step's output (many-to-one)
        x = self.fc(x)  # Shape: (batch_size, output_dim)
        return x
    

# import torch
# import torch.nn as nn
# import math

# class ProbAttention(nn.Module):
#     def __init__(self, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
#         super(ProbAttention, self).__init__()
#         self.factor = factor
#         self.scale = scale
#         self.output_attention = output_attention
#         self.dropout = nn.Dropout(attention_dropout)

#     def _prob_QK(self, Q, K, sample_k, n_top):
#         # Q [B, H, L, D]
#         B, H, L_Q, D = Q.shape
#         _, _, L_K, _ = K.shape

#         # calculate the sampled Q_K
#         K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, D)
#         index_sample = torch.randint(L_K, (L_Q, sample_k))
#         K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
#         Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

#         # find the Top_k query with sparisty measurement
#         M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
#         M_top = M.topk(min(n_top, L_Q), sorted=False)[1]  # Ensure n_top does not exceed L_Q

#         # use the reduced Q to calculate Q_K
#         Q_reduce = Q[torch.arange(B)[:, None, None],
#                     torch.arange(H)[None, :, None],
#                     M_top, :]
#         Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))

#         return Q_K, M_top


#     def forward(self, queries, keys, values, attn_mask):
#         B, L, H, D = queries.shape
#         _, S, _, _ = keys.shape

#         U_part = self.factor * math.ceil(math.log(L))
#         u = self.factor * math.ceil(math.log(S))
        
#         U_part = min(U_part, L)
#         u = min(u, S)
        
#         scores_top, index = self._prob_QK(queries, keys, u, U_part)
        
#         scale = self.scale or 1./math.sqrt(D)
#         if scale is not None:
#             scores_top = scores_top * scale
        
#         if attn_mask is not None:
#             attn_mask = attn_mask.unsqueeze(1).expand(B, H, L, S)
#             attn_mask = attn_mask.gather(dim=-1, index=index)
#             scores_top = scores_top.masked_fill(attn_mask, -float('inf'))

#         scores_top = self.dropout(torch.softmax(scores_top, dim=-1))

#         # values = values.unsqueeze(-3).expand(B, H, S, D)
#         values = values.expand(B, H, S, D)

#         values = values.gather(dim=2, index=index.unsqueeze(-1).expand(B, H, U_part, D))
        
#         out = torch.matmul(scores_top, values)

#         return out.transpose(2, 1).contiguous().view(B, L, -1)

# class Informer(nn.Module):
#     def __init__(self, input_dim, output_dim, d_model=512, n_heads=8, n_layers=3, dropout=0.1, factor=5):
#         super(Informer, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.d_model = d_model
#         self.n_heads = n_heads

#         # Embedding layer for the input
#         self.embedding = nn.Linear(input_dim, d_model)

#         # ProbAttention layers
#         self.attention_layers = nn.ModuleList([
#             ProbAttention(factor=factor, scale=None, attention_dropout=dropout)
#             for _ in range(n_layers)
#         ])

#         # Feed-forward layers
#         self.feed_forward = nn.Sequential(
#             nn.Linear(d_model, d_model * 4),
#             nn.ReLU(),
#             nn.Linear(d_model * 4, d_model)
#         )

#         # Layer normalization
#         self.layer_norm1 = nn.LayerNorm(d_model)
#         self.layer_norm2 = nn.LayerNorm(d_model)

#         # Dropout
#         self.dropout = nn.Dropout(dropout)

#         # Fully connected output layer
#         self.fc = nn.Linear(d_model, output_dim)

#     def forward(self, x):
#         # x shape: (batch_size, sequence_length, input_dim)
#         x = self.embedding(x)  # Shape: (batch_size, sequence_length, d_model)
        
#         B, L, D = x.shape
#         H = self.n_heads
#         x = x.view(B, L, H, D // H)

#         for attention in self.attention_layers:
#             # Self-attention
#             residual = x
#             x = self.layer_norm1(x.view(B, L, -1)).view(B, L, H, -1)
#             x = attention(x, x, x, attn_mask=None)
#             x = x.view(B, L, -1)
#             x = self.dropout(x)
#             x = residual.view(B, L, -1) + x

#             # Feed-forward
#             residual = x
#             x = self.layer_norm2(x)
#             x = self.feed_forward(x)
#             x = self.dropout(x)
#             x = residual + x

#         x = x[:, -1, :]  # Take the last time step's output (many-to-one)
#         x = self.fc(x)  # Shape: (batch_size, output_dim)
#         return x