import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

# class ProbAttention(nn.Module):
#     def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
#         super(ProbAttention, self).__init__()
#         self.factor = factor
#         self.scale = scale
#         self.mask_flag = mask_flag
#         self.output_attention = output_attention
#         self.dropout = nn.Dropout(attention_dropout)

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        # Q [B, H, L, D]
        B, H, L_Q, E = Q.shape
        _, _, L_K, _ = K.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.sum(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V.shape[-1]).clone()
        else:
            assert(L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag and attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).expand(B, H, L_Q, L_V)
            scores.masked_fill_(attn_mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    # def forward(self, queries, keys, values, attn_mask):
    #     B, L, H, D = queries.shape
    #     _, S, _, _ = keys.shape

    #     queries = queries.transpose(2, 1)
    #     keys = keys.transpose(2, 1)
    #     values = values.transpose(2, 1)

    #     U_part = self.factor * np.ceil(np.log(S)).astype('int').item()
    #     u = self.factor * np.ceil(np.log(L)).astype('int').item()
        
    #     U_part = U_part if U_part < L else L
    #     u = u if u < S else S
        
    #     scores_top, index = self._prob_QK(queries, keys, u, U_part)

    #     scale = self.scale or 1./np.sqrt(D)
    #     if scale is not None:
    #         scores_top = scores_top * scale
        
    #     context = self._get_initial_context(values, L)
    #     context, attn = self._update_context(context, values, scores_top, index, L, attn_mask)
        
    #     return context.transpose(2,1).contiguous()

    def forward(self, queries, keys, values, n_heads, attn_mask):
        B, L, D = queries.shape
        _, S, _ = keys.shape
        
        head_dim = D // n_heads
        queries = queries.view(B, L, n_heads, head_dim).transpose(1, 2)
        keys = keys.view(B, S, n_heads, head_dim).transpose(1, 2)
        values = values.view(B, S, n_heads, head_dim).transpose(1, 2)

        U_part = self.factor * np.ceil(np.log(S)).astype('int').item()
        u = self.factor * np.ceil(np.log(L)).astype('int').item()
        
        U_part = U_part if U_part < L else L
        u = u if u < S else S
        
        scores_top, index = self._prob_QK(queries, keys, u, U_part)

        if self.mask_flag and attn_mask is not None:
            scores_top = scores_top.masked_fill(attn_mask[:, :, :U_part], float('-inf'))

        scale = self.scale or 1./np.sqrt(head_dim)
        if scale is not None:
            scores_top = scores_top * scale
        
        context = self._get_initial_context(values, L)
        context, attn = self._update_context(context, values, scores_top, index, L, attn_mask)
        
        context = context.transpose(2,1).contiguous().view(B, L, D)
        return context

# class InformerEncoder(nn.Module):
#     def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1, activation="relu", factor=5):
#         super(InformerEncoder, self).__init__()
#         d_ff = d_ff or 4*d_model
#         self.attention = ProbAttention(mask_flag=True, attention_dropout=dropout, factor=factor) # mask_flag=True means no look on future data
#         self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
#         self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
#         self.activation = F.relu if activation == "relu" else F.gelu

class InformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1, activation="relu", factor=5):
        super(InformerEncoder, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = ProbAttention(mask_flag=True, attention_dropout=dropout, factor=factor) # mask_flag=True means no look on future data
        self.n_heads = n_heads
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        new_x = self.attention(x, x, x, self.n_heads, attn_mask)
        x = x + self.dropout(new_x)

        x = self.norm1(x)

        y = x.transpose(1, 2)
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y))
        x = x + y.transpose(1, 2)

        return self.norm2(x)
        

# class Informer(nn.Module):
#     def __init__(self, input_dim, output_dim, d_model=512, n_heads=8, n_layers=3, dropout=0.1, factor=5):
#         super(Informer, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.d_model = d_model

#         self.embedding = nn.Linear(input_dim, d_model)
#         self.pos_encoder = PositionalEncoding(d_model)
        
#         self.encoder_layers = nn.ModuleList([
#             InformerEncoder(d_model, n_heads, dropout=dropout, factor=factor)
#             for _ in range(n_layers)
#         ])
        
#         self.decoder = nn.Linear(d_model, output_dim)

class Informer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, n_heads, n_layers=3, dropout=0.1, factor=5, l1_lambda=0, l2_lambda=0):
        super(Informer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        self.encoder_layers = nn.ModuleList([
            InformerEncoder(d_model, n_heads, dropout=dropout, factor=factor)
            for _ in range(n_layers)
        ])
        
        self.decoder = nn.Linear(d_model, output_dim)

    def get_l1_loss(self):
        l1_loss = 0
        for param in self.parameters():
            l1_loss += torch.sum(torch.abs(param))
        return self.l1_lambda * l1_loss

    def get_l2_loss(self):
        l2_loss = 0
        for param in self.parameters():
            l2_loss += torch.sum(param ** 2)
        return self.l2_lambda * l2_loss

    def forward(self, src, mask=None):
        src = self.embedding(src)
        src = self.pos_encoder(src)
        
        for enc_layer in self.encoder_layers:
            # src = enc_layer(src)
            src = enc_layer(src, mask)
        
        last_hidden = src[:, -1, :]
        output = self.decoder(last_hidden)
        
        return output


# Usage:
# model = Informer(input_dim=X_train.shape[2], output_dim=10, d_model=512, n_heads=8, n_layers=3, dropout=0.1)