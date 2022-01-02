import torch
import torch.nn as nn
import torch.nn.functional as F
from data import WMT14
import math
import numpy as np
from torch.autograd import Variable
from prettytable import PrettyTable


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        enc_layers: int = 6,
        dec_layers: int = 6,
        d_model: int = 512,
        dff: int = 2048,
        num_heads: int = 8,
        max_seq_len: int = 100,
        enc_dp: float = 0.1,
        dec_dp: float = 0.1,
    ):
        super(Transformer, self).__init__()
        self.encoder_layers = enc_layers
        self.decoder_layers = dec_layers
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.dff = dff
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.enc_dropout = enc_dp
        self.dec_dropout = dec_dp
        self.encoder = Encoder(
            self.encoder_layers,
            self.src_vocab_size,
            self.d_model,
            self.dff,
            self.num_heads,
            self.enc_dropout,
            self.max_seq_len,
        )
        self.decoder = Decoder(
            self.tgt_vocab_size,
            self.d_model,
            self.dff,
            self.num_heads,
            self.decoder_layers,
            self.dec_dropout,
            self.max_seq_len,
        )
        self.last = nn.Linear(self.d_model, self.tgt_vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask):
        x_ = self.encoder(src, src_mask)
        x = self.decoder(tgt, x_, src_mask, tgt_mask)
        x = self.last(x)
        x = F.log_softmax(x, dim=-1)
        return x


class Encoder(nn.Module):
    def __init__(
            self,
            num_enc_layers: int,
            src_vocab_size: int,
            d_model: int,
            dff: int,
            num_heads: int,
            enc_dropout: float,
            max_seq_len: int,
            pad_idx: int = 0
    ):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.dff = dff
        self.num_heads = num_heads
        self.enc_dp = enc_dropout
        self.max_seq_len = max_seq_len
        self.src_vocab_size = src_vocab_size
        self.enc_embed = nn.Embedding(
            self.src_vocab_size,
            self.d_model,
            padding_idx=pad_idx
        )
        self.pe = PositionalEncoding(
            self.d_model,
            self.enc_dp,
            self.max_seq_len
        )
        self.layers = nn.ModuleList(
            EncoderLayer(
                MultiHeadAttentionLayer(
                    self.d_model,
                    self.num_heads,
                    self.enc_dp
                ),
                FFN(self.d_model, self.dff, self.enc_dp),
                self.d_model,
                self.enc_dp
            ) for _ in range(num_enc_layers)
        )

    def forward(self, x, src_mask):
        x = self.enc_embed(x) * math.sqrt(self.d_model)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, self_attention, ffn, num_features: int, dropout: float):
        super(EncoderLayer, self).__init__()
        self.self_attention = self_attention
        self.ffn = ffn
        self.num_features = num_features
        self.dp_attn = nn.Dropout(p=dropout)
        self.dp_ffn = nn.Dropout(p=dropout)
        self.norm_attn = LayerNorm(num_features)
        self.norm_ffn = LayerNorm(num_features)

    def forward(self, x, mask=None):
        x_ = self.self_attention(x, x, x, mask)
        x = self.norm_attn(x + self.dp_attn(x_))
        x_ = self.ffn(x)
        x = self.norm_ffn(x + self.dp_ffn(x_))

        return x


class Decoder(nn.Module):
    def __init__(
            self,
            tgt_vocab_size: int,
            d_model: int,
            dff: int,
            num_heads: int,
            num_dec_layers: int,
            dec_dropout: float,
            max_seq_len: int,
            pad_idx: int = 0
    ):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.dec_dp = dec_dropout
        self.dec_embed = nn.Embedding(
            tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.pe = PositionalEncoding(self.d_model, dec_dropout, max_seq_len)
        self.layers = nn.ModuleList([
            DecoderLayer(
                MultiHeadAttentionLayer(self.d_model, num_heads, self.dec_dp),
                MultiHeadAttentionLayer(self.d_model, num_heads, self.dec_dp),
                FFN(self.d_model, dff, self.dec_dp),
                self.d_model,
                self.dec_dp,
            ) for _ in range(num_dec_layers)
        ])

    def forward(self, x, enc_x,  enc_mask, dec_mask):
        x = self.dec_embed(x) * math.sqrt(self.d_model)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, enc_x, enc_mask, dec_mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(
            self,
            mattn,
            attn,
            ffn,
            num_features: int,
            dec_dropout: float
    ):
        super(DecoderLayer, self).__init__()
        self.num_features = num_features
        self.dec_dp = dec_dropout
        self.m_attention = mattn
        self.attention = attn
        self.ffn = ffn
        self.norm_mattn = LayerNorm(self.num_features)
        self.norm_attn = LayerNorm(self.num_features)
        self.norm_ffn = LayerNorm(self.num_features)
        self.dp = nn.Dropout(p=self.dec_dp)

    def forward(self, x, enc_x, enc_mask,  dec_mask):
        mem = enc_x
        x_ = self.m_attention(x, x, x, dec_mask)
        x = self.norm_mattn(x + self.dp(x_))
        x_ = self.attention(x, mem, mem, enc_mask)
        x = self.norm_attn(x + self.dp(x_))
        x_ = self.ffn(x)
        x = self.norm_ffn(x + self.dp(x_))

        return x


class FFN(nn.Module):
    def __init__(self, d_model: int, dff: int, ffn_dropout: float):
        super(FFN, self).__init__()
        self.w1 = nn.Linear(d_model, dff)
        self.w2 = nn.Linear(dff, d_model)
        self.dp = nn.Dropout(p=ffn_dropout)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.w2(self.dp(self.act(self.w1(x))))
        return x


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super(MultiHeadAttentionLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.linear_Q = nn.Linear(d_model, d_model)
        self.linear_K = nn.Linear(d_model, d_model)
        self.linear_V = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.attention = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, Q, K, V, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        n_batches = Q.size(0)

        Q = self.linear_Q(Q)\
            .view(n_batches, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.linear_K(K)\
            .view(n_batches, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.linear_V(V)\
            .view(n_batches, -1, self.num_heads, self.d_k).transpose(1, 2)

        x, self.attention = scaled_dot_product_attention(Q, K, V,
                                                         mask=mask,
                                                         dropout=self.dropout)

        x = x.transpose(1, 2).contiguous()\
            .view(n_batches, -1, self.num_heads * self.d_k)
        x = self.linear_out(x)
        return x

class LayerNorm(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-6):
        super(LayerNorm, self).__init__()
        self.p1 = nn.Parameter(torch.ones(num_features))
        self.p2 = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        x = self.p1 * (x - mean) / (std + self.eps) + self.p2
        return x


class PositionalEncoding(nn.Module):
    """PE function from http://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding"""
    def __init__(self, d_model: int, enc_dropout: float, max_seq_len: int):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=enc_dropout)
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + torch.autograd.Variable(self.pe[:, :x.size(1)],
                                        requires_grad=False)
        x = self.dropout(x)
        return x

def scaled_dot_product_attention(Q, K, V, mask=None, dropout=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, V), p_attn



def count_parameters(model: nn.Module):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param

    print(table)
    print(f"Total trainable params: {total_params}")
    return total_params


def subsequent_mask(size: int):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def  main():
    dataset = WMT14('wmt14')
    dataset.load_vocab()
    d_model = 512
    max_seq_len = 50
    batch_size = 1
    data_gen = dataset.data_generator(batch_size, max_seq_len)

    src_vocab_size = len(dataset.src_idx2word)
    tgt_vocab_size = len(dataset.tgt_idx2word)
    src_embed_layer = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
    tgt_embed_layer = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)

    for src, tgt in data_gen:
        print(src.shape)
        print(tgt.shape)
        src = torch.tensor(src)
        print(src)
        src_mask = (src != 0).unsqueeze(-2)
        print(src_mask)
        tgt = torch.tensor(tgt)
        print(tgt)
        tgt = tgt[:, :-1]
        print(tgt)
        tgt_y = tgt[:, 1:]
        print(tgt_y)
        tgt_mask = (tgt != 0).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        print(tgt_mask)
        print(tgt_mask.size())
        src_embed = src_embed_layer(src)
        tgt_embed = tgt_embed_layer(tgt)
        print(src_embed.size())
        print(tgt_embed.size())
        break

    encoder = Encoder(
        num_enc_layers=6,
        src_vocab_size=src_vocab_size,
        d_model=d_model,
        dff=2048,
        num_heads=8,
        enc_dropout=0.2,
        max_seq_len=max_seq_len,
    )

    test_src = torch.randint(1000, (4, 50))
    print(test_src.size())
    out = encoder(test_src, src_mask)
    print(out.size())

    #count_parameters(encoder)

    decoder = Decoder(
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        dff=2048,
        num_heads=8,
        num_dec_layers=6,
        dec_dropout=0.1,
        max_seq_len=max_seq_len,
        pad_idx=0,
    )
    test_tgt = torch.randint(1000, (4, 50))
    test_tgt = test_tgt[:,:-1]
    res = decoder(test_tgt, out, src_mask, tgt_mask)
    print(res.size())

    transformer = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        enc_layers=6,
        dec_layers=6,
        d_model=512,
        dff=2048,
        num_heads=8,
        max_seq_len=100,
        enc_dp=0.1,
        dec_dp=0.1,
    )

    tout = transformer(test_src, test_tgt, src_mask, tgt_mask)
    print(tout.size())



if __name__ == "__main__":
    main()
