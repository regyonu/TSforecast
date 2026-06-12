import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
from layers.Autoformer_EncDec import series_decomp


class Model(nn.Module):
    """
    Transformer + decomposition:
      trend     → Linear(seq_len → pred_len)
      seasonal  → Transformer encoder-decoder
      output    = trend_pred + seasonal_pred
    """

    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = False

        self.decomposition = series_decomp(configs.moving_avg)

        if configs.channel_independence:
            self.enc_in = 1
            self.dec_in = 1
            self.c_out  = 1
        else:
            self.enc_in = configs.enc_in
            self.dec_in = configs.dec_in
            self.c_out  = configs.c_out

        self.trend_projection = nn.Linear(configs.seq_len, configs.pred_len, bias=True)

        self.enc_embedding = DataEmbedding(
            self.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor,
                                      attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )

        self.dec_embedding = DataEmbedding(
            self.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor,
                                      attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads
                    ),
                    AttentionLayer(
                        FullAttention(False, configs.factor,
                                      attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for _ in range(configs.d_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, self.c_out, bias=True)
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

            label_len = x_dec.shape[1] - self.pred_len
            x_dec = x_dec.clone()
            x_dec[:, :label_len, :] = (x_dec[:, :label_len, :] - means) / stdev

        # ── 1. Decompose ──────────────────────────────────────────────────
        seasonal_enc, trend_enc = self.decomposition(x_enc)
        seasonal_dec, trend_dec = self.decomposition(x_dec)

        # ── 2. Trend branch ───────────────────────────────────────────────
        trend_out = self.trend_projection(
            trend_enc.permute(0, 2, 1)
        ).permute(0, 2, 1)                                     # (B, pred_len, C)

        # ── 3. Seasonal branch ────────────────────────────────────────────
        enc_out = self.enc_embedding(seasonal_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.dec_embedding(seasonal_dec, x_mark_dec)
        seasonal_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        seasonal_out = seasonal_out[:, -self.pred_len:, :]     # (B, pred_len, C)

        # ── 4. Combine ────────────────────────────────────────────────────
        out = seasonal_out + trend_out                         # (B, pred_len, C)

        if self.use_norm:
            out = out * stdev + means

        return out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
