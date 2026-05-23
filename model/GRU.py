import torch
import torch.nn as nn
from utils.revin import RevIN


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.pred_len = configs.pred_len
        self.c_out = configs.c_out
        self.d_model = configs.d_model
        self.use_revin = getattr(configs, 'use_revin', True)

        # RevIN normalization
        # Used across all models for fair comparison
        if self.use_revin:
            self.revin = RevIN(configs.enc_in, affine=True)

        # Simple linear projection
        # No temporal or positional encoding
        self.input_proj = nn.Linear(configs.enc_in, self.d_model)

        # Pure GRU backbone
        self.gru = nn.GRU(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=configs.e_layers,
            batch_first=True,
            dropout=configs.dropout if configs.e_layers > 1 else 0
        )

        # Forecast head
        self.projection = nn.Linear(
            self.d_model,
            self.pred_len * self.c_out
        )

    def forward(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        mask=None
    ):

        # Normalize
        if self.use_revin:
            x_enc = self.revin(x_enc, 'norm')

        # Input projection
        # x_mark_enc is intentionally ignored
        x = self.input_proj(x_enc)  # [B, T, d_model]

        # GRU forward
        out, _ = self.gru(x)       # [B, T, d_model]

        
        out = out[:, -1, :]      # [B, d_model]

        # Forecast projection
        out = self.projection(out)

        # Reshape to forecasting format
        out = out.view(
            out.size(0),
            self.pred_len,
            self.c_out
        )                           # [B, pred_len, c_out]

        # Denormalize
        if self.use_revin:
            out = self.revin(out, 'denorm')

        return out
