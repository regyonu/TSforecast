import torch
import torch.nn as nn


class Model(nn.Module):
    """
    PURE GRU BASELINE (vanilla)
    - Raw GRU encoder
    - Last hidden state aggregation
    - Linear forecasting head
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        self.pred_len = configs.pred_len
        self.c_out = configs.c_out
        self.hidden_dim = configs.d_model

        # PURE GRU (no embedding layer)
        self.gru = nn.GRU(
            input_size=configs.enc_in,
            hidden_size=self.hidden_dim,
            num_layers=configs.e_layers,
            batch_first=True,
            dropout=configs.dropout if configs.e_layers > 1 else 0
        )

        # Forecast head
        self.projection = nn.Linear(self.hidden_dim, self.pred_len * self.c_out)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):

        # 1. GRU encoding (raw input)
        out, _ = self.gru(x_enc)

        # 2. Take last hidden state
        out = out[:, -1, :]

        # 3. Forecast
        out = self.projection(out)

        # 4. Reshape to [B, pred_len, c_out]
        out = out.view(out.size(0), self.pred_len, self.c_out)

        return out
