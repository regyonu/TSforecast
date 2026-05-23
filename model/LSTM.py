import torch
import torch.nn as nn
from utils.revin import RevIN


class Model(nn.Module):
    """
    Baseline Recurrent Model:
    - RevIN normalization (optional, for fair comparison)
    - Linear input projection
    - LSTM backbone
    - Last step (final hidden state) aggregation
    - Linear forecasting head
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        self.pred_len = configs.pred_len
        self.c_out = configs.c_out
        self.d_model = configs.d_model
        self.use_revin = getattr(configs, "use_revin", True)

        # RevIN (applied consistently across models for fairness)
        if self.use_revin:
            self.revin = RevIN(configs.enc_in, affine=True)

        # Input projection (simple feature lifting only)
        self.input_proj = nn.Linear(configs.enc_in, self.d_model)

        # LSTM backbone (pure recurrent model)
        self.lstm = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=configs.e_layers,
            batch_first=True,
            dropout=configs.dropout if configs.e_layers > 1 else 0
        )

        # Forecast head
        self.projection = nn.Linear(self.d_model, self.pred_len * self.c_out)


    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):

        # -----------------------
        # 1. Normalization
        # -----------------------
        if self.use_revin:
            x_enc = self.revin(x_enc, mode='norm')

        # -----------------------
        # 2. Input projection
        # -----------------------
        x = self.input_proj(x_enc)  # [B, T, d_model]

        # -----------------------
        # 3. lstm encoding
        # -----------------------
        out, _ = self.lstm(x)  # [B, T, d_model]

        # -----------------------
        # 4. Temporal aggregation (last step)
        # -----------------------
        out = out[:, -1, :]  # [B, d_model]

        # -----------------------
        # 5. Forecasting head
        # -----------------------
        out = self.projection(out)  # [B, pred_len * c_out]

        out = out.view(
            out.size(0),
            self.pred_len,
            self.c_out
        )

        # -----------------------
        # 6. Denormalization
        # -----------------------
        if self.use_revin:
            out = self.revin(out, mode='denorm')

        return out
