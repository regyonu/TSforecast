import torch
import torch.nn as nn

from utils.revin import RevIN


class Model(nn.Module):
    """
    Pure RNN baseline with:
    - RevIN normalization
    - Linear input projection
    - Sequence-output forecasting 

    Architecture:
    RevIN -> Linear Projection -> RNN -> Sequence Forecast Head
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        # =========================================================
        # Basic configs
        # =========================================================
        self.pred_len = configs.pred_len
        self.c_out = configs.c_out
        self.d_model = configs.d_model

        self.use_revin = getattr(configs, 'use_revin', True)

        # =========================================================
        # RevIN
        # =========================================================
        if self.use_revin:
            self.revin = RevIN(
                num_features=configs.enc_in,
                affine=True
            )

        # =========================================================
        # Input projection
        # [B, T, enc_in] -> [B, T, d_model]
        # =========================================================
        self.input_proj = nn.Linear(
            configs.enc_in,
            self.d_model
        )

        # =========================================================
        # RNN backbone
        # =========================================================
        self.rnn = nn.RNN(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=configs.e_layers,
            nonlinearity='tanh',
            batch_first=True,
            dropout=configs.dropout if configs.e_layers > 1 else 0
        )

        # =========================================================
        # Forecast head
        # Applied timestep-wise
        # [B, pred_len, d_model]
        # -> [B, pred_len, c_out]
        # =========================================================
        self.projection = nn.Linear(
            self.d_model,
            self.c_out
        )

    def forward(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        mask=None
    ):

        # =========================================================
        # 1. RevIN normalization
        # =========================================================
        if self.use_revin:
            x_enc = self.revin(x_enc, 'norm')

        # =========================================================
        # 2. Input projection
        # =========================================================
        x = self.input_proj(x_enc)

        # x shape:
        # [B, seq_len, d_model]

        # =========================================================
        # 3. RNN forward
        # =========================================================
        out, _ = self.rnn(x)

        # out shape:
        # [B, seq_len, d_model]

        # =========================================================
        # 4. Take last pred_len hidden states
        # =========================================================
        out = out[:, -self.pred_len:, :]

        # shape:
        # [B, pred_len, d_model]

        # =========================================================
        # 5. Forecast projection
        # =========================================================
        out = self.projection(out)

        # shape:
        # [B, pred_len, c_out]

        # =========================================================
        # 6. RevIN denormalization
        # =========================================================
        if self.use_revin:
            out = self.revin(out, 'denorm')

        return out
