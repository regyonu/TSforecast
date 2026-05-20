import torch
import torch.nn as nn
from layers.Embed import DataEmbedding
from utils.revin import RevIN

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.c_out = configs.c_out
        self.use_revin = getattr(configs, 'use_revin', True)

        # RevIN on raw input features
        if self.use_revin:
            self.revin = RevIN(configs.enc_in, affine=True)

        # DataEmbedding: projects enc_in -> d_model, adds positional + temporal
        self.embedding = DataEmbedding(
            configs.enc_in,
            configs.d_model,
            configs.embed,       
            configs.freq,        
            configs.dropout
        )

        self.lstm = nn.LSTM(
            input_size=configs.d_model,   # now takes d_model, not enc_in
            hidden_size=configs.d_model,
            num_layers=configs.e_layers,
            batch_first=True,
            dropout=configs.dropout if configs.e_layers > 1 else 0
        )
        self.projection = nn.Linear(configs.d_model, self.pred_len * self.c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # 1. Normalize
        if self.use_revin:
            x_enc = self.revin(x_enc, 'norm')

        # 2. Embed: [B, T, enc_in] -> [B, T, d_model]
        x = self.embedding(x_enc, x_mark_enc)

        # 3. LSTM
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last timestep: [B, d_model]

        # 4. Project to [B, pred_len, c_out]
        out = self.projection(out)
        out = out.view(out.size(0), self.pred_len, self.c_out)

        # 5. Denormalize
        if self.use_revin:
            out = self.revin(out, 'denorm')

        return out
