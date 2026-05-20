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

        if self.use_revin:
            self.revin = RevIN(configs.enc_in, affine=True)

        self.embedding = DataEmbedding(
            configs.enc_in,
            configs.d_model,
            configs.embed,       
            configs.freq,        
            configs.dropout
        )

        
        self.gru = nn.GRU(
            input_size=configs.d_model,   
            hidden_size=configs.d_model,
            num_layers=configs.e_layers,
            batch_first=True,
            dropout=configs.dropout if configs.e_layers > 1 else 0
        )
        self.projection = nn.Linear(configs.d_model, self.pred_len * self.c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.use_revin:
            x_enc = self.revin(x_enc, 'norm')

        x = self.embedding(x_enc, x_mark_enc)

       
        out, _ = self.gru(x)
        out = out[:, -1, :]  

        out = self.projection(out)
        out = out.view(out.size(0), self.pred_len, self.c_out)

        if self.use_revin:
            out = self.revin(out, 'denorm')

        return out
