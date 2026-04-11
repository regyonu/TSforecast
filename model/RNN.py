import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.c_out = configs.c_out
        
        self.rnn = nn.RNN(
            input_size=configs.enc_in,
            hidden_size=configs.d_model,
            num_layers=configs.e_layers,
            batch_first=True,
            dropout=configs.dropout if configs.e_layers > 1 else 0
        )
        self.projection = nn.Linear(configs.d_model, self.pred_len * self.c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        out, _ = self.rnn(x_enc)
        out = out[:, -1, :] 
        out = self.projection(out)
        return out.view(out.size(0), self.pred_len, self.c_out)