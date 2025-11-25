"""
@author: Lochana Marasinghe
@date: 10/17/2025
@description: 
"""
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size,  dropout=0):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last timestep
        out = self.fc(out)
        return out

# class GRUModel(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2, bidirectional=False):
#         super(GRUModel, self).__init__()
#         self.gru = nn.GRU(
#             input_size=input_size,   # number of features per time step
#             hidden_size=hidden_size, # number of hidden units
#             num_layers=num_layers,   # stacked GRU layers
#             batch_first=True,        # (batch, seq, feature)
#             dropout=dropout,
#             bidirectional=bidirectional
#         )
#         direction_factor = 2 if bidirectional else 1
#         self.fc = nn.Linear(hidden_size * direction_factor, output_size)
#
#     def forward(self, x, lengths):
#         packed = torch.nn.utils.rnn.pack_padded_sequence(
#             x, lengths.cpu(), batch_first=True, enforce_sorted=False
#         )
#         packed_out, _ = self.gru(packed)
#
#         out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
#
#         batch_idx = torch.arange(out.size(0))
#         last_idx = lengths - 1
#         out_last = out[batch_idx, last_idx, :]
#         out = self.fc(out_last)
#         return out


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2, bidirectional=False):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,   # number of features per time step
            hidden_size=hidden_size, # number of hidden units
            num_layers=num_layers,   # stacked GRU layers
            batch_first=True,        # (batch, seq, feature)
            dropout=dropout,
            bidirectional=bidirectional
        )
        direction_factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * direction_factor, output_size)

    def forward(self, x, lengths):
        out, _ = self.gru(x)            # out: (batch, seq, hidden)
        out = out[:, -1, :]             # take last time step
        out = self.fc(out)              # predict yield
        return out