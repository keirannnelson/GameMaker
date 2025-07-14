import torch
from torch import nn

class GruPlayerPredictor(nn.Module):
    def __init__(self,
                 gru_input_size,
                 gru_hidden_size,
                 linear_output_size,
                 gru_num_layers=1
                 ):
        
        super().__init__()

        self.linear_input_size = gru_hidden_size * 3
        self.gru = nn.GRU(gru_input_size, gru_hidden_size, gru_num_layers, batch_first=True)
        self.linear = nn.Linear(self.linear_input_size, linear_output_size)

    def forward(self, x):
        x, h_n = self.gru(x)
        x = self.linear(torch.reshape(x, (-1, self.linear_input_size)))
        return x
    
    def loss(self, y_model, y_target):
        mse = nn.MSELoss()
        return mse(y_model, y_target)