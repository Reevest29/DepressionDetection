import torch
from torch import nn
class LSTM_CNN(nn.Module):

    def __init__(self, in_size, num_hidden_layers, hidden_dim, out_size):
        super(LSTM_CNN, self).__init__()
        self.hidden_dim = hidden_dim

        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(in_size, hidden_dim)

        self.block_

        # The linear layer that maps from hidden state space to out space
        self.out_layer = nn.Linear(hidden_dim, out_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out_space = self.out_layer(lstm_out.view(len(x), -1))
        return out_space
    
if __name__ == "__main__":
    lstm_cnn = LSTM_CNN(in_size=10,num_hidden_layers=5,hidden_dim=100,out_size=2)

    x = torch.rand((5,10))
    out = lstm_cnn(x)
    print(out.shape)
    assert(torch.all(tuple(out.shape) == (5,2)))

git config --global user.name "Thomas Reeves"
git config --global user.email "tjtommy5000@gmail.com"