import torch
from torch import nn
from torchvision import models
class LSTM_CNN(nn.Module):

    def __init__(self, input_size, lstm_input_size, num_lstm_layers, lstm_hidden_dim, out_size):
        super(LSTM_CNN, self).__init__()
        self.lstm_hidden_dim = lstm_hidden_dim

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=32,kernel_size=(5,5)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(5,5)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=16,kernel_size=(5,5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Flatten()
        )


        cnn_out_size = None
        with torch.no_grad():
            test_ten = torch.ones(input_size).unsqueeze(0)
            out = self.cnn(test_ten)
            cnn_out_size = out.shape[-1]

        # layer to project cnn output to lstm input
        self.proj = nn.Linear(cnn_out_size,lstm_input_size)
        # with dimensionality lstm_hidden_dim.
        self.lstm = nn.LSTM(lstm_input_size, lstm_hidden_dim, num_lstm_layers,batch_first=True)

        # The linear layer that maps from hidden state space to out space
        self.out_layer = nn.Linear(lstm_hidden_dim, out_size)

    def forward(self, x):
        B,N,H,W = x.shape
        x = x.reshape(B*N,1,H,W) # ignore sequences and run cnn on all images.
        features = self.cnn(x) 
        proj_feat = self.proj(features).reshape(B,N,-1) # restore sequences
        lstm_out, _ = self.lstm(proj_feat)
        out_space = self.out_layer(lstm_out)
        return out_space[:,-1]
    
if __name__ == "__main__":
    lstm_cnn = LSTM_CNN(input_size=(1,128,100),lstm_input_size=100,num_lstm_layers=5,lstm_hidden_dim=100,out_size=2)
    # B, n_seq, H_in
    x = torch.rand((5,20,128,100))
    out = lstm_cnn(x)
    print(out.shape)
