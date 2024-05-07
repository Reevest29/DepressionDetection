#!/usr/bin/env python
# coding: utf-8
import os
#from tqdm import tqdm
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.utils import resample
#import seaborn as sns
#from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, make_scorer
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


x_train= torch.load('upsample/audio_train_features.pt')
print(x_train.shape)
#x_train.dtype

#gender_tensor--> torch.Size([163])
#y_tensor--> torch.Size([163])


# In[25]:


gender_train= torch.load('upsample/gender_train_features.pt')
#gender_train.shape
y_train= torch.load('upsample/y_train.pt')
#y_train.shape

train = TensorDataset(x_train, y_train)
train_loader = DataLoader(train, batch_size = 2, shuffle = False)

class LSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, attention_size):
        super(LSTMWithAttention, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # Output layer for binary classification
        self.attention = nn.Linear(hidden_size, attention_size)  # Attention layer

    def forward(self, x):
        # LSTM layer
        out, _ = self.lstm(x)
        
        # Attention mechanism
        attn_weights = torch.softmax(torch.tanh(self.attention(out)), dim=1)
        attn_applied = torch.bmm(attn_weights.permute(0, 2, 1), out)
        
        # Classification layer
        out = self.fc(attn_applied[:, -1, :])  # Use only the last timestep for classification
        
        return torch.sigmoid(out)  # Apply sigmoid activation for binary classification

attention_size = 128
input_size = x_train.shape[2]
sequence_len=x_train.shape[1]
hidden_size = 128
num_layers = 7
output_size = 1
learning_rate = 0.01
num_epochs = 100
batch_size = 32

model = LSTMWithAttention(input_size, hidden_size, num_layers, attention_size)

# Initialize the model, loss function, and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        inputs = inputs.float()
        labels = labels.float().view(-1, 1)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

x_test= torch.load('upsample/audio_test_features.pt')
y_test=torch.load('upsample/y_test.pt')
test = TensorDataset(x_test, y_test)
test_loader = DataLoader(test, batch_size = 2, shuffle = False)


def test_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    all_preds = []
    all_true_labels = []

    with torch.no_grad():
        for X, y in dataloader:
            test_preds = model(X.to(torch.float32))
            test_preds=test_preds.squeeze()
            test_preds= [1 if pred >= 0.5 else 0 for pred in test_preds] 
            #print(test_preds)
            correct = [1 for i in range(len(test_preds)) if test_preds[i] == y_test[i]]
            all_preds.extend(test_preds)
            all_true_labels.extend(y.tolist())
    
    f1 = f1_score(all_true_labels, all_preds)
    correct = sum(correct)/size
    print(f"Accuracy: {(100*correct):>0.1f}%")
    print(f"F1 Score: {f1:.4f}")
    return f1

test_loop(test_loader, model, criterion, optimizer)

print("DONE WITH TESTING")



