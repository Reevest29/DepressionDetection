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


x_train= torch.load('features/audio_train_features.pt')
print(x_train.shape)
#x_train.dtype

#gender_tensor--> torch.Size([163])
#y_tensor--> torch.Size([163])


# In[25]:


gender_train= torch.load('features/gender_train_features.pt')
#gender_train.shape
y_train= torch.load('features/y_train.pt')
#y_train.shape

train = TensorDataset(x_train, y_train)
train_loader = DataLoader(train, batch_size = 2, shuffle = False)


# In[33]:


# Define your LSTM model
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out


# In[65]:


# Define hyperparameters
input_size = x_train.shape[2]
sequence_len=x_train.shape[1]
hidden_size = 128
num_layers = 7
output_size = 1
learning_rate = 0.01
num_epochs = 100
batch_size = 32


# In[66]:


# Initialize the model, loss function, and optimizer
model = LSTMClassifier(input_size, hidden_size, num_layers, output_size)
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


x_test= torch.load('features/audio_test_features.pt')
y_test=torch.load('features/y_test.pt')
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



