# MLP.py 

# Multi Layer Perceptron (MLP)

# We follow these steps:

# 1. Define the architecture of the MLP, including the number of layers, neurons per layer, activation function, and regularization techniques (e.g., dropout).
# 2. Load the data from a file and prepare it for training.
# 3. Set hyperparameters such as learning rate, batch size, loss function, and optimizer.
# 4. Train the MLP on the loaded data.

# same data as project 2
lpath = "https://raw.githubusercontent.com/dawud-shakir/logistic_regression/main/in/mfcc_13_ids.csv"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd

# Architecture

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob=0.0):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.activation = nn.ReLU()  # activation function

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load data and prep for training
def load_mfccs(file_path):
    df = pd.read_csv(file_path)
    return df.iloc[:,:-1].to_numpy(), df.iloc[:,-1].to_numpy()   # mfccs, labels


# Load and preprocess audio files
data, labels = load_mfccs(lpath)


# Hyperparameters
input_size = 10  # input dimensionality of input data
hidden_size = 64  # number of neurons in the hidden layer
output_size = 2  # number of output classes
learning_rate = 0.001
batch_size = 32
num_cycles = 10 # epochs for training
dropout_prob = 0.2  # dropout probability (0 to disable)

# Tensors
X = torch.Tensor(data)
Y = torch.LongTensor(labels) 

dataset = TensorDataset(X, Y)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# MLP model
model = MLP(input_size, hidden_size, output_size, dropout_prob)

# Loss function 
loss_function = nn.CrossEntropyLoss()  # for classification

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training
for cycle in range(num_cycles): 
    cycle_loss = 0.0
    for inputs, labels in data_loader:
        optimizer.zero_grad()       # clear previous gradients
        outputs = model(inputs)     # forward pass: calculate predicted outputs
        loss = loss_function(outputs, labels)   # calculate loss (applies softmax internally)
        loss.backward()             # backward pass calculates the gradients of the loss with respect to parameters
        optimizer.step()            # update parameters based on gradient
        cycle_loss += loss.item()
    print(f"cycle {cycle + 1}, loss: {cycle_loss / len(data_loader)}")
