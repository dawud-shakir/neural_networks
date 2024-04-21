# PyTorch.py


### Multi-Layer Perceptron (MLP) with PyTorch
### This demo creates a simple MLP with hidden layers, a Relu activation function,
### and mean squared error loss for training.

import torch
import torch.nn as nn
import torch.optim as optim

### Model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

### Layers
input_size = 10
hidden_size = 20
output_size = 1

model = MLP(input_size, hidden_size, output_size)

### Loss function
criterion = nn.MSELoss()

### Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

### Data
inputs = torch.randn(5, input_size)  # batch size = 5
targets = torch.randn(5, output_size) 

### Train
num_iterations = 100    # epochs
for i in range(num_iterations):
    optimizer.zero_grad()               # reset gradients
    outputs = model(inputs)             # 
    loss = criterion(outputs, targets)  #  
    
    loss.backward()                     # 
    optimizer.step()                    #

    

    print(f'Epoch [{i + 1} / {num_iterations}], Loss: {loss.item():.4f}')




