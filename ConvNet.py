# ConvNet.py

import os 

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl

### SPEC = SPECTROGRAM
SPEC_FOLDER_PATH = os.getcwd() + "/spectrograms" 

### Do "this" to spectrograms... 
### convert to tensor
### scale values to be between 0.0 and 1.0
### normalize by [] values     
SPEC_TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])    # why these normals?

input_size = 1
hidden_size = 128
output_size = 10



class ConvNet(pl.LightningModule):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(-1, 64 * 7 * 7) # or x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = self.loss(output, target)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

# Load MNIST data
#transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
#mnist_train = datasets.MNIST('data', train=True, download=True, transform=transform)
#mnist_test = datasets.MNIST('data', train=False, download=True, transform=transform)

# Load SPECTROGRAM data
mnist_train = datasets.ImageFolder(root=SPEC_FOLDER_PATH, transforms=SPEC_TRANSFORM)
#mnist_train = datasets.MNIST('data', train=True, download=True, transform=transform)
#mnist_test = datasets.MNIST('data', train=False, download=True, transform=transform)

train_set, val_set = random_split(mnist_train, lengths=[TRAIN_RATIO, VALIDATION_RATIO], generator=generator)

train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=64)

### Convolutional NN Model
model = ConvNet()

#trainer = pl.Trainer(max_epochs=5, gpus=1) # Train for 5 epochs with GPU
trainer = pl.Trainer(max_epochs=5) # Train for 5 epochs


### Train
trainer.fit(model, train_loader)

### Test
test1 = trainer.test(model, dataloaders=test_loader)