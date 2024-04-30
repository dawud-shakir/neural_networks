import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

import torchvision.transforms as transforms

import librosa


# Data

num_of_channels = 1

# Hyperparams

learning_rate = 1e-3

momentum = 0.9


total_epochs = 3 # training iterations

weight_decay = 0

batch_size = 4  # e.g., 32

train_ratio = 0.8  # e.g., 80% train, 20% test

# Sim
show_images = 1
seed_random = 1

# Input size is 128x128
resize = 128


# Define a 5-conv/maxpool CNN with PyTorch Lightning with validation and testing
class FiveConvNet(pl.LightningModule):
    def __init__(self, input_channels, num_classes):
        super(FiveConvNet, self).__init__()
        
        # Convolutional Layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)  # 1st convolution
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)  # 2nd convolution
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)  # 3rd convolution
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3)  # 4th convolution
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3)  # 5th convolution

        # Pooling Layers
        self.pool = nn.MaxPool2d(2, 2)  # 2x2 pooling

        # Calculate the output size for the FC layer
        output_size = 128  # Assuming a 128x128 input
        for _ in range(5):  # 5 convolutional layers with pooling
            output_size = ((output_size - 2) // 2) - 2  # Each convolution and pooling layer reduces the size
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * output_size * output_size, 512)  # FC layer
        self.fc2 = nn.Linear(512, num_classes)  # Output layer for classification

    def forward(self, x):
        # Apply convolutions, ReLU, and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))

        x = torch.flatten(x, 1)  # Flatten before FC layers
        x = F.relu(self.fc1(x))  # ReLU after first FC layer
        x = self.fc2(x)  # Output layer
        return x

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)
        self.log("train_loss", loss)  # Log the training loss
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)  # Calculate validation loss
        self.log("val_loss", loss)  # Log the validation loss
        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)  # Calculate test loss
        self.log("test_loss", loss)  # Log the test loss
        return loss

    def configure_optimizers(self):
        # Configure SGD as the optimizer with learning rate and momentum
        return torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)

    def train_dataloader(self):
        # Define the training DataLoader
        # Placeholder example to split dataset and return DataLoader
        data = ...  # Your dataset
        train_data, val_test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data.y)
        return DataLoader(train_data, batch_size=32, shuffle=True)

    def val_dataloader(self):
        # Define the validation DataLoader
        data = ...  # Your validation dataset
        _, val_test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data.y)
        val_data, _ = train_test_split(val_test_data, test_size=0.5, random_state=42)
        return DataLoader(val_data, batch_size=32, shuffle=False)

    def test_dataloader(self):
        # Define the test DataLoader
        data = ...  # Your test dataset
        _, val_test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data.y)
        _, test_data = train_test_split(val_test_data, test_size=0.5, random_state=42)
        return DataLoader(test_data, batch_size=32, shuffle=False)
    
if __name__ == "__main__":

    ### Seed for reproducibility
    
    ### Preprocessing Transformations
    transform = transforms.Compose([
        transforms.Resize((resize, resize)),  # Resize to a uniform size
        transforms.Grayscale(num_output_channels=num_of_channels),  # grayscale (1 channel) 
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.5,), (0.5,)),  # Normalize
    ])

