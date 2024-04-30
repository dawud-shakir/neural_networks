# CNN_Simple.py

# Very simple .. only 1 sample (spectrogram)

import os, random
from sklearn.model_selection import train_test_split
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
#from torch.utils.data import DataLoader, TensorDataset

import torch.utils.data as data
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import librosa

import cv2

classes = (
    "blues", 
    "classical", 
    "country", 
    "disco", 
    "hiphop", 
    "jazz", 
    "metal", 
    "pop", 
    "reggae", 
    "rock"
    )

num_of_classes = len(classes)

if __name__ == "__main__":    # main for threading

    lpath = os.getcwd() + "/spectrograms/train/blues/blues.00000.png"

    label = 1 # 1 = "blues"

    if 0:
        spectrogram = cv2.imread(lpath)
        cv2.imshow("Blues", spectrogram)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # About color channels: https://pillow.readthedocs.io/en/stable/handbook/concepts.html

    # im = Image.open(lpath).convert('L')  # grayscale (1 channel, 8-bit pixels, grayscale)
    # im = Image.open(lpath).convert('RGB')  # rgb (3 channels, 3x8-bit pixels, true color)
    im = Image.open(lpath).convert('RGBA')  # rgba (4 channels, 4x8-bit pixels, with transparency)


    num_of_channels = len(im.getbands())
    spectrogram = im


    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            
            # Convolutional layers
            # 3x3 kernel is often used because it is small and effective
   
            # 
            self.conv1 = nn.Conv2d(num_of_channels, 16, kernel_size=3, stride=1, padding=1)  # 1st conv
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # 2nd conv
            
            # Pooling layers (downsample)
            self.pool = nn.MaxPool2d(2, 2)   # 2x2 kernel reduces spatial dimension by 1/2
            
            # Fully connected layers
            # self.fc1 = nn.Linear(num_of_channels * 224 * 224, 256)  # Input to FC
            self.fc1 = nn.Linear(100352, 256)
            self.fc2 = nn.Linear(256, num_of_classes)  # Output FC for 10 classes

        def forward(self, x):
            # Apply convolution and pooling
            x = self.pool(F.relu(self.conv1(x)))  
            x = self.pool(F.relu(self.conv2(x)))  
            
            # Flatten for fully connected layers
            x = torch.flatten(x, 1)  # Flatten all but the batch dimension
            
            # Apply fully connected layers
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            
            return x
    
    # Custom dataset for spectrogram + label
    class CustomSpectrogramDataset(data.Dataset):
        def __init__(self, image_tensor, label):
            self.image_tensor = image_tensor
            self.label = label

        def __len__(self):
            return 1
        
        def __getitem__(self, idx):
            if idx >= self.__len__():
                raise IndexError("Index out of range")
            return self.image_tensor, self.label

   
    ### Preprocessing Transformations

    # Resize to 224x224 (a common standard)

    transform = transforms.Compose([
        # this is to maintain aspect ratio
        transforms.Resize(224),          # resize just the smaller dimension to 224
        transforms.Pad((32, 0, 32, 0)),  # pad to 224x224
        transforms.CenterCrop(224),      # final size is 224x224

        transforms.ToTensor(),  # this also normalizes [0,1]
    ])

    

    tensor = transform(spectrogram)  # apply transformation to image
 
    print("tensor shape:", tensor.shape)
    dataset = CustomSpectrogramDataset(tensor, 1)
    loader = data.DataLoader(dataset, batch_size=1, shuffle=True)
    
    network = SimpleCNN()
    criterion = nn.CrossEntropyLoss()  # loss function
    optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)  # gradient optimizer

    ### Train the network
    for epoch in range(2):  # just 2 cycles
        running_loss = 0.0

        for i, data in enumerate(loader, 0):
            inputs, labels = data

            ### zero gradients
            optimizer.zero_grad()

            ### forward + backward + optimize 
            probs = network(inputs)   # propagate forward
            loss = criterion(probs, labels)   # what was the loss?
            loss.backward()     # propagate backwards
            optimizer.step()    # improve gradient

            ### print loss 
            running_loss += loss.item()
            if 1:  # print every mini-batch
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}")
                running_loss = 0.0

        print(f"===== epoch {epoch + 1}: finished training =====")

    ### Test the network 
    correct = 0
    total = 0
    with torch.no_grad():   # no gradient for testing
        for data in loader:
            images, labels = data
            probs = network(images)
            _, predicted = torch.max(probs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy of the network on {len(loader) * 1} test images: {100 * correct // total} %")

