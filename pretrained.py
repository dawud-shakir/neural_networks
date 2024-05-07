# pretrained.py

# Standard libraries
import numpy as np
import pandas as pd
import os
import random

# PyTorch Lightning and other libraries for data processing
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models  # VGG16 pretrained model
from torchvision.models import VGG16_Weights
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
import seaborn as sns
import matplotlib.pyplot as plt

import librosa
import time   # timer


# Constants and hyperparameters
MODEL = "Pretrained"
BATCH_SIZE = 32
#INPUT_SIZE = 32
#HIDDEN_SIZE = 128
DROP_OUT_RATE = 0.2
LEARNING_RATE = 0.001
PENALTY = 0.001
MAX_ITERATIONS = 50
TRAIN_SIZE = 0.80
SEED_RANDOM = True
CHANNELS = 1  # For grayscale spectrograms

# Aggressive seed for reproducibility
if SEED_RANDOM:
    random.seed(0)  # Python
    np.random.seed(0)  # NumPy
    torch.manual_seed(0)  # PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Function to convert audio to Mel spectrogram (in decibels)
def audio_to_mels(file_path, sr=22050, n_mels=128, hop_length=512):
    # Load the audio file
    y, sr = librosa.load(file_path, sr=sr)
    
    # Calculate Mel spectrogram
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    
    # Convert to decibels
    mels = librosa.power_to_db(mels, ref=np.max)
    
    # Truncate so audio files are equal length
    mels = mels[:, 0:1290] # 1290 is the minimum length of audio files

    # Normalize with z-score
    mels = (mels - mels.mean()) / mels.std()
    
    # Add channel dimension
    mels = torch.tensor(mels).unsqueeze(0).float()  # [1 x 128 x 1290]
    
    return mels

# Dataset for loading and preprocessing audio files
class AudioDataset(Dataset):
    def __init__(self, file_paths, labels, sr=22050, n_mels=128, hop_length=512):
        self.file_paths = file_paths
        self.labels = labels
        self.sr = sr
        self.n_mels = n_mels
        self.hop_length = hop_length
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Convert audio to Mel spectrogram
        mels = audio_to_mels(file_path, self.sr, self.n_mels, self.hop_length)
        
        return mels, torch.tensor(label).long()

# Data module for training, validation, and test data handling
class AudioDataLoader(pl.LightningDataModule):
    def __init__(self, dataset, batch_size=32, train_size=0.8):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.train_size = train_size

    def setup(self, stage=None):
        # Split into train and validation+test sets
        train_data, val_test_data = train_test_split(
            self.dataset, train_size=self.train_size
        )
        # Further split for validation and test sets
        val_data, test_data = train_test_split(val_test_data, train_size=0.5)
       
        self.train_dataset = train_data
        self.val_dataset = val_data
        self.test_dataset = test_data

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

# CNN model with PyTorch Lightning
class CNN(pl.LightningModule):
    def __init__(self, output_size):
        super(CNN, self).__init__()
        
        ### Pretrained weights ###
        print(f"Loaded pretrained (vgg16) weights")

        self.pretrained = models.vgg16(weights=VGG16_Weights.DEFAULT)

        ### Freeze pretrained ###
        for param in self.pretrained.features.parameters():
            param.requires_grad = False     # weights will not be update during training

                
        ### CNN layers are after pretraining 
        self.conv1 = nn.Conv2d(512, 32, kernel_size=3, stride=1, padding=1)  # 512 (in), 32 (out)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)   # 32 (in), 64 (out)
        self.pool = nn.MaxPool2d(2, 2)      # reduce size by 1/2
        
        # Size after twice pooling is 659,456 
        # calculated as ..
        flat_size = 64 * (128 // 4) * (1290 // 4)  # dividing by 4 because there are 2 pools
        
        # Fully connected layers are after convolves
        # self.fc1 = nn.Linear(flat_size, hidden_size, bias=True) # bias is true by default

        self.fc1 = nn.Linear(640, 128, bias=True)                            # 32 (in), 128 (out)
        self.fc2 = nn.Linear(128, output_size, bias=True)                   # 128 (in), 10 (out)
        
        self.dropout = nn.Dropout(DROP_OUT_RATE)  # randomly drop nodes to prevent over-reliance
        self.relu = nn.ReLU()  # activation function provides non-linearity
        self.cost = nn.CrossEntropyLoss()  # loss function


    def forward(self, x):
       
        batch = x.shape[0]  # incoming batch size grows smaller (e.g., 32, 16, 8 ..)
        # Expected size for incoming x
        assert x.size() == torch.Size([batch, CHANNELS, 128, 1290])
        
        # VGG16 expects 3 channels 
        if CHANNELS == 1:
            x = x.repeat(1, 3, 1, 1)   # make 1 channel into 3 channels by duplicating thrice
        
        # Expected size before pretraining
        assert x.size() == torch.Size([batch, 3, 128, 1290])

        ### Pass x thru pretrained module ###
        x = self.pretrained.features(x)  # pass it to VGG16

        # Expected size after pretraining 
        assert x.size() == torch.Size([batch, 512, 4, 40])
       
        # Apply convolutional and pooling layers
        x = self.pool(self.relu(self.conv1(x)))

        # Expected size after 1st convolve, relu, pool 
        assert x.size() == torch.Size([batch, 32, 2, 20])
      
        x = self.pool(self.relu(self.conv2(x)))

        # Expected size after 2nd convolve, relu, pool 
        assert x.size() == torch.Size([batch, 64, 1, 10])


        # Flatten the output 
        x = x.view(x.size(0), -1)   # preserve batch size

        # Expected size after flatten
        assert x.size() == torch.Size([batch, 640]) 

        # Apply fully connected layers
        x = self.relu(self.fc1(x))  
        x = self.dropout(x)
        
        # Expected final size after 1st fully connected
        assert x.size() == torch.Size([batch, 128])
        
        x = self.fc2(x)
        
        return x
    
    def evaluate(self, batch, stage=None):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.cost(y_hat, y)  # y-guess, y-true
        preds = torch.argmax(y_hat, dim=1)  # best guess from classes
        accuracy = (preds == y).float().mean()   # count and mean predictions

        if stage:
            self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=False, logger=True) 
            self.log(f"{stage}_acc", accuracy, on_epoch=True, prog_bar=False, logger=True)

        return loss, preds, accuracy

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.evaluate(batch, "Train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, _ = self.evaluate(batch, "Val")
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE, weight_decay=PENALTY)
        return optimizer

# This extension for storing predictions and labels during testing
class CustomCNN(CNN):
    def on_test_start(self):
        # Initialize storage for predictions, labels, accuracy, and loss
        self.all_preds = []
        self.all_labels = []
        self.accuracies = []
        self.losses = []
       
    def test_step(self, batch, batch_idx):
        loss, preds, accuracy = self.evaluate(batch)
        # Store predictions, labels, accuracy, and loss
        self.all_preds.append(preds)
        self.all_labels.append(batch[1])
        self.accuracies.append(accuracy)
        self.losses.append(loss)
        return {"loss": loss, "preds": preds, "labels": batch[1]}

    def on_test_epoch_end(self):
        # Concatenate all predictions and labels
        all_preds = torch.cat(self.all_preds).cpu().numpy()
        all_labels = torch.cat(self.all_labels).cpu().numpy()

        # Calculate and display metrics
        class_report = classification_report(all_labels, all_preds, output_dict=True)

        overall_accuracy = torch.mean(torch.stack(self.accuracies)).item() * 100
        overall_loss = torch.mean(torch.stack(self.losses)).item()

        print(f"Overall Test Accuracy: {overall_accuracy:.2f}%")
        print(f"Overall Test Loss: {overall_loss:.4f}")

        # Optional visualization of metrics
        class_names = list(class_report.keys())
        precision_values = [class_report[class_name]["precision"] for class_name in class_names if isinstance(class_report[class_name], dict)]
        recall_values = [class_report[class_name]["recall"] for class_name in class_names if isinstance(class_report[class_name], dict)]
        f1_values = [class_report[class_name]["f1-score"] for class_name in class_names if isinstance(class_report[class_name], dict)]

        plt.figure(figsize=(10, 6))
        plt.bar(class_names[:-1], precision_values, label="Precision")
        plt.bar(class_names[:-1], recall_values, alpha=0.7, label="Recall")
        plt.bar(class_names[:-1], f1_values, alpha=0.5, label="F1-Score")
        plt.xlabel("Classes")
        plt.ylabel("Metric Values")
        plt.title(f"Precision, Recall, and F1-Score ({MODEL})")
        plt.legend()
        plt.tight_layout()
        plt.show()

# Data loading and preparation
github = os.getcwd() + "/data/train"

# Get all audio files in the 'train' directory
au_files = librosa.util.find_files(github, ext='au', recurse=True)
labels = [os.path.basename(path).split(".")[0] for path in au_files]

# Encode labels (blues=0, classical=1, etc.)
y = LabelEncoder().fit_transform(labels)

# Create AudioDataset and Data Module
dataset = AudioDataset(au_files, y)

# Create DataLoader
data_module = AudioDataLoader(dataset, batch_size=BATCH_SIZE)

# Create the model and train it

### 
tic = time.time()
model = CustomCNN(output_size=len(np.unique(y)))    # give model the number of genres
load_time = time.time() - tic


trainer = pl.Trainer(max_epochs=MAX_ITERATIONS, logger=CSVLogger(save_dir="logs/"))


### Train model
tic = time.time()
trainer.fit(model, data_module)
fit_time = time.time() - tic

### Test model
tic = time.time()
trainer.test(model, dataloaders=data_module)
test_time = time.time() - tic



# Optional: Plot metrics from training
metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
metrics.set_index("epoch", inplace=True)

plt.figure(figsize=(10, 6))

# Plot training and validation accuracy
sns.lineplot(data=metrics, x=metrics.index, y='Train_acc_epoch', color='r', label='Training Accuracy')
sns.lineplot(data=metrics, x=metrics.index, y='Val_acc', color='orange', label='Validation Accuracy')
plt.ylabel("Accuracy")
plt.xlabel("Epochs")

# Plot training and validation loss on a secondary y-axis
ax2 = plt.gca().twinx()
sns.lineplot(data=metrics, x=metrics.index, y='Train_loss_epoch', ax=ax2, color='b', label='Training Loss')
sns.lineplot(data=metrics, x=metrics.index, y='Val_loss', ax=ax2, color='g', label='Validation Loss')

plt.ylabel("Loss")
plt.xlabel("Epochs")    # done twice because there are two x-axes

plt.title(f"Training and Validation Metrics ({MODEL})")

plt.tight_layout()  # Avoid overlapping elements

plt.show()



print("-"*5 + f" time to load: {load_time}" + "-"*5)
print("-"*5 + f" time to fit: {fit_time}" + "-"*5)
print("-"*5 + f" time to test: {test_time}" + "-"*5)
