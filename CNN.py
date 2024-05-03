import os
import torch
import torchaudio
import librosa
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import models, transforms

from pytorch_lightning.loggers import TensorBoardLogger
tensorboard_logger = TensorBoardLogger("lightning_logs", name="audio_classifier")

# Callback to collect training and validation metrics
class MetricsCollectionCallback(pl.Callback):
    def __init__(self):
        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []

    def on_train_epoch_end(self, trainer, pl_module):
        self.train_loss.append(trainer.callback_metrics["train_loss"].item())
        self.train_acc.append(trainer.callback_metrics["train_acc"].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        self.val_loss.append(trainer.callback_metrics["val_loss"].item())
        self.val_acc.append(trainer.callback_metrics["val_acc"].item())

# Hypers


MAX_ITER = 1    # also 10


# (AU)dio file to Mel spectrogram
def audio_to_mel_spectrogram(file_path, n_mels=128):
    y, sr = librosa.load(file_path, sr=None)    # sample rate of 22050 (default)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return mel_db

# Dataset definition for loading and preprocessing audio files
class AudioDataset(Dataset):
    def __init__(self, file_paths, labels, max_shape = [128, 1290]):
        self.file_paths = file_paths
        self.labels = labels
        self.n_mels = max_shape[0]  
        self.n_steps = max_shape[1]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        # Au(dio) to Mel spectrogram
        mel_spectrogram = audio_to_mel_spectrogram(file_path, self.n_mels)
        mel_spectrogram = mel_spectrogram[:, 0:self.n_steps] # truncate
        mel_spectrogram = torch.tensor(mel_spectrogram).float()
        mel_spectrogram = (mel_spectrogram - mel_spectrogram.mean()) / mel_spectrogram.std()
        mel_spectrogram = mel_spectrogram.unsqueeze(0)  # Adding channel dimension

        return mel_spectrogram, torch.tensor(label).long()

# Data module for handling training and validation data
class AudioDataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size=32, test_size=0.2, random_state=0):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.test_size = test_size
        self.random_state = random_state

    def setup(self, stage=None):
        train_data, val_data = train_test_split(self.dataset, test_size=self.test_size, random_state=self.random_state)
        self.train_dataset = train_data
        self.val_dataset = val_data

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

# CNN model definition
class AudioClassifier(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=1e-3, penalty=0.25):
        super().__init__()
        self.learning_rate = learning_rate
        self.penalty = penalty
        
        # Define a simple CNN architecture
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(penalty)
        #self.fc1 = nn.Linear(64 * 32 * 32, 128)  # fully-connected size based on spectrogram dimensions
        self.fc1 = nn.Linear(659456, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        train_loss = F.cross_entropy(outputs, labels) # softmax is included internally
        train_acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log("train_loss", train_loss, on_step=False, on_epoch=True)
        self.log("train_acc", train_acc, prog_bar=True)
        
        return train_loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        val_loss = F.cross_entropy(outputs, labels) # softmax is included internally
        val_acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log("val_loss", val_loss, on_step=False, on_epoch=True)
        self.log("val_acc", val_acc, prog_bar=True)
        return val_loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


# Get all audio files in the 'train' directory
au_files = librosa.util.find_files(os.getcwd() + "/data/train/", ext='au', recurse=True)

# Check for correct number of files
assert len(au_files) == 900, f"Expected 900 .au files, found {len(au_files)}"


labels = [os.path.basename(path).split(".")[0] for path in au_files]

# Clip for testing 



# 2 from each genre 
X = au_files[0:900:90] + au_files[1:900:90]
y = labels[0:900:90] + labels[1:900:90]

# Encode labels (blues=0, classical=1, ..., rock=9)
y = LabelEncoder().fit_transform(y)

all_classes = np.unique(labels)
num_classes = len(all_classes)

# One-hot encode labels
# y = np.zeros((len(au_files), num_classes))
# for i, label in enumerate(labels):
#     for j, clas in enumerate(all_classes):
#         y[i, j] = 1 if clas == label else 0





#df = pd.DataFrame(data_info)

# Create the dataset and data module
dataset = AudioDataset(X, y)
data_module = AudioDataModule(dataset)

# Initialize the model with the number of classes
classifier = AudioClassifier(num_classes)

metrics_collector = MetricsCollectionCallback()
# PyTorch Lightning trainer (callback for loss)
trainer = pl.Trainer(max_epochs=MAX_ITER, logger=tensorboard_logger, callbacks=[metrics_collector, pl.callbacks.ModelCheckpoint(save_top_k=1, mode='min', monitor='val_loss')])

# Train the model
trainer.fit(classifier, data_module)

# Training loss and validation accuracy data
metrics = trainer.callback_metrics

# Plot loss and accuracy
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(metrics_collector.train_loss, label="Train Loss")
plt.plot(metrics_collector.val_loss, label="Validation Loss")
plt.xlabel('Iteration')
plt.ylabel("Loss")
plt.title(f"Loss over {MAX_ITER} epochs")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(metrics_collector.train_acc, label='Training Accuracy')
plt.plot(metrics_collector.val_acc, label='Validation Accuracy')
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.title(f"Accuracy over {MAX_ITER} epochs")
plt.legend()

plt.tight_layout()
plt.show()
