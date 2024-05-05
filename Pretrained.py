# Pretrained.py

# Pretrained CNN with VGG8 dataset 


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


from torchvision.models import VGG16_Weights
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score


# (AU)dio file to Mel spectrogram
# Mels are better than stft and chroma for sound classification 


# returns mels in decibels
def audio_to_mels(au_file):
    y, sr = librosa.load(au_file, sr=22050)    # sample rate of 22050 (default)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)   # 128 mel bins (default)
    mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)   # to decibel basis (log)    
   
    # with the defaults, the shortest au file in our data has 1290 samples
    mel_db = mel_db[:, 0:1290] # truncate
    mel_db = torch.tensor(mel_db).float() # would a float64 seperate more?  
    mel_db = (mel_db - mel_db.mean()) / mel_db.std()    # zscore normalization (x-mu/sigma)
    mel_db = mel_db.unsqueeze(0)  # add color channel between mel_db.shape[0] and mel_db.shape[2]

    return mel_db


c = 1       # 

# demo with one file


au_file = os.getcwd() + "/data/train/classical/classical.00000.au"      
label = os.path.basename(au_file).split(".")[0]      
y_true = 1 # classical is the 2nd 

print(f"{c}. au_file = {os.path.basename(au_file)}")
c += 1
print(f"{c}. label = {label}")
c += 1

basename = os.path.basename(au_file)
X = audio_to_mels(au_file) 
print(f"{c}. Converted {basename} audio image (mel spectrogram) into decibels")
c += 1

assert X.shape == torch.Size([1, 128, 1290])
print("X is a 3D tensor with shape ", X.shape)


### Pretrained weights ###
vgg = models.vgg16(weights=VGG16_Weights.DEFAULT)
print(f"{c}. Loaded pretrained (vgg16) weights")
c += 1

### Freeze pretrained ###
for param in vgg.features.parameters():
    param.requires_grad = False     # Now will not be update during training
print(f"{c}. Froze training for pretrained")        
c += 1

X = X.repeat(3, 1, 1) # make the 1 color channel 3 colors (rgb) 
assert X.shape == torch.Size([3, 128, 1290])
print(f"{c}. Made X a three-color channel")
c += 1


print("X is now a 3D tensor with shape ", X.shape)

X = vgg.features(X)  # pass it to VGG16 

### Passed X thru pretrained module

print(f"{c}. Passed X thru pretrained module")
c += 1

### X is now a 3D tensor with shape  torch.Size([512, 4, 40])
assert X.shape == torch.Size([512, 4, 40])

print("X is now a 3D tensor with shape ", X.shape)


learning_rate = 1e-3    # how fast does model learn (step)?
penalty = 0.25  # weight decay
        
### Really Simple CNN 

conv = nn.Conv2d(512, 32, kernel_size=3, stride=1, padding=1)  # 512 (in), 32 (out)
reLU = nn.ReLU()    # non-linearity by clipping negative values to 0
maxpool = nn.MaxPool2d(2, 2)    # 2x2 pooling will reduce X's width and height by 1/2

print(":"*5, "conv(X)", ":"*5)
c += 1

print(":"*5, "reLU(X)", ":"*5)
c += 1

print(":"*5, "MaxPool(X)", ":"*5)
c += 1

X = maxpool(reLU(conv(X)))

print("X is now a 3D tensor with shape ", X.shape)

### X.shape = torch.Size([32, 2, 20])

print(":"*5, "dropout(X)", ":"*5)

dropout = nn.Dropout(0.25)
X = dropout(X)  # drop out weights randomly to avoid over-dependance on a specific weight

print("X is now a 3D tensor with shape ", X.shape)

### X.shape = torch.Size([32, 2, 20])


print(":"*5, "flatten(X)", ":"*5)
### Flatter X to a 1D array
X = X.flatten()     # combine everything into one dimension 

print("X is now a 1D tensor with shape ", X.shape)


fc1 = nn.Linear(X.size()[0], 128, bias=True)    # with bias (default)    
reLU = nn.ReLU()    # non-linearity by clipping negative values to 0
fc2 = nn.Linear(128, 10, bias=True)   

print(f"{c}. 1st fully-connected layer")
c += 1

print(":"*5, "reLU(X)", ":"*5)

X = reLU(fc1(X))    # from 32 * 2 * 20


print(f"{c}. 2nd fully-connected layer", ":"*5)
c += 1


X = fc2(X)

print(f"{c}. The highest of these 10 probabilities (one per class) is our best guess")

cols = 1
y_hat = torch.argmax(X) # position of best guess
print('\n')
print(f"best guess = {y_hat}")
print(f"correct answer = {y_true}") 
