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


OUTPUT_PATH = os.getcwd() + "/color_channels.png"

lpath = os.getcwd() + "/spectrograms/train/blues/blues.00000.png"

# About color channels: https://pillow.readthedocs.io/en/stable/handbook/concepts.html
original_image = Image.open(lpath)

im1 = Image.open(lpath).convert('L').resize((224,224))  # grayscale (1 channel, 8-bit pixels, grayscale)
im2 = Image.open(lpath).convert('RGB').resize((224,224))  # rgb (3 channels, 3x8-bit pixels, true color)
im3 = Image.open(lpath).convert('RGBA').resize((224,224))  # rgba (4 channels, 4x8-bit pixels, with transparency)


print("original image size:", [original_image.width, original_image.height])
print("original image bands (colors):", original_image.getbands())

axis = "off"

plt.subplot(2, 2, 1)  # 1st subplot (top-left)
plt.imshow(original_image)
plt.title(f"Original Image\nRGBA {original_image.width} x {original_image.height}")
plt.axis(axis)  # hide axis

plt.subplot(2, 2, 2)  # 2nd subplot (top-right)
plt.imshow(im1, cmap="gray")  # grayscale
plt.title(f"Grayscale {im1.width} x {im1.height}")
plt.axis(axis)

plt.subplot(2, 2, 3)  # 3rd subplot (bottom-left)
plt.imshow(im2)
plt.title(f"RGB {im2.width} x {im2.height}")
plt.axis(axis)

plt.subplot(2, 2, 4)  # 4th subplot (bottom-right)
plt.imshow(im3)
plt.title(f"RGBA {im3.width} x {im3.height}")
plt.axis(axis)

#plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()
plt.savefig(OUTPUT_PATH, dpi=600)
plt.close()

