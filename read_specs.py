# read_spec.py

import os

from PIL import Image
import torchvision.transforms as transforms
#import torch

# Output directory containing the spectrograms
spectrogram_dir = os.getcwd() + "/spectrograms"

# Find all PNG files recursively
spectrogram_paths = []
for root, _, files in os.walk(spectrogram_dir):
    for file in files:
        if file.endswith('.png'):
            spectrogram_paths.append(os.path.join(root, file))

print(f"Found {len(spectrogram_paths)} spectrograms")



# Define a transform to convert images to tensors
image_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale (1 channel)
    transforms.ToTensor()  # Convert to tensor
])

# Function to read and convert an image to a tensor
def read_spectrogram(spectrogram_path):
    # Read the image
    image = Image.open(spectrogram_path)

    # Apply the transformation to convert to a tensor
    tensor = image_transform(image)

    return tensor


# Store the spectrogram tensors
spectrogram_tensors = []

# Read each spectrogram image and convert it into a tensor
for spectrogram_path in spectrogram_paths:
    tensor = read_spectrogram(spectrogram_path)
    spectrogram_tensors.append(tensor)

print(f"Converted {len(spectrogram_tensors)} spectrograms to tensors")