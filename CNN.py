# CNN.py
### adapted from Prof. Trilce's notebook (barebones MLP) 
### adapted from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html




import os, random
from sklearn.model_selection import train_test_split
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, TensorDataset

import torch.utils.data as data
from torch.utils.data import random_split
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import librosa

# lightening

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

PREDICT_KAGGLE_DATASET = 0
# default values in pytorchlibrary 
# lr=1e-3
# momentum=0
# weight_decay=0


MAX_ITERATIONS = 10

### About color channels: https://pillow.readthedocs.io/en/stable/handbook/concepts.html

##### "L"       = grayscale (1 channel, 8-bit pixels, grayscale)
##### "RGB"     = 3 channels, 3 x 8-bit pixels, true color
##### "RGBA"    = 4 channels, 4 x 8-bit pixels, true color with transparency

COLORS = "RGB" # "L", "RGBA"
NUM_OF_CHANNELS = len(COLORS)
RESIZE_BY = (224,224)
#RESIZE_BY = (3840, 2880) # spectrogram images are saved at this size

OPTIMIZER = "ADAM"  
#OPTIMIZER = "SGD"   # standard gradient descent

PENALTY = 1e-3    # weight decay (lambda) is usually from 1e-5 to 1e-2

BATCH_SIZE = 1  # 32 gave 6% val accuracy, 1 gave 30% val accuracy
LEARNING_RATE = 1e-3   # eta
MOMENTUM = 0.9 #  momentum of algorithm

DROP_OUT_RATE = 0.5    # drop out "this" many nodes per CNN layer 

TRAIN_RATIO = 0.8       # 720
VALIDATION_RATIO = 0.2  # 180
TEST_RATIO = 0.0

SPECS_PATH = os.getcwd() + "/spectrograms/train" # path to root of spectrogram images

### kernel
KERNEL_DIMS = (3,3) # joe suggested (spectro_height, 1 .. 3+)
KERNEL_STRIDE = 1 # 1
KERNEL_PADDING = 0 # 1

NUM_CONVOLUTIONS = 6

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

NUM_OF_CLASSES = len(classes)




if __name__ == "__main__":    # main for threading


    class SimpleCNN(pl.LightningModule): # (nn.Module)
        def __init__(self, input_size, output_size):
            super(SimpleCNN, self).__init__() 
            
            # Convolutional layers
            # 3x3 kernel, padding=KERNEL_PADDING, stride=KERNEL_STRIDE
   
            # 
            self.conv1 = nn.Conv2d(input_size, 16, kernel_size=3, stride=KERNEL_STRIDE, padding=KERNEL_PADDING)  # 1st conv
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=KERNEL_STRIDE, padding=KERNEL_PADDING)  # 2nd conv
            
            # Let's go wild with convolutions: 2 -> 4 -> 8 -> 16 -> 32 -> 64
       

            self.conv1 = nn.Conv2d(NUM_OF_CHANNELS, 2, kernel_size=KERNEL_DIMS, stride=KERNEL_STRIDE, padding=KERNEL_PADDING)  # 1st conv
            self.conv2 = nn.Conv2d(2, 4, kernel_size=KERNEL_DIMS, stride=KERNEL_STRIDE, padding=KERNEL_PADDING)  # 2nd conv
            self.conv3 = nn.Conv2d(4, 8, kernel_size=KERNEL_DIMS, stride=KERNEL_STRIDE, padding=KERNEL_PADDING)  # 4th conv
            self.conv4 = nn.Conv2d(8, 16, kernel_size=KERNEL_DIMS, stride=KERNEL_STRIDE, padding=KERNEL_PADDING)  # 6th conv
            self.conv5 = nn.Conv2d(16, 32, kernel_size=KERNEL_DIMS, stride=KERNEL_STRIDE, padding=KERNEL_PADDING)  # 7th conv
            self.conv6 = nn.Conv2d(32, 64, kernel_size=KERNEL_DIMS, stride=KERNEL_STRIDE, padding=KERNEL_PADDING)  # 8th conv
            


            # Pooling layers (downsample)
            self.pool = nn.MaxPool2d(2, 2)   # 2 x 2 kernel reduces dimension by 1/2 x 1/2
            
            # Fully connected layers
            # Assuming your input image size is (224, 224)
            # after all the convolutions and max-poolings
            # Final output size will depend on your input size and strides used in the convolutions
            # This is an example assuming no padding, stride of 1 for convolutions, and stride of 2 for max pooling

            # Assuming no padding and stride of 1 in convolutions
            #output_size_after_convs = (224 - 3 + 1) * (224 - 3 + 1) * 64 

            output_size_after_convs = (RESIZE_BY[0] - KERNEL_DIMS[0] + 1) * (RESIZE_BY[1] - KERNEL_DIMS[1] + 1) * 64
            output_size_after_pooling = int(output_size_after_convs / (2 * 2 * NUM_CONVOLUTIONS))

            self.fc1 = nn.Linear(output_size_after_pooling, 256)
   
            # input to FC
            self.fc2 = nn.Linear(256, output_size)  # output from FC (classes)
            self.dropout = nn.Dropout(DROP_OUT_RATE)                 # randomly disable "this many" per cycle
            self.relu = nn.ReLU()                                   # clip negative values to 0
            self.cost = nn.CrossEntropyLoss()                       # cross entropy between y-true and y-predict (loss)

            self.mse_cost = nn.MSELoss()                            # mean square error (loss)

        def forward(self, x):
            # Why convolution?
            # 
            # Apply convolution and pooling
            x = self.conv1(x)  
            
            #
            # 
            #
            x = self.relu(x)  
            
            x = self.pool(x)  
            
            ### todo: use a for loop or an array conv[0], conv[1]..

            x = self.pool(F.relu(self.conv2(x)))  
            x = self.pool(F.relu(self.conv3(x)))  
            x = self.pool(F.relu(self.conv4(x)))  
            x = self.pool(F.relu(self.conv5(x)))  
            x = self.pool(F.relu(self.conv6(x)))  


            # Why flatten?
            # Suppose x is a tensor with shape [batch_size, channels, height, width]. If we apply torch.flatten(x, 1), the 
            # resulting tensor will have shape [batch_size, channels * height * width]. This transformation happens while moving
            # from the convolutional layer to the dense layers, where flat input is used.
            x = torch.flatten(x, 1)  # flatten all but the batch dimension
     
            # Why fully connected layer?
            # Picasso problem: Convolution Neural Networks are translation invariant. The network can recognize patterns regardless
            # of their position in the input data. Simply because the portrait has eyes and a mouth, it will be identified as a valid face. 
            # To solve this, the layer after the convolution layer is flattened is a fully connected neural network. 
                        
            x = self.fc1(x)
            
            # Why relu?
            # 
            #
            x = self.relu(x)
            
            # Why dropout?
            #
            #
            x = self.dropout(x)

            # Why a second fully connected layer?
            x = self.fc2(x)
            return x
            
        def evaluate(self, batch, stage=None):
        
            x, y = batch    # x-true and y-true
            y_hat = self.forward(x)  # y-predict
            loss = self.cost(y_hat, y) # cross-entroopy 
            preds = torch.argmax(y_hat, dim=1)
            acc = (preds == y).float().mean()
    
            if stage:
                self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=False, logger=True) 
                self.log(f"{stage}_acc", acc, on_epoch=True, prog_bar=True, logger=True)
                
            return loss

        def training_step(self, batch, batch_idx):
            return(self.evaluate(batch, "train"))

        def validation_step(self, batch, batch_idx):
            return(self.evaluate(batch, "val"))

        def test_step(self, batch, batch_idx):
            return(self.evaluate(batch, "test"))

        def on_train_epoch_end(self):
            print(f"Train Loss: {self.trainer.callback_metrics['train_loss'].item()}, Train Accuracy: {100 * self.trainer.callback_metrics['train_acc'].item():.2f}%")

        def on_validation_epoch_end(self):
            print(f"Validation Loss: {self.trainer.callback_metrics['val_loss'].item()}, Validation Accuracy: {100 * self.trainer.callback_metrics['val_acc'].item():.2f}%")

        def configure_optimizers(self):
            optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE, weight_decay=PENALTY)
            return optimizer

        

    
    ### Preprocessing Transformations

    # Resize to 224x224 (a common standard)

    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert(COLORS)),  # colors
        transforms.Resize(RESIZE_BY),

        # this is just to keep aspect ratio (4:3)
        #transforms.Resize(224),          # resize just the smaller dimension to 224
        #transforms.Pad((32, 0, 32, 0)),  # pad to 224x224
        #transforms.CenterCrop(224),      # final size is 224x224

        transforms.ToTensor(),           # this also normalizes [0,1]
    ])

    # ImageFolder loads everything 
    dataset = torchvision.datasets.ImageFolder(root=SPECS_PATH, transform=transform)

    #X = dataset[:,0]
    #Y = dataset[:,1]





    generator = torch.Generator().manual_seed(0)
    train_set, val_set = random_split(dataset, lengths=[TRAIN_RATIO, VALIDATION_RATIO], generator=generator)

    train_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    # no shuffle: we don't want to play around with validation and test data
    val_loader = data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    #test_loader = data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)


    ### Model
    model = SimpleCNN(input_size=NUM_OF_CHANNELS,output_size=NUM_OF_CLASSES)
    print(model)

    ### Train
    trainer = pl.Trainer(max_epochs=MAX_ITERATIONS, logger=CSVLogger(save_dir="logs/"))
    model.train
    trainer.fit(model, train_loader, val_loader)


    ### Test
    test1 = trainer.test(model, dataloaders=test_loader)

    if PREDICT_KAGGLE_DATASET:

        ### Load Kaggle dataset
        df_kaggle = pd.read_csv(github + os.sep + f"kaggle_mfcc_{num_mfcc}.csv")
        X_kaggle = df_kaggle.iloc[:, 0:]  # unknown kaggle testing data
        
        X_kaggle = StandardScaler().fit_transform(X_kaggle)   # standardize

        kaggle_dataset = TensorDataset(torch.tensor(X_kaggle, dtype=torch.float32)) # convert to tensor dataset

        ### Predict using MLP
        
        model.eval()  # because model is already trained, switch to evaluation mode (inplace)

        with torch.no_grad():  # disable gradient calculation, only need predictions
            numbered_preds = model(kaggle_dataset.tensors[0])            # pass Kaggle dataset to model
            numbered_preds = numbered_preds.argmax(dim=1).tolist()       # predicted numbers (0..9)
            predictions = unique_labels[numbered_preds]                  # outputs are predictions to labels

        ### Build submission
        files_in_test_dir = pd.read_csv(github + os.sep + "list_test.txt", header=None)   # from data/test/
        
        kaggle_submission = pd.DataFrame()
        kaggle_submission.insert(0, "id", files_in_test_dir)
        kaggle_submission.insert(1, "class", predictions)

        print(len(predictions), "Kaggle predictions:")
        print(kaggle_submission)

        ### Write to csv
        kaggle_submission.to_csv(save_kaggle_submission_as, index=False)  # no index for submission file
        print("Kaggle submission file:", save_kaggle_submission_as)

    # need a test_loader as well ??

    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()  # loss function

    if OPTIMIZER == "ADAM":
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY_RATE)
    elif OPTIMIZER == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)  # gradient optimizer
    else:
        exit(f"error: optimizer {OPTIMIZER} not supported")

    ### Train the network

    for epoch in range(2):  # just 2 cycles
        running_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            ### zero gradients
            optimizer.zero_grad()

            ### forward + backward + optimize 
            probs = model(inputs)   # propagate forward
            loss = criterion(probs, labels)   # what was the loss?
            loss.backward()     # propagate backwards
            optimizer.step()    # improve gradient

            running_loss += loss.item()

            ### print loss 

            if 1:  # print every mini-batch
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}")
                running_loss = 0.0

        print(f"===== epoch {epoch + 1}: finished training =====")

    ### Test the network 

    

    correct = 0
    total = 0
    with torch.no_grad():   # no gradient for testing
        for data in val_loader:
            images, labels = data
            probs = model(images)
            _, predicted = torch.max(probs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy of the network on {len(val_loader) * BATCH_SIZE} test images: {100 * correct // total} %")


#     # Predictions for each class
#     correct_pred = {classname: 0 for classname in classes}
#     total_pred = {classname: 0 for classname in classes}

#     with torch.no_grad():
#         for data in test_loader:
#             images, labels = data
#             outputs = net(images)
#             _, predictions = torch.max(outputs, 1)

#             for label, prediction in zip(labels, predictions):
#                 if label == prediction:
#                     correct_pred[classes[label]] += 1
#                 total_pred[classes[label]] += 1

#     # Accuracy for each class
#     for classname, correct_count in correct_pred.items():
#         accuracy = 100 * float(correct_count) / total_pred[classname]
#         print(f"Accuracy for class {classname} is {accuracy:.1f} %")


## older version below ... 



    #tensor = transform(spectrogram)  # apply transformation to image
 
     # Custom dataset for spectrogram + label
    # class CustomSpectrogramDataset(data.Dataset):
    #     def __init__(self, image_tensor, label):
    #         self.image_tensor = image_tensor
    #         self.label = label

    #     def __len__(self):
    #         return 1
        
    #     def __getitem__(self, idx):
    #         if idx >= self.__len__():
    #             raise IndexError("Index out of range")
    #         return self.image_tensor, self.label

    #print("tensor shape:", tensor.shape)
    #dataset = CustomSpectrogramDataset(tensor, 1)
    #dataset = CustomSpectrogramDataset([tensor,tensor], [1,1])


# dataset = torchvision.datasets.ImageFolder(root=os.getcwd() + "/spectrograms", transform=transform)


# import os, random
# from sklearn.model_selection import train_test_split
# import torch
# import torchvision.transforms as transforms
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import torchvision
# from torch.utils.data import DataLoader, TensorDataset
# from PIL import Image
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# import librosa

# # Data

# num_of_channels = 3

# # Hyperparams

# learning_rate = 1e-3

# momentum = 0.9


# total_epochs = 3 # training iterations

# weight_decay = 0

# batch_size = 4  # e.g., 32

# train_ratio = 0.8  # e.g., 80% train, 20% test

# # Sim
# show_images = 1
# seed_random = 1

# # Input size is 128x128
# input_size = 128

# # Conv1: 5x5 kernel with no padding, 2x2 pooling
# output_size_after_conv1 = (input_size - 5 + 1) // 2  # After convolution and pooling
# # Conv2: 5x5 kernel with no padding, 2x2 pooling
# output_size_after_conv2 = (output_size_after_conv1 - 5 + 1) // 2

# # Number of output channels from the final convolutional layer (16 from `conv2`)
# num_output_channels = 16

# # Calculate the input size for the first fully connected layer
# fc_input_size = num_output_channels * output_size_after_conv2 * output_size_after_conv2

# ### Preprocessing Transformations
# transform = transforms.Compose([
#     transforms.Resize((input_size, input_size)),  # Resize to a uniform size
#     transforms.Grayscale(num_output_channels=num_of_channels),  # Convert to grayscale (1 channel)
#     transforms.ToTensor(),  # Convert to tensor
#     transforms.Normalize((0.5,), (0.5,)),  # Normalize
# ])

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # Define a simple CNN for 1000x600 input
# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super(SimpleCNN, self).__init__()
        
#         # Convolutional layers
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=KERNEL_STRIDE, padding=KERNEL_PADDING)  # 1st conv
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=KERNEL_STRIDE, padding=KERNEL_PADDING)  # 2nd conv
        
#         # Pooling layers
#         self.pool = nn.MaxPool2d(2, 2)  # 2x2 pooling
        
#         # Fully connected layers
#         self.fc1 = nn.Linear(32 * (500) * (500), 256)  # Input to FC
#         self.fc2 = nn.Linear(256, 10)  # Output FC for 10 classes

#     def forward(self, x):
#         # Apply convolution and pooling
#         x = self.pool(F.relu(self.conv1(x)))  # 1000x600 to 500x300 after pooling
#         x = self.pool(F.relu(self.conv2(x)))  # 500x300 to 250x150
        
#         # Flatten for fully connected layers
#         x = torch.flatten(x, 1)  # Flatten all but the batch dimension
        
#         # Apply fully connected layers
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
        
#         return x
    
# if __name__ == "__main__":

#     ### Seed for reproducibility
#     if seed_random:
#         random.seed(0) # Python

#         np.random.seed(0) # NumPy

#         torch.manual_seed(0) # PyTorch
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False


#     # Find paths to spectrograms (alphabetical order)
#     def find_spectrogram_paths(lpath, extension='png'):
#         return librosa.util.find_files(lpath, ext=extension, recurse=True)  # recurse into subdirectories

#     # Spectrogram to tensor (normalized)
#     def read_spectrogram_to_tensor(spectrogram_path):


#         image = Image.open(spectrogram_path)  # open spectrogram image
#         tensor = transform(image)  # apply transformation to image

        
         
#         return tensor

#     ### Read and convert spectrograms to tensors

#     lpath = os.getcwd()

#     # Training spectrograms
#     spectrogram_paths = find_spectrogram_paths(lpath + "/spectrograms/train")

#     # Parse genre label ("blues", "rock", etc.) from file path
#     labels = [os.path.basename(os.path.dirname(path)) for path in spectrogram_paths]

#     # Convert labels to tensor for CNN
#     label_encoder = LabelEncoder()
#     encoded_labels = label_encoder.fit_transform(labels)
#     label_tensors = torch.tensor(encoded_labels, dtype=torch.long)

#     # Spectrograms to tensors
#     spectrogram_tensors = [read_spectrogram_to_tensor(path) for path in spectrogram_paths]

#     # TensorDataset with spectrograms and encoded genre labels (numbers)
#     spectrogram_dataset = TensorDataset(torch.stack(spectrogram_tensors), label_tensors)

#     ### Train and val-test sets
#     X_train, X_test_val, y_train, y_test_val = train_test_split(
#         spectrogram_tensors, label_tensors, train_size=train_ratio, random_state=0, stratify=label_tensors)

#     # Standardize: (X-mean(X))/std(X)
#     #scaler = StandardScaler() 
#     #X_train = scaler.fit_transform(X_train) 
#     #X_test_val = scaler.transform(X_test_val)       # Note: Standardize by X_train's mean/std

#     ### Split validation dataset further into test and validation
#     X_test, X_val, y_test, y_val = train_test_split(
#         X_test_val, y_test_val, train_size=0.8, random_state=0, stratify=y_test_val)

#     ### Datasets
#     train_dataset = TensorDataset(torch.stack(X_train), y_train)  # Stack and ensure correct tensor shapes
    
    
    
#     val_dataset = TensorDataset(torch.stack(X_val), y_val)  # Ensure consistent tensor dtype and shape
#     test_dataset = TensorDataset(torch.stack(X_test), y_test)  # As above

#     ### Data loaders
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size)

#     # Ensure DataLoader has data
#     if len(train_loader) == 0:
#         raise ValueError("The training dataset is empty. Check dataset loading.")

#     '''
#     # Define a Convolutional Neural Network (CNN)
#     class Net(nn.Module):
#         def __init__(self):
#             super(Net, self).__init__()
#             # Convolutions
#             self.conv1 = nn.Conv2d(num_of_channels, 6, 5)  # 1 input channel, 6 output, kernel 5
#             # Pooling
#             self.pool = nn.MaxPool2d(2, 2)  # 2x2 pooling
#             # More Convolutions
#             self.conv2 = nn.Conv2d(6, 16, 5)  # 6 input, 16 output, kernel 5
#             # Fully connected layers
#             self.fc1 = nn.Linear(16 * 29 * 29, 120)  # 13,456 in, 120 out
#             self.fc2 = nn.Linear(120, 84)  # 120 in, 84 out
#             self.fc3 = nn.Linear(84, 10)  # Final fc layer for 10 genre classes



#             # super(Net, self).__init__()
#             # # Convolutions
#             # self.conv1 = nn.Conv2d(num_of_channels, 6, 5)  # 1 input channel, 6 output, kernel 5
#             # # Pooling
#             # self.pool = nn.MaxPool2d(2, 2)  # 2x2 pooling
#             # # More Convolutions
#             # self.conv2 = nn.Conv2d(6, 16, 5)  # 6 input, 16 output, kernel 5
#             # # Fully connected layers
#             # self.fc1 = nn.Linear(16 * 29 * 29, 120)  # 13,456 in, 120 out
#             # self.fc2 = nn.Linear(120, 84)  # 120 in, 84 out
#             # self.fc3 = nn.Linear(84, 10)  # Final fc layer for 10 genre classes

#         def forward(self, x):
#             x = self.pool(F.relu(self.conv1(x)))  # reLU and maxpool
#             x = self.pool(F.relu(self.conv2(x)))  # reLU and maxpool
#             x = torch.flatten(x, 1)  # flatten
#             x = F.relu(self.fc1(x))  # reLU
#             x = F.relu(self.fc2(x))  # reLU
#             x = self.fc3(x)  # Final linear layer classifies
#             return x
#     '''

#     # Define a Convolutional Neural Network (CNN)
#     class Net(nn.Module):
#         def __init__(self):
#             super(Net, self).__init__()
#             # Convolutions
#             self.conv1 = nn.Conv2d(1, 6, kernel_size=5)  # 1 input channel, 6 output, kernel 5
#             # Pooling
#             self.pool = nn.MaxPool2d(2, 2)  # 2x2 pooling
#             # More Convolutions
#             self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # 6 input, 16 output, kernel 5
#             # Fully connected layers
#             self.fc1 = nn.Linear(16 * 29 * 29, 120)  # Corrected FC layer input size
#             self.fc2 = nn.Linear(120, 84)  # FC layer 120 in, 84 out
#             self.fc3 = nn.Linear(84, 10)  # Final FC layer for 10 genre classes

#         def forward(self, x):
#             # Forward pass with convolution, activation, and pooling
#             x = self.pool(F.relu(self.conv1(x)))  # reLU and maxpool
#             x = self.pool(F.relu(self.conv2(x)))  # reLU and maxpool
#             x = torch.flatten(x, 1)  # Flatten before FC layers
#             x = F.relu(self.fc1(x))  # reLU after first FC layer
#             x = F.relu(self.fc2(x))  # reLU after second FC layer
#             x = self.fc3(x)  # Final linear layer classifies
#             return x

#     # Initialize network
#     #net = Net()
#     net = SimpleCNN()

#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print("device=", device)

#     # Train the CNN
#     criterion = nn.CrossEntropyLoss()  # Loss function
#     optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)  # Optimizer

#     for epoch in range(total_epochs):  # Loop over dataset
#         running_loss = 0.0

#         for i, data in enumerate(train_loader, 0):
#             inputs, labels = data

#             # Zero the parameter gradients
#             optimizer.zero_grad()

#             # Forward + backward + optimize
#             outputs = net(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             # Print loss 
#             running_loss += loss.item()
#             if i % 10 == 9:  # print every 10 mini-batches
#                 print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}")
#                 running_loss = 0.0

#         print(f"===== epoch {epoch + 1}: finished training =====")

#     # Test the network 
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data in val_loader:
#             images, labels = data
#             outputs = net(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

  
#     ### Genre class names

#     classes = (
#         "blues", 
#         "classical", 
#         "country", 
#         "disco", 
#         "hiphop", 
#         "jazz", 
#         "metal", 
#         "pop", 
#         "reggae", 
#         "rock"
#         )
    
#     bad_accuracy = 100 * 1/len(classes) # random

#     print(f"Accuracy of the network on the {len(test_loader) * batch_size} test images: {100 * correct // total} % (should be better than random: {bad_accuracy} %)")


#     ### How 
#     # Predictions for each class
#     correct_pred = {classname: 0 for classname in classes}
#     total_pred = {classname: 0 for classname in classes}

#     with torch.no_grad():
#         for data in test_loader:
#             images, labels = data
#             outputs = net(images)
#             _, predictions = torch.max(outputs, 1)

#             for label, prediction in zip(labels, predictions):
#                 if label == prediction:
#                     correct_pred[classes[label]] += 1
#                 total_pred[classes[label]] += 1

#     # Accuracy for each class
#     for classname, correct_count in correct_pred.items():
#         accuracy = 100 * float(correct_count) / total_pred[classname]
#         print(f"Accuracy for class {classname} is {accuracy:.1f} %")


