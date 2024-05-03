# Trainer.py

# Template for NN

#### Threading ##########################
NUM_WORKERS = 0   # 0=no threading




import numpy as np
import pandas as pd
from pathlib import Path

import os

### plotting


import pandas as pd
import seaborn as sn  # for model accuracy and loss plots              
import matplotlib.pyplot as plt 

from torchview import draw_graph    # model architecture
from torchviz import make_dot       # gradient propagation


### neural network modules and 

import torch
import torch.nn as nn
import torch.optim as optim
import torchsummary as summary



import numpy as np
import random
from sklearn.model_selection import train_test_split    
from sklearn.preprocessing import StandardScaler  # for standardization
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torchvision import models, transforms
from torchvision.models import VGG16_Weights

import librosa  # for audio preprocessing

from sklearn.preprocessing import LabelEncoder

### Au file paths and labels #####

# Get all audio files in the 'train' directory
au_files = librosa.util.find_files(os.getcwd() + "/data/train/", ext='au', recurse=True)

# Check for correct number of files
assert len(au_files) == 900, f"Expected 900 .au files, found {len(au_files)}"

labels = [os.path.basename(path).split(".")[0] for path in au_files]
if __name__=="__main__":        # for workers (threads)



    ### Plotting ########
    SHOW_GRAPHS = 1
    SAVE_PLOTS = 0
    SAVE_PATH = os.getcwd() + "/plots/"
    SAVE_FIG1_AS = SAVE_PATH + "CNN_acc_loss.png" # plot of accuracy and loss
    SAVE_FIG2_AS = SAVE_PATH + "CNN_architecture.png" # plot of architecture
    SAVE_FIG3_AS = SAVE_PATH + "CNN_gradient.png" # plot of gradient 

    PREDICT_KAGGLE_DATASET = True
    SAVE_KAGGLE_SUBMISSION_AS = SAVE_PATH + f"CNN_kaggle_.csv" 

    ### Hyper-Parameters #####

    # two splits ..

    # .. these are split
    TRAIN_SIZE = 0.80      
    TEST_VAL_SIZE = 1-TRAIN_SIZE
    # .. then these are split
    VAL_SIZE = TRAIN_SIZE * TEST_VAL_SIZE
    TEST_SIZE = (1-VAL_SIZE)

    # Architecture #########
    BATCH_SIZE = 32

    DROP_OUT_RATE = 0.2 # drop "this" many neurons during forward propagation

    MAX_ITERATIONS = 1 # number of cycles (epochs)

    LEARNING_RATE=0.001
    PENALTY = 0.001         

    NUM_CLASSES = 10    # mutli-class

    ### Seed for reproducibility ##########

    SEED_RANDOM = True

    if SEED_RANDOM:
        random.seed(0) # Python
        np.random.seed(0) # NumPy
        torch.manual_seed(0) # PyTorch

        #torch.backends.cudnn.deterministic = True # hurts performance
        #torch.backends.cudnn.benchmark = False




    ##### Clip for testing #################
        ### 2 from each genre  
    # X = au_files[0:900:90] + au_files[1:900:90]
    # y = labels[0:900:90] + labels[1:900:90]

    # Encode labels to unique numbers
    # y_ids = np.zeros(y)
    # for id,label in enumerate(all_classes):
    #     y_ids[y==label] = id 

    # One-hot encode labels
    # y = np.zeros((len(au_files), num_classes))
    # for i, label in enumerate(labels):
    #     for j, clas in enumerate(all_classes):
    #         y[i, j] = 1 if clas == label else 0

    # Replaced custom encoder with built-in one (it's safer)
    # Encode labels (blues=0, classical=1, ..., rock=9)

    labels = LabelEncoder().fit_transform(labels)     

    all_classes = np.unique(labels)
    num_classes = len(all_classes)


    # (AU)dio file to Mel spectrogram 
    def audio_to_spectrogram(file_path):
        ### Spectrogram #####

        # professor's notebook used stfts
        #spec_type = "stft"
        # mel spectrograms capture more genre information 
        spec_type = "mel"

        # shape of spectrogram: SAMPLES_IN_AUDIO x TIMESTEPS_IN_AUDIO

        SAMPLES_IN_AUDIO = 128      # ffts, mels, mfccs, etc.
        TIMESTEPS_IN_AUDIO = 1290   # number of slices thru decibaled waveform

        # audio processing

        #FRAME_SIZE = 2048           # each frame has this many FFTs
        
        N_FFT = 2048                  # 2048 is the default number of Fast Fourier Transforms per frame
        HOP_SIZE = 512              # distance between two (usually overlapping to avoid information loss) frames


        # rate of 22050 Hz = 2048 samples in 93 milliseconds
        SAMPLE_RATE = 22050     # 22050 Hz is the default




        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)    # sample rate of (default)
        if spec_type == "stft":

            #  The default value in librosa is n_fft=2048, which is 
            # fft = time-domain to frequency domain


            spectrogram = librosa.stft(y=y, sr=sr, hop_length=HOP_SIZE, n_fft=N_FFT)
            spectrogram = np.abs(spectrogram) ** 2
            
        elif spec_type == "mel":
            
            spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=HOP_SIZE, n_mels=SAMPLES_IN_AUDIO)
        
        decibels = librosa.power_to_db(spectrogram, ref=np.max) # log of y 


        decibels = decibels[:, 0:TIMESTEPS_IN_AUDIO] # truncate
        decibels = torch.tensor(decibels).float()
        decibels = (decibels - decibels.mean()) / decibels.std()
        decibels = decibels.unsqueeze(0)  # add channel dimension

        return decibels
            
    # Custom dataset for loading and preprocessing audio files
    class AuFile(Dataset):
        def __init__(self, file_paths, labels, max_shape = [128, 1290]):
            self.file_paths = file_paths
            self.labels = labels
            self.n_mels = max_shape[0]  
            self.n_steps = max_shape[1]

            self.spectrograms = {}

        def __len__(self):
            return len(self.file_paths)

        def __getitem__(self, idx):
            file_path = self.file_paths[idx]
            label = self.labels[idx]

            # Label to tensor (long)
            label_tensor = torch.tensor(label).long()


            # Au(dio) to Mel spectrogram
            if idx in self.spectrograms:
                spectrogram_tensor = self.spectrograms[idx]
            else:
                spectrogram_tensor = audio_to_spectrogram(file_path)
                self.spectrograms[idx] = spectrogram_tensor         # save in memory

            return spectrogram_tensor, label_tensor


    ### Adapted from Prof. Trilce's MLP notebook
    class CNN(pl.LightningModule):
        def __init__(self, num_channels=1):
            super(CNN, self).__init__()
            self.num_channels = num_channels


            # Define a simple CNN architecture
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            #self.fc1 = nn.Linear(64 * 32 * 32, 128)  # fully-connected size based on spectrogram dimensions
            self.fc1 = nn.Linear(659456, 128)
            self.fc2 = nn.Linear(128, NUM_CLASSES)

            self.dropout = nn.Dropout(DROP_OUT_RATE)                 # randomly disable "this many" per cycle
            self.relu = nn.ReLU()                                   # clip negative values to 0
            self.cross_entropy_cost = nn.CrossEntropyLoss()                       # cross entropy between y-true and y-predict (loss)

            self.msa_cost = nn.L1Loss()                            # mean absolute error (L1)
            

            self.mse_cost = nn.MSELoss()                            # mean square error (L2)
            
        def forward(self, x):
            # x = self.relu(self.fc1(x))
            # x = self.dropout(x)
            # x = self.fc2(x)
            # return x

            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = self.dropout(x)
            x = x.view(x.size(0), -1)
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)

            return x

        def evaluate(self, batch, stage=None):
            
            x, y = batch    # x-true and y-true
            y_hat = self.forward(x)  # y-predict
            loss = self.cross_entropy_cost(y_hat, y) # cross-entroopy 
            preds = torch.argmax(y_hat, dim=1)
            acc = (preds == y).float().mean()
    
            if stage:
                self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True) 
                self.log(f"{stage}_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
                
            return loss


        def training_step(self, batch, batch_idx):
            return(self.evaluate(batch, "train"))

        def validation_step(self, batch, batch_idx):
            return(self.evaluate(batch, "val"))

        def test_step(self, batch, batch_idx):
            return(self.evaluate(batch, "test"))



        def on_train_epoch_end(self):
            print('\n')
            print(f"Train Loss: {self.trainer.callback_metrics['train_loss'].item()}")
            print(f"Train Accuracy: {100 * self.trainer.callback_metrics['train_acc'].item():.2f}%")

        def on_validation_epoch_end(self):
            print('\n')
            print(f"Validation Loss: {self.trainer.callback_metrics['val_loss'].item()}")
            print(f"Validation Accuracy: {100 * self.trainer.callback_metrics['val_acc'].item():.2f}%")

        def configure_optimizers(self):
            optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE, weight_decay=PENALTY)
            return optimizer


    # # Transfer learning model using VGG16 pre-trained dataset
    # class CnnTransferNet(nn.Module):
    #     def __init__(self, num_classes):
    #         super(CnnTransferNet, self).__init__()
            
    #         self.vgg = models.vgg16(weights=VGG16_Weights.DEFAULT)
            
    #         # Freeze layers
    #         for param in self.vgg.features.parameters():
    #             param.requires_grad = False
            
    #         # Classifier for dataset
    #         num_features = self.vgg.classifier[0].in_features
    #         self.vgg.classifier = nn.Sequential(
    #             nn.Linear(num_features, 256),
    #             nn.ReLU(),
    #             nn.Dropout(0.5),
    #             nn.Linear(256, num_classes) 
    #         )
            
    #     def forward(self, x):
    #         return self.vgg(x)




    ### Split X and Y into training and validation datasets

    HOW_MANY_AUs = 400
    X, _, y, _ = train_test_split(au_files, labels, train_size=HOW_MANY_AUs, random_state=0, stratify=labels)

    # X = au_files
    # y = labels


    X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, train_size=TRAIN_SIZE, random_state=0, stratify=y)

    # # Standardize: (X-mean(X))/std(X)
    # scaler = StandardScaler() 
    # X_train = scaler.fit_transform(X_train) 
    # X_test_val = scaler.transform(X_test_val)       # Note: Standardize by X_train's mean/std

    ### Split validation dataset further into test and validation 
    X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, train_size=0.8, random_state=0, stratify=y_test_val)

    ### Datasets
    # train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    # val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    # test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))    



    train_dataset = AuFile(X_train,y_train)
    val_dataset = AuFile(X_val, y_val)
    test_dataset = AuFile(X_test, y_test)



    ### Data loaders
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)


    # shuffle is false because we generally don't mess with validation and test sets
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    ### Model
    # input_size = X_train.shape[1]           # input layer size is the number of mfccs, e.g., 13
    # HIDDEN_SIZE = HIDDEN_SIZE               # hiddlen layer size, e.g., 64
    # output_size = len(np.unique(y_labels))  # output layer size is the number of classes, e.g., 10

    model = CNN()
    print(model)

    ### Train
    trainer = pl.Trainer(max_epochs=MAX_ITERATIONS, logger=CSVLogger(save_dir="logs/"), log_every_n_steps=1)
    trainer.fit(model, train_loader, val_loader)


    ### Test
    test1 = trainer.test(model, dataloaders=test_loader)

    


    
    if 1:
        ### Plot loss and accuracy

        metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
        del metrics["step"]
        metrics.set_index("epoch", inplace=True)
        plt.plot(metrics[['train_acc_epoch','val_acc','train_loss_epoch','val_loss']])
        #print(metrics.head())
        if SHOW_GRAPHS:
            sn.relplot(data=metrics[['train_acc_epoch','val_acc','train_loss_epoch','val_loss']], kind="line")
        if SAVE_PLOTS:
            sn.savefig(SAVE_FIG1_AS)

        ### Network architecture

        # # device='meta' -> no memory is consumed for visualization
        # model_graph = draw_graph(model, input_size=(BATCH_SIZE, input_size), device='meta')
        # if SHOW_GRAPHS:
        #     model_graph.visual_graph
        # if SAVE_PLOTS:
        #     model_graph.visual_graph.render(filername=SAVE_FIG2_AS)
        ### Gradient propagation

        # X,Y = next(iter(test_loader))
        # device = next(model.parameters()).device
        # X = X.to(device)
        # yhat = model(X)

        # gradient_graph = make_dot(yhat.mean(), params=dict(model.named_parameters()))
        # if SAVE_PLOTS:
        #     gradient_graph.render(filename=SAVE_FIG3_AS)

        plt.show()

    pass