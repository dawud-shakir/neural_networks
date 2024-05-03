# MLP.py (Multi Layer Perceptron)

import numpy as np
import pandas as pd
from pathlib import Path

import os

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

# 
# Need these files in root directory: 
#        mfcc_13_labels.csv -or- mfcc_128_labels.cvs            (known train dataset)
#        kaggle_mfcc_13.csv -or- kaggle_mfcc_128.csv            (unknown test dataset)
#        list_test.txt                                          (from data/test)
#

github = "https://raw.githubusercontent.com/dawud-shakir/logistic_regression/main/in"   # same mfccs as logistic regression

### Number of MFCCs per sample (13 or 128)
num_mfcc = 13

SHOW_GRAPHS = 1
SAVE_PLOTS = 1
SAVE_PATH = os.getcwd() + "/plots/"
SAVE_FIG1_AS = SAVE_PATH + "MLP_acc_loss.png" # graph of log loss
SAVE_FIG2_AS = SAVE_PATH + "MLP_architecture.png" # graph of architecture
SAVE_FIG3_AS = SAVE_PATH + "MLP_gradient.png" # graph of gradient 

PREDICT_KAGGLE_DATASET = False   # for kaggle test set
SAVE_KAGGLE_SUBMISSION_AS = SAVE_PATH + f"MLP_kaggle_{num_mfcc}.csv" 

### Hyper-Parameters

SEED_RANDOM = True
TRAIN_SIZE = 0.80     
MAX_ITERATIONS = 50  # number of back-and-forward cycles
LEARNING_RATE=0.001
PENALTY = 0.001        #  

# new stuff
BATCH_SIZE = 32
HIDDEN_SIZE = 64
DROP_OUT_RATE = 0.2
### Adapted from Prof. Trilce's MLP notebook
class MLP(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=DROP_OUT_RATE):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)           # biases are added internally 
        self.fc2 = nn.Linear(hidden_size, output_size)          # "fc" = "fully connected" (dense)
        self.dropout = nn.Dropout(dropout_rate)                 # randomly disable "this many" per cycle
        self.relu = nn.ReLU()                                   # clip negative values to 0
        self.cost = nn.CrossEntropyLoss()                       # cross entropy between y-true and y-predict (loss)

        self.mse_cost = nn.MSELoss()                            # mean square error (loss)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
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


### Seed for reproducibility
if SEED_RANDOM:
    random.seed(0) # Python

    np.random.seed(0) # NumPy

    torch.manual_seed(0) # PyTorch
    #torch.backends.cudnn.deterministic = True # hurts performance
    torch.backends.cudnn.benchmark = False

### Load MFCC and LABEL training data
df = pd.read_csv(github + os.sep + f"mfcc_{num_mfcc}_labels.csv")      # same as 2nd project 

### MFCCs 
X = df.iloc[:,:-1]
X_mfcc = X

### Labels to unique numbers for PyTorch
y_labels = df.iloc[:,-1]   # "blues", "rock", etc.
unique_labels = np.unique(y_labels)

y_ids = np.zeros(y_labels.shape)
for id,label in enumerate(unique_labels):
    y_ids[y_labels==label] = id 
    
y_mfcc = y_ids

### X and Y 

X = X_mfcc      # mfccs
y = y_mfcc      # labels are numbers: 0 thru 9

### Split X and Y into training and validation datasets

X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, train_size=TRAIN_SIZE, random_state=0, stratify=y)

# Standardize: (X-mean(X))/std(X)
scaler = StandardScaler() 
X_train = scaler.fit_transform(X_train) 
X_test_val = scaler.transform(X_test_val)       # Note: Standardize by X_train's mean/std

### Split validation dataset further into test and validation 
X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, train_size=0.8, random_state=0, stratify=y_test_val)

### Datasets
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))    

### Data loaders
BATCH_SIZE=BATCH_SIZE   # e.g., 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


### Adapted from Prof. Trilce's notebook (barebones MLP) 

### Model
input_size = X_train.shape[1]           # input layer size is the number of mfccs, e.g., 13
HIDDEN_SIZE = HIDDEN_SIZE               # hiddlen layer size, e.g., 64
output_size = len(np.unique(y_labels))  # output layer size is the number of classes, e.g., 10

model = MLP(input_size, HIDDEN_SIZE, output_size)
print(model)

### Train
trainer = pl.Trainer(max_epochs=MAX_ITERATIONS, logger=CSVLogger(save_dir="logs/"))
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
    kaggle_submission.to_csv(SAVE_KAGGLE_SUBMISSION_AS, index=False)  # no index for submission file
    print("Kaggle submission file:", SAVE_KAGGLE_SUBMISSION_AS)




import pandas as pd
import seaborn as sn  # for model accuracy and loss plots              
import matplotlib.pyplot as plt 

from torchview import draw_graph    # model architecture
from torchviz import make_dot       # gradient propagation

if 1:
    ### Plot loss and accuracy

    metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
    del metrics["step"]
    metrics.set_index("epoch", inplace=True)
    #print(metrics.head())
    if SHOW_GRAPHS:
        sn.relplot(data=metrics[['train_acc_epoch','val_acc','train_loss_epoch','val_loss']], kind="line")
    #if SAVE_PLOTS:
        #sn.savefig(SAVE_FIG1_AS)

    ### Network architecture

    # device='meta' -> no memory is consumed for visualization
    model_graph = draw_graph(model, input_size=(BATCH_SIZE, input_size), device='meta')
    if SHOW_GRAPHS:
        model_graph.visual_graph
    if SAVE_PLOTS:
        model_graph.visual_graph.render(filername=SAVE_FIG2_AS)
    ### Gradient propagation

    X,Y = next(iter(test_loader))
    device = next(model.parameters()).device
    X = X.to(device)
    yhat = model(X)

    gradient_graph = make_dot(yhat.mean(), params=dict(model.named_parameters()))
    if SAVE_PLOTS:
        gradient_graph.render(filename=SAVE_FIG3_AS)

    plt.show()
exit()