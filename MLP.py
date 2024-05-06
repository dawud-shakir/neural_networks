import numpy as np
import pandas as pd
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
import seaborn as sns
import matplotlib.pyplot as plt



# Constants and hyperparameters
MODEL = "MLP"

BATCH_SIZE = 32
HIDDEN_SIZE = 64
DROP_OUT_RATE = 0.2
LEARNING_RATE = 0.001
PENALTY = 0.001
MAX_ITERATIONS = 50
TRAIN_SIZE = 0.80
SEED_RANDOM = True

# Seeding for reproducibility
if SEED_RANDOM:
    random.seed(0)  # Python
    np.random.seed(0)  # NumPy
    torch.manual_seed(0)  # PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Data loading and preparation
github = "https://raw.githubusercontent.com/dawud-shakir/logistic_regression/main/in"
num_mfcc = 13
df = pd.read_csv(github + os.sep + f"mfcc_{num_mfcc}_labels.csv")

# Features (MFCC) and labels
X = df.iloc[:, :-1]
y_labels = df.iloc[:, -1]
all_classes = np.unique(y_labels)

# Encode labels into unique numbers for multi-class classification
y_mfcc = np.zeros(y_labels.shape)
for idx, label in enumerate(all_classes):
    y_mfcc[y_labels == label] = idx

# Standardization and data split
X_train, X_test_val, y_train, y_test_val = train_test_split(X, y_mfcc, train_size=TRAIN_SIZE, stratify=y_mfcc, random_state=0)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test_val = scaler.transform(X_test_val)

X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, train_size=0.8, stratify=y_test_val, random_state=0)

# Create datasets and data loaders
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Define MLP model with PyTorch Lightning
class MLP(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=DROP_OUT_RATE):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Fully connected layer 1
        self.fc2 = nn.Linear(hidden_size, output_size)  # Fully connected layer 2
        self.dropout = nn.Dropout(dropout_rate)  # Dropout
        self.relu = nn.ReLU()  # Activation function
        self.cost = nn.CrossEntropyLoss()  # Loss function

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def evaluate(self, batch, stage=None):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.cost(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        accuracy = (preds == y).float().mean()

         
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


# This extension is just for storing predictions and labels during testing
class CustomMLP(MLP):
    def on_test_start(self): 
       
        self.all_preds = []
        self.all_labels = []
        self.accuracies = []
        self.losses = []
       

    def test_step(self, batch, batch_idx):
        loss, preds, accuracy = self.evaluate(batch)
        # Store predictions, labels, accuracy, and loss for later use
        self.all_preds.append(preds)
        self.all_labels.append(batch[1])
        self.accuracies.append(accuracy)
        self.losses.append(loss)
        return {"loss": loss, "preds": preds, "labels": batch[1]}

    def on_test_epoch_end(self):
        # Concatenate all predictions and labels
        all_preds = torch.cat(self.all_preds).cpu().numpy()
        all_labels = torch.cat(self.all_labels).cpu().numpy()

        # Calculate precision, recall, and F1-score for each class
        class_report = classification_report(all_labels, all_preds, target_names=all_classes, output_dict=True)

        print("Classification Report:")
        for class_name, metrics in class_report.items():
            if isinstance(metrics, dict):  # Ignore 'accuracy', which is a float
                print(
                    f"Class '{class_name}': Precision={metrics['precision']:.2f}, Recall={metrics['recall']:.2f}, F1-Score={metrics['f1-score']:.2f}"
                )

        # Calculate overall test accuracy
        overall_accuracy = torch.mean(torch.stack(self.accuracies)).item() * 100
        print(f"Overall Test Accuracy: {overall_accuracy:.2f}%")

        # Calculate overall test loss
        overall_loss = torch.mean(torch.stack(self.losses)).item()
        print(f"Overall Test Loss: {overall_loss:.4f}")

        # Visualize the metrics for precision, recall, F1-Score, and loss
        class_names = list(class_report.keys())
        precision_values = [class_report[class_name]["precision"] for class_name in class_names if isinstance(class_report[class_name], dict)]
        recall_values = [class_report[class_name]["recall"] for class_name in class_names if isinstance(class_report[class_name], dict)]
        f1_values = [class_report[class_name]["f1-score"] for class_name in class_names if isinstance(class_report[class_name], dict)]

        plt.figure(figsize=(10, 6))
        plt.bar(class_names[:-1], precision_values, label="Precision")
        plt.bar(class_names[:-1], recall_values, alpha=0.7, label="Recall")  # Transparency for overlapping bars
        plt.bar(class_names[:-1], f1_values, alpha=0.5, label="F1-Score")  # Transparency for overlapping bars
        plt.xlabel("Classes")
        plt.ylabel("Metric Values")
        plt.title(f"Precision, Recall, and F1-Score for Each Class ({MODEL})")
        plt.legend()
        plt.tight_layout()
        plt.show()



# Initialize and train the model
INPUT_SIZE = X_train.shape[1]
OUTPUT_SIZE = len(all_classes)

model = CustomMLP(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

trainer = pl.Trainer(max_epochs=MAX_ITERATIONS, logger=CSVLogger(save_dir="logs/"))
trainer.fit(model, train_loader, val_loader)

# Test and get precision, recall, and F1-score
trainer.test(model, dataloaders=test_loader)

# Plotting training and validation metrics
metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
metrics.set_index("epoch", inplace=True)

plt.figure(figsize=(10, 6))

# Plot training and validation accuracy
sns.lineplot(data=metrics, x=metrics.index, y='Train_acc_epoch', color='r', label='Training Accuracy')
sns.lineplot(data=metrics, x=metrics.index, y='Val_acc', color='orange', label='Validation Accuracy')
plt.ylabel("Accuracy")


# Plot training and validation loss on a secondary y-axis
ax2 = plt.gca().twinx()
sns.lineplot(data=metrics, x=metrics.index, y='Train_loss_epoch', ax=ax2, color='b', label='Training Loss')
sns.lineplot(data=metrics, x=metrics.index, y='Val_loss', ax=ax2, color='g', label='Validation Loss')

plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.title("Training and Validation Metrics (MLP)")

plt.tight_layout()  # Avoid overlapping
plt.show()
