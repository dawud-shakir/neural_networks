# RayTune.py

### https://docs.ray.io/en/latest/tune/getting-started.html#tune-tutorial

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
from ray.train import Checkpoint
import ray

# Define the Convolutional Neural Network
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3)
        self.fc = nn.Linear(192, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3))
        x = x.view(-1, 192)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

# Define training and testing functions
def train_func(model, optimizer, data_loader):
    model.train()
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

def test_func(model, data_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = 100 * correct / len(data_loader.dataset)
    return accuracy

# Main training function for Ray Tune
def train_mnist(config):
    # Data transformation and loading
    mnist_transforms = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])

    train_loader = DataLoader(
        datasets.MNIST(os.path.expanduser("~/data"), train=True, download=True, transform=mnist_transforms),
        batch_size=64,
        shuffle=True)
    
    test_loader = DataLoader(
        datasets.MNIST(os.path.expanduser("~/data"), train=False, transform=mnist_transforms),
        batch_size=64,
        shuffle=True)

    # Define device
    global device  # To ensure `device` is accessible globally
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConvNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])

    # Training loop with checkpointing
    for epoch in range(10):
        train_func(model, optimizer, train_loader)
        acc = test_func(model, test_loader)

        checkpoint = None
        if (epoch + 1) % 5 == 0:
            checkpoint_dir = os.path.join(os.path.expanduser("~/ray_checkpoints"), f"epoch_{epoch + 1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model.pth"))
            checkpoint = Checkpoint.from_directory(checkpoint_dir)

        # Report back to Ray Tune
        train.report({"mean_accuracy": acc}, checkpoint=checkpoint)

# Hyperparameter search space
search_space = {
    "lr": tune.sample_from(lambda spec: 10 ** (-1 * np.random.rand())),  # Reasonable learning rate
    "momentum": tune.uniform(0.1, 0.9),
}

# Initialize Ray properly
ray.init(address=None)  # Use "auto" if connecting to an external Ray cluster

# Tuning with Asynchronous HyperBand Scheduler 
    
tuner = tune.Tuner(
    train_mnist,
    tune_config=tune.TuneConfig(
        num_samples=20,
        
        # scheduler = ASHAScheduler(
        # time_attr="training_iteration",
        # metric="mean_accuracy",
        # mode="max",
        # max_t=20,  # Maximum number of iterations
        # grace_period=5,  # Minimum iterations before considering stopping
        # reduction_factor=3  # Successive budgets' ratio
        # )
        scheduler=ASHAScheduler(metric="mean_accuracy", mode="max"),
    ),
    param_space=search_space,
)

# Run the tuning process
results = tuner.fit()

# Check and plot results
dfs = {result.path: result.metrics_dataframe for result in results}

for df_path, df in dfs.items():
    if 'mean_accuracy' in df.columns:
        df['mean_accuracy'].plot(label=df_path)
    else:
        print(f"Warning: 'mean_accuracy' not found in DataFrame at {df_path}")

#plt.legend()
plt.xlabel("Sample")
plt.ylabel("Mean Accuracy")
plt.title("Mean Accuracy from Ray Tune Results")
plt.show()

