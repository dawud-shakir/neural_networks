
# 1. Regression Loss Functions
#        Mean Squared Error Loss
#        Mean Squared Logarithmic Error Loss
#        Mean Absolute Error Loss
# 2. Binary Classification Loss Functions
#        Binary Cross-Entropy
#        Hinge Loss
#        Squared Hinge Loss
# 3. Multi-Class Classification Loss Functions
#        Multi-Class Cross-Entropy Loss
#        Sparse Multiclass Cross-Entropy Loss
#        Kullback Leibler Divergence Loss

# https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/


# Mean Square Error (MSE) loss function

from sklearn.datasets import make_regression
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html
# Imports
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split  # Better practice for data splitting
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.losses import MeanSquaredError  # Explicit import
from matplotlib import pyplot as plt  # Standard abbreviation

# Generate regression dataset
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=1)

# Standardize dataset
scaler_X = StandardScaler()  # Consistent naming
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()  # Flatten to convert back to 1D

# Split into training and test datasets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=1)  # Split the dataset evenly

# Define the model
model = Sequential()
model.add(Dense(25, input_dim=20, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(1, activation='linear'))

# Define the optimizer with learning rate correction
opt = SGD(learning_rate=0.01, momentum=0.9)  # Corrected parameter

# Compile the model with MSE loss
model.compile(loss=MeanSquaredError(),  
              optimizer=opt,
#              metrics=[MeanSquaredError()]
)  

# Fit the model with validation data
history = model.fit(
    trainX, 
    trainy, 
    validation_data=(testX, testy), 
    epochs=100, 
    verbose=1  # Verbose output to monitor progress
)

# Evaluate the model
train_mse = model.evaluate(trainX, trainy, verbose=0)
test_mse = model.evaluate(testX, testy, verbose=0)

# Display training and testing MSE
print(f'Train MSE: {train_mse:.3f}, Test MSE: {test_mse:.3f}')

# Plot loss (MSE) during training
plt.figure(figsize=(8, 6))  # Adjust the plot size
plt.title('Mean Squared Error (MSE) During Training')
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Test')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.show()


### Mean Squared Logarithmic Error (MSLE) loss function

# Imports
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split  # Better practice than manual slicing
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.losses import MeanSquaredLogarithmicError  # Explicit import for MSLE loss
from keras.metrics import MeanSquaredError  # Explicit import for metrics
from matplotlib import pyplot as plt  # Standard abbreviation

# Generate regression dataset
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=1)

# Standardize dataset
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Split into train and test datasets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=1)

# Define model
model = Sequential()
model.add(Dense(25, input_dim=20, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(1, activation='linear'))

# Define optimizer with learning rate correction
opt = SGD(learning_rate=0.01, momentum=0.9)

# Compile model with MSLE loss function and MSE as the metric
model.compile(
    loss=MeanSquaredLogarithmicError(), 
    optimizer=opt, 
    metrics=[MeanSquaredError()]
)

# Fit model and store training history
history = model.fit(
    trainX, 
    trainy, 
    validation_data=(testX, testy), 
    epochs=100, 
    verbose=1
)

# Evaluate the model
_, train_mse = model.evaluate(trainX, trainy, verbose=0)
_, test_mse = model.evaluate(testX, testy, verbose=0)

print('Train MSE: {:.3f}, Test MSE: {:.3f}'.format(train_mse, test_mse))

# Plot loss and MSE during training
plt.figure(figsize=(12, 8))

# Plot loss
plt.subplot(211)
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot Mean Squared Error
plt.subplot(212)
plt.title('Mean Squared Error')
plt.plot(history.history['mean_squared_error'], label='train')
plt.plot(history.history['val_mean_squared_error'], label='test')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()

# Display the plots
plt.tight_layout()  # Ensure there's no overlap
plt.show()


# Imports
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split  # Better practice than manual slicing
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.losses import MeanAbsoluteError  # Explicit import for MAE loss
from keras.metrics import MeanSquaredError  # Explicit import for metrics
from matplotlib import pyplot as plt  # Standard abbreviation for matplotlib.pyplot

# Generate regression dataset
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=1)

# Standardize dataset
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Split into training and test datasets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=1)

# Define model
model = Sequential()
model.add(Dense(25, input_dim=20, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(1, activation='linear'))

# Define optimizer with learning rate correction
opt = SGD(learning_rate=0.01, momentum=0.9)

# Compile model with MAE loss function and MSE as a metric
model.compile(
    loss=MeanAbsoluteError(), 
    optimizer=opt, 
    metrics=[MeanSquaredError()]
)

# Fit model with validation data and store training history
history = model.fit(
    trainX, 
    trainy, 
    validation_data=(testX, testy), 
    epochs=100, 
    verbose=1
)

# Evaluate the model
_, train_mse = model.evaluate(trainX, trainy, verbose=0)
_, test_mse = model.evaluate(testX, testy, verbose=0)

# Print evaluation results
print('Train MSE: {:.3f}, Test MSE: {:.3f}'.format(train_mse, test_mse))

# Plot loss and MSE during training
plt.figure(figsize=(12, 8))

# Plot loss
plt.subplot(211)
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot Mean Squared Error
plt.subplot(212)
plt.title('Mean Squared Error')
plt.plot(history.history['mean_squared_error'], label='train')
plt.plot(history.history['val_mean_squared_error'], label='test')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()

# Display the plots
plt.tight_layout()  # Ensures no overlap between plots
plt.show()


# Binary Cross-Entropy (CE)
# Imports
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split  # Better practice for splitting
from sklearn.preprocessing import StandardScaler  # Standardize the features
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.losses import BinaryCrossentropy  # Explicit import for loss function
from keras.metrics import Accuracy  # Explicit import for metrics
from matplotlib import pyplot as plt  # Standard abbreviation

# Generate 2D classification dataset
X, y = make_circles(n_samples=1000, noise=0.1, random_state=1)


# scatter plot of the circles dataset with points colored by class
from sklearn.datasets import make_circles
from numpy import where
from matplotlib import pyplot

# select indices of points with each class label
for i in range(2):
	samples_ix = where(y == i)
	pyplot.scatter(X[samples_ix, 0], X[samples_ix, 1], label=str(i))
pyplot.legend()
pyplot.show()


# Standardize dataset
scaler = StandardScaler()  # Standardization is often beneficial for neural networks
X = scaler.fit_transform(X)

# Split into training and testing sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=1)

# Define the model
model = Sequential()
model.add(Dense(50, input_dim=2, activation='relu', kernel_initializer='he_uniform'))  # Input layer
model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

# Define the optimizer and compile the model
opt = SGD(learning_rate=0.01, momentum=0.9)  # Corrected parameter
model.compile(
    loss=BinaryCrossentropy(),  # Corrected loss function
    optimizer=opt, 
    metrics=[Accuracy()]  # Explicit metrics import
)

# Fit the model and store training history
history = model.fit(
    trainX, 
    trainy, 
    validation_data=(testX, testy), 
    epochs=200, 
    verbose=1  # Set to 1 for better visibility of training progress
)

# Evaluate the model
_, train_acc = model.evaluate(trainX, trainy, verbose=0)
_, test_acc = model.evaluate(testX, testy, verbose=0)

# Print evaluation results
print(f'Train Accuracy: {train_acc:.3f}, Test Accuracy: {test_acc:.3f}')

# Plot loss and accuracy during training
plt.figure(figsize=(12, 8))  # Adjusting the figure size

# Plot loss
plt.subplot(211)
plt.title('Loss')
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Test')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot accuracy
plt.subplot(212)
plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Test')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Display the plots
plt.tight_layout()  # Ensures no overlap between plots
plt.show()


# Hinge Loss

# Import necessary libraries
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split  # For better data splitting
from sklearn.preprocessing import StandardScaler  # Useful for feature scaling
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.losses import Hinge  # Explicit import for hinge loss
from keras.metrics import Accuracy  # Explicit import for accuracy
from matplotlib import pyplot as plt  # Standard abbreviation
import numpy as np  # Explicit import for where()

# Generate 2D classification dataset
X, y = make_circles(n_samples=1000, noise=0.1, random_state=1)

# Change y from {0, 1} to {-1, 1}
y = np.where(y == 0, -1, y)  # Simplified syntax for where()

# Optionally standardize the dataset
scaler = StandardScaler()  # Helpful for consistent scaling
X = scaler.fit_transform(X)

# Split into training and test datasets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=1)  # Better practice

# Define the model
model = Sequential()
model.add(Dense(50, input_dim=2, activation='relu', kernel_initializer='he_uniform'))  # ReLU activation for hidden layers
model.add(Dense(1, activation='tanh'))  # Tanh activation for output layer

# Define the optimizer with correct learning rate syntax
opt = SGD(learning_rate=0.01, momentum=0.9)

# Compile the model with hinge loss and accuracy metric
model.compile(loss=Hinge(), optimizer=opt, metrics=[Accuracy()])

# Fit the model with validation data and verbose output
history = model.fit(
    trainX, 
    trainy, 
    validation_data=(testX, testy), 
    epochs=200, 
    verbose=1  # Verbose output to monitor training progress
)

# Evaluate the model
train_acc = model.evaluate(trainX, trainy, verbose=0)[1]  # Extract accuracy
test_acc = model.evaluate(testX, testy, verbose=0)[1]

print(f'Train Accuracy: {train_acc:.3f}, Test Accuracy: {test_acc:.3f}')

# Plot loss during training
plt.figure(figsize=(10, 6))  # Increased plot size
plt.subplot(211)
plt.title('Hinge Loss During Training')
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Test')
plt.xlabel('Epochs')
plt.ylabel('Hinge Loss')
plt.legend()

# Plot accuracy during training
plt.subplot(212)
plt.title('Accuracy During Training')
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Test')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()  # Ensure plots don't overlap
plt.show()
