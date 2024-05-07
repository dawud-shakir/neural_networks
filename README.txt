# README (TODO: convert to latex)


They include data loading, preprocessing, model definition, training, evaluation, and visualization functionalities.  After training, their test method is called on the trained model to make predictions on the test set.
       A cassification report (precision, recall, F1-score), overall test accuracy, and overall test loss are stored and metrics for each class are plotted.
1. mlp.py: 


2. cnn.py
This demo is a CNN model for audio classification. 


3. pretrained.py
This demo is a pre-trained VGG16 model for audio classification.



Our MFCC13 and MFCC128 datasets can be accessed:

The "data/" folder is available from professor Trilce's course website or at 

github = "https://raw.githubusercontent.com/dawud-shakir/logistic_regression/main/data/"

---- Parameters -----


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




PENALTY = 1e-3    # weight decay (lambda) is usually from 1e-5 to 1e-2

### BATCH_SIZE
# Smaller batch sizes (e.g., 1, 8) can lead to more updates but slower training. 
# Larger batch sizes (e.g., 128, 256) can be more stable but require more memory.

BATCH_SIZE = 32 # [1, 8, 16, 32, 128, 256]

### LEARNING_RATE
## How much to adjust the model's weights based on the gradient from the loss function.
## A higher learning rate causes larger updates to the weights, which leads to faster
## convergence (or instability in some cases). A lower learning rate results in smaller updates, 
## which leads to more stablity but a slower convergence.

LEARNING_RATE = [1e-5, 1e-1]    

### MOMENTUM 
# Momentum is used to maintain a direction during training.
# Typical Range: 0 to 0.9 allows the model to explore the impact 
# of momentum without leading to excessive oscillations.
MOMENTUM = [0.0, 0.9] # uniform distribution


### DROP_OUT_RATE
# Generally between 0.1 and 0.5. Higher dropout rates can help prevent overfitting 
# but might impact training effectiveness and stability.
DROP_OUT_RATE = [0.1, 0.5]    # drop out "this" many nodes from layer 

### TRAIN, VALIDATE, TEST
## A common training/validation/testing split is 70/15/15 or 80/10/10.
TRAIN_RATIO = 0.8       # 720
VALIDATION_RATIO = 0.2  # 180
TEST_RATIO = 0.0



MLP.py

 This script creates and trains a multi-layer perceptron (MLP) model for classifying audio recordings using Mel-frequency cepstral coefficients (MFCCs).

Data

    The script loads MFCC features and labels from a CSV file.
    It then splits the data into training, validation, and test sets, maintaining class distribution using stratification.
    Standarization is applied to the features (MFCCs) before feeding them to the model.

Model

    The model is a multi-layer perceptron (MLP) with a single hidden layer.
    The number of hidden units is set to a hyperparameter HIDDEN_SIZE.
    Dropout is used for regularization to prevent overfitting (controlled by DROP_OUT_RATE).
    The model uses ReLU activation in the hidden layer and softmax activation in the output layer for multi-class classification.

Training

    Adam optimizer is used with a learning rate (LEARNING_RATE) and weight decay (PENALTY) for regularization.
    The script trains the model for a fixed number of epochs (MAX_ITERATIONS).
    PyTorch Lightning is used for training and validation.
    Early stopping or other regularization techniques are not explicitly included in this code.

Evaluation

    After training, the model is evaluated on the test set.
    The script calculates precision, recall, and F1-score for each class using the classification_report function.
    It also calculates overall test accuracy and loss.
    Finally, the script plots the training and validation accuracy and loss curves.

Key Points

    This script focuses on building and training a basic MLP model for audio classification using MFCC features.
    Hyperparameters like HIDDEN_SIZE, LEARNING_RATE, and DROP_OUT_RATE are set manually and could be optimized using techniques like grid search or random search.
    The script doesn't include techniques like early stopping or learning rate scheduling, but these could be incorporated for better performance.


CNN.py


Convolutional Neural Network (CNN) for Audio Classification

This script defines and trains a CNN model for classifying audio recordings using Mel-frequency cepstral coefficients (MFCCs).

Data Loading and Preprocessing:

    Data Path: The script assumes audio files are stored in the data/train directory with a .au extension.
    Librosa: It uses the librosa library to read audio files, extract MFCC features, and convert them to decibels (dB).
    Label Encoding: Filenames are used as labels, and a LabelEncoder transforms them into numerical values.
    Data Split: The script splits the data into training and validation sets using train_test_split from scikit-learn.
    Dataset and Dataloader: A custom AudioDataset class is defined to load and preprocess audio files on the fly during training.
        It truncates MFCCs to a specific shape ([128, 1290]).
        It standardizes (z-scores) the features for better model performance.
        It adds a channel dimension ([1, 128, 1290]) for the CNN.
        The AudioDataLoader class inherits from pl.LightningDataModule and handles splitting data into training, validation, and test sets using another round of train_test_split.

Model Architecture (CNN):

    The model is a CustomCNN class that inherits from pl.LightningModule.
    It defines a typical CNN architecture with convolutional layers, pooling layers, and fully connected layers.
        Convolutional layers: Two convolutional layers are used with 32 and 64 filters.
        Pooling layers: Max pooling is used for dimensionality reduction.
        Dropout layer: Dropout with a rate of 0.2 is used for regularization to prevent overfitting.
        Fully connected layers: Two fully connected layers are used.
            The first layer transforms the features from the convolutional layers into a lower-dimensional space (128).
            The second layer has an output size equal to the number of classes for classification.
    The model uses ReLU activation in the convolutional and hidden layers and softmax activation in the output layer for multi-class classification.

Training and Evaluation:

    PyTorch Lightning: The script leverages PyTorch Lightning for training and validation.
    Adam Optimizer: The Adam optimizer is used with a learning rate (LEARNING_RATE) and weight decay (PENALTY) for regularization.
    Evaluation Metrics: The evaluate method calculates loss, accuracy, and predictions on a batch.
    Training Loop: The training_step and validation_step methods call the evaluate method to calculate loss on training and validation data, respectively.
    Custom Evaluation for Testing: The CustomCNN class overrides the test_step method to store predictions, labels, accuracy, and loss during testing for later evaluation.
  

Additional Notes:

    The script defines constants and hyperparameters like MODEL, CHANNELS, BATCH_SIZE, HIDDEN_SIZE, LEARNING_RATE, etc.
    It uses aggressive seeding (SEED_RANDOM) to ensure reproducibility of the training process.
    The script saves the training and validation metrics (metrics.csv) and plots them after training.
    It also saves the final test performance metrics and model visualizations.
    Training and testing times are printed for reference.


Pretrained.py





Our Kaggle submissions were created thusly:

\begin{code}

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

\end{code}

