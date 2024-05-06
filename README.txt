# README (TODO: convert to latex)

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

    Model Definition: An MLP class with an initializer, forward propagation, evaluation functions, training, validation, and test steps.
    Dataset Preparation: Preprocessing and splitting the dataset into training, validation, and test sets.
    Training and Testing: Training the MLP using a Trainer and then evaluating performance on test data, including calculating confidence intervals for accuracy.
    Plotting: Generating plots for accuracy and loss metrics.


CNN.py




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

