import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler  # for standardization

import torch
from torch.utils.data import TensorDataset

github = "https://raw.githubusercontent.com/dawud-shakir/logistic_regression/main/in/"

predict_kaggle_dataset = True
kaggle_dataset_file = f"/kaggle_mfcc_13.csv"


predict_kaggle_dataset = True



df = pd.read_csv(github + mfcc_and_label_dataset_file)      # same as 2nd project 
y_labels = df.iloc[:,-1]   # "blues", "rock", etc.
unique_labels = np.unique(y_labels)

if predict_kaggle_dataset:
    ### Load Kaggle dataset
    df_kaggle = pd.read_csv(github + kaggle_dataset_file)
    X_kaggle = df_kaggle.iloc[:, 0:]  # unknown kaggle testing data
    
    X_kaggle = StandardScaler().fit_transform(X_kaggle)   # standardize

    kaggle_dataset = TensorDataset(torch.tensor(X_kaggle, dtype=torch.float32))

    # Predict
    model.eval()  # because model is already trained, use evaluation mode
    with torch.no_grad():  # disable gradient calculation, only need predictions
        outputs = model(kaggle_dataset.tensors[0])  # Pass Kaggle dataset to model
        predictions = unique_labels[outputs.argmax(dim=1).tolist()]  # Get label of predicted IDs



    files_in_test_dir = pd.read_csv(github + "list_test.txt", header=None)   # from data/test/
    
    kaggle_submission = pd.DataFrame()
    kaggle_submission.insert(0, "id", files_in_test_dir)
    kaggle_submission.insert(1, "class", predictions)

    print(len(predictions), "Kaggle predictions:")
    print(kaggle_submission)

    kaggle_submission.to_csv(save_kaggle_submission_as, index=False)  # no index for submission file
    print("Kaggle submission:", save_kaggle_submission_as)


