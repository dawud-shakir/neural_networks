# README (TODO: convert to latex)

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




CNN.py




Pretrained.py
