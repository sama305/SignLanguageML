import os

# Directory the project lies in
CWD = os.getcwd()

# Directory where all category folders will be found
DATASET_DIR = os.path.join(CWD, "dataset/training")

# Directory where all category folders will be found
TESTSET_DIR = os.path.join(CWD, "dataset/testing/asl_alphabet_test")

# Category names. Each name has a corresponding index (e.g., A=0, B=1, etc.)
CATEGORIES = [
    "A", "B", "C", "D", "E", "F",
    "G", "H", "I", "J", "K", "L",
    "M", "N", "O", "P", "Q", "R",
    "S", "T", "U", "V", "W", "X",
    "Y", "Z", "space", "nothing"
]

# Default input shape (height and width)
IMG_SIZE = 65

# Default umber of epochs by default
NUM_EPOCHS = 9

# Validation set ratio
VAL_RATIO = 0.2

# Name of model output filename (and file extension)
MODEL_FILENAME = "sl_model.h5"

# Total number of data points (pictures)
DATA_NUM = 84000

# Names for various training/test files
TR_FEAT_FILENAME = "X_train.npy"
TR_LABL_FILENAME = "y_train.npy"
TS_FEAT_FILENAME = "X_test.npy"
TS_LABL_FILENAME = "y_test.npy"