# main.py
# Main entry point, contains a GUI for casting commands on the program. From
# this file you can train the model, generate the dataset/testing files, and
# test the model, among other things.

import os
import dataset
import cnn
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

CWD = os.getcwd()
# Directory where all category folders will be found
DATASET_DIR = os.path.join(CWD, "dataset")

# Category names. Each name has a corresponding index (e.g., A=0, B=1, etc.)
CATEGORIES = [
    "A", "B", "C", "D", "E", "F",
    "G", "H", "I", "J", "K", "L",
    "M", "N", "O", "P", "Q", "R",
    "S", "T", "U", "V", "W", "X",
    "Y", "Z", "space", "nothing"
]

# Input shape height and width
IMG_SIZE = 65

# Number of epochs by default
NUM_EPOCHS = 3

# Validation set ratio
VAL_RATIO = 0.2

def gen_dataset_prompt():
    size = input("Image size (empty for default, " + str(IMG_SIZE) + "): ")
    if (size):
        size = int(size)
    else:
        size = IMG_SIZE
    
    dataset.create_training_data(DATASET_DIR, CATEGORIES, IMG_SIZE)
    print("<!> Created/updated files: X_features.npy, y_labels.npy <!>\n")

def train_model_prompt():
    epochs = input("# of epochs (empty for default, " + str(NUM_EPOCHS) + "): ")
    if (epochs):
        epochs = int(epochs)
    else:
        epochs = NUM_EPOCHS

    vset = input("# of epochs (empty for default, " + str(VAL_RATIO) + "): ")
    if (vset):
        vset = float(epochs)
    else:
        vset = VAL_RATIO

    cnn.train_model(epochs, vset)

def show_sample_images():
    images = []
    for category in CATEGORIES:
        path = os.path.join(DATASET_DIR, category)
        random_index = random.randint(0, 3000 - 1)
        random_file = os.listdir(path)[random_index]
        img_array = cv2.imread(os.path.join(path, random_file), cv2.IMREAD_GRAYSCALE)
        images.append(img_array)
    images = np.array(images)
    i = random.randint(0, 28)
    plt.imshow(images[i], cmap="gray")
    plt.title("Category: " + CATEGORIES[i])
    plt.show()

# !!!!!!!!!!!!!!!!!!!!!!!!!!!
# ! MAIN PROGRAM ENTRY HERE !
# !!!!!!!!!!!!!!!!!!!!!!!!!!!

if (os.path.exists(os.path.join(CWD, "X_features.npy")) or
    os.path.exists(os.path.join(CWD, "X_features.npy")) or
    os.path.exists(os.path.join(CWD, "X_features.npy")) or
    os.path.exists(os.path.join(CWD, "X_features.npy"))):
    pass

print ("Welcome to the interactive GUI for SignLanguageML")
while (True):
    print("Please pick an option:")
    print("\t0) Quit")
    print("\t1) Generate dataset")
    print("\t2) Train the model on the dataset")
    print("\t3) Retrieve a model summary")
    print("\t4) Observe a random sample")
    try:
        inp = int(input(">> "))
    except:
        inp = -1

    if (inp == 0):
        break
    elif (inp == 1):
        gen_dataset_prompt()
    elif (inp == 2):
        train_model_prompt()
    elif (inp == 3):
        cnn.model_summary()
    elif (inp == 4):
        show_sample_images()
