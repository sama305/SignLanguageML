import numpy as np
import matplotlib.pyplot as plt
import os, sys
import cv2
import pickle

DATADIR = os.path.join(os.getcwd(), "dataset")
CATEGORIES = [
    "A", "B", "C", "D", "E", "F",
    "G", "H", "I", "J", "K", "L",
    "M", "N", "O", "P", "Q", "R",
    "S", "T", "U", "V", "W", "X",
    "Y", "Z", "space", "nothing"
]
IMG_SIZE = 65


print("Processing dataset:")

training_data = []

def create_training_data():
    prog = 0
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            prog += 1
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            training_data.append([resized_array, class_num])
            print(str(round(prog / 84000.0 * 100, 2)) + "%", end="\r")

create_training_data()

import random
random.shuffle(training_data)
# training_data = training_data[:1000]

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype("float32") / 255.0


# pickle_out = open("X_features.pickle", "wb")
# pickle.dump(X, pickle_out)
# pickle_out.close()

# pickle_out = open("y_labels.pickle", "wb")
# pickle.dump(X, pickle_out)
# pickle_out.close()

np.save("X_features.npy", X)
np.save("y_labels.npy", y)