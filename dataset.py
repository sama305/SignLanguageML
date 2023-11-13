

import numpy as np
import os, sys
import cv2
import pickle
import random
from keras.utils import to_categorical

def create_training_data(dataset_path, categories, size):
    training_data = []
    prog = 0

    print("Processing dataset:")

    # In this for loop nest, we are going through each image in each categorical
    # folder and (1) turning them into grayscale (2) resizing them to a desired
    # size
    for category in categories:
        path = os.path.join(dataset_path, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            prog += 1
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            resized_array = cv2.resize(img_array, (size, size))
            training_data.append([resized_array, class_num])
            print(str(round(prog / 84000.0 * 100, 2)) + "%", end="\r")
    print("100%")
    
    # Shuffling data to make all categories equally likely
    random.shuffle(training_data)

    # The training_data list is structure as follows: [(X, y), (X, y), ...],
    # so we can easily extract the features (X) and labels (y) into separate
    # lists. It is important we do this after shuffling or else the labels
    # would be wrong.
    y = []
    X = []
    for features, label in training_data:
        X.append(features)
        y.append(label)

    # Converting the feature and label arrays into a format that can be accepted
    # by keras
    X = np.array(X).reshape(-1, size, size, 1).astype("float32") / 255.0
    y = to_categorical(y, num_classes=28)

    # Finally, we save these into numpy files
    np.save("X_features.npy", X)
    np.save("y_labels.npy", y)

def create_testing_data(testset_path, categories, size):
    pass

def load_dataset():
    return (np.load("X_features.npy"), np.load("y_labels.npy"))

# Alternate saving method (???) doesn't really work
# pickle_out = open("X_features.pickle", "wb")
# pickle.dump(X, pickle_out)
# pickle_out.close()

# pickle_out = open("y_labels.pickle", "wb")
# pickle.dump(X, pickle_out)
# pickle_out.close()
