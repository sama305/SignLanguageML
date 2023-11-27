

import numpy as np
import os
import config
import cv2
import random
from keras.utils import to_categorical

def process_image(path, newsize):
    img_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    resized_array = cv2.resize(img_array, (newsize, newsize))
    return resized_array

def create_training_data(size):
    training_data = []
    prog = 0

    print("Compiling dataset:")

    # In this for loop nest, we are going through each image in each categorical
    # folder and (1) turning them into grayscale (2) resizing them to a desired
    # size
    for category in config.CATEGORIES:
        path = os.path.join(config.DATASET_DIR, category)
        class_num = config.CATEGORIES.index(category)
        for img in os.listdir(path):
            prog += 1
            training_data.append([process_image(os.path.join(path, img), size), class_num])
            print(f'{prog} of {config.DATA_NUM} generated ({prog / config.DATA_NUM *100:.0f}%)', end="\r")
    print("\nShuffling...")
    
    # Shuffling data to make all categories equally likely
    random.shuffle(training_data)

    # The training_data list is structure as follows: [(X, y), (X, y), ...],
    # so we can easily extract the features (X) and labels (y) into separate
    # lists. It is important we do this after shuffling or else the labels
    # would be wrong.
    print("Structuring...")
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
    print("Saving...")
    np.save(config.TR_FEAT_FILENAME, X)
    np.save(config.TR_LABL_FILENAME, y)

def load_dataset():
    return (np.load(config.TR_FEAT_FILENAME), np.load(config.TR_LABL_FILENAME))

# Alternate saving method (???) doesn't really work
# pickle_out = open("X_features.pickle", "wb")
# pickle.dump(X, pickle_out)
# pickle_out.close()

# pickle_out = open("y_labels.pickle", "wb")
# pickle.dump(X, pickle_out)
# pickle_out.close()
