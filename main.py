# ~~~ main.py ~~~
# Main entry point, contains a GUI for casting commands on the program. From
# this file you can train the model, generate the dataset/testing files, and
# test the model, among other things.

import os
import dataset
import config
import cnn
import cv2
import test
import random
import numpy as np
import matplotlib.pyplot as plt

def gen_all():
    dataset.create_training_data(config.IMG_SIZE)
    print("<!> All images sucessfully processed!")
    print("Creating/compiling/training the model...")
    cnn.train_model(config.NUM_EPOCHS, config.VAL_RATIO)

def gen_dataset_prompt():
    dataset.create_training_data(config.IMG_SIZE)
    print(f"<!> Created/updated files: {config.TR_FEAT_FILENAME}, {config.TR_LABL_FILENAME}")

def train_model_prompt():
    epochs = input("# of epochs (empty for default, " + str(config.NUM_EPOCHS) + "): ")
    if (epochs):
        epochs = int(epochs)
    else:
        epochs = config.NUM_EPOCHS

    vset = input("Validation set ratio (empty for default, " + str(config.VAL_RATIO) + "): ")
    if (vset):
        vset = float(vset)
    else:
        vset = config.VAL_RATIO

    cnn.train_model(epochs, vset)
    print(f"<!> Created/updated files: {config.MODEL_FILENAME}")

def show_sample_images():
    images = []
    for category in config.CATEGORIES:
        path = os.path.join(config.DATASET_DIR, category)
        random_index = random.randint(0, 3000 - 1)
        random_file = os.listdir(path)[random_index]
        images.append(dataset.process_image(os.path.join(path, random_file), config.IMG_SIZE))
    images = np.array(images)
    i = random.randint(0, 27)
    plt.imshow(images[i], cmap="gray")
    plt.title("Category: " + config.CATEGORIES[i])
    plt.show()

def test_img_prompt():
    choice = input("Do you want to test all images? [Y/n]: ")
    if choice.lower() == 'y':
        test.test_many_imgs()
    else:
        path = input("Filename: ")
        pred = test.test_img(os.path.join(config.TESTSET_DIR, path))
        plt.imshow(dataset.process_image(os.path.join(config.TESTSET_DIR, path), config.IMG_SIZE), cmap="gray")
        plt.title(path + " --predicted-> " + pred)
        plt.show()


# !!!!!!!!!!!!!!!!!!!!!!!!!!!
# ! MAIN PROGRAM ENTRY HERE !
# !!!!!!!!!!!!!!!!!!!!!!!!!!!

if (__name__ == "__main__"):
    # If a missing training file or model is detected, prompt user
    if not (os.path.exists(os.path.join(config.CWD, config.TR_FEAT_FILENAME)) and
        os.path.exists(os.path.join(config.CWD, config.TR_LABL_FILENAME)) and
        os.path.exists(os.path.join(config.CWD, config.MODEL_FILENAME))):
        print("There seems to be one or more missing training/model files.")
        i = input("Would you like to generate/update ALL model/training files (this will overwrite any existing data) [y/n]:")
        if (str(i).lower() == "y"):
            print("<!> Generating all necessary files with default values (this can take some time!)...")
            gen_all()

    print ("Welcome to the interactive GUI for SignLanguageML")
    print("Please pick an option:")
    print("\t0) Quit")
    print("\t1) Generate dataset")
    print("\t2) Train the model on the dataset")
    print("\t3) Retrieve a model summary")
    print("\t4) Observe a random sample")
    print("\t5) Make a prediction")
    while (True):
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
        elif (inp == 5):
            test_img_prompt()

        print()
