# ~~~ cnn.py ~~~
# This file creates and trains the convolutional neural network 

import dataset
import config
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical

# loading with pickle (doesn't really work)
# X = pickle.load(open("X_features.pickle", "rb"))
# y = pickle.load(open("y_labels.pickle", "rb"))

def train_model(num_epochs, v_ratio):
    # loading the data in
    # X is the features, y is the labels
    # NOTE: we don't have test data yet
    # (X_train, y_train, X_test, y_test) = dataset.load_dataset()
    try:
        (X_train, y_train) = dataset.load_dataset()
    except FileNotFoundError:
        print(f"error: file(s) {config.TR_FEAT_FILENAME}, {config.TR_LABL_FILENAME} don't exist")
        print("note: try generating them using option (1)")
        return

    # This model has 3 layers with an input shape of 65x65x1 (1 since the images
    # are grayscale) and can have 28 possible categories
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(config.IMG_SIZE, config.IMG_SIZE, 1))) # 1 = grayscale
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(len(config.CATEGORIES), activation='softmax')) # Softmax, since multiclass
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Actually train and save the result of the model
    # Batch size seems to affect accuracy (>=256 results in lower accuracy)
    model.fit(X_train, y_train, epochs=num_epochs, batch_size=64, validation_split=v_ratio)
    model.save(config.MODEL_FILENAME)

def model_summary():
    try:
        model = load_model(config.MODEL_FILENAME)
        model.summary()
    except OSError:
        print("error: file " + config.MODEL_FILENAME + " not found")
        print("note: try creating/training it using option (2)")

# Evaluate the model on the test set
# accuracy = model.evaluate(X_test, y_test, verbose=0)[1]
# print(f'Test Accuracy: {accuracy * 100:.2f}%')

