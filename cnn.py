# ~~~ cnn.py ~~~
# This file creates and trains the convolutional neural network 

import numpy as np
import dataset
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical

# loading with pickle (doesn't really work)
# X = pickle.load(open("X_features.pickle", "rb"))
# y = pickle.load(open("y_labels.pickle", "rb"))

def train_model(num_epochs, v_ratio):
    # loading with np
    (x_train, y_train) = dataset.load_dataset()

    # This model has 3 layers with an input shape of 65x65x1 (1 since the images
    # are grayscale) and can have 28 possible categories
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(65, 65, 1))) # 1 = grayscale
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(28, activation='softmax')) # Softmax, since multiclass
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Actually train and save the result of the model
    model.fit(x_train, y_train, epochs=num_epochs, batch_size=256, validation_split=v_ratio)
    model.save('slmodel.h5')

def model_summary():
    model = load_model('slmodel.h5')
    model.summary()

# Evaluate the model on the test set
# accuracy = model.evaluate(x_test, y_test, verbose=0)[1]
# print(f'Test Accuracy: {accuracy * 100:.2f}%')

