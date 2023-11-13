import numpy as np
import pickle
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical

# loading with pickle
# X = pickle.load(open("X_features.pickle", "rb"))
# y = pickle.load(open("y_labels.pickle", "rb"))

# loading with np
x_train = np.load("X_features.npy")
y_train = np.load("y_labels.npy")
y_train = to_categorical(y_train, num_classes=28)

# Define the model
model = Sequential()

# Add a convolutional layer
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(65, 65, 1))) # grayscale

# Add a max pooling layer
model.add(MaxPooling2D((2, 2)))

# Add another convolutional layer
model.add(Conv2D(64, (3, 3), activation='relu'))

# Add another max pooling layer
model.add(MaxPooling2D((2, 2)))

# Flatten the output
model.add(Flatten())

# Add a dense layer
model.add(Dense(128, activation='relu'))

# Add the output layer
model.add(Dense(28, activation='softmax'))  # For binary classification, change 1 to the number of classes for multi-class

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Display the model summary
model.fit(x_train, y_train, epochs=2, validation_split=0.2)
model.summary()
model.save('slmodel.h5')

# Evaluate the model on the test set
# accuracy = model.evaluate(x_test, y_test, verbose=0)[1]
# print(f'Test Accuracy: {accuracy * 100:.2f}%')

