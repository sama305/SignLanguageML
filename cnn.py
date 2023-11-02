import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


# Load the CIFAR-10 dataset
(X, y), _ = keras.datasets.cifar10.load_data()

# Convert labels to one-hot encoding if needed
# y = keras.utils.to_categorical(y, num_classes=10)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
# Convert pixel values to a range between 0 and 1
X_train = X_train.astype('float32') / 255

# Calculate the mean pixel value of the training set
mean_pixel = X_train.mean(axis=(0, 1, 2))

# Subtract the mean pixel value from the images (zero-centering)
X_train -= mean_pixel

# Define the CNN model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model on the training data
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
model.save('slmodel.keras')

