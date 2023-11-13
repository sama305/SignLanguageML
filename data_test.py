import keras
from PIL import Image
import numpy as np
import os
import cv2

IMG_SIZE = 65

model = keras.models.load_model('slmodel.h5')
# Load the image
images = []  # Convert to grayscale

TESTDIR = os.path.join(os.getcwd(), "dataset/testing/asl_alphabet_test")
for img in os.listdir(TESTDIR):
    img_array = cv2.imread(os.path.join(TESTDIR, img), cv2.IMREAD_GRAYSCALE)
    resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    images.append(resized_array)

# Convert the image to a NumPy array
image_array = np.array(images).reshape(-1, 65, 65, 1).astype('float32') / 255.0

# Add batch dimension and channel dimension to match the expected input shape
# image_array = np.expand_dims(np.expand_dims(image_array, axis=0), axis=-1)

# Now, you can use this 'image_array' for prediction with your model
predictions = model.predict(image_array)

# Print the predictions
print("Predictions:")
print(predictions)


