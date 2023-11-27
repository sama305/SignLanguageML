import tensorflow as tf
import dataset
import config
import os
import numpy as np
from keras.models import load_model

def test_img(path):
    processed_img = dataset.process_image(path, config.IMG_SIZE)
    test_img_arr = processed_img.reshape(-1, config.IMG_SIZE, config.IMG_SIZE, 1).astype("float32") / 255.0
    model = load_model(config.MODEL_FILENAME)
    try:
        prediction = model.predict([test_img_arr], verbose = 0)
        str_pred = config.CATEGORIES[prediction.argmax(axis=-1)[0]]
    except:
        raise Exception("<!!!> Prediction failed <!!!>")
    return str_pred
    
def test_many_imgs():
    model = load_model(config.MODEL_FILENAME)
    test_img_arr_arr = []
    for img in os.listdir(config.TESTSET_DIR):
        processed_img = dataset.process_image(os.path.join(config.TESTSET_DIR, img), config.IMG_SIZE)
        processed_img = processed_img.reshape(-1, config.IMG_SIZE, config.IMG_SIZE, 1).astype("float32") / 255.0
        test_img_arr_arr.append(processed_img)
    try:
        prediction = model.predict(test_img_arr_arr)
        print(prediction)
        # print(config.CATEGORIES[prediction.argmax(axis=-1)[0]])
    except:
        raise Exception("<!!!> Prediction failed <!!!>")