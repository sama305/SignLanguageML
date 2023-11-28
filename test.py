import dataset
import config
import os
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
    
def test_img_bulk():
    model = load_model(config.MODEL_FILENAME)
    correct_predictions = 0
    total_predictions = 0
    for img in os.listdir(config.TESTSET_DIR):
        processed_img = dataset.process_image(os.path.join(config.TESTSET_DIR, img), config.IMG_SIZE)
        test_img_arr = processed_img.reshape(-1, config.IMG_SIZE, config.IMG_SIZE, 1).astype("float32") / 255.0
        try:
            total_predictions += 1
            prediction = model.predict([test_img_arr], verbose = 0)
            if config.CATEGORIES[prediction.argmax(axis=-1)[0]] == img[0]:
                correct_predictions += 1
            print(f'{total_predictions} of {len(os.listdir(config.TESTSET_DIR))} images tested with accuracy of {1.0 * correct_predictions / total_predictions * 100:.0f}%', end="\r")
        except:
            raise Exception("<!!!> Prediction failed <!!!>")