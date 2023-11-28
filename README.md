# Sign Language ML

This project was created for CIS4930.

## Installation & Setup

After cloning this repository, there are a few steps you need to follow to get this proejct set up.

First, and most importantly, you must install the dataset [here](https://drive.google.com/drive/folders/1Nr4YSa6-LoxhVAkVdENXDAv8BX5Ab0xk?usp=drive_link). After you install it, **place the dataset folder in the root of this project**, meaning the highest level.

After you must install **Python v3.11**. You can install it [here](https://www.python.org/downloads/release/python-3116/) (the version is important as this is the most recent version supported by Keras).

Next, you must install/upgrade several packages using `pip` (if you don't have pip, install/upgrade it with `python3.11 -m pip install -U pip`):
* NumPy
* Keras
* cv2
* MatPlotLib
```
python3.11 -m pip install -U numpy
python3.11 -m pip install -U keras
python3.11 -m pip install -U cv2
python3.11 -m pip install -U matplotlib
```

## Usage

Once the project has been setup, you can use it:
```
python3.11 main.py
```

After a few seconds, a GUI should appear. This GUI shows several commands you can execute by pressing a corresponding number. For instance, you can quit the program by using `0`.

If it is your first time using the program, it will suggest that you generate the preprocessed dataset and train a model.

### Commands
1. Quit (`0`)
    * This command simply quits the program
2. Generate Dataset (`1`)
    * Running this command will begin the dataset preprocessing sequence. 
    * This *must* be done before training a model, as the model looks for corresponding `.npy` files to use in the training process.
    * This can take a minute or two, but there is a progress bar so you know exactly how far you are.
    * Once completed, the files `X_train.npy` and `y_train.npy` are generated.
3. Train Model (`2`)
    * This command will begin the training sequence for the model.
    * This can take very long (up to 5-10 minutes with default settings) so, for testing purpose, you can enter a smaller amount of epochs at the cost of a lower overall accuracy.
    * Once completed, a `sl_model.h5` file is generated.
4. Model Summary (`3`)
    * Running this command will give you a summary of the model.
    * It includes the number of parameters and input shape, as well as some other important variables.
5. View Random Sample (`4`)
    * This command will visually output a random preprocessed image.
6. Prediction (`5`)
    * This command allows you to test the accuracy of the model with test images.
    * Test images are provided but you could theoretically add your own, as all preprocessing is done to the images automatically.
    * You have two options, single test and bulk test. Single test will ask for an input (in the format of `X_test.jpg`, where X would be any letter along with `nothing` or `space`) and conduct a prediction on that. Bulk test will test all images and give you the accuracy.