from tensorflow.keras.models import load_model

# this is where we would test out our dat
new_data

# Load the trained model
loaded_model = load_model('slmodel.h5')

# Make predictions using the loaded model
predictions = loaded_model.predict(new_data)

