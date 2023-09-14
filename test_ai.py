# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 21:17:46 2023

@author: Gormery
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load the saved model
model = tf.keras.models.load_model('rps.h5')

# Define a function to classify an image
def classify_image(image_path):
    # Load the image
    img = image.load_img(image_path, target_size=(150, 150))
    # Convert the image to an array
    x = image.img_to_array(img)
    # Reshape the array to match the input shape of the model
    x = np.expand_dims(x, axis=0)
    # Scale the array values to be between 0 and 1
    x = x / 255.0
    # Use the model to predict the class probabilities for the image
    preds = model.predict(x)
    print(preds)
    # Return the predicted class label (rock, paper, or scissors)
    return ['paper', 'rock', 'scissors'][np.argmax(preds)]

# Test the function on an example image
image_path = './img6.png'
predicted_class = classify_image(image_path)
print(f"The model predicted that the image {image_path} is {predicted_class}.")
