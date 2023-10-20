import warnings
import os
import numpy as np
from flask import Flask, request, render_template
from PIL import Image
from keras import Sequential
from keras.layers import Dense
import tensorflow_hub as hub
import cv2

app = Flask(__name__)

# Load pre-trained models
path3 = "https://tfhub.dev/google/efficientnet/b0/classification/1"
efficient_model = hub.KerasLayer(path3, input_shape=(224, 224, 3), trainable=False)

num_class = 4

efficient_pre_model = Sequential()
efficient_pre_model.add(efficient_model)
efficient_pre_model.add(Dense(units=num_class, activation="softmax"))
efficient_pre_model.load_weights('models/brain_model.h5')

class_labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = np.array(image)
    image = image / 255.0
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction='Please upload an image.')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', prediction='Please upload a valid image file.')

    try:
        img = Image.open(file)
        img = np.array(img)
        img = preprocess_image(img)

        efficient_pred = efficient_pre_model.predict(np.expand_dims(img, axis=0))

        efficient_pred_class = class_labels[np.argmax(efficient_pred)]

        return render_template('index.html', efficient_pred=efficient_pred_class)
    except:
        return render_template('index.html', prediction='An error occurred while processing the image.')

if __name__ == '__main__':
    app.run(debug=True)
