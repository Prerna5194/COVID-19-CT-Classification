from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np


# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'C:\\Users\\Ram\\Desktop\\WeSchool\\TRIM-6\\External Assignment\\Covid_Model.h5'
from tensorflow import keras
import tensorflow as tf


config = tf.ConfigProto(
    device_count={'GPU': 1},
    intra_op_parallelism_threads=1,
    allow_soft_placement=True
)

config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6

session = tf.Session(config=config)

keras.backend.set_session(session)

# Load your trained model
model = load_model(MODEL_PATH)
model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    # Preprocessing the image
    try:
        with session.as_default():
            with session.graph.as_default():
                test_image = image.load_img(img_path, target_size = (150, 150), color_mode= 'grayscale')
                test_image = image.img_to_array(test_image)
                test_image = np.expand_dims(test_image, axis = 0)
                
                preds = model.predict(test_image)
                if preds[0][0] == 1:
                    prediction = 'NonCovid'
                else:
                    prediction = 'Covid'
                return prediction
    except Exception as ex:
        print('Covid Prediction Error', ex, ex.__traceback__.tb_lineno)
    
    
    
    
    test_image = image.load_img(img_path, target_size = (150, 150), color_mode= 'grayscale')
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    
    preds = model.predict(test_image)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'Dataset', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        result = model_predict(file_path, model)
        #pred_class = decode_predictions(preds, top=1)
# =============================================================================
#         if result[0][0] == 1:
#             prediction = 'NonCovid'
#         else:
#             prediction = 'Covid'
# =============================================================================
        #result = str(pred_class[0][0])
        
        
        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        # Convert to string
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
