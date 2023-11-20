from pathlib import Path
from sys import argv

import numpy as np

from keras.applications.mobilenet import preprocess_input
from keras.models import load_model, Model
from keras.utils import load_img
from keras.utils import img_to_array
from tensorflow import executing_eagerly
from tensorflow import compat

from catdog import DATAROOT, MODELSROOT
from catdog.train import validation_generator


DATA_HOME = Path(__file__).parent / 'data'


def make_prediction():
    modelname, samplefile = argv[1:]

    model: Model = load_model(MODELSROOT / modelname)

    img_path = DATAROOT / 'samples' / samplefile
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    prediction = model.predict(preprocessed_img)

    compat.v1.disable_eager_execution()
    print('FOUND_SAMPLE? ', img_path.exists())
    print('EXECUTING_EAGERLY? ', executing_eagerly())
    print(prediction)
    print(validation_generator.class_indices)
