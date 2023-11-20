import math

from time import time as timestamp

from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.models import Model
from keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.src.engine.training import Model

from catdog import PROJECTNAME, DATAROOT, MODELSROOT
from catdog.enums import Activations


TRAIN_DATA_DIR = DATAROOT / 'train'
VALIDATION_DATA_DIR = DATAROOT / 'val'
TRAIN_SAMPLES = 500
VALIDATION_SAMPLES = 500
NUM_CLASSES = 2
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 64

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DATA_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=12345,
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    VALIDATION_DATA_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    shuffle=False,
    seed=12345,
    class_mode='categorical'
)


def model_producer():
    input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)
    base_model: Model = MobileNet(include_top=False, input_shape=input_shape)

    for layer in base_model.layers[:]:
        layer.trainable = False

    input = Input(shape=input_shape)
    custom_model = base_model(input)
    custom_model = GlobalAveragePooling2D()(custom_model)
    custom_model = Dense(64, activation=Activations['relu'].name)(custom_model)
    custom_model = Dropout(0.5)(custom_model)
    predictions = Dense(NUM_CLASSES, activation=Activations['softmax'].name)(custom_model)

    return Model(inputs=input, outputs=predictions)


def execute_regimen():
    model = model_producer()
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['acc'],
    )
    num_steps = math.ceil(float(TRAIN_SAMPLES) / BATCH_SIZE)
    model.fit_generator(
        train_generator,
        steps_per_epoch=num_steps,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=num_steps,
    )
    model.save(f'{MODELSROOT / PROJECTNAME}-{int(timestamp())}.h5')
