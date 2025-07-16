import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras import Model


testdir = "D:/crop disease/test"
weightsfilepath = "D:/crop disease/bestweights.hdf5"


class_labels = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]


def define_model(in_shape=(224, 224, 3), out_shape=len(class_labels)):
    base_model = VGG16(include_top=False, input_shape=in_shape, weights=None)
    for layer in base_model.layers[:-4]:  # Freeze all except last 4 layers
        layer.trainable = False
    x = Flatten()(base_model.output)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = BatchNormalization()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = BatchNormalization()(x)
    output = Dense(out_shape, activation='softmax')(x)
    model = Model(inputs=base_model.inputs, outputs=output)
    model.load_weights(weightsfilepath)
    return model


def load_image(filename):
    img = load_img(filename, target_size=(224, 224))
    img = img_to_array(img) / 255.0
    return np.expand_dims(img, axis=0)


model = define_model()


for filename in os.listdir(testdir):
    filepath = os.path.join(testdir, filename)
    img = load_image(filepath)
    prediction = model.predict(img)
    predicted_class_name = class_labels[np.argmax(prediction)]
    print(f"{filename} predicted as {predicted_class_name}")
