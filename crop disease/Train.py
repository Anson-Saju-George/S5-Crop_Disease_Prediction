import numpy as np
import seaborn as sns
import sys
import os
from matplotlib import pyplot
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras.optimizers import SGD
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

# Define paths
traindir = r"C:\Users\HP-OMEN\OneDrive\Desktop\crop disease\train"
validdir = r"C:\Users\HP-OMEN\OneDrive\Desktop\crop disease\valid"

# Define the model
def define_model(in_shape=(224, 224, 3), out_shape=38):
    model = VGG16(include_top=False, input_shape=in_shape, weights="../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")
    for layer in model.layers:
        layer.trainable = False

    # Allow last VGG block to be trainable
    for layer_name in ['block5_conv1', 'block5_conv2', 'block5_conv3', 'block5_pool']:
        model.get_layer(layer_name).trainable = True

    # Add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    fcon1 = Dense(4096, activation='relu', kernel_initializer='he_uniform')(flat1)
    fdrop1 = Dropout(0.25)(fcon1)
    fbn1 = BatchNormalization()(fdrop1)
    fcon2 = Dense(4096, activation='relu', kernel_initializer='he_uniform')(fbn1)
    fdrop2 = Dropout(0.25)(fcon2)
    fbn2 = BatchNormalization()(fdrop2)
    output = Dense(out_shape, activation='softmax')(fbn2)
    
    model = Model(inputs=model.inputs, outputs=output)
    opt = SGD(lr=0.01, momentum=0.9, decay=0.005)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Plot diagnostic learning curves
def summarize_diagnostics(history):
    sns.set()
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='valid')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Loss')
    pyplot.legend()
    
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='valid')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Accuracy')
    pyplot.legend()
    
    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig(filename + '_plot.png')
    pyplot.close()

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 128
training_iterator = train_datagen.flow_from_directory(traindir, target_size=(224, 224), batch_size=batch_size, class_mode='categorical')
validation_iterator = valid_datagen.flow_from_directory(validdir, target_size=(224, 224), batch_size=batch_size, class_mode='categorical')

# Train the model
model = define_model()
model.summary()

weightsfilepath = "bestweights.hdf5"
checkpoint = ModelCheckpoint(weightsfilepath, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True, mode='max')
callbacks_list = [checkpoint]

history = model.fit(training_iterator, steps_per_epoch=len(training_iterator),
                    validation_data=validation_iterator, validation_steps=len(validation_iterator), 
                    epochs=8, callbacks=callbacks_list, verbose=2)

summarize_diagnostics(history)
model.save('plantdisease_vgg16model.h5')
