#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import keras
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers
from datetime import datetime

#train_dir = '/home/jimmy/catkin_ws/src/pretrain/clean-dataset/train'
#validation_dir = '/home/jimmy/catkin_ws/src/pretrain/clean-dataset/validation'
train_dir = '/home/advrobot/keras_Fine_Tuning_VGG16/train'
validation_dir = '/home/advrobot/keras_Fine_Tuning_VGG16/test'
save_model_name = 'ev_safty_check_20epochs_drop25.h5'
image_size = 224

#Experiment 2 : Train Last 4 layers without data augmentation

#Load the VGG model
vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

# Freeze all the layers, not included the last 4 layer
for layer in vgg_conv.layers[:-4]:
    layer.trainable = False

# Check the trainable status of the individual layers
for layer in vgg_conv.layers:
    print(layer, layer.trainable)




# Create the model
model = models.Sequential()

# Add the vgg convolutional base model
model.add(vgg_conv)

# Add new layers
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.25))
#model.add(layers.Dense(3, activation='softmax'))
#two classified
model.add(layers.Dense(2, activation='softmax'))

# Show a summary of the model. Check the number of trainable parameters
model.summary()

# No Data augmentation 
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Change the batchsize according to your system RAM
train_batchsize = 10
val_batchsize = 10

# Data Generator for Training data
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_size, image_size),
        batch_size=train_batchsize,
        class_mode='categorical')

# Data Generator for Validation data
validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(image_size, image_size),
        batch_size=val_batchsize,
        class_mode='categorical',
        shuffle=False)


# Get the label to class mapping from the generator
label2index = validation_generator.class_indices
print('label2index = ',label2index)