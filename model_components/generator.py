"""
Load and preprocess data

using Keras' ImageDataGenerator to generate batches of image data 
with real-time data augmentation. Specifically, it sets up data generators 
for training and validation datasets from a directory structure.
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from extract_data import target_dir as data_dir

#adjust  batch side to 32 or 16
batch_size = 16
# adjust target size for images to 150x150 or 128x128
target_size = (128,128)
# datagen = ImageDataGenerator(rescale=0.2, validation_split=0.2)
datagen = ImageDataGenerator(
    rescale=1./255,          # Rescale the image pixel values from 0-255 to 0-1
    validation_split=0.2     # Set aside 20% of the data for validation
)

train_generator = datagen.flow_from_directory(
    data_dir,                # Directory where the data is located
    target_size=target_size,  # Resize all images to 150x150 pixels
    batch_size=batch_size,           # Number of images to be yielded from the generator per batch
    class_mode='categorical',# Return labels as one-hot encoded vectors
    subset='training'        # Specify that this is the training subset
)

validation_generator = datagen.flow_from_directory(
    data_dir,                # Directory where the data is located
    target_size=target_size,  # Resize all images to 150x150 pixels
    batch_size=batch_size,           # Number of images to be yielded from the generator per batch
    class_mode='categorical',# Return labels as one-hot encoded vectors
    subset='validation'      # Specify that this is the validation subset
)
