# Imports
import numpy as np
import tensorflow as tf


def preprocessing():

    # load data from labelled folders "positive" and "negative"
    data = tf.keras.utils.image_dataset_from_directory('raw_data/data',
                                                       batch_size=32, # we can change this
                                                       image_size=(768, 1024), # keep the ratio of both datasets, but size to recommended size for MaskRCNN
                                                       color_mode='grayscale',
                                                       crop_to_aspect_ratio=True) # set as true to avoid ratio distorsion

    # resizing images to make it square and the recommended size for the Mask R-CNN model.
    data_resize = data.map(lambda x,y: (tf.image.resize_with_pad(x, 1024, 1024), y))

    # preprocessing step to scaled data
    data_scaled = data_resize.map(lambda x,y: (x/255., y))

    # Split into train / val / test
    train_size = int(len(data_scaled)*.7)
    val_size = int(len(data_scaled)*.2)
    test_size = int(len(data_scaled) - (val_size + train_size))

    train = data_scaled.take(train_size)
    val = data_scaled.skip(train_size).take(val_size)
    test = data_scaled.skip(train_size+val_size).take(test_size)
