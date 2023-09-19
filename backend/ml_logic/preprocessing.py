import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt


def preprocessing():

    parent_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_path = os.path.join(parent_path, 'raw_data', 'data')

    # load data from labelled folders "positive" and "negative"
    data = tf.keras.utils.image_dataset_from_directory(data_path,
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

    return train, val, test
