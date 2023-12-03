import cv2
import numpy as np
import os
import sys
import tensorflow as tf
tf.get_logger().setLevel('INFO')

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = list()
    labels = []
    for i in range(NUM_CATEGORIES):
        for file in os.listdir(data_dir + os.sep + str(i)):
            img = cv2.imread(data_dir + os.sep + str(i) + os.sep + file)
            img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_LINEAR)
            images.append(img)
            labels.append(i)
    return images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv3D(
                input_shape=(IMG_WIDTH, IMG_HEIGHT, 3, 1),
                kernel_size=(3, 3, 1), activation='relu',
                filters=3
            ),
            #tf.keras.layers.AveragePooling3D(pool_size=(2, 2, 1)),
            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax')
        ]
    )
    ''' 
    Things I noticed:
    Convolution and pooling simplify things too much,
    3 filters on convolution
    Too much dropout, and dropout on 3 layers makes the neural network
    unable to learn quickly enough
    Consistent amount of neurons across layers seems to work best
    At least 3 layers
    4 layers of 128 nodes start to plateau at around 25 epochs
    3 layers seems to work best
    More epochs leads mainly to overfitting, not better accuracy
    3 layers of 256 nodes works great - 0.9238
    relu seems to be the best one
    '''
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


if __name__ == "__main__":
    main()