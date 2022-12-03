"""
Solution stub for Question 2 (Neural Networks).

Fill in the implementations of the `mlp2` and `cnn` functions.

See https://www.tensorflow.org/tutorials for a Tensorflow tutorial.
"""
from __future__ import print_function
import numpy as np
import tensorflow as tf

# These should be the only tensorflow classes you need:
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D

# get_data returns (train_x, train_y), (test_x, test_y)
# argument determines whether images are shifted to top-left or bottom-right
# X values are an array of 30x30 images
# Y values are an array of 10 one-hot encoded labels
from cnn_utils import get_data

# show_examples creates an image that shows some example data from two datasets
# side by side
from cnn_utils import show_examples


def mlp1(train_x, train_y, test1_x, test1_y, test2_x, test2_y):
    """
    Train and evaluate a feedforward network with a single hidden layer.
    """
    model = Sequential([
      Flatten(input_shape=(30, 30)), # Need to flatten before Dense layers
      Dense(512, activation='relu'),
      Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_x, train_y, epochs=5)

    print("Evaluating MLP1 on test set 1")
    model.evaluate(test1_x, test1_y)
    print("Evaluating MLP1 on test set 2")
    return model.evaluate(test2_x, test2_y)

def mlp2(train_x, train_y, test1_x, test1_y, test2_x, test2_y):
    """
    Train and evaluate a feedforward network with two hidden layers.
    """
    # First layer will need argument `input_shape=(30,30)`
    model = Sequential([
        #`input_shape=(30,30)`,
        Flatten(input_shape=(30,30)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')        
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_x, train_y, epochs=5)

    print("Evaluating MLP2 on test set 1")
    model.evaluate(test1_x, test1_y)
    print("Evaluating MLP2 on test set 2")
    return model.evaluate(test2_x, test2_y)

def cnn(train_x, train_y, test1_x, test1_y, test2_x, test2_y):
    """
    Train and evaluate a feedforward network with two hidden layers.
    """
    # Add a single "channels" dimension at the end
    trn_x = train_x.reshape([-1, 30, 30, 1])
    tst1_x = test1_x.reshape([-1, 30, 30, 1])
    tst2_x = test2_x.reshape([-1, 30, 30, 1])

    # First layer will need argument `input_shape=(30,30,1)`
    model = Sequential([
        Conv2D(32, kernel_size=(5,5), activation="relu", input_shape=(30,30,1), strides=(1,1)),
        MaxPooling2D(pool_size=(2,2), strides=(2,2)),
        Conv2D(64, kernel_size=(5,5), activation="relu"),
        MaxPooling2D(pool_size=(2,2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(trn_x, train_y, epochs=5)

    print("Evaluating CNN on test set 1")
    model.evaluate(tst1_x, test1_y)
    print("Evaluating CNN on test set 2")
    return model.evaluate(tst2_x, test2_y)

def main():
    (train1_x, train1_y), (test1_x, test1_y) = get_data('top_left')
    (train2_x, train2_y), (test2_x, test2_y) = get_data('bottom_right')

    # Left column is images from top_left dataset
    # Right column is corresponding images from bottom_right dataset
    show_examples(test1_x, test1_y, test2_x, test2_y, 'examples.png')

    mlp1(train1_x, train1_y, test1_x, test1_y, test2_x, test2_y)
    mlp2(train1_x, train1_y, test1_x, test1_y, test2_x, test2_y)
    cnn(train1_x, train1_y, test1_x, test1_y, test2_x, test2_y)


if __name__ == '__main__':
    main()
