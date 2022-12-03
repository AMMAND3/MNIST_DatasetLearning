"""
Utility functions for cnn.py
"""
import tensorflow as tf
import numpy as np

def get_data(name):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    
    with tf.compat.v1.Session():
         trn_y = tf.one_hot(y_train, 10).eval()
         tst_y = tf.one_hot(y_test, 10).eval()

    if name == 'top_left':
        trn_x = decenter(x_train, -2)
        tst_x = decenter(x_test, -2)
    elif name == 'bottom_right':
        trn_x = decenter(x_train, 2)
        tst_x = decenter(x_test, 2)
    else:
        raise ValueError("Only valid names are ['top_left', 'bottom_right'], not '%s'" % name)

    return (trn_x,trn_y), (tst_x,tst_y)

def decenter(X,pad=2):
    out = np.zeros([X.shape[0], X.shape[1]+abs(pad), X.shape[2]+abs(pad)])
    if pad > 0:
        out[:,pad:X.shape[1]+pad, pad:X.shape[2]+pad] = X
    else:
        out[:,0:X.shape[1], 0:X.shape[2]] = X
    return out

def show_examples(X1, Y1, X2, Y2, fname='examples.png'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(1,5))
    for digit in range(10):
        i = Y1[:,digit].argmax()
        plt.subplot(10, 2, 2*digit+1)
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)
        
        plt.imshow(X1[i], cmap='gray')

        plt.subplot(10, 2, 2*digit+2)
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)

        plt.imshow(X2[i], cmap='gray')
    plt.savefig(fname)
