import os
from abc import ABC

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, BatchNormalization, Convolution2D, LayerNormalization, ReLU, MaxPooling2D, \
    Flatten, Dropout, Dense, UpSampling2D, Conv2D, Concatenate
from tqdm import tqdm
import ast

np.random.seed(0)
tf.random.set_seed(0)


class Conv(tf.keras.Model):
    def __init__(self, n_out, stride=1):
        super(Conv, self).__init__()

        self.cblock = tf.keras.Sequential()

        self.cblock.add(Conv2D(n_out, kernel_size=3, strides=(stride, stride), use_bias=False, padding="same"))
        self.cblock.add(BatchNormalization())  # Check the axis
        self.cblock.add(ReLU())

    @tf.function
    def call(self, inputs, training=None, mask=None):
        predicted = self.cblock(inputs, training)
        return predicted


class UNet(tf.keras.Model):

    def __init__(self, input_shape=(224, 224), lr=0.001):

        super(UNet, self).__init__()

        self._input_shape = input_shape
        self._lr = lr

        fwd_blocks = [64, 128, 256, 256, 256]
        num_fwd = [2, 3, 3, 3, 3]
        back_blocks = [64, 128, 256, 256]
        num_back = [2, 3, 3, 3]

        h_in, w_in = self._input_shape
        h, w = [], []
        for i in range(len(num_fwd)):
            h.append(h_in)
            w.append(w_in)
            h_in = int(np.ceil(float(h_in) / 2))
            w_in = int(np.ceil(float(w_in) / 2))

        # Forward
        self.fwd = []
        n_in = 3

        for i in range(len(fwd_blocks)):
            c_group = tf.keras.Sequential()
            for j in range(num_fwd[i]):
                stride = 2 if i > 0 and j == 0 else 1
                c_group.add(Conv(n_out=fwd_blocks[i], stride=stride))
            self.fwd.append(c_group)
        # Backward
        stride = 1

        self.upsample = []
        self.back = []
        for i in range(len(back_blocks) - 1, -1, -1):
            self.upsample = [UpSampling2D(size=(2, 2), interpolation='bilinear')] + self.upsample
            c_group = tf.keras.Sequential()
            for j in range(num_back[i]):
                c_group.add(Conv(n_out=back_blocks[i], stride=stride))
            self.back = [c_group] + self.back

        self.final = Conv2D(3, kernel_size=1, use_bias=False, padding="same")

        # Refine
        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(lr=self._lr)
        # Loss
        # self.loss = tf.keras.losses.BinaryCrossentropy()
        self.loss = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        # Metrics
        self.train_accuracy = tf.keras.metrics.BinaryAccuracy()
        self.test_accuracy = tf.keras.metrics.BinaryAccuracy()

        # Tracker
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @tf.function
    def call(self, inputs, training=None, mask=None):
        image = inputs  # x is the image
        out = image
        outputs = []
        # print('Forward')
        for i in range(len(self.fwd)):
            out = self.fwd[i](out)
            # print(out.shape)
            if i != len(self.fwd) - 1:
                outputs.append(out)

        # print('Backward')
        for i in range(len(self.back) - 1, -1, -1):
            out = self.upsample[i](out)
            out = Concatenate(axis=3)([out, outputs[i]])
            out = self.back[i](out)
            # print(out.shape)

        # print('Final')
        out = self.final(out)
        # print(out.shape)
        image += out
        return image


if __name__ == '__main__':
    net = UNet(input_shape=(224, 224))
    net.build(input_shape=(1, 224, 224, 3))
    initializer = tf.random_uniform_initializer()
    # image = tf.Variable(initial_value=initializer(shape=(1, 224, 224, 3), dtype=tf.float32), trainable=False)
    # a = net.call(image)
    print(net.summary())
