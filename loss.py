import tensorflow as tf
from tensorflow import keras
import random
import numpy as np
"""
this py define the loss function that compute the NL and PL
cp_label means the complementary label(random generated)
"""
class CustomLoss(keras.losses.Loss):
    def __init__(self, class_num, param=0.01, name="custom_loss"):
        super().__init__(name=name)
        self.class_num = class_num
        self.th = 1/self.class_num
        self.rd = random.random()
        self.param = param

    def call(self, y_true, y_pred):

        self.batch_size = tf.shape(y_true)[0]
        y_true = tf.reshape(y_true, [self.batch_size, 1])

        self.cp_label = y_true +1
        rd = tf.random.categorical([[1.0]*self.class_num-2], self.batch_size)
        self.cp_label = (self.cp_label + tf.reshape(rd, [self.batch_size,1])) % (self.class_num-1)
        self.cp_label_onehot = tf.reshape(tf.one_hot(self.cp_label, axis=1, depth=self.class_num), [self.batch_size, self.class_num])                    #shape = (batch_size, class_nums)
        self.y_true_onehot = tf.reshape(tf.one_hot(y_true, axis=1, depth=self.class_num), [self.batch_size, self.class_num])                             #shape = (batch_size, class_nums)
        self.predict_label = tf.reshape(tf.argmax(y_pred, axis=1),
                                        [self.batch_size, 1])                                                                                            #shape = (batch_size, 1)

        NL_score = self.NL(y_pred, self.cp_label)
        PL_score = self.PL(y_pred, self.predict_label)
        score = NL_score + self.param * PL_score

        return score

    def NL(self, y_pred, cp_label):
        # build a index to gather the socre from the socre matrix
        index = tf.cast(tf.reshape(tf.linspace(0, self.batch_size-1,
                                               self.batch_size),
                                   [self.batch_size, 1]),
                        dtype='int64')                                                          #shape = (batch_size, 1)
        index = tf.concat((index, cp_label), axis=1)                                            # shape = (batch_size, 2)
        py = tf.gather_nd(y_pred, index)                                                        # shape = (1,batch_size)

        # calculate the NL cross_entropy between the cp_label and the predict score
        cp_label_onehot = tf.reshape(tf.one_hot(cp_label, axis=-1, depth=self.class_num),       #shape = (batch_size, class_nums)
                                     [self.batch_size, self.class_num])


        cross_entropy = cp_label_onehot * tf.math.log(1-y_pred)                                 #shape = (batch_size, class_nums)
        cross_entropy = tf.reduce_sum(cross_entropy, axis=1)                                    #shape = (batch_size, 1)

        weight = -(1 - py)                                                                      #shape = (1, batch_size)
        out = tf.matmul(tf.reshape(weight, [1, self.batch_size]),
                        tf.reshape(cross_entropy, [self.batch_size, 1]))                        #shape = (1,1)
        return tf.reshape(out, [1])                                                             #shape = (1, )

    def PL(self, y_pred, pred_label):
        # select the predict score that satisfied the th
        one = tf.ones_like(y_pred)
        zero = tf.zeros_like(y_pred)
        label = tf.where(y_pred < self.th, x=zero, y=one)                   #shape = (batch_size, class_num)
        label = tf.reduce_sum(label, axis=-1)                               #shape = (batch_size)

        one = tf.ones_like(label)
        zero = tf.zeros_like(label)
        label = tf.where(label < 2, x=one, y=zero)
        D = y_pred * tf.reshape(label, [self.batch_size, 1])                #shape = (batch_size, class_num)

        # calculate the PL
        num = self.batch_size
        index = tf.cast(tf.reshape(tf.linspace(0, num - 1,
                                               num),
                                   [num, 1]),
                        dtype='int64')                                      # shape = (n, 1)
        D_label = tf.reshape(tf.argmax(D, axis=1),
                            [num, 1])                                       # shape = (n, 1)
        index = tf.concat((index, D_label), axis=1)                         # shape = (n, 2)
        py = tf.gather_nd(D, index)                                         # shape = (1, n)
        py = 1 - tf.math.square(py)                                         # shape = (1, n)
        py = tf.reshape(py, [1,num])                                        # shape = (n)
        weight = tf.reduce_prod(py)

        one_hot = self.y_true_onehot * tf.reshape(label, [self.batch_size, 1])
        cross_entropy = one_hot * tf.math.log(y_pred)            # shape = (batch_size, class_nums)
        cross_entropy = tf.reduce_sum(cross_entropy, axis=1)                # shape = (batch_size, 1)

        out = -weight * cross_entropy
        out = tf.reduce_sum(out, axis=0)

        return out
