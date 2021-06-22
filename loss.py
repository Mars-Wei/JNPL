import tensorflow as tf
from tensorflow import keras
import random
"""
this py define the loss function that compute the NL and PL
cp_label means the complementary label(random generated)
"""
class CustomLoss(keras.losses.Loss):
    def __init__(self, class_num, name="custom_loss"):
        super().__init__(name=name)
        self.class_num = class_num
        self.th = 1/self.class_num
        self.rd = random.random()

    def call(self, y_true, y_pred):
        self.batch_size = y_true.shape[0]
        self.cp_label = tf.reshape(tf.random.categorical([[1.0]*self.class_num], self.batch_size), [self.batch_size,1])                                  #shape = (batch_size,1)
        self.cp_label_onehot = tf.reshape(tf.one_hot(self.cp_label, axis=1, depth=self.class_num), [self.batch_size, self.class_num])                    #shape = (batch_size, class_nums)
        self.y_true_onehot = tf.reshape(tf.one_hot(y_true, axis=1, depth=self.class_num), [self.batch_size, self.class_num])                        #shape = (batch_size, class_nums)
        self.predict_label = tf.reshape(tf.argmax(y_pred, axis=1),
                                        [self.batch_size, 1])                                                                                            #shape = (batch_size, 1)

        NL_score = self.NL(y_pred, self.cp_label)
        PL_score = self.PL(y_pred, self.predict_label)
        score = NL_score + PL_score

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
        D = None
        p = None
        for i in range(y_pred.shape[0]):
            flag = True
            for j in range(y_pred.shape[1]):
                if j == pred_label[i]:
                    p = y_pred[i][j]
                    continue
                if y_pred[i][j] > self.th:
                    flag = False
                    break
            if flag and self.rd < p:
                if D is None:
                    D = tf.reshape(y_pred[i], [1, self.class_num])
                else:
                    D = tf.concat((D, tf.reshape(y_pred[i], [1, self.class_num])),
                                  axis=0)

        # calculate the PL                                                  D's shape = (n, class_num)
        if D is None:
            return 0
        index = tf.cast(tf.reshape(tf.linspace(0, D.shape[0] - 1,
                                               D.shape[0]),
                                   [D.shape[0], 1]),
                        dtype='int64')                                      # shape = (n, 1)
        D_label = tf.reshape(tf.argmax(D, axis=1),
                            [D.shape[0], 1])                                # shape = (n, 1)
        index = tf.concat((index, D_label), axis=1)                         # shape = (n, 2)
        py = tf.gather_nd(D, index)                                         # shape = (1, n)
        py = 1 - tf.math.square(py)                                         # shape = (1, n)
        py = tf.reshape(py,[D.shape[0]])                                    # shape = (n)
        weight = py[0]
        for i in range(1, py.shape[0]):
            weight = weight * py[i]                                         # shape = (1)

        cross_entropy = self.y_true_onehot * tf.math.log(y_pred)            # shape = (batch_size, class_nums)
        cross_entropy = tf.reduce_sum(cross_entropy, axis=1)                # shape = (batch_size, 1)

        out = -weight * cross_entropy
        out = tf.reduce_sum(out, axis=0)

        return out








