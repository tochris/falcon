"""Functions for logging during training models"""

import numpy as np
import tensorflow as tf

class Log_extend_array():

    def __init__(self, name='extend', dtype=None):
        self.bool_array = False

    def add(self, new_array):
        if tf.is_tensor(new_array):
            new_array = new_array.numpy()
        if self.bool_array == True:
            self.array = np.vstack((self.array, new_array))
        else:
            self.array = new_array
            self.bool_array = True

    def result(self):
        return self.array

    def reset(self):
        del(self.array)
        self.bool_array = False


class Log_extend_mean_array():

    def __init__(self, name='extend_mean', dtype=None):
        self.bool_array = False

    def add(self, new_array):
        if tf.is_tensor(new_array):
            new_array = new_array.numpy()
        if self.bool_array == True:
            self.array = np.vstack((self.array, new_array))
        else:
            self.array = new_array
            self.bool_array = True

    def result(self):
        res = np.mean(self.array, axis=0)
        return res

    def reset(self):
        del(self.array)
        self.bool_array = False
