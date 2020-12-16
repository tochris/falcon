import numpy as np
import tensorflow as tf
import math


class fashionMNIST:
    """Fashion MNIST dataset (Paper: Xiao, H.; Rasul, K.; and Vollgraf, R. 2017.
    Fashion-mnist: a novel image dataset for benchmarking machine learning
    algorithms.)"""
    def __init__(
        self,
        train_batch_size=100,
        valid_batch_size=None,
        test_batch_size=None,
        shuffle_buffer=5000,
        flatten=False,
        **kwargs
    ):
        """
        Args:
            train_batch_size: int, batch size for training
            valid_batch_size: int, batch size for validation
            test_batch_size: int, batch size for testing
            shuffle_buffer: int, size of shuffle buffer
            flatten: bool, if True the dataset is flattend (28*28=784)
        """
        self.data_name = "fashionMNIST"
        self.flatten = flatten

        self.shuffle_buffer = shuffle_buffer
        self.train_batch_size = train_batch_size
        if valid_batch_size == None:
            self.valid_batch_size = train_batch_size
        else:
            self.valid_batch_size = valid_batch_size
        if test_batch_size == None:
            self.test_batch_size = train_batch_size
        else:
            self.test_batch_size = test_batch_size

        self.load_data()
        self.prepare_generator()

    def load_data(self):
        '''load dataset and devide into training, validation and testing data'''
        def to_categorical(data):
            """converts to one-hot encoded array
            Args:
                data: categorical tf tensor or numpy array with 1D [sample size]
            """
            data = np.array(data)
            encoded = tf.keras.utils.to_categorical(data)
            return encoded

        print("loading %s data..." % (self.data_name))
        (x_train, y_train),\
        (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        y_train, y_test = y_train.astype(np.int32), y_test.astype(np.int32)
        y_train, y_test = to_categorical(y_train), to_categorical(y_test)
        x_valid = x_train[55000:]
        y_valid = y_train[55000:]
        x_train = x_train[:55000]
        y_train = y_train[:55000]

        if self.flatten == False:
            train_data = np.reshape(x_train, (np.shape(x_train)[0], 28, 28, 1))
            train_label = y_train
            valid_data = np.reshape(x_valid, (np.shape(x_valid)[0], 28, 28, 1))
            valid_label = y_valid
            test_data = np.reshape(x_test, (np.shape(x_test)[0], 28, 28, 1))
            test_label = y_test
            (
                self.n_train_samples,
                self.n_rows,
                self.n_columns,
                self.n_channels_input,
            ) = train_data.shape
            _, self.n_classes = train_label.shape
            self.n_valid_samples, _, _, _ = valid_data.shape
            self.n_test_samples, _, _, _ = test_data.shape
        else:
            #flatten dataset
            train_data = np.reshape(x_train, (np.shape(x_train)[0], 28*28, 1))
            train_label = y_train
            valid_data = np.reshape(x_valid, (np.shape(x_valid)[0], 28*28, 1))
            valid_label = y_valid
            test_data = np.reshape(x_test, (np.shape(x_test)[0], 28*28, 1))
            test_label = y_test
            (
                self.n_train_samples,
                self.time_steps,
                self.n_channels_input,
            ) = train_data.shape
            _, self.n_classes = train_label.shape
            self.n_valid_samples, _, _ = valid_data.shape
            self.n_test_samples, _, _ = test_data.shape

        self.train_dataset = tf.data.Dataset.from_tensor_slices(
            (train_data, train_label)
        )
        self.valid_dataset = tf.data.Dataset.from_tensor_slices(
            (valid_data, valid_label)
        )
        self.test_dataset = tf.data.Dataset.from_tensor_slices(
            (test_data, test_label)
        )

        self.train_steps_per_epoch = math.floor(55000/float(self.train_batch_size))
        self.valid_steps_per_epoch = math.floor(5000/float(self.valid_batch_size))
        self.test_steps_per_epoch = math.floor(10000/float(self.test_batch_size))
        print(
            "n_train_samples %d, n_valid_samples %d, n_test_samples %d"
            % (self.n_train_samples, self.n_valid_samples, self.n_test_samples)
        )
        if self.flatten == False:
            print(
                "n_rows %d, n_columns %d, n_channels_input %d, n_classes %d"
                % (
                    self.n_rows,
                    self.n_columns,
                    self.n_channels_input,
                    self.n_classes
                  )
            )
        else:
            print(
                "time_steps %d, n_channels_input %d, n_classes %d"
                % (
                    self.time_steps,
                    self.n_channels_input,
                    self.n_classes
                  )
            )
        print("*** finished loading data ***")

    def prepare_generator(self):
        """prepare data generator for training, validation and testing data"""
        self.train_ds = self.train_dataset.shuffle(self.shuffle_buffer).batch(
            self.train_batch_size
        )  # .repeat(epochs_to_repeat)
        self.valid_ds = self.valid_dataset.batch(self.valid_batch_size)
        self.test_ds = self.test_dataset.batch(self.test_batch_size)
