import numpy as np
import math
import tensorflow as tf
import keras
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.datasets import fetch_20newsgroups


class Newsgroups20:
    """Newsgroups20 classification dataset with glove embeddings"""
    def __init__(
        self,
        max_seq_length=1000,
        length_vocab=20000,
        embedding_layer_size=100,
        path_glove_embeddings=None,
        train_batch_size=100,
        valid_batch_size=None,
        test_batch_size=None,
        shuffle_buffer=5000,
        data_path="./",
        **kwargs
    ):
        """
        Args:
            max_seq_length: int, length in characters
            length_vocab: int, vocabulary size
            embedding_layer_size: int
            path_glove_embeddings: string, path to glove embeddings file
            train_batch_size: int, batch size for training
            valid_batch_size: int, batch size for validation
            test_batch_size: int, batch size for testing
            shuffle_buffer: int, size of shuffle buffer
            data_path: string, path which points to dataset
        """
        self.data_name = "Newsgroups20"
        self.data_path = data_path
        self.max_seq_length = max_seq_length
        self.length_vocab = length_vocab
        self.embedding_layer_size = embedding_layer_size
        self.path_glove_embeddings = path_glove_embeddings

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


    def init_embedding_matrix(self):
        """initialize glove embeddings"""
        embeddings_index = {}
        with open(self.path_glove_embeddings) as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, 'f', sep=' ')
                embeddings_index[word] = coefs

        embedding_matrix = np.zeros((self.length_vocab, self.embedding_layer_size))
        for word, i in self.word_index.items():
            if i<self.length_vocab:
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    # words not found in embedding index will be all-zeros.
                    embedding_matrix[i] = embedding_vector
        self.init_word_embedding_matrix = embedding_matrix


    def get_word_level_input(self, train_texts, test_texts):
        """tokenize train and test sequences
        Args:
            train_texts: list of strings, text for training
            test_texts: list of strings, test for testing
        Returns:
            tokenized train and test arrays
        """
        tokenizer = Tokenizer(num_words=self.length_vocab)
        texts = []
        texts.extend(train_texts)
        texts.extend(test_texts)
        tokenizer.fit_on_texts(texts)

        train_sequences = tokenizer.texts_to_sequences(train_texts)
        train_data = pad_sequences(
            train_sequences,
            maxlen=self.max_seq_length,
            padding='post'
        )
        test_sequences = tokenizer.texts_to_sequences(test_texts)
        test_data = pad_sequences(
            test_sequences,
            maxlen=self.max_seq_length,
            padding='post'
        )

        self.word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(self.word_index))
        return train_data, test_data


    def load_data(self):
        """load dataset and devide into training, validation and testing data"""
        def to_categorical(data):
            """converts to one-hot encoded array
            Args:
                data: categorical tf tensor or numpy array with 1D [sample size]
            """
            data = np.array(data)
            encoded = tf.keras.utils.to_categorical(data)
            return encoded

        #load data
        dataset_train = fetch_20newsgroups(
            data_home=self.data_path,
            categories=None,
            subset='train'
        )
        train_texts = dataset_train.data # Extract texts
        train_texts = [te.lower() for te in train_texts]
        train_targets = dataset_train.target # Extract targets
        dataset_test = fetch_20newsgroups(
            data_home=self.data_path,
            categories=None,
            subset='test'
        )
        test_texts = dataset_test.data # Extract texts
        test_texts = [te.lower() for te in test_texts]
        test_targets = dataset_test.target # Extract targets
        #delete too long sequences
        train_data_length = len(train_texts)
        test_data_length = len(test_texts)
        #convert data
        x_train, x_test = self.get_word_level_input(
                                            train_texts,
                                            test_texts
                                            )
        y_train, y_test = to_categorical(train_targets), to_categorical(test_targets)

        print('Max sequence length: ', self.max_seq_length)
        print('number of train data samples deleted due to maximum length limit: ',
            train_data_length-len(x_train)
        )
        print('number of test data samples deleted due to maximum length limit: ',
            test_data_length-len(x_test)
        )
        print('remaining number of train data samples: ', len(x_train))
        print('remaining number of test data samples: ', len(x_test))

        x_valid = x_train[9000:]
        y_valid = y_train[9000:]
        x_train = x_train[:9000]
        y_train = y_train[:9000]

        print('class distribution train: ', np.sum(y_train,axis=0))
        print('class distribution valid: ', np.sum(y_valid,axis=0))
        print('class distribution test: ', np.sum(y_test,axis=0))

        (
            self.n_train_samples,
            self.sequence_length,
        ) = x_train.shape
        _, self.n_classes = y_train.shape
        self.n_valid_samples, _ = x_valid.shape
        self.n_test_samples, _ = x_test.shape



        self.init_embedding_matrix() #initialize glove embeddings

        input_to_model = tf.keras.Input(
            shape=(self.sequence_length),
            name="input_embedding"
        )
        word_embeddings_layer = tf.keras.layers.Embedding(
            self.length_vocab,
            self.embedding_layer_size,
            embeddings_initializer=\
            keras.initializers.Constant(self.init_word_embedding_matrix),
            input_length=self.sequence_length,
            trainable=False, name="embedding")(input_to_model)
        self.embedding = tf.keras.Model(
            input_to_model, word_embeddings_layer, name="rnn_embedding"
        )
        x_train_embedding = self.embedding(x_train)
        x_valid_embedding = self.embedding(x_valid)
        x_test_embedding = self.embedding(x_test)

        self.train_dataset = tf.data.Dataset.from_tensor_slices(
            (x_train_embedding, y_train)
        )
        self.valid_dataset = tf.data.Dataset.from_tensor_slices(
            (x_valid_embedding, y_valid)
        )
        self.test_dataset = tf.data.Dataset.from_tensor_slices(
            (x_test_embedding, y_test)
        )

        self.valid_dataset_raw = tf.data.Dataset.from_tensor_slices(
            (x_valid, y_valid)
        )
        self.test_dataset_raw = tf.data.Dataset.from_tensor_slices(
            (x_test, y_test)
        )
        self.train_steps_per_epoch = math.floor(
            9000/float(self.train_batch_size))
        self.valid_steps_per_epoch = math.floor(
            len(x_train)-9000/float(self.valid_batch_size))
        self.test_steps_per_epoch = math.floor(
            len(x_test)/float(self.test_batch_size))

        print(
            "n_train_samples %d, n_valid_samples %d, n_test_samples %d"
            % (self.n_train_samples, self.n_valid_samples, self.n_test_samples)
        )
        print(
            "sequence_length %d, length_vocab %d, n_classes %d"
            % (self.sequence_length, self.length_vocab, self.n_classes)
        )
        print("*** finished loading data ***")

    def prepare_generator(self):
        """prepare data generator for training, validation and testing data"""
        self.train_ds = self.train_dataset.shuffle(self.shuffle_buffer).batch(
            self.train_batch_size
        )  
        self.valid_ds = self.valid_dataset.batch(self.valid_batch_size)
        self.test_ds = self.test_dataset.batch(self.test_batch_size)
        self.valid_ds_raw = self.valid_dataset_raw.batch(self.valid_batch_size)
        self.test_ds_raw = self.test_dataset_raw.batch(self.test_batch_size)
