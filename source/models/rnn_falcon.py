import tensorflow as tf
import numpy as np
from .falcon_model import Model_Falcon


class Model_RNN_falcon(Model_Falcon):
    """RNN Falcon model"""

    def __init__(
        self,
        build_new_model,
        load_path,
        rnn_cell_type,
        n_units,
        n_hidden_layers,
        input_seq_length,
        input_channels,
        dropout_rate,
        epochs,
        lambda_l2_loss,
        lambda_ent_loss,
        annealing_max_ent_loss,
        lambda_advcalib_loss,
        annealing_max_advcalib_loss,
        n_classes,
        bins_calibration,
        **params):
        """
        Args:
            build_new_model: bool, whether to build or load a model
            load_path: string, path to trained model (if build_new_model is False)
            rnn_cell_type: string, "RNN", "LSTM", "GRU"
            n_units: int, number of units per cell
            n_hidden_layers: int, number of layers
            input_seq_length: int
            input_channels: int
            dropout_rate: float, rate of droped units
            epochs: int
            lambda_l2_loss: float
            lambda_ent_loss: float, lambda for predictive entropy loss
            annealing_max_ent_loss: float, maximum annealing for
                predictive entropy loss
            lambda_advcalib_loss: float, lambda for adversarial calibration loss
            annealing_max_advcalib_loss: float, max annealing for
                adversarial calibration loss
            n_classes: int, number of classes of softmax output
            bins_calibration: list of floats, limits of calibration bins
        """

        self.rnn_cell_type = rnn_cell_type
        self.n_units = n_units
        self.n_hidden_layers = n_hidden_layers
        self.input_seq_length = input_seq_length
        self.input_channels = input_channels
        self.dropout_rate = dropout_rate
        self.n_classes = n_classes

        super().__init__(
            build_new_model,
            load_path,
            epochs,
            lambda_l2_loss,
            lambda_ent_loss,
            annealing_max_ent_loss,
            lambda_advcalib_loss,
            annealing_max_advcalib_loss,
            n_classes,
            bins_calibration
        )


    def build_main_model(self):
        """build tf model"""

        print("building model...")
        if self.rnn_cell_type == 'RNN':
            func_RNN = tf.keras.layers.SimpleRNN
        elif self.rnn_cell_type == 'LSTM':
            func_RNN = tf.keras.layers.LSTM
        elif self.rnn_cell_type == 'GRU':
            func_RNN = tf.keras.layers.GRU
        else:
            raise ValueError('rnn_cell_type needs to be either RNN, LSTM or GRU')

        input_to_model = tf.keras.Input(shape=(self.input_seq_length,self.input_channels), name="input")
        if self.n_hidden_layers > 1:
            x = input_to_model
            for i in range(self.n_hidden_layers - 1):
                x = func_RNN(
                    self.n_units,
                    dropout=self.dropout_rate,
                    return_sequences=True,
                    kernel_regularizer=tf.keras.regularizers.l2(l=self.lambda_l2_loss),
                    name="RNN_layer_" + str(i),
                )(x)
            x = func_RNN(
                self.n_units,
                dropout=self.dropout_rate,
                kernel_regularizer=tf.keras.regularizers.l2(l=self.lambda_l2_loss),
                name="RNN_layer_" + str(self.n_hidden_layers-1),
            )(x)
        elif self.n_hidden_layers == 1:
            x = func_RNN(
                self.n_units,
                dropout=self.dropout_rate,
                kernel_regularizer=tf.keras.regularizers.l2(l=self.lambda_l2_loss),
                name="RNN_layer_0"
            )(input_to_model)
        else:
            raise ValueError('n_hidden_layer must be > 0')

        model_logits = tf.keras.layers.Dense(
            self.n_classes,
            kernel_regularizer=tf.keras.regularizers.l2(l=self.lambda_l2_loss),
            name="logits",
        )(x)
        model_preds = tf.keras.layers.Softmax(name="softmax")(model_logits)
        # build model
        self.model = tf.keras.Model(input_to_model, model_preds, name="rnn")

    def build_sub_models(self):
        """build tf models based on main model that returns the logits"""
        self.model_logits = tf.keras.Model(
            self.model.input, self.model.get_layer("logits").output, name="rnn_logits"
        )
