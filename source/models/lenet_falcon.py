import tensorflow as tf
import numpy as np
from .falcon_model import Model_Falcon


class Model_LeNet_falcon(Model_Falcon):
    """LeNet Falcon model"""

    def __init__(
        self,
        build_new_model,
        load_path,
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
        input_to_model = tf.keras.Input(shape=(28, 28, 1), name="input")
        x = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(5, 5),
            padding="same",
            activation="relu",
            input_shape=(28, 28, 1),
            name="conv_1",
        )(input_to_model)
        x = tf.keras.layers.MaxPool2D(strides=2, name="pool_1")(x)
        x = tf.keras.layers.Conv2D(
            filters=48,
            kernel_size=(5, 5),
            padding="valid",
            activation="relu",
            name="conv_2",
        )(x)
        x = tf.keras.layers.MaxPool2D(strides=2, name="pool_2")(x)
        x = tf.keras.layers.Flatten(name="flatten")(x)
        x = tf.keras.layers.Dense(
            256,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(l=self.lambda_l2_loss),
            name="dense_1",
        )(x)
        x = tf.keras.layers.Dropout(self.dropout_rate, name="dropout_1")(x)
        x = tf.keras.layers.Dense(
            84,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(l=self.lambda_l2_loss),
            name="dense_2",
        )(x)
        x = tf.keras.layers.Dropout(self.dropout_rate, name="dropout_2")(x)
        model_logits = tf.keras.layers.Dense(
            self.n_classes,
            kernel_regularizer=tf.keras.regularizers.l2(l=self.lambda_l2_loss),
            name="logits",
        )(x)
        model_preds = tf.keras.layers.Softmax(name="softmax")(model_logits)
        # build model
        self.model = tf.keras.Model(input_to_model, model_preds, name="lenet")

    def build_sub_models(self):
        """build tf models based on main model that returns the logits"""
        self.model_logits = tf.keras.Model(
            self.model.input, self.model.get_layer("logits").output, name="lenet_logits"
        )
