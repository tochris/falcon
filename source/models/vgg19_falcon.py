import tensorflow as tf
import numpy as np
from .falcon_model import Model_Falcon


class Model_VGG19_falcon(Model_Falcon):
    """VGG19 Falcon model"""

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
        input_to_model = tf.keras.Input(shape=(32, 32, 3), name="input")
        x = tf.keras.layers.Conv2D(
            64,
            kernel_size=3,
            padding="SAME",
            activation=tf.nn.relu,
            input_shape=(32, 32, 3),
        )(input_to_model)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(
            64, kernel_size=3, padding="SAME", activation=tf.nn.relu
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(
            pool_size=[2, 2], strides=[2, 2], padding="SAME"
        )(x)

        x = tf.keras.layers.Conv2D(
            128, kernel_size=3, padding="SAME", activation=tf.nn.relu
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(
            128, kernel_size=3, padding="SAME", activation=tf.nn.relu
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(
            pool_size=[2, 2], strides=[2, 2], padding="SAME"
        )(x)

        x = tf.keras.layers.Conv2D(
            256, kernel_size=3, padding="SAME", activation=tf.nn.relu
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(
            256, kernel_size=3, padding="SAME", activation=tf.nn.relu
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(
            256, kernel_size=3, padding="SAME", activation=tf.nn.relu
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(
            256, kernel_size=3, padding="SAME", activation=tf.nn.relu
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(
            pool_size=[2, 2], strides=[2, 2], padding="SAME"
        )(x)

        x = tf.keras.layers.Conv2D(
            512, kernel_size=3, padding="SAME", activation=tf.nn.relu
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(
            512, kernel_size=3, padding="SAME", activation=tf.nn.relu
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(
            512, kernel_size=3, padding="SAME", activation=tf.nn.relu
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(
            512, kernel_size=3, padding="SAME", activation=tf.nn.relu
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(
            pool_size=[2, 2], strides=[2, 2], padding="SAME"
        )(x)

        x = tf.keras.layers.Conv2D(
            512, kernel_size=3, padding="SAME", activation=tf.nn.relu
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(
            512, kernel_size=3, padding="SAME", activation=tf.nn.relu
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(
            512, kernel_size=3, padding="SAME", activation=tf.nn.relu
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(
            512, kernel_size=3, padding="SAME", activation=tf.nn.relu
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(
            pool_size=[2, 2], strides=[2, 2], padding="SAME"
        )(x)

        x = tf.keras.layers.Flatten(name="flatten")(x)
        x = tf.keras.layers.Dense(1024, activation=tf.nn.relu)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)

        x = tf.keras.layers.Dense(1024, activation=tf.nn.relu)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)

        x = tf.keras.layers.Dense(self.n_classes, name="logits")(x)
        x = tf.keras.layers.Softmax(name="softmax")(x)

        self.model = tf.keras.Model(input_to_model, x, name="vgg19")

    def build_sub_models(self):
        """build tf models based on main model that returns the logits"""
        self.model_logits = tf.keras.Model(
            self.model.input, self.model.get_layer("logits").output, name="vgg19_logits"
        )
