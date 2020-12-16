import tensorflow as tf
import numpy as np
from .falcon_model import Model_Falcon
import keras


class Model_ResNet50_falcon(Model_Falcon):
    """ResNet50 Falcon model"""

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
        raise Exception('No model building possible in this case! Load a model!')

    def build_sub_models(self):
        """build tf models based on main model that returns the logits"""
        self.model_logits = tf.keras.Model(
            self.model.input, self.model.layers[-2].output, name="lenet_logits"
        )
