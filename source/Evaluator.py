import os
import numpy as np
from .utilsevaluation.evaluator import Metrics_Evaluator


class Evaluator:
    """evaluate model and calculate metrics"""
    def __init__(
        self,
        model,
        perturb_generator,
        data_name="Data",
        folder_path="results/Evaluation/"
    ):
        """
        Args:
            model: object of Model class
            perturb_generator: object of PerturbationGenerator
            data_name: string
            folder_path: string, determines path where Evaluator is stored
        """
        self.evaluator = Metrics_Evaluator()
        self.model = model
        self.perturb_generator = perturb_generator
        self.folder_path = os.path.join(
            folder_path,
            data_name,
            os.path.split(os.path.split(self.model.save_path)[0])[1]
        )

    def reset(self):
        """Delete old evaluator object (and its storage) and replaces it
        with a newly initialized one. Delete also logits in model.
        Careful: No save or backup is happening, so be sure nothing is lost.
        """
        self.evaluator = Metrics_Evaluator()
        self.model.logits_test = {}
        self.model.labels_test = {}

    def evaluate(
        self,
        data,
        test_metric,
        perturb_type,
        epsilon,
        from_cache=True,
        to_cache=True,
        save_to_file=False,
        bins_calibration = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        **params
        ):
        """evaluate metrics
        Args:
            data: tf.data.Dataset object of (x_data, y_labels), prepared with
            batch_size, shuffle etc.
            test_metric: metric from utilsevaluation.measures
            perturb_type: string, in case dataset should be perturbed
            epsilon: int, level of perturbation
            from_cache: if False the logits and labels are calculated via the Tensorflow
                graph, if True the logits are restored from cache
            to_cache: bool, if True the logits and labels are stored to cache
            save_to_file: bool, if True the logits and labels are saved to file
            bins_calibration: list of floats, limits of calibration bins
        """
        logits, labels = self.model.logits(
            data,
            self.perturb_generator,
            perturb_type,
            epsilon,
            from_cache = from_cache,
            to_cache = to_cache,
            save_to_file = save_to_file
        )
        test_values = test_metric(
            logits,
            labels,
            bins_calibration=bins_calibration,
        )
        # add metric to evaluator storage
        perturb_type = perturb_type.replace('/','_')
        self.evaluator.add(
            test_metric.__name__,
            perturb_type,
            epsilon,
            test_values
        )

    def save(self, folder_path = None):
        """Method to save evaluation storage"""
        if folder_path is None:
            folder_path = self.folder_path
        self.evaluator.save(folder_path)

    def load(self, folder_path = None):
        """Method to load evaluation storage"""
        if folder_path is None:
            folder_path = self.folder_path
        self.evaluator.load(folder_path)
