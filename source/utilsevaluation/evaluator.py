import os
import numpy as np
import pandas as pd
from ..utils.helpers import load_obj, save_obj

class Metrics_Evaluator:
    """Storing evaluation metrics. Metrics are stored in a pandas dataframe."""
    def __init__(self):
        self.storage = pd.DataFrame(
            columns = ["Metric", "Index", "Perturbation", "Epsilon", "Value"]
        )

    def add(self, metric, perturbation, epsilon, values):
        """Appends new rows to storage data frame.
        Args:
            metric: string, metric such as "ECE"
            perturbation: string
            epsilon: index, level of perturbation
            values: list of floats, an additional index is stored to
                specify the position of each value in the list
        """
        n = len(values)
        to_append = pd.DataFrame({
            "Metric" : np.repeat(metric, n),
            "Index" : range(n),
            "Perturbation" : np.repeat(perturbation, n),
            "Epsilon" : np.repeat(epsilon, n),
            "Value" : values
        })
        self.storage = pd.concat(
            [self.storage, to_append],
            ignore_index = True
        )

    def save(self, path = "./", file = "evaluator_storage.pkl", with_csv=False):
        """Save evaluation storage object.
        Args:
            path: string, directory path of where the target is located
            file: string, target file name
            with_csv: bool, whether to save the storage additionaly as a csv file
        """
        if not os.path.exists(path):
            os.makedirs(path)
        if with_csv:
            # safe each loss type in a csv file for additional readability
            for lt in set(self.storage.Metric):
                df = self.storage
                df.loc[df.Metric==lt].to_csv(path + lt + ".csv")
        save_obj(self.storage, path, file)

    def load(self, path = "./", file = "evaluator_storage.pkl"):
        """Load evaluation storage object from file.
        Args:
            path: string, directory path of where the target is located
            file: string, target file name
        """
        self.storage = load_obj(path, file)
