import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from numpy.random import default_rng
import source.data.imagenet_corrupted as imagenet_corrupted
from tensorflow.python.ops import array_ops

class PerturbationGenerator:
    """Perturbation of data"""
    def __init__(self, dataset_name = 'MNIST', **kwargs):
        """
        Args:
            dataset_name: string
        """

        self.dataset_name = dataset_name
        self.image_generator = ImageDataGenerator()
        # set params
        for key, value in kwargs.items():
            self.__setattr__(key, value)

        if self.dataset_name in ['MNIST','CIFAR','CIFAR10','MNISTseq','newsgroups20','Newsgroups20']:
            self.params_perturb = {}
            # Rotation angle in degrees.
            self.params_perturb["rot_right"] = [
                0, 10, 20, 30, 40, 50, 60, 70, 80, 90
            ]
            # Rotation angle in degrees.
            self.params_perturb["rot_left"] = [
                0, 350, 340, 330, 320, 310, 300, 290, 280, 270
            ]
            # Shift in the x direction.
            self.params_perturb["xshift"] = [
                0, 2, 4, 6, 8, 10, 12, 14, 16, 18
            ]
            # Shift in the y direction.
            self.params_perturb["yshift"] = [
                0, 2, 4, 6, 8, 10, 12, 14, 16, 18
            ]
            # Shift in the xy direction.
            self.params_perturb["xyshift"] = [
                0, 2, 4, 6, 8, 10, 12, 14, 16, 18
            ]
            # Shear angle in degrees.
            self.params_perturb["shear"] = [
                0, 10, 20, 30, 40, 50, 60, 70, 80, 90
            ]
            # Zoom in the x direction.
            self.params_perturb["xzoom"] = [
                1.0, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10
            ]
            # Zoom in the y direction.
            self.params_perturb["yzoom"] = [
                1.0, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10
            ]
            # Zoom in the xy direction.
            self.params_perturb["xyzoom"] = [
                1.0, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10
            ]
            # Swap words randomly
            self.params_perturb["char_swap"] = [
                0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
            ]
            # adversarials generated with FGSM
            self.params_perturb["fgsm"] = np.array([
                   0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45
            ])

        elif self.dataset_name in [
            'Imagenet',
            'ObjectNet_not_imagenet',
            'ObjectNet_only_imagenet']:
            # adversarials generated with FGSM
            self.params_perturb = {}
            self.params_perturb["fgsm"] = np.array([
                    0.001, 0.002, 0.004, 0.008, 0.016,
                    0.032, 0.064, 0.128, 0.256, 0.512
                ])*255
        else:
            self.params_perturb = {}

        # map perturbation to keras fctn parameter
        self.perturb_2_param = {}
        self.perturb_2_param["rot_right"] = ["theta"]
        self.perturb_2_param["rot_left"] = ["theta"]
        self.perturb_2_param["shear"] = ["shear"]
        self.perturb_2_param["xzoom"] = ["zx"]
        self.perturb_2_param["yzoom"] = ["zy"]
        self.perturb_2_param["xyzoom"] = ["zx", "zy"]
        self.perturb_2_param["xshift"] = ["tx"]
        self.perturb_2_param["yshift"] = ["ty"]
        self.perturb_2_param["xyshift"] = ["tx", "ty"]

        #self.params_perturb["imagenet2012_corrupted/gaussian_noise"] = [0, 1, 2, 3, 4, 5]
        self.perturb_corruption = ["imagenet2012_corrupted/gaussian_noise",
                                   "imagenet2012_corrupted/shot_noise",
                                   "imagenet2012_corrupted/impulse_noise",
                                   "imagenet2012_corrupted/defocus_blur",
                                   "imagenet2012_corrupted/glass_blur",
                                   "imagenet2012_corrupted/motion_blur",
                                   "imagenet2012_corrupted/zoom_blur",
                                   "imagenet2012_corrupted/snow",
                                   "imagenet2012_corrupted/frost",
                                   "imagenet2012_corrupted/fog",
                                   "imagenet2012_corrupted/brightness",
                                   "imagenet2012_corrupted/contrast",
                                   "imagenet2012_corrupted/elastic_transform",
                                   "imagenet2012_corrupted/pixelate",
                                   "imagenet2012_corrupted/jpeg_compression",
                                   "imagenet2012_corrupted/gaussian_blur",
                                   "imagenet2012_corrupted/saturate",
                                   "imagenet2012_corrupted/spatter",
                                   "imagenet2012_corrupted/speckle_noise"]

    def possible_epsilons(self, perturb_type):
        """Return all possible possible epsilons"""
        if perturb_type == "None":
            return [1]
        elif perturb_type in self.perturb_corruption:
            return [0, 1, 2, 3, 4, 5]
        else:
            return self.params_perturb[perturb_type]

    def perturb_dataset(self, dataset, perturb_type, epsilon,
                        data_corrupted_path, batch_size=100):
        """returns a perturbated tf.data object
        Args:
            dataset: tf.data.Dataset object of (x_data, y_labels),
                prepared with batch_size, shuffle etc.
            perturb_type: string, includes dataset name and perturbation type
                (e.g. 'imagenet2012_corrupted/gaussian_noise')
            epsilon: int, level of perturbation
            data_corrupted_path: string
            batch_size: int
        """
        if perturb_type in self.perturb_corruption:
            if epsilon != 0:
                data = imagenet_corrupted.Imagenet_corrupted(
                    corruption_type = perturb_type,
                    epsilon = epsilon,
                    data_path = data_corrupted_path,
                    test_batch_size = batch_size
                )
                return data.test_ds
            else:
                return dataset
        return None


    def perturb_batch(self,
                      X,
                      perturb_type=None,
                      epsilon=0,
                      model=None,
                      label=None,
                      embedding=None,
                      length_vocab=None):
        """returns a perturbated tf.data object
        Args:
            X: data array
            perturb_type: string, includes dataset name and perturbation type
                (e.g. 'rot_left')
            epsilon: int, level of perturbation
            model: object of Model class
            label=None,
            embedding=None,
            length_vocab=None
        Returns:
            perturbed samples
        """
        if perturb_type is None or perturb_type == "None":
            if embedding is None:
                return X
            else:
                return embedding(X)

        elif perturb_type in list(self.perturb_2_param.keys()):
            Xnp = X.numpy()

            architecture_rnn = ['RNN_falcon'] #architectures with flat datasets
            if model.architecture in architecture_rnn:
                Xnp = np.reshape(Xnp,(np.shape(Xnp)[0],28,28,1))
            keras_paras = self.perturb_2_param[perturb_type]
            keras_param_dict = {}
            for para in keras_paras:
                keras_param_dict[para] = self.params_perturb[perturb_type][epsilon]

            X_pert = []
            for x in Xnp:
                X_pert.append(
                    tf.keras.preprocessing.image.apply_affine_transform(
                        x, **keras_param_dict
                    )[np.newaxis, ...]
                )
            X_pert = np.concatenate(X_pert)

            if model.architecture in architecture_rnn:
                X_pert = np.reshape(X_pert,(np.shape(X_pert)[0],28*28,1))

        elif perturb_type == "char_swap":
            Xnp = X.numpy()
            shape_dim1 = np.shape(Xnp)[0]
            shape_dim2 = np.shape(Xnp)[1]
            random_mask = np.random.choice(
                a=[True, False],
                size=(shape_dim1, shape_dim2),
                p=[
                    self.params_perturb["char_swap"][epsilon],
                    1-self.params_perturb["char_swap"][epsilon]
                ]
            )
            for i in range(np.shape(random_mask)[0]):
                for j in range(np.shape(random_mask)[1]):
                    if random_mask[i][j] == True:
                        Xnp[i][j] = np.random.randint(1,length_vocab,1)[0]
            X_pert = embedding(Xnp)

        elif perturb_type == "fgsm":
            assert (
                model is not None
            ), "Provide model from model factory to generate adversarials"
            assert label is not None, "Provide label to generate adversarials"
            eps = self.params_perturb[perturb_type][epsilon]
            signed_grad = self.get_signed_grad(X, label, model)
            X_pert = X + eps * signed_grad
            if self.dataset_name not in ["Imagenet", "newsgroups20"]:
                X_pert = tf.clip_by_value(X_pert, 0, 1)

        return X_pert

    def adv_batch(self, X, perturb_type=None, epsilon=0, model=None, label=None):
        """get batch of samples
        Args:
            X: data array
            perturb_type: string, includes dataset name and perturbation type
                (e.g. 'rot_left')
            epsilon: int, level of perturbation
            model: object of Model class
            label=None,
        Returns:
            adversatrial sample with desired epsilon
        """
        if perturb_type is None or perturb_type == "None":
            return X
        elif perturb_type == "fgsm":
            assert (
                model is not None
            ), "Provide model from model factory to generate adversarials"
            assert label is not None, "Provide label to generate adversarials"
            signed_grad = self.get_signed_grad(X, label, model)
            X_pert = X + epsilon * signed_grad
        return X_pert

    def get_signed_grad(self, X, label, model):
        """get signed gradient of model
        Args:
            X: data array
            label=None,
            model: object of Model class
        Returns:
            signed gardient
        """
        with tf.GradientTape() as tape:
            tape.watch(X)
            loss = model.loss_adv(X, label)
        gradient = tape.gradient(loss, X)
        signed_grad = tf.sign(gradient)
        return signed_grad

    def print_perturbations(self):
        """list all implemented pertubations"""
        print(self.params_perturb.keys())
