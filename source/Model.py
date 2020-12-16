import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Progbar
import random
import datetime
import time

from .Perturbation_Generator import PerturbationGenerator
from .models.lenet_falcon import Model_LeNet_falcon
from .models.vgg19_falcon import Model_VGG19_falcon
from .models.resnet50_falcon import Model_ResNet50_falcon
from .models.rnn_falcon import Model_RNN_falcon
from .models.rnn_nlp_word_falcon import Model_RNN_NLP_falcon
from .utils.log_functions import Log_extend_array, Log_extend_mean_array
from .utils.plotutils import (
    save_line_chart,
    save_dict_to_structured_txt,
    save_dict_to_csv,
    save_dict_to_pkl,
    load_dict_from_pkl,
)

class Model:
    """Class for evaluating and training a model
    construct object either via .load(...) or via .build(...)
    """

    def __init__(
        self,
        architecture,
        save_path_general="./",
        save_path_with_hyperparams=True,
        random_seed=1,
        **kwargs
    ):
        """
        Args:
            architecture: string, defines the model's architecture:
                    Options: e.g. 'LeNet_falcon', 'ResNet50_falcon'
            save_path_general: string, path for saving trained models
            save_path_with_hyperparams: bool, if True: A folder with the
                architecture's name and all hyperparamters is created,
                if False: A folder with only the architecture's name is created
            random_seed: int
        """

        self.architecture = architecture
        self.save_path_general = save_path_general
        self.save_path_with_hyperparams = save_path_with_hyperparams
        self.random_seed = random_seed
        self.data_corrupted_path = None
        self.stored_in_cache = False  # whether logits and labels are stored in cache
        self.logits_test = {}
        self.labels_test = {}

        for key, value in kwargs.items():
            self.__setattr__(key, value)
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

    def build(self, build_new_model=True, load_path=None, **params):
        """build new model
        Args:
            build_new_model: bool, whether to build or load a model
            load_path: string, path to trained model (if build_new_model is False)
        """

        for key, value in params.items():
            self.__setattr__(key, value)

        if self.architecture == "LeNet_falcon":
            self.model = Model_LeNet_falcon(
                build_new_model=build_new_model,
                load_path=load_path,
                dropout_rate=self.dropout_rate,
                epochs=self.epochs,
                lambda_l2_loss=self.lambda_l2_loss,
                lambda_ent_loss=self.lambda_ent_loss,
                annealing_max_ent_loss=1.0,
                lambda_advcalib_loss=self.lambda_advcalib_loss,
                annealing_max_advcalib_loss=0.5,
                n_classes=10,
                bins_calibration=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                params=params
            )

        if self.architecture == "VGG19_falcon":
            self.model = Model_VGG19_falcon(
                build_new_model=build_new_model,
                load_path=load_path,
                dropout_rate=self.dropout_rate,
                epochs=self.epochs,
                lambda_l2_loss=self.lambda_l2_loss,
                lambda_ent_loss=self.lambda_ent_loss,
                annealing_max_ent_loss=1.0,
                lambda_advcalib_loss=self.lambda_advcalib_loss,
                annealing_max_advcalib_loss=0.5,
                n_classes=10,
                bins_calibration=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                params=params
            )

        if self.architecture == "RNN_falcon":
            self.model = Model_RNN_falcon(
                build_new_model=build_new_model,
                load_path=load_path,
                rnn_cell_type = self.rnn_cell_type,
                n_units=self.n_units,
                n_hidden_layers=self.n_hidden_layers,
                input_seq_length=784,
                input_channels=1,
                dropout_rate=self.dropout_rate,
                epochs=self.epochs,
                lambda_l2_loss=self.lambda_l2_loss,
                lambda_ent_loss=self.lambda_ent_loss,
                annealing_max_ent_loss=1.0,
                lambda_advcalib_loss=self.lambda_advcalib_loss,
                annealing_max_advcalib_loss=0.5,
                n_classes=10,
                bins_calibration=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                params=params
            )

        if self.architecture == "RNN_NLP_falcon":
            self.model = Model_RNN_NLP_falcon(
                build_new_model=build_new_model,
                load_path=load_path,
                rnn_cell_type = self.rnn_cell_type,
                n_units=self.n_units,
                n_hidden_layers=self.n_hidden_layers,
                input_seq_length=self.input_seq_length,
                dropout_rate=self.dropout_rate,
                epochs=self.epochs,
                lambda_l2_loss=self.lambda_l2_loss,
                lambda_ent_loss=self.lambda_ent_loss,
                annealing_max_ent_loss=1.0,
                lambda_advcalib_loss=self.lambda_advcalib_loss,
                annealing_max_advcalib_loss=0.5,
                embedding_layer_size=self.embedding_layer_size,
                n_classes=20,
                path_glove_embeddings=self.path_glove_embeddings,
                dataset=self.dataset,
                bins_calibration=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                params=params
            )

        if self.architecture == "ResNet50_falcon":
            self.model = Model_ResNet50_falcon(
                build_new_model=build_new_model,
                load_path=load_path,
                dropout_rate=self.dropout_rate,
                epochs=self.epochs,
                lambda_l2_loss=self.lambda_l2_loss,
                lambda_ent_loss=self.lambda_ent_loss,
                annealing_max_ent_loss=0.0001,
                lambda_advcalib_loss=self.lambda_advcalib_loss,
                annealing_max_advcalib_loss=1.0,
                n_classes=1000,
                bins_calibration=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                params=params
            )

    def set_save_dir(self, **kwargs):
        """return the model's directory name where results are stored"""
        if self.save_path_with_hyperparams == True:
            params_self = self.__dict__
            keys_for_dir = (
                {}
            )
            # attributes of the model which are part of the directory path
            # if the respective model contains them
            keys_for_dir["architecture"] = "ar"
            keys_for_dir["rnn_cell_type"] = "cell"
            keys_for_dir["epochs"] = "ep"
            keys_for_dir["learning_rate"] = "lrate"
            keys_for_dir["batch_size"] = "bs"
            keys_for_dir["dropout_rate"] = "dout"
            keys_for_dir["lambda_l2_loss"] = "l2"
            keys_for_dir["n_units"] = "un"
            keys_for_dir["n_hidden_layers"] = "hl"
            keys_for_dir["random_seed"] = "rs"
            params_in_dir = {
                keys_for_dir[key_for_dir]: params_self[key_for_dir]
                for key_for_dir in keys_for_dir.keys()
                if key_for_dir in params_self.keys()
            }
            params_list = [
                "_" + str(key) + str(value)
                for key, value in params_in_dir.items()
                if key is not "ar"
            ]
            params_list.insert(0, params_in_dir["ar"])
            params_list.insert(0, self.save_path_general)
            self.save_path = "".join(params_list) + "/"
            print("Model will be saved to the following directory: ", self.save_path)
            print()
        else:
            self.save_path = str(str(self.save_path_general) + str(architecture) + "/")

    def train_step(self, x_data, labels, log_dict, train_step):
        def tf_train_step():
            with tf.GradientTape() as tape:
                preds = self.model.prediction_train(x_data)
                loss = self.model.loss(labels, preds, train_step)
            gradients = tape.gradient(loss, self.model.trainable_variables())
            self.model.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables())
            )
            log_dict["loss"](loss)
            log_dict["accuracy"](labels, preds)

            preds = np.clip(preds, 1.17e-37, 3.40e37)
            entropy = np.mean([-np.sum(np.multiply(pred, np.log(pred))) for pred in preds])
            log_dict["entropy"](entropy)
        tf_train_step()
        return log_dict

    def train_step_logits(self, x_data, labels, log_dict, train_step):
        def tf_train_step():
            with tf.GradientTape() as tape:
                preds = self.model.prediction_train(x_data)
                logits = self.model.logits_train(x_data)
                loss = self.model.loss(labels, logits, train_step)
            gradients = tape.gradient(loss, self.model.trainable_variables())
            self.model.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables())
            )
            log_dict["loss"](loss)
            log_dict["accuracy"](labels, preds)
        tf_train_step()
        return log_dict

    def train_step_ensembles(self, x_data, labels, log_dict, train_step, perturbation_generator=None, epsilon=0):
        def tf_train_step():
            loss_sub_model_list = []
            for i_model in range(self.nb_ensembles):
                if epsilon != 0:
                     x_data_i = perturbation_generator.adv_batch(
                         x_data,
                         perturb_type=self.train_perturbation_mode,
                         epsilon=epsilon,
                         model=self,
                         sub_model_num=i_model,
                         label=labels,
                     )
                else:
                     x_data_i = x_data
                with tf.GradientTape() as tape:
                    preds = self.model.prediction_train(x_data_i, model_num=i_model)
                    loss = self.model.loss(labels, preds, train_step, model_num=i_model)
                gradients = tape.gradient(
                    loss, self.model.trainable_variables(model_num=i_model)
                )
                self.model.optimizer.apply_gradients(
                    zip(gradients, self.model.trainable_variables(model_num=i_model))
                )
                loss_sub_model_list.append(tf.reduce_mean(loss))
                log_dict["loss"](loss)
                log_dict["accuracy"](labels, preds)
            log_dict["loss_sub_models"].add(loss_sub_model_list)
        tf_train_step()
        return log_dict

    def train_step_advcalib(self, x_data, labels, log_dict, train_step):
        def tf_train_step_advcalib():
            with tf.GradientTape() as tape:
                preds = self.model.prediction_train(x_data)
                logits = self.model.logits_train(x_data)
                loss = self.model.loss_advcalib(labels, logits, train_step)
            gradients = tape.gradient(loss, self.model.trainable_variables())
            self.model.optimizer_advcalib.apply_gradients(
                zip(gradients, self.model.trainable_variables())
            )
        tf_train_step_advcalib()

    def test_step(self, x_data, labels, log_dict, n_test_iter_now=None):
        def tf_test_step():
            preds = self.model.prediction_test(x_data, n_test_iter_now=n_test_iter_now)
            loss = self.model.loss(labels, preds, train_step=None)
            log_dict["loss"](loss)
            log_dict["accuracy"](labels, preds)
        tf_test_step()
        return log_dict

    def test_step_logits(self, x_data, labels, log_dict, n_test_iter_now=None):
        def tf_test_step_logits():
            preds = self.model.prediction_test(x_data, n_test_iter_now=n_test_iter_now)
            loss = self.model.loss(labels, preds, train_step=None)
            logits = self.model.logits_test(x_data, n_test_iter_now=n_test_iter_now)
            log_dict["loss"](loss)
            log_dict["accuracy"](labels, preds)
            log_dict["logits"].add(logits)
            log_dict["labels"].add(labels)
        tf_test_step_logits()
        return log_dict

    def test_step_probs(self, x_data, labels, log_dict, n_test_iter_now=None):
        def tf_test_step_probs():
            preds = self.model.prediction_test(x_data, n_test_iter_now=n_test_iter_now)
            loss = self.model.loss(labels, preds, train_step=None)
            logits = self.model.prediction_test(x_data, n_test_iter_now=n_test_iter_now)
            log_dict["loss"](loss)
            log_dict["accuracy"](labels, preds)
            log_dict["logits"].add(logits)
            log_dict["labels"].add(labels)
        tf_test_step_probs()
        return log_dict

    def train(
        self,
        train_dataset,
        valid_dataset,
        perturbation_generator,
        epochs,
        batch_size,
        learning_rate,
        probability_train_perturbation,
        train_perturbation_mode='fgsm',
        **kwargs
    ):
        """
        all training happens here
        Args:
            train_dataset: tf.data.Dataset object of (x_data, y_labels), prepared with
            batch_size, shuffle etc.
            valid_dataset: tf.data.Dataset object of (x_data, y_labels), prepared with
            batch_size etc.
            perturbation_generator: object from class PerturbationGenerator
            epochs: int
            batch_size: int
            learning_rate: float
            probability_train_perturbation: float, proportion of training with
                adversarial calibration loss
            train_perturbation_mode: string, 'fgsm' for Falcon model
        """

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.probability_train_perturbation = probability_train_perturbation
        self.train_perturbation_mode = train_perturbation_mode

        self.set_save_dir()
        self.log_dir_train = self.save_path + "train/"
        self.log_dir_test = self.save_path + "test/"
        self.model.build_optimizers(learning_rate)

        # create log metrics
        log_dict_train = {}
        log_dict_train["loss"] = tf.keras.metrics.Mean(name="train_loss")
        log_dict_train["accuracy"] = tf.keras.metrics.CategoricalAccuracy(
            name="train_accuracy"
        )
        log_dict_train["entropy"] = tf.keras.metrics.Mean(name="train_entropy")
        log_dict_train["loss_sub_models"] = Log_extend_mean_array(name="loss_sub_models")
        log_dict_valid = {}
        log_dict_valid["loss"] = tf.keras.metrics.Mean(name="valid_test_loss")
        log_dict_valid["accuracy"] = tf.keras.metrics.CategoricalAccuracy(
            name="valid_test_accuracy"
        )

        # set up summary writeres for Tensorboard
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tb_train_log_dir = self.log_dir_train + current_time + "/train"
        tb_valid_log_dir = self.log_dir_train + current_time + "/valid"
        train_summary_writer = tf.summary.create_file_writer(tb_train_log_dir)
        with train_summary_writer.as_default():
            tf.summary.trace_on(graph=True, profiler=True)

        # create log history over training
        log_history_dict_train = {}
        log_history_dict_train["loss"] = []
        log_history_dict_train["accuracy"] = []
        log_history_dict_train["entropy"] = []
        log_history_dict_train["loss_sub_models"] = []
        log_history_dict_valid = {}
        log_history_dict_valid["loss"] = []
        log_history_dict_valid["accuracy"] = []

        ##Training##
        global_step = 0
        for epoch in range(self.epochs):
            global_step += 1
            tf.summary.trace_off()

            num_training_samples=self.dataset.train_steps_per_epoch
            num_valid_samples=self.dataset.valid_steps_per_epoch
            pb_i = Progbar(num_training_samples, stateful_metrics=['acc','loss','ent'])
            pb_t = Progbar(num_valid_samples, stateful_metrics=['acc','loss'])
            for x_data, y_data in train_dataset:
                if np.random.uniform() < probability_train_perturbation:
                    log_dict_train = self.train_step(
                        x_data, y_data, log_dict_train, global_step
                    )
                    if perturbation_generator is not None:
                        n_eps = len(perturbation_generator.possible_epsilons(
                            perturb_type=train_perturbation_mode
                        ))
                        epsilon = random.randint(0,n_eps-1)
                        x_data = perturbation_generator.perturb_batch(
                            x_data,
                            perturb_type=train_perturbation_mode,
                            epsilon=epsilon,
                            model=self,
                            label=y_data,
                        )
                    self.train_step_advcalib(x_data, y_data, log_dict_train, global_step)
                else:
                    log_dict_train = self.train_step(
                        x_data, y_data, log_dict_train, global_step
                    )
                #progressbar
                pb_i.add(1, values=[
                            ('acc',log_dict_train["accuracy"].result()),
                            ('loss',log_dict_train["loss"].result()),
                            ('ent',log_dict_train["entropy"].result())
                        ]
                    )
            # test on validation set
            if valid_dataset != None:
                for x_data, y_data in valid_dataset:
                    log_dict_valid = self.test_step(
                       x_data, y_data, log_dict_valid, n_test_iter_now=1
                    )
            tf.summary.trace_on()
            print(
                "Epoch "
                + str(epoch)
                + ", Minibatch Loss= "
                + "{:.4f}".format(log_dict_train["loss"].result())
                + ", Training Accuracy= "
                + "{:.3f}".format(log_dict_train["accuracy"].result())
                + ", Valid Accuracy= "
                + "{:.3f}".format(log_dict_valid["accuracy"].result())
            )

            # save to summary for Tensorboard
            with train_summary_writer.as_default():
                tf.summary.scalar("loss_train", log_dict_train["loss"].result(), step=epoch)
                tf.summary.scalar(
                    "accuracy_train", log_dict_train["accuracy"].result(), step=epoch
                )
                tf.summary.scalar(
                    "entropy_train", log_dict_train["entropy"].result(), step=epoch
                )
                tf.summary.scalar("loss_valid", log_dict_valid["loss"].result(), step=epoch)
                tf.summary.scalar(
                    "accuracy_valid", log_dict_valid["accuracy"].result(), step=epoch
                )

            # add certain metrics to history per epoch
            log_history_dict_train["loss"].append(
                log_dict_train["loss"].result().numpy()
            )
            log_history_dict_train["accuracy"].append(
                log_dict_train["accuracy"].result().numpy()
            )
            log_history_dict_train["entropy"].append(
                log_dict_train["entropy"].result().numpy()
            )
            log_history_dict_valid["loss"].append(
                log_dict_valid["loss"].result().numpy()
            )
            log_history_dict_valid["accuracy"].append(
                log_dict_valid["accuracy"].result().numpy()
            )

            # reset log metrics
            log_dict_train["loss"].reset_states()
            log_dict_train["accuracy"].reset_states()
            log_dict_train["entropy"].reset_states()
            log_dict_valid["loss"].reset_states()
            log_dict_valid["accuracy"].reset_states()

        # plot accuracy and loss lists (log_histories)
        save_line_chart(
            self.log_dir_train,
            "accuracy_train_history",
            log_history_dict_train["accuracy"],
            diag_title="Train Accuracy",
            xlabel="epoch",
            ylabel="accuracy",
            pyplotBool=True,
            write_to_txtfile_Bool=True,
        )
        save_line_chart(
            self.log_dir_train,
            "loss_train_history",
            log_history_dict_train["loss"],
            diag_title="Train Loss",
            xlabel="epoch",
            ylabel="loss",
            pyplotBool=True,
            write_to_txtfile_Bool=True,
        )
        save_line_chart(
            self.log_dir_train,
            "entropy_train_history",
            log_history_dict_train["entropy"],
            diag_title="Train Entropy",
            xlabel="epoch",
            ylabel="loss",
            pyplotBool=True,
            write_to_txtfile_Bool=True,
        )
        save_line_chart(
            self.log_dir_train,
            "accuracy_test_history",
            log_history_dict_valid["accuracy"],
            diag_title="Valid Accuracy",
            xlabel="epoch",
            ylabel="accuracy",
            pyplotBool=True,
            write_to_txtfile_Bool=True,
        )
        save_line_chart(
            self.log_dir_train,
            "loss_test_history",
            log_history_dict_valid["loss"],
            diag_title="Valid Loss",
            xlabel="epoch",
            ylabel="loss",
            pyplotBool=True,
            write_to_txtfile_Bool=True,
        )

        # plot results
        keys_to_plot_train = {"loss", "accuracy"}
        keys_to_plot_valid = {"loss", "accuracy"}
        log_dict_test_plot_train = {
            key + "_train": value[-1]
            for key, value in log_history_dict_train.items()
            if key in keys_to_plot_train
        }
        log_dict_test_plot_test = {
            key + "_valid": value[-1]
            for key, value in log_history_dict_valid.items()
            if key in keys_to_plot_valid
        }
        log_dict_test_plot = {**log_dict_test_plot_train, **log_dict_test_plot_test}
        save_dict_to_structured_txt(
            self.log_dir_train, log_dict_test_plot, filename="results_train"
        )

        print("Training Finished!")

    def test(
        self,
        dataset,
        perturbation_generator=None,
        perturb_type=None,
        epsilon=None,
        cache_results=False,
        folder_name_add="",
    ):
        """
        all testing happens here
        Args:
            dataset: tf.data.Dataset object of (x_data, y_labels), prepared with
            batch_size etc.
            perturbation_generator: object from class PerturbationGenerator
            perturb_type: string, in case dataset should be perturbed
            epsilon: int, level of perturbation
            from_cache: if False the logits are calculated via the Tensorflow graph,
                if True the logits are restored from cache
            folder_name_add: string, additional string for folder name
        Returns:
            logits and respective labels
        """

        # define detailed foldernname for saving results of the respective
        # perturb_type and epsilon
        log_dir_test_detailed = (
            self.log_dir_test + str(perturb_type) + "_" + str(epsilon) + "/"
        )

        # create log metrics
        log_dict_test = {}
        log_dict_test["loss"] = tf.keras.metrics.Mean(name="loss")
        log_dict_test["accuracy"] = tf.keras.metrics.CategoricalAccuracy(
            name="accuracy"
        )
        log_dict_test["logits"] = Log_extend_array(name="logits")
        log_dict_test["labels"] = Log_extend_array(name="labels")

        #get corrupted tf.data if applicable
        batch_perturb = True
        if perturbation_generator is not None:
            dataset_perturb = perturbation_generator.perturb_dataset(
                dataset=dataset,
                perturb_type=perturb_type,
                epsilon=epsilon,
                data_corrupted_path=self.data_corrupted_path,
                batch_size=self.batch_size
            )
            if dataset_perturb == None:
                batch_perturb = True
            else:
                batch_perturb = False
                dataset = dataset_perturb

        for x_data, y_data in dataset:
            if batch_perturb == True:
                if perturbation_generator is not None:
                    if self.dataset.data_name=="Newsgroups20" and \
                    perturb_type in ["char_swap", "None"]:
                        x_data = perturbation_generator.perturb_batch(
                            x_data,
                            perturb_type=perturb_type,
                            epsilon=epsilon,
                            model=self,
                            label=y_data,
                            embedding=self.dataset.embedding,
                            length_vocab=self.dataset.length_vocab
                        )
                    else:
                        x_data = perturbation_generator.perturb_batch(
                            x_data,
                            perturb_type=perturb_type,
                            epsilon=epsilon,
                            model=self,
                            label=y_data
                        )
                elif self.dataset.data_name=="Newsgroups20":
                    x_data = self.dataset.embedding(x_data)

            log_dict_test = self.test_step_logits(x_data, y_data, log_dict_test)

        logits_test = log_dict_test["logits"].result()
        labels_test = log_dict_test["labels"].result()
        if cache_results == True:
            self.logits_test[str(perturb_type) + "_" + str(epsilon)] = logits_test
            self.labels_test[str(perturb_type) + "_" + str(epsilon)] = labels_test

        # plot results
        keys_to_plot = {"loss", "accuracy"}
        log_dict_test_plot = {
            key + "_train": value.result().numpy()
            for key, value in log_dict_test.items()
            if key in keys_to_plot
        }

        # print metrics
        print(
            "Test Loss= "
            + "{:.4f}".format(log_dict_test["loss"].result().numpy())
            + ", Test Accuracy= "
            + "{:.3f}".format(log_dict_test["accuracy"].result().numpy())
        )

        return logits_test, labels_test

    def predict(self, x_data):
        """get predictions"""
        return self.model.prediction_test(x_data)

    def loss_adv(self, x_data, y_data, sub_model_num=None):
        """get crossentropy loss, e.g., for adversarial sample generation
        Args:
            x_data: array of input data
            y_data: array of ouput data
            sub_model_num: int, is not None if the model has several submodels the
                number of the submodel is specified here (e.g. for ensembles)
        Returns:
            crossentropy loss
        """
        if sub_model_num == None:
            preds = self.model.prediction_test(x_data)
            return self.model.loss_adv(y_data, preds)
        else:
            preds = self.model.prediction_test(x_data, model_num=sub_model_num)
            return self.model.loss_adv(y_data, preds)

    def logits(
        self,
        dataset=None,
        perturb_generator=None,
        perturb_type=None,
        epsilon=None,
        from_cache=False,
        to_cache=False,
        save_to_file=False,
    ):
        """get logits for a respective dataset
        Args:
            dataset: tf.data.Dataset object of (x_data, y_labels), prepared with
            batch_size etc.
            perturbation_generator: object from class PerturbationGenerator
            perturb_type: string, in case dataset should be perturbed
            epsilon: int, level of perturbation
            from_cache: if False the logits and labels are calculated via the Tensorflow
                graph, if True the logits are restored from cache
            to_cache: bool, if True the logits and labels are stored to cache
            save_to_file: bool, if True the logits and labels are saved to file
        Returns:
            logits and respective labels
        """

        key = str(perturb_type) + "_" + str(epsilon)
        if from_cache == True:
            if key in self.logits_test.keys() and key in self.labels_test.keys():
                logits_test, labels_test = self.logits_test[key], self.labels_test[key]
            else:
                if dataset == None:
                    raise ValueError("Dataset is not specified!")
                logits_test, labels_test = self.test(
                    dataset, perturb_generator, perturb_type, epsilon, to_cache
                )
        else:
            if dataset == None:
                raise ValueError("Dataset is not specified!")
            logits_test, labels_test = self.test(
                dataset, perturb_generator, perturb_type, epsilon, to_cache
            )
        if save_to_file == True:
            save_dict_to_pkl(self.log_dir_test, self.logits_test, "logits_test")
            save_dict_to_pkl(
                self.log_dir_test, self.labels_test, "labels_test"
            )
        return logits_test, labels_test

    def save(self, save_logits=True):
        """save trained models with additional information.
        Args:
            save_logits: bool, whether logits and labels are saved
        """
        # save params
        params_self = self.__dict__
        delete_keys_list = ["model", "dataset", "logits_test", "labels_test", "mirrored_strategy"]
        params = {
            key: value
            for key, value in params_self.items()
            if key not in delete_keys_list
        }
        save_dict_to_pkl(self.save_path, params, "model_params")
        save_dict_to_csv(self.save_path, params, "model_params")
        save_dict_to_structured_txt(self.save_path, params, "model_params")
        # save model
        print('save model...')
        self.model.save(self.save_path + "model.h5")
        # save logits
        if save_logits == True:
            if self.logits_test and self.labels_test:
                save_dict_to_pkl(self.log_dir_test, self.logits_test, "logits_test")
                save_dict_to_pkl(self.log_dir_test, self.labels_test, "labels_test")

    def load(self, load_path, load_logits=False, params_load={}, **kwargs):
        """loads the model's parameters and the model
        Args:
            load_path: string, determines the path to the model's folder
            load_logits: bool, whether logits and labels are loaded
        """
        # load params from file and initiate them as local variables
        params = load_dict_from_pkl(load_path, "model_params")
        for key, value in params.items():
            self.__setattr__(key, value)
        for key, value in params_load.items():
            self.__setattr__(key, value)
        for key, value in kwargs.items():
            self.__setattr__(key, value)
        # overwrite paths of loaded model
        self.save_path = load_path
        self.save_path_general = os.path.split(os.path.split(self.save_path)[0])[0] + "/"
        self.log_dir_train = load_path + "train/"
        self.log_dir_test = load_path + "test/"
        if load_logits == True:
            if os.path.isfile(
                self.log_dir_test + "logits_test" + ".pkl"
            ) and os.path.isfile(self.log_dir_test + "labels_test" + ".pkl"):
                self.logits_test = load_dict_from_pkl(self.log_dir_test, "logits_test")
                self.labels_test = load_dict_from_pkl(self.log_dir_test, "labels_test")
        # load model
        if self.architecture != 'ResNet50_svill':
            self.build(build_new_model=False, load_path=load_path+'model.h5')
        else:
            self.build(build_new_model=False, load_path=load_path)
