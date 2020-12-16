import tensorflow as tf
import numpy as np

class Model_Falcon:
    """Template for model with loss function and additional methods"""
    def __init__(
        self,
        build_new_model,
        load_path,
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
        self.build_new_model = build_new_model
        self.load_path = load_path
        self.epochs = epochs
        self.lambda_l2_loss = lambda_l2_loss
        self.lambda_ent_loss = lambda_ent_loss
        self.annealing_max_ent_loss = annealing_max_ent_loss
        self.lambda_advcalib_loss = lambda_advcalib_loss
        self.annealing_max_advcalib_loss = annealing_max_advcalib_loss
        self.n_classes = n_classes
        self.bins_calibration = bins_calibration
        self.calculate_annealing_factors()
        self.build()

    def build(self, **kwargs):
        """builds or loads model
        if build_new_model: True, build tf model
        if build_new_model: False, load model from load_path
        """
        if self.build_new_model == True:
            self.build_main_model()
        else:
            if self.load_path != None:
                self.load(self.load_path)
            else:
                raise ValueError("load_path is not specified")
        self.build_sub_models()
        self.model.summary()

    def build_optimizers(self, learning_rate, **kwargs):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def build_optimizers(self, learning_rate, **kwargs):
        """initialize Adam and SGD optimizers"""
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.optimizer_advcalib = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    def calculate_annealing_factors(self):
        """calculate annealing_factor for predictive entropy loss and for
        adversarial calibration loss"""

        if self.lambda_ent_loss > 0:
            self.annealing_factor_ent_loss = (
                self.epochs * self.annealing_max_ent_loss
            ) / (self.lambda_ent_loss)
        else:
            self.annealing_factor_ent_loss = (
                self.epochs * self.annealing_max_ent_loss
            )
        print(
            "annealing factor for predictive entropy loss: %i steps" % \
            self.annealing_factor_ent_loss
        )

        if self.lambda_advcalib_loss > 0:
            self.annealing_factor_advcalib_loss = (
                self.epochs * self.annealing_max_advcalib_loss
            ) / (self.lambda_advcalib_loss)
        else:
            self.annealing_factor_advcalib_loss = self.epochs * self.annealing_max_advcalib_loss
        print("annealing factor for adversarial calibration loss: %i steps" % \
            self.annealing_factor_advcalib_loss
        )

    @tf.function
    def loss(self, labels, preds, train_step=None):
        """ loss function with: Crossentropy Loss, predictive entropy loss and
        l2 weight regularization"""

        # crossentropy loss
        loss_crossentropy = tf.losses.categorical_crossentropy(
            y_true=labels, y_pred=preds
        )

        # predictive entropy loss
        if train_step != None:
            annealing_coef_ent_loss = tf.minimum(
             self.lambda_ent_loss,
             (train_step / self.annealing_factor_ent_loss),
            )
        else:
            annealing_coef_ent_loss = self.lambda_ent_loss

        regul_labels = np.zeros((1, self.n_classes))
        regul_labels.fill(0.1)
        regul_labels = tf.constant(regul_labels, dtype=tf.float32)

        greater_than = tf.constant(0.0, dtype=tf.float32)
        p = tf.cast(
            tf.greater(labels, greater_than), dtype=tf.float32
        )  # convert tensor
        regul_labels = regul_labels * (1 - p)
        regul_labels = tf.transpose(
            (tf.transpose(regul_labels, [1, 0]) / \
            tf.reduce_sum(regul_labels, axis=1)),
            [1, 0],
        )
        regul_preds = preds * (1 - p)
        loss_entropy = tf.losses.categorical_crossentropy(
            y_true=regul_labels, y_pred=regul_preds
        )
        ent_loss = annealing_coef_ent_loss * loss_entropy

        # l2 weight regularization
        l2_regularization_loss = tf.reduce_sum(self.model.losses)
        return loss_crossentropy + ent_loss + l2_regularization_loss

    @tf.function
    def loss_advcalib(self, labels, logits, train_step=None):
        """ loss function with: adversarial calibration loss
        Args:
            labels: tf array with one hot encoded labels
            logits: tf array with logits
            train_step: int, current training step
        Returns:
            tensorflow graph based adversarial calibration loss
        """

        def calculate_confidence_vec(probability_list, match_list, bins_limits):
            def calculate_binned_acc_list(
                probability_list, match_list, bins_limits, binplace
            ):
                binned_accuracy_list = tf.cast(
                    tf.range(0.0, tf.shape(bins_limits)[0] + 1), dtype="float32"
                )
                def fn(bin_num):
                    return tf.cond(
                        tf.cast(
                            tf.reduce_max(
                                tf.cast(tf.equal(bin_num, binplace), dtype="float32")
                            ),
                            dtype="bool",
                        ),
                        lambda: tf.reduce_mean(
                            tf.cast(
                                tf.gather_nd(
                                    match_list, tf.where(tf.equal(binplace, bin_num))
                                ),
                                dtype="float32",
                            )
                        ),
                        lambda: tf.cast(tf.convert_to_tensor(0.0), dtype="float32"),
                    )
                binned_accuracy_list = tf.map_fn(fn, binned_accuracy_list)
                return binned_accuracy_list
            binplace = tf.cast(
                tf.searchsorted(bins_limits, probability_list, side="left"),
                dtype="float32",
            )  # Returns which bin an array element belongs to
            binned_accuracy_vec = calculate_binned_acc_list(
                probability_list, match_list, bins_limits, binplace
            )
            confidene_vec = tf.gather(
                binned_accuracy_vec, tf.cast(binplace, dtype="int32")
            )
            return confidene_vec, binned_accuracy_vec, tf.cast(binplace, dtype="int32")

        if train_step != None:
            annealing_coef_advcalib_loss = tf.minimum(
                self.lambda_advcalib_loss,
                tf.cast(train_step / self.annealing_factor_advcalib_loss, tf.float32),
            )
        else:
            annealing_coef_advcalib_loss = self.lambda_advcalib_loss

        class_prob = tf.clip_by_value(tf.nn.softmax(logits), 1.17e-37, 3.40e37)
        highest_class_prob = tf.squeeze(tf.reduce_max(class_prob, axis=1))

        pred = tf.argmax(logits, 1)
        truth = tf.argmax(labels, 1)
        match = tf.squeeze(
            tf.reshape(tf.cast(tf.equal(pred, truth), tf.float32), (-1, 1))
        )

        confidence_vec, binned_acc_vec, binplace = calculate_confidence_vec(
            highest_class_prob, match, self.bins_calibration
        )
        confidence_vec = tf.stop_gradient(confidence_vec)
        loss_advcalib = tf.nn.l2_loss(confidence_vec - highest_class_prob)
        loss_advcalib = annealing_coef_advcalib_loss * loss_advcalib
        return loss_advcalib

    @tf.function
    def loss_adv(self, labels, preds, train_step=None, **kwargs):
        """loss function used for adversarial sample generation with
        crossentropy loss
        """
        # crossentropy loss
        loss_crossentropy = tf.losses.categorical_crossentropy(
            y_true=labels, y_pred=preds
        )
        return loss_crossentropy

    @tf.function
    def prediction_train(self, x_data, **kwargs):
        """get predictions of model in training mode"""
        return self.model(x_data, training=True)

    @tf.function
    def prediction_test(self, x_data, **kwargs):
        """get predictions of model in testing mode"""
        return self.model(x_data, training=False)

    @tf.function
    def logits_train(self, x_data, **kwargs):
        """get logits of model in training mode"""
        return self.model_logits(x_data, training=True)

    @tf.function
    def logits_test(self, x_data, **kwargs):
        """get logits of model in testing mode"""
        return self.model_logits(x_data, training=False)

    def trainable_variables(self, **kwargs):
        """get trainable variables"""
        return self.model.trainable_variables

    def save(self, path):
        """save model to path"""
        self.model.save(path)

    def load(self, path):
        """load model from path"""
        print('load model from following path: ', path)
        self.model = tf.keras.models.load_model(path)
        print("Model was loaded from file.")
