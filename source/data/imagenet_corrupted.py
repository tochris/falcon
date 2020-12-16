# Several functions are taken from Google Research and adapted
# (https://github.com/google-research/google-research/blob/master/
# uq_benchmark_2019/imagenet/resnet_preprocessing.py):
# -distorted_bounding_box_crop
# -_at_least_x_are_equal
# -center_crop_and_resize
# -random_crop_and_resize
# -preprocess_for_eval
# -preprocess_for_train
# -preprocess_for_image

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.ops import array_ops

class Imagenet_corrupted:
    """ImageNet2012 corrupted dataset (Paper: Hendrycks, D.; and Dietterich,
    T. 2019. Benchmarking neural network robustness to common corruptions and
    perturbations.)"""
    def __init__(
        self,
        corruption_type=None,
        epsilon=0,
        test_batch_size=100,
        image_size = 224,
        padding = 32,
        seed = 0,
        data_path= "./",
        download=False,
        **kwargs
    ):
        """
        Args:
            corruption_type: string, includes dataset name and corruption type
                (e.g. 'imagenet2012_corrupted/glass_blur')
            epsilon: int, level of perturbation
            test_batch_size: int, batch size for testing
            image_size: int, one dimension of the resulting quadratic image
            padding: int
            seed: int, random seed
            data_path: string, path which points to dataset
            download: bool, whether to downolad dataset
        """

        self.data_name = "IMAGENET_corrupted"
        self.corruption_type = corruption_type
        self.epsilon = epsilon
        self.data_path = data_path
        self.download = download

        self.image_size = image_size
        self.padding = padding
        self.seed = seed

        self.load_data()

        self.test_batch_size = test_batch_size

        self.prepare_generator()

    def load_data(self):
        '''load dataset and devide into training, validation and testing data'''
        print("loading %s data..." % (self.data_name))
        assert self.corruption_type != None, "Error: epsilon must be greater than 0"
        assert self.epsilon > 0, "Error: epsilon must be greater than 0"
        corrupt_data_name = self.corruption_type + '_' + str(self.epsilon)
        self.test_dataset, info = tfds.load(
            name=corrupt_data_name,
            split="validation[50%:]",
            data_dir=self.data_path,
            with_info=True,
            download=self.download
        )

        def distorted_bounding_box_crop(image,
                                bbox,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100,
                                scope=None):
            """Generates cropped_image using one of the bboxes randomly distorted.
            See tf.image.sample_distorted_bounding_box for more documentation.
            Args:
            image: Tensor of binary image data.
            bbox: Tensor of bounding boxes arranged [1, num_boxes, coords] where
              each coordinate is [0, 1) and the coordinates are arranged as [ymin,
              xmin, ymax, xmax]. If num_boxes is 0 then use the whole image.
            min_object_covered: An optional float. Defaults to 0.1. The cropped area
              of the image must contain at least this fraction of any bounding box
              supplied.
            aspect_ratio_range: An optional list of floats. The cropped area of the
              image must have an aspect ratio = width / height within this range.
            area_range: An optional list of floats. The cropped area of the image must
              contain a fraction of the supplied image within in this range.
            max_attempts: An optional int. Number of attempts at generating a cropped
              region of the image of the specified constraints. After max_attempts
              failures, return the entire image.
            scope: Optional str for name scope.
            Returns:
            cropped image Tensor
            """
            shape = array_ops.shape(image)
            sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
                shape,
                bounding_boxes=bbox,
                seed=self.seed,
                min_object_covered=min_object_covered,
                aspect_ratio_range=aspect_ratio_range,
                area_range=area_range,
                max_attempts=max_attempts,
                use_image_if_no_bounding_boxes=True)
            bbox_begin, bbox_size, _ = sample_distorted_bounding_box

            # Crop the image to the specified bounding box.
            offset_y, offset_x, _ = tf.unstack(bbox_begin)
            target_height, target_width, _ = tf.unstack(bbox_size)
            crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
            image = tf.image.crop_to_bounding_box(image,
                                                  offset_y,
                                                  offset_x,
                                                  target_height,
                                                  target_width)
            return image

        def _at_least_x_are_equal(a, b, x):
            """At least x of a and b Tensors are equal."""
            match = tf.equal(a, b)
            match = tf.cast(match, tf.int32)
            return tf.greater_equal(tf.reduce_sum(match), x)

        def center_crop_and_resize(image):
            """Make a center crop of image_size from image"""
            shape = array_ops.shape(image)
            image_height = shape[0]
            image_width = shape[1]

            padded_center_crop_size = tf.cast(
              ((self.image_size / (self.image_size + self.padding)) *
               tf.cast(tf.minimum(image_height, image_width), tf.float32)),
              tf.int32)

            offset_height = ((image_height - padded_center_crop_size) + 1) // 2
            offset_width = ((image_width - padded_center_crop_size) + 1) // 2
            crop_window = tf.stack([offset_height, offset_width,
                                  padded_center_crop_size, padded_center_crop_size])

            image = tf.image.crop_to_bounding_box(image,
                                                  offset_height,
                                                  offset_width,
                                                  padded_center_crop_size,
                                                  padded_center_crop_size)
            image = tf.image.resize(image, (self.image_size, self.image_size))

            return image

        def random_crop_and_resize(image):
            """Make a random crop of image_size from image"""
            bbox = tf.constant(
                [0.0, 0.0, 1.0, 1.0],
                dtype=tf.float32,
                shape=[1, 1, 4]
            )
            image = distorted_bounding_box_crop(
              image,
              bbox,
              min_object_covered=0.1,
              aspect_ratio_range=(3. / 4, 4. / 3.),
              area_range=(0.08, 1.0),
              max_attempts=10,
              scope=None)
            original_shape = array_ops.shape(image)
            bad = _at_least_x_are_equal(original_shape, tf.shape(image), 3)

            image = tf.cond(
              bad,
              lambda: center_crop_and_resize(image),
              lambda: tf.image.resize(image, (self.image_size, self.image_size)))

            return image

        def preprocess_for_eval(image):
            """
            Preprocesses the given image for evaluation
            Args:
                image: 'Tensor', representing an image
            Returns:
                The preprocessed image tensor for evaluation
            """
            image = tf.cast(image, tf.float32)
            image = center_crop_and_resize(image)
            image = tf.reshape(image, [self.image_size, self.image_size, 3])
            return image

        def preprocess_for_train(image):
            """
            Preprocesses the given image for training
            Args:
                image: 'Tensor', representing an image
            Returns:
                The preprocessed image tensor for training
            """
            image = tf.cast(image, tf.float32)
            image = random_crop_and_resize(image)
            image = tf.image.random_flip_left_right(image)
            image = tf.reshape(image, [self.image_size, self.image_size, 3])
            return image

        def preprocess_image(image, is_training, use_bfloat16=False):
            """
            Preprocess an image tensor for training or testing
            Args:
                image: 'Tensor', representing an image
                is_training: 'bool', if True preprocessing for training
                use_bfloat16: 'bool', for whether to use bfloat16.
            Returns:
                The preprocessed image tensor
            """
            image = tf.cast(image, tf.float32)
            image = tf.clip_by_value(image, 0., 255.) / 255.
            if is_training:
                image = preprocess_for_train(image)
            else:
                image = preprocess_for_eval(image)
            #if use_bfloat16==True convert to bfloat16
            image = tf.image.convert_image_dtype(
              image, dtype=tf.bfloat16 if use_bfloat16 else tf.float32)
            return image

        self.test_dataset = self.test_dataset.map(
            lambda d: {
                "image":preprocess_image(d["image"], is_training=False),
                "label":d["label"]
            }
        )
        #one_hot encoded labels
        self.test_dataset = self.test_dataset.map(
            lambda d: {
                "image":d["image"],
                "label":tf.one_hot(d["label"], 1000)
            }
        )
        #transform to tuple
        self.test_dataset = self.test_dataset.map(
            lambda d: (d["image"],d["label"])
        )
        #print("*** finished loading data ***")

    def prepare_generator(self):
        """prepare data generator for testing data"""
        self.test_ds = self.test_dataset.batch(self.test_batch_size)
        self.test_ds.prefetch(tf.data.experimental.AUTOTUNE)
