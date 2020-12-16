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
import math
from tensorflow.python.ops import array_ops

class Imagenet:
    """ImageNet2012 dataset from http://www.image-net.org"""
    def __init__(
        self,
        train_batch_size=100,
        valid_batch_size=None,
        test_batch_size=None,
        shuffle_buffer=5000,
        image_size = 224,
        padding = 32,
        seed = 0,
        data_path="./",
        download=False,
        **kwargs
    ):
        """
        Args:
            train_batch_size: int, batch size for training
            valid_batch_size: int, batch size for validation
            test_batch_size: int, batch size for testing
            shuffle_buffer: int, size of shuffle buffer
            image_size: int, one dimension of the resulting quadratic image
            padding: int
            seed: int, random seed
            data_path: string, path which points to dataset
            download: bool, whether to downolad dataset
        """

        self.data_name = "IMAGENET"
        self.data_path = data_path
        self.download = download

        self.image_size = image_size
        self.padding = padding
        self.seed = seed

        self.shuffle_buffer = shuffle_buffer
        self.train_batch_size = train_batch_size
        if valid_batch_size == None:
            self.valid_batch_size = train_batch_size
        else:
            self.valid_batch_size = valid_batch_size
        if test_batch_size == None:
            self.test_batch_size = train_batch_size
        else:
            self.test_batch_size = test_batch_size

        self.load_data()
        self.prepare_generator()

    def load_data(self):
        '''load dataset and devide into training, validation and testing data'''
        print("loading %s data..." % (self.data_name))
        self.nb_train_sampels = 1281167
        self.nb_valid_sampels = 25000
        self.nb_test_sampels = 25000
        self.train_dataset, info = tfds.load(
            name="imagenet2012",
            split="train",
            data_dir=self.data_path,
            with_info=True,
            download=self.download
        )
        self.valid_dataset, info = tfds.load(
            name="imagenet2012",
            split="validation[:50%]",
            data_dir=self.data_path,
            with_info=True,
            download=self.download
        )
        self.test_dataset, info = tfds.load(
            name="imagenet2012",
            split="validation[50%:]",
            data_dir=self.data_path,
            with_info=True,
            download=self.download
        )
        self.train_steps_per_epoch = math.floor(
            self.nb_train_sampels/float(self.train_batch_size)
        )
        self.valid_steps_per_epoch = math.floor(
            self.nb_valid_sampels/float(self.valid_batch_size)
        )
        self.test_steps_per_epoch = math.floor(
            self.nb_test_sampels/float(self.test_batch_size)
        )
        print('Train set - steps per epoch:', self.train_steps_per_epoch)
        print('Valid set - steps per epoch:', self.valid_steps_per_epoch)
        print('Test set - steps per epoch:', self.test_steps_per_epoch)
        print(info)

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
                image: Tensor, representing an image
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
                image: Tensor, representing an image
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
                image: Tensor, representing an image
                is_training: bool, if True preprocessing for training
                use_bfloat16: bool, for whether to use bfloat16.
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

        self.train_dataset = self.train_dataset.map(
            lambda d: {"image":preprocess_image(
                d["image"],is_training=True), "label":d["label"]
            }
        )
        self.valid_dataset = self.valid_dataset.map(
            lambda d: {"image":preprocess_image(
                d["image"],is_training=False), "label":d["label"]
            }
        )
        self.test_dataset = self.test_dataset.map(
            lambda d: {"image":preprocess_image(
                d["image"],is_training=False), "label":d["label"]
            }
        )

        #one_hot encoded labels
        self.train_dataset = self.train_dataset.map(
            lambda d: {"image":d["image"], "label":tf.one_hot(d["label"], 1000)}
        )
        self.valid_dataset = self.valid_dataset.map(
            lambda d: {"image":d["image"], "label":tf.one_hot(d["label"], 1000)}
        )
        self.test_dataset = self.test_dataset.map(
            lambda d: {"image":d["image"], "label":tf.one_hot(d["label"], 1000)}
        )

        #transform to tuple
        self.train_dataset = self.train_dataset.map(
            lambda d: (d["image"],d["label"])
        )
        self.valid_dataset = self.valid_dataset.map(
            lambda d: (d["image"],d["label"])
        )
        self.test_dataset = self.test_dataset.map(
            lambda d: (d["image"],d["label"])
        )

        print("*** finished loading data ***")

    def prepare_generator(self):
        """prepare data generator for training, validation and testing data"""
        self.train_ds = self.train_dataset.shuffle(self.shuffle_buffer).batch(
            self.train_batch_size
        )
        self.valid_ds = self.valid_dataset.batch(self.valid_batch_size)
        self.test_ds = self.test_dataset.batch(self.test_batch_size)
        self.train_ds.prefetch(tf.data.experimental.AUTOTUNE)
        self.valid_ds.prefetch(tf.data.experimental.AUTOTUNE)
        self.test_ds.prefetch(tf.data.experimental.AUTOTUNE)

    def calculate_info(self):
        """print and store general information about the dataset"""
        def calculate_steps_per_epoch(ds):
            num_elements = 0
            for element in ds:
                num_elements += 1
            return steps
        print('calculating dataset info...')
        self.train_steps_per_epoch = calculate_steps_per_epoch(self.train_ds)
        self.valid_steps_per_epoch = calculate_steps_per_epoch(self.valid_ds)
        self.test_steps_per_epoch = calculate_steps_per_epoch(self.test_ds)
        print('Train set - steps per epoch:', self.train_steps_per_epoch)
        print('Valid set - steps per epoch:', self.valid_steps_per_epoch)
        print('Test set - steps per epoch:', self.test_steps_per_epoch)
