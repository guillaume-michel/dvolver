import glob

import tensorflow as tf

from tensorflow.contrib.data.python.ops import batching
from tensorflow.contrib.data.python.ops import interleave_ops
from tensorflow.contrib.data.python.ops import threadpool

from na_tf_ops.cutout import cutout

TARGET_HEIGHT = 224
TARGET_WIDTH = 224
TARGET_DEPTH = 3

NB_CLASSES = 1001
central_fraction = 0.875
min_object_covered = 0.1
aspect_ratio_range = (0.75, 1.33)
area_range = (0.05, 1.0)
max_attempts = 100

def get_default_inference_shape(batch_size=1,
                                data_format='channels_first'):
    """Returns the input shape for one image"""
    if data_format == 'channels_first':
        return [batch_size, TARGET_DEPTH, TARGET_HEIGHT, TARGET_WIDTH]
    elif data_format == 'channels_last':
        return [batch_size, TARGET_HEIGHT, TARGET_WIDTH, TARGET_DEPTH]
    else:
        raise ValueError('Wrong data_format specified to get_default_inference_shape function')


def get_search_mode_files(data_dir):
    raise ValueError('You may not want to perform search mode on ImageNet')


def get_full_mode_files(data_dir):
    TRAIN_LIST = tf.gfile.Glob(data_dir + '/' + 'train-*')
    VALIDATION_LIST = tf.gfile.Glob(data_dir + '/' + 'validation-*')

    num_train_samples = 1281167
    num_test_samples = 50000

    return TRAIN_LIST, num_train_samples, VALIDATION_LIST, num_test_samples


def get_input_fn(filenames,
                 batch_size=1,
                 num_threads=2,
                 perform_shuffle=False,
                 perform_augmentation=False,
                 per_image_standardization=False,
                 enable_cutout=False,
                 repeat_count=1):
    """
    Input pipeline for ImageNet tfrecord
    """

    def parse(example_proto):
        features = {
            'image/class/label': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
            'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
            "image/height": tf.FixedLenFeature((), tf.int64, default_value=0),
            "image/width": tf.FixedLenFeature((), tf.int64, default_value=0),
        }

        parsed_features = tf.parse_single_example(example_proto, features)

        encoded_image = parsed_features['image/encoded']

        height = tf.cast(parsed_features['image/height'], tf.int32)
        width = tf.cast(parsed_features['image/width'], tf.int32)

        label = tf.cast(parsed_features['image/class/label'], tf.int32)
        label = tf.reshape(label, [])

        image = tf.image.decode_image(encoded_image, channels=3)
        image = tf.reshape(image, [height, width, 3])

        return image, label


    def resize(image):
        # resize_bilinear needs a 4-D tensor
        image = tf.expand_dims(image, 0)
        # resize to target dimensions. output image's type is float
        image = tf.image.resize_bilinear(image, [TARGET_HEIGHT, TARGET_WIDTH])
        # remove extra dimension introduced for resize_bilinear
        image = tf.squeeze(image, [0])

        return image


    def distort_image(image):
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0],
                           dtype=tf.float32,
                           shape=[1, 1, 4])

        bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=bbox,
            min_object_covered=min_object_covered,
            aspect_ratio_range=aspect_ratio_range,
            area_range=area_range,
            max_attempts=max_attempts,
            use_image_if_no_bounding_boxes=True)

        # Crop the image to the specified bounding box.
        image = tf.slice(image, bbox_begin, bbox_size)
        image.set_shape([None, None, 3])

        return image


    def augment(image):

        # distort image
        image = distort_image(image)

        # resize_bilinear
        image = resize(image)

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)

        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

        image = tf.clip_by_value(image, 0.0, 1.0)

        if enable_cutout:
            image = cutout(image,
                           p=0.5,
                           s_l=0.02,
                           s_h=0.4,
                           r_1=0.3,
                           r_2=3.3,
                           v_l=0,
                           v_h=1.0)

        return image


    def preprocess_fn(example_proto):

        # decode example from proto
        image, label = parse(example_proto)

        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        if perform_augmentation:
            # data augmentation and resize
            image = augment(image)
        else:
            # central crop like slim
            image = tf.image.central_crop(image, central_fraction=central_fraction)
            # resize
            image = resize(image)

        if per_image_standardization:
            # Subtract off the mean and divide by the variance of the pixels
            image = tf.image.per_image_standardization(image)
        else:
            # Convert from [0, 255] -> [-1.0, 1.0] floats.
            image = tf.subtract(image, 0.5)
            image = tf.multiply(image, 2.0)

        # convert from HWC to CHW
        image = tf.transpose(image, [2, 0, 1])

        # Convert label from a scalar uint8 tensor to an int32 scalar.
        label = tf.cast(label, tf.int32)

        return image, label


    ds = tf.data.TFRecordDataset.list_files(filenames)

    ds = ds.apply(
        interleave_ops.parallel_interleave(
            tf.data.TFRecordDataset, cycle_length=10))

    ds = ds.prefetch(buffer_size=batch_size)

    if perform_shuffle:
        ds = ds.shuffle(buffer_size=10000)

    ds = ds.repeat(repeat_count)
    ds = ds.apply(
        batching.map_and_batch(map_func=preprocess_fn,
                               batch_size=batch_size,
                               num_parallel_batches=2))

    ds = ds.prefetch(buffer_size=1)

    if num_threads:
        ds = threadpool.override_threadpool(ds,
                                            threadpool.PrivateThreadPool(num_threads,
                                                                         display_name='input_pipeline_thread_pool'))

    return ds
