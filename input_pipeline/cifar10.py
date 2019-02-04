import tensorflow as tf
from na_tf_ops.cutout import cutout

TARGET_HEIGHT = 32
TARGET_WIDTH = 32
TARGET_DEPTH = 3

PAD_SIZE = 4

NB_CLASSES = 10

def get_default_inference_shape(batch_size=1,
                                data_format='channels_first'):
    """Returns the input shape for one image"""
    if data_format == 'channels_first':
        return [batch_size, 3, 32, 32]
    elif data_format == 'channels_last':
        return [batch_size, 32, 32, 3]
    else:
        raise ValueError('Wrong data_format specified to get_default_inference_shape function')


def get_search_mode_files(data_dir):
    TRAIN_LIST = [data_dir + '/' + 'train_' + str(i) + '.tfrecords' for i in range(9)]
    VALIDATION_LIST = [data_dir + '/' + 'train_9.tfrecords'] # use last training tfrecords has validation set

    num_train_samples = 45000
    num_test_samples = 5000

    return TRAIN_LIST, num_train_samples, VALIDATION_LIST, num_test_samples


def get_full_mode_files(data_dir):
    TRAIN_LIST = [data_dir + '/' + 'train_' + str(i) + '.tfrecords' for i in range(10)]
    VALIDATION_LIST = [data_dir + '/' + 'test_' + str(i) + '.tfrecords' for i in range(2)]

    num_train_samples = 50000
    num_test_samples = 10000

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
    Input pipeline for cifar-10 tfrecord
    """

    def data_augmentation(images):
        """
        Data augmentation for CIFAR-10
        images should be in NHWC format
        """

        def augment(image):
            # random crop 32x32 in image of size 40x40
            image = tf.random_crop(image, [TARGET_HEIGHT, TARGET_WIDTH, TARGET_DEPTH])
            # Randomly flip the image horizontally.
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image,
                                               max_delta=63)
            image = tf.image.random_contrast(image,
                                             lower=0.2, upper=1.8)

            if enable_cutout:
                image = cutout(image,
                               p=0.5,
                               s_l=0.02,
                               s_h=0.4,
                               r_1=0.3,
                               r_2=3.3,
                               v_l=0,
                               v_h=255)

            return image

        if PAD_SIZE > 0:
            # pad full batch at once. each image is now 40x40
            images = tf.pad(images, [[0, 0], [PAD_SIZE, PAD_SIZE], [PAD_SIZE, PAD_SIZE], [0, 0]])

        # random crop and flip
        images = tf.map_fn(lambda image: augment(image), images)

        return images


    def parse(example_protos, batch_size):
        features = {"label": tf.FixedLenFeature((), tf.int64, default_value=0),
                    "image": tf.FixedLenFeature((), tf.string, default_value="")}

        parsed_features = tf.parse_example(example_protos, features)

        images = tf.decode_raw(parsed_features['image'], tf.uint8)
        images = tf.reshape(images, [batch_size, TARGET_DEPTH, TARGET_HEIGHT, TARGET_WIDTH], name='raw_reshape')

        # Convert to float
        images = tf.cast(images, tf.float32)

        if perform_augmentation or per_image_standardization:
            # convert from NCHW to NHWC
            images = tf.transpose(images, [0, 2, 3, 1])

        if perform_augmentation:
            # data augmentation
            images = data_augmentation(images)

        if per_image_standardization:
            # Subtract off the mean and divide by the variance of the pixels
            images = tf.map_fn(tf.image.per_image_standardization, images)
        else:
            # Convert from [0, 255] -> [-1.0, 1.0] floats.
            images = 2.0*(images * (1. / 255) - 0.5)

        if perform_augmentation or per_image_standardization:
            # convert from NHWC to NCHW
            images = tf.transpose(images, [0, 3, 1, 2])

        # Convert label from a scalar uint8 tensor to an int32 scalar.
        labels = tf.cast(parsed_features['label'], tf.int32)
        labels = tf.reshape(labels, [batch_size])

        return images, labels


    dataset = tf.data.TFRecordDataset(filenames)  # Read tfrecord file

    dataset = dataset.cache()

    dataset = dataset.repeat(repeat_count)  # Repeats dataset this # times

    if perform_shuffle:
        # Randomizes input
        dataset = dataset.shuffle(buffer_size=batch_size*8)

    dataset = dataset.batch(batch_size)  # Batch size to use
    dataset = dataset.map(lambda e: parse(e, batch_size),
                          num_parallel_calls=num_threads)  # Transform each elem by applying parse fn (vectorized on the batch)


    #dataset = dataset.prefetch(2) # for parallel processing of CPU and GPU

    return dataset
