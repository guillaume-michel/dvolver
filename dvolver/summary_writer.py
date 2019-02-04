# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import numpy as np
from io import BytesIO

from PIL import Image

import tensorflow as tf
from tensorflow.core.framework import summary_pb2
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.framework import tensor_shape_pb2

class SummaryWriter(object):

    def __init__(self, log_dir, lazy_flush=False):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)
        self.lazy_flush = lazy_flush


    def add_scalar(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self._add_summary(summary, step)


    def add_image(self, tag, image, step):
        """Log an image."""

        # Write the image to a string
        s = BytesIO()
        Image.fromarray(image).save(s, format="png")

        # Create an Image object
        img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                   height=image.shape[0],
                                   width=image.shape[1])

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, image=img_sum)])
        self._add_summary(summary, step)


    def add_text(self, tag, text, step):
        """Log a text."""
        pluginData = tf.SummaryMetadata.PluginData(plugin_name='text')

        smd = tf.SummaryMetadata(plugin_data=pluginData)

        tensor = tensor_pb2.TensorProto(dtype='DT_STRING',
                                        string_val=[text.encode(encoding='utf_8')],
                                        tensor_shape=tensor_shape_pb2.TensorShapeProto(dim=[tensor_shape_pb2.TensorShapeProto.Dim(size=1)]))

        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, metadata=smd, tensor=tensor)])
        self._add_summary(summary, step)


    def _add_summary(self, summary, step):
        self.writer.add_summary(summary, step)

        if not self.lazy_flush:
            self.writer.flush()
