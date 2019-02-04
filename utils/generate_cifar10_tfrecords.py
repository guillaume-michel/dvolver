#!/usr/bin/python3

"""Read CIFAR-10 data from pickled numpy arrays and writes TFRecords.

Generates tf.train.Example protos and writes them to TFRecord files from the
python version of the CIFAR-10 dataset downloaded from
https://www.cs.toronto.edu/~kriz/cifar.html.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import pickle
import os

import tarfile
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np

CIFAR_FILENAME = 'cifar-10-python.tar.gz'
CIFAR_DOWNLOAD_URL = 'https://www.cs.toronto.edu/~kriz/' + CIFAR_FILENAME
CIFAR_LOCAL_FOLDER = 'cifar-10-batches-py'

flatten = lambda l: [item for sublist in l for item in sublist]

def unison_shuffled_copies(a, b, seed=42):
    assert len(a) == len(b)

    # use seed to be determinist
    np.random.seed(seed)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def download_and_extract(data_dir):
  # download CIFAR-10 if not already downloaded.
  tf.contrib.learn.datasets.base.maybe_download(CIFAR_FILENAME, data_dir,
                                                CIFAR_DOWNLOAD_URL)
  tarfile.open(os.path.join(data_dir, CIFAR_FILENAME),
               'r:gz').extractall(data_dir)


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def read_pickle_from_file(filename):
  with tf.gfile.Open(filename, 'rb') as f:
    data_dict = pickle.load(f, encoding='latin-1')
  return data_dict


def convert_to_tfrecord(labels, datas, output_file):
  """Converts a file to TFRecords."""
  print('Generating %s' % output_file)

  rows = 32
  cols = 32
  depth = 3

  try:
    os.remove(output_file)
  except OSError:
    pass

  with tf.python_io.TFRecordWriter(output_file) as record_writer:
    num_entries_in_batch = len(labels)
    for i in range(num_entries_in_batch):
      example = tf.train.Example(features=tf.train.Features(
        feature={
          'height': _int64_feature(rows),
          'width': _int64_feature(cols),
          'depth': _int64_feature(depth),
          'format': _bytes_feature(tf.compat.as_bytes('RAW')),
          'label': _int64_feature(labels[i]),
          'image': _bytes_feature(datas[i].tobytes())
        }))
      record_writer.write(example.SerializeToString())


def load_and_balance_data(input_dir, files, split_count, class_count=10):

  labels = []
  datas = []

  input_files = [os.path.join(input_dir, f) for f in files]

  for input_file in input_files:
    data_dict = read_pickle_from_file(input_file)

    datas.append(data_dict['data'])
    labels.append(data_dict['labels'])

  # flatten
  labels = np.array(flatten(labels))
  datas = np.concatenate(datas, axis=0)

  # needed for the balance to work
  assert (len(labels)/split_count) % class_count == 0

  # sort by labels
  inds = labels.argsort()
  labels = labels[inds]
  datas = datas[inds]

  # get list of arrays (one for each class)
  datas = np.vsplit(datas, class_count)
  labels = np.vsplit(labels.reshape(labels.shape[0], 1), class_count)

  # interleaves rows of each labels to balance labels
  shape_datas = (len(datas)*datas[0].shape[0], datas[0].shape[1])
  shape_labels = (len(labels)*labels[0].shape[0], labels[0].shape[1])

  datas = np.hstack(datas).reshape(shape_datas)
  labels = np.hstack(labels).reshape(shape_labels)

  # split into split_count arrays
  datas = np.vsplit(datas, split_count)
  labels = np.vsplit(labels.reshape(labels.shape[0], 1), split_count)

  # shuffle rows for each array pair
  # returns a list of split_count pairs
  # each pair is (labels, datas) with balanced labels
  return [unison_shuffled_copies(l,d) for l,d in zip(labels, datas)]


def main(data_dir):
  print('Download from {} and extract.'.format(CIFAR_DOWNLOAD_URL))
  download_and_extract(data_dir)

  train_files = ['data_batch_%d' % i for i in xrange(1, 6)]
  test_files = ['test_batch']

  input_dir = os.path.join(data_dir, CIFAR_LOCAL_FOLDER)

  # ------- TRAIN -----------
  # balance and split train datas
  train_sets =  load_and_balance_data(input_dir, train_files, split_count=10, class_count=10)
  output_train_files = [os.path.join(data_dir, 'train_' + str(i) + '.tfrecords') for i in np.arange(len(train_sets))]

  for i, (labels, datas) in enumerate(train_sets):
    convert_to_tfrecord(labels, datas, output_train_files[i])

  # ------- TEST -----------
  # balance and split test datas
  test_sets =  load_and_balance_data(input_dir, test_files, split_count=2, class_count=10)
  output_test_files = [os.path.join(data_dir, 'test_' + str(i) + '.tfrecords') for i in np.arange(len(test_sets))]

  for i, (labels, datas) in enumerate(test_sets):
    convert_to_tfrecord(labels, datas, output_test_files[i])

  print('Done!')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data-dir',
      type=str,
      default='',
      help='Directory to download and extract CIFAR-10 to.')

  args = parser.parse_args()
  main(args.data_dir)
