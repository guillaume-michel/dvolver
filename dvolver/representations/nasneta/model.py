import copy
import functools
import importlib

import tensorflow as tf

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import saver
from tensorflow.python.training import device_setter

from tensorflow.core.framework import node_def_pb2

from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import ops

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops

from .optimizer import get_optimizer_fn

from dvolver import *

from dvolver.representations.nasneta import nasnet_utils

from dvolver.representations.nasneta.nasnet import *
from dvolver.representations.nasneta.nasnet import _update_hparams, _build_nasnet_base

tf_nasnet = importlib.import_module('dvolver.representations.nasneta.nasnet')

arg_scope = tf.contrib.framework.arg_scope
slim = tf.contrib.slim

def dvolver_nas_config(model_params):
  return tf.contrib.training.HParams(
      stem_multiplier=model_params.stem_multiplier,
      drop_path_keep_prob=model_params.drop_path_keep_prob,
      num_cells=model_params.num_cells,
      use_aux_head=model_params.use_aux_head,
      num_conv_filters=model_params.num_conv_filters,
      dense_dropout_keep_prob=model_params.dense_dropout_keep_prob,
      filter_scaling_rate=model_params.filter_scaling_rate,
      num_reduction_layers=model_params.num_reduction_layers,
      data_format='NCHW',
      skip_reduction_layer_input=model_params.skip_reduction_layer_input,
      # This is used for the drop path probabilities since it needs to increase
      # the drop out probability over the course of training.
      total_training_steps=model_params.max_steps,
  )


def get_nasnet_arg_scope(model_type):
  return getattr(tf_nasnet,
                 'nasnet_' + model_type + '_arg_scope')


def get_used_hiddenstates(hiddenstate_indices, B):
  used_hiddenstates = [0] * (2 + B)

  for i in hiddenstate_indices:
    used_hiddenstates[i] = 1

  return used_hiddenstates


def get_normal_cell_params(params):
  assert len(params) % 2 == 0
  return params[:len(params)//2]


def get_reduction_cell_params(params):
  assert len(params) % 2 == 0
  return params[len(params)//2:]


class Dvolver_NasNetACell(nasnet_utils.NasNetABaseCell):
  """Dvolver NASNetA Generic (Normal/Reduction) Cell."""

  def __init__(self, params, num_conv_filters, drop_path_keep_prob, total_num_cells,
               total_training_steps):
    assert (len(params)-1) % 4 == 0 # 4 parameters per block

    B = int((len(params)-1) / 4)

    regular_topology = params[:-1]
    extra_concat = params[-1:][0]

    operations = regular_topology[1::2]
    hiddenstate_indices = regular_topology[0::2]
    used_hiddenstates = get_used_hiddenstates(hiddenstate_indices, B)

    for i in extra_concat:
      used_hiddenstates[i] = 0 # add extra connection for h_i

    super(Dvolver_NasNetACell, self).__init__(num_conv_filters, operations,
                                              used_hiddenstates,
                                              hiddenstate_indices,
                                              drop_path_keep_prob,
                                              total_num_cells,
                                              total_training_steps)


def get_additional_cells(model_type):
  if model_type == 'cifar':
    return 0
  else:
    return 2

def get_stem_type(model_type):
  if model_type == 'cifar':
    return 'cifar'
  else:
    return 'imagenet'


def dvolver_build_nasnet(model_type,
                         params,
                         images,
                         num_classes,
                         is_training=True,
                         final_endpoint=None,
                         config=None):
  """Build NASNet model"""

  if not config:
    raise ValueError('config should not be None')

  hparams = copy.deepcopy(config)
  _update_hparams(hparams, is_training)

  # Calculate the total number of cells in the network
  # Add the reduction cells
  total_num_cells = hparams.num_cells + hparams.num_reduction_layers
  # If ImageNet, then add stem cells
  total_num_cells += get_additional_cells(model_type)

  normal_cell = Dvolver_NasNetACell(get_normal_cell_params(params),
                                    hparams.num_conv_filters,
                                    hparams.drop_path_keep_prob,
                                    total_num_cells,
                                    hparams.total_training_steps)
  reduction_cell = Dvolver_NasNetACell(get_reduction_cell_params(params),
                                       hparams.num_conv_filters,
                                       hparams.drop_path_keep_prob,
                                       total_num_cells,
                                       hparams.total_training_steps)

  with arg_scope([slim.dropout, nasnet_utils.drop_path, slim.batch_norm],
                 is_training=is_training):
    with arg_scope([slim.avg_pool2d,
                    slim.max_pool2d,
                    slim.conv2d,
                    slim.batch_norm,
                    slim.separable_conv2d,
                    nasnet_utils.factorized_reduction,
                    nasnet_utils.global_avg_pool,
                    nasnet_utils.get_channel_index,
                    nasnet_utils.get_channel_dim],
                   data_format=hparams.data_format):
      return _build_nasnet_base(images,
                                normal_cell=normal_cell,
                                reduction_cell=reduction_cell,
                                num_classes=num_classes,
                                hparams=hparams,
                                is_training=is_training,
                                stem_type=get_stem_type(model_type),
                                final_endpoint=final_endpoint)


def get_network_fn(model_type, params, num_classes, weight_decay=0.0, is_training=False):
  """Returns a network_fn such as `logits, end_points = network_fn(images)`.
  Args:
    model_type: NASNet-A model type (cifar, mobile or large)
    params: DVOLVER cell representation
    num_classes: The number of classes to use for classification. If 0 or None,
      the logits layer is omitted and its input features are returned instead.
    weight_decay: The l2 coefficient for the model weights.
    is_training: `True` if the model is being used for training and `False`
      otherwise.
  Returns:
    network_fn: A function that applies the model to a batch of images. It has
      the following signature:
          net, end_points = network_fn(images)
      The `images` input is a tensor of shape [batch_size, height, width, 3]
      with height = width = network_fn.default_image_size. (The permissibility
      and treatment of other sizes depends on the network_fn.)
      The returned `end_points` are a dictionary of intermediate activations.
      The returned `net` is the topmost layer, depending on `num_classes`:
      If `num_classes` was a non-zero integer, `net` is a logits tensor
      of shape [batch_size, num_classes].
      If `num_classes` was 0 or `None`, `net` is a tensor with the input
      to the logits layer of shape [batch_size, 1, 1, num_features] or
      [batch_size, num_features]. Dropout has not been applied to this
      (even if the network's original classification does); it remains for
      the caller to do this or not.
  Raises:
    ValueError: If network `name` is not recognized.
  """
  @functools.wraps(dvolver_build_nasnet)
  def network_fn(images, **kwargs):
    arg_scope = get_nasnet_arg_scope(model_type)(weight_decay=weight_decay)
    with slim.arg_scope(arg_scope):
      return dvolver_build_nasnet(model_type, params, images, num_classes, is_training=is_training, **kwargs)

  return network_fn


def get_model_fn(rep_params, model_params, enable_summary):
  """
  Returns a function that will build the NASNet-A model
  Args:
    * rep_params: list of parameters with the following representation:
              B = 5 # number of blocks per cell
              bi = block i
              h0 = hidden state on the left for the block
              h1 = hidden state on the right for the block
              op0 = left operation for the block
              op1 = right operation for the block
              combination = always ADD
              individual representation: [# normal cell
                                          b0.h0, b0.op0, b0.h1, b0.op1,
                                          b1.h0, b1.op0, b1.h1, b1.op1,
                                          b2.h0, b2.op0, b2.h1, b2.op1,
                                          b3.h0, b3.op0, b3.h1, b3.op1,
                                          b4.h0, b4.op0, b4.h1, b4.op1,
                                          # reduction cell
                                          b0.h0, b0.op0, b0.h1, b0.op1,
                                          b1.h0, b1.op0, b1.h1, b1.op1,
                                          b2.h0, b2.op0, b2.h1, b2.op1,
                                          b3.h0, b3.op0, b3.h1, b3.op1,
                                          b4.h0, b4.op0, b4.h1, b4.op1,
                                         ]

  """

  optimizer_fn = get_optimizer_fn(model_params, rep_params)

  weight_decay = model_params.weight_decay
  nb_classes = model_params.nb_classes
  data_format = model_params.data_format
  label_smoothing = model_params.label_smoothing
  aux_weight = model_params.aux_weight
  clip_gradient_norm = model_params.clip_gradient_norm
  train_mode = TrainMode(model_params.train_mode)
  total_training_steps = model_params.max_steps
  model_type = model_params.model_type

  config = dvolver_nas_config(model_params)

  def inference(features, mode):
    """Pure inference function"""

    network_fn = get_network_fn(model_type,
                                rep_params,
                                num_classes=nb_classes,
                                weight_decay=weight_decay,
                                is_training=(mode == tf.estimator.ModeKeys.TRAIN))

    inputs = features

    # if data_format == 'channels_first'
    # Reshape X to 4-D tensor: [batch_size, channels, height, width]
    # if data_format == 'channels_last'
    # Reshape X to 4-D tensor: [batch_size, height, width, channels]

    if data_format == 'channels_last':
      inputs = tf.transpose(inputs, [0, 2, 3, 1], name='transposeNCHW-NHWC')
      config.set_hparam('data_format', 'NHWC')

    logits, end_points = network_fn(inputs,
                                    config=config)

    return logits, end_points


  def inference_wrapper(features, mode):
    logits, _ = inference(features, mode)

    return logits


  def model_fn(features, labels, mode, params):
    """NASNet-A model body.
    Args:
      features: a list of tensors, one for each tower
      labels: a list of tensors, one for each tower
      mode: ModeKeys.TRAIN or EVAL
      params: Hyperparameters suitable for tuning
    Returns:
      A EstimatorSpec object.
    """

    optimizer = optimizer_fn()

    logits, end_points = inference(features, mode)

    # if enable_summary:
    #   tf.summary.image('input', tf.transpose(features, [0, 2, 3, 1]), max_outputs=10)

    # compute total_loss
    losses = []

    # get current scopes to avoid cross tower stuff in replicated multi gpu training
    current_var_scope = tf.get_variable_scope()

    # Auxiliary Loss
    with tf.name_scope('aux_loss'):
      if 'AuxLogits' in end_points:
        onehot_labels = tf.one_hot(labels, nb_classes)
        aux_loss = tf.losses.softmax_cross_entropy(logits=end_points['AuxLogits'],
                                                   onehot_labels=onehot_labels,
                                                   weights=aux_weight,
                                                   label_smoothing=label_smoothing)
        losses.append(aux_loss)
        if enable_summary:
          tf.summary.scalar('aux_loss', aux_loss)

    # Crosse entropy loss
    with tf.name_scope('train_loss'):
      cross_loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels, weights=1.0)
      if enable_summary:
        tf.summary.scalar('cross_loss', cross_loss)

    # regularization losses
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    if enable_summary:
      tf.summary.scalar('regularization_loss', tf.add_n(regularization_losses, name='regularization_losses'))

    losses += [cross_loss] + regularization_losses

    # final total loss
    total_loss = tf.add_n(losses, name='total_loss')

    # needed for batch norm
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    if model_params.moving_average_decay:
      moving_average_variables = slim.get_model_variables(current_var_scope)
      variable_averages = tf.train.ExponentialMovingAverage(model_params.moving_average_decay,
                                                            tf.train.get_global_step())
      update_ops.append(variable_averages.apply(moving_average_variables))

    with tf.control_dependencies(update_ops):
      #train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
      gradients_and_vars = optimizer.compute_gradients(total_loss,
                                                       var_list=tf.trainable_variables(current_var_scope.name))

      gradients = [gradient for gradient, var in gradients_and_vars]

      gradients, _ = tf.clip_by_global_norm(gradients, clip_gradient_norm)

      gradients_and_vars = [(gradient, x[1]) for gradient, x in zip(gradients, gradients_and_vars)]

      train_op = optimizer.apply_gradients(gradients_and_vars, global_step=tf.train.get_global_step())

    # Calculate root mean squared error as additional eval metric
    class_ids = math_ops.argmax(logits, 1, name="class_ids")
    class_ids = array_ops.expand_dims(class_ids, axis=(1,))
    weights = 1
    if enable_summary:
      eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=class_ids, weights=weights, name="accuracy"),
        "top 5": tf.metrics.recall_at_k(tf.cast(labels,tf.int64), logits, 5),
      }
    else:
      eval_metric_ops = {}

    # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=total_loss,
                                      train_op=train_op,
#                                      training_hooks=train_hooks,
                                      eval_metric_ops=eval_metric_ops)


  return inference_wrapper, model_fn
