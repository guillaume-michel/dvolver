#!/usr/bin/python3
"""
Fully train a single architecture defined by a csv file and a specific representation
"""
import argparse
import numpy as np
import pandas as pd

import time
import random
import logging
import threading
import json
import importlib
import itertools

from tqdm import tqdm
tqdm.monitor_interval = 0  # see https://github.com/tqdm/tqdm/issues/481

import tensorflow as tf

from objective import make_objective

from dvolver import *

tf.logging.set_verbosity(tf.logging.DEBUG)

class WorkerArgs():
    pass


def getWorkerArgs(d):
    worker_args = WorkerArgs()

    for key,value in d.items():
        setattr(worker_args, key, value)

    setattr(worker_args, 'num_train_samples', 0)
    setattr(worker_args, 'num_test_samples', 0)

    return worker_args


def get_objective_function(worker_args):

    train_mode = TrainMode(worker_args.train_mode)

    if train_mode == TrainMode.SEARCH:
        TRAIN_LIST, worker_args.num_train_samples, VALIDATION_LIST, worker_args.num_test_samples = input_pipeline.get_search_mode_files(worker_args.data_dir)

    elif train_mode == TrainMode.FULL:
        TRAIN_LIST, worker_args.num_train_samples, VALIDATION_LIST, worker_args.num_test_samples = input_pipeline.get_full_mode_files(worker_args.data_dir)

    else:
        raise ValueError('Unsupported TrainMode' + train_mode)

    return make_objective(worker_args, train_list=TRAIN_LIST, test_list=VALIDATION_LIST, fresh_train=False)


def train(arch_file,
          worker_args,
          verbose=False):

    objectiveFunc = get_objective_function(worker_args)

    archs = read_reference_file(representation, arch_file)

    if len(archs) != 1:
        raise ValueError('Multiple architectures per arch_file is not yet supported')

    x = archs[0]
    x.gen = 'full'

    fitness = objectiveFunc(x)

    x.fitness.values = fitness

    print(individual_to_str(x))


def main(args):

    verbose = args.verbose
    arch_file = args.arch_file

    worker_args = representation.add_worker_args(args, {
        'data_dir': args.data_dir,
        'job_dir': args.job_dir,
        'nb_classes': input_pipeline.NB_CLASSES,
        'data_format': args.data_format,
        'max_steps': args.max_steps,
        'train_batch_size': args.train_batch_size,
        'eval_batch_size': args.eval_batch_size,
        'learning_rate': args.learning_rate,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'enable_cutout': (not args.disable_cutout),
        'log_device_placement': args.log_device_placement,
        'save_summary_steps': args.save_summary_steps,
        'keep_checkpoint_every_n_hours' : args.keep_checkpoint_every_n_hours,
        'keep_checkpoint_max' : args.keep_checkpoint_max,
        'log_step_count_steps': args.log_step_count_steps,
        'save_checkpoints_steps': args.save_checkpoints_steps,
        'save_checkpoints_secs': args.save_checkpoints_secs,
        'preproc_threads': args.preproc_threads,
        'representation_name': 'dvolver.representations.' + args.representation,
        'train_mode': str(TrainMode.FULL),
        'throttle_secs': args.throttle_secs,
        'input_pipeline_name': 'input_pipeline.' + args.input_pipeline
    })


    worker_args = getWorkerArgs(worker_args)

    train(arch_file=arch_file,
          worker_args=worker_args,
          verbose=verbose)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--verbose', action='store_true', default=False, help='Enable verbose mode.')

    parser.add_argument('--data-dir', type=str, required=True, help='base directory where CIFAR-10 tfrecords are.')
    parser.add_argument('--job-dir', type=str, required=True, help='The directory where the models will be stored.')
    parser.add_argument('--data-format', type=str, default='channels_first', help='Image format to use.')
    parser.add_argument('--max-steps', type=int, default=937500, help='The number of steps to use for training.')
    parser.add_argument('--train-batch-size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--eval-batch-size', type=int, default=25, help='Batch size for validation.')
    parser.add_argument('--learning-rate', type=float, default=0.025, help="This is the inital learning rate value.")
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for MomentumOptimizer.')
    parser.add_argument('--weight-decay', type=float, default=4e-5, help='Weight decay for convolutions.')
    parser.add_argument('--disable-cutout', action='store_true', default=False, help='Whether to disable cutout in data augmentation.')
    parser.add_argument('--log-device-placement', action='store_true', default=False, help='Whether to log device placement.')
    parser.add_argument('--log-step-count-steps', type=int, default=1000, help='The number of steps to wait between each logs.')
    parser.add_argument('--save-checkpoints-steps', type=int, default=18750, help='The number of steps between each checkpoint and evaluation.')
    parser.add_argument('--save-checkpoints-secs', type=int, default=None, help='The number of secs between each checkpoint and evaluation.')
    parser.add_argument('--save-summary-steps', type=int, default=18750, help='The number of steps between each summary.')
    parser.add_argument('--keep-checkpoint-every-n-hours', type=float, default=1., help='frequency of kept checkpoints (not deleted by checkpoint_keep_max)')
    parser.add_argument('--keep-checkpoint-max', type=int, default=5, help="maximum number of checkpoints to keep")
    parser.add_argument('--throttle_secs', type=int, default=1800, help='Minimal duration between sucessive evaluations in seconds')
    parser.add_argument('--preproc-threads', type=int, default=4, help='The number of dedicated threads for preprocessing.')
    parser.add_argument('--representation', type=str, default='nasneta', help='choice of representation and search space.')
    parser.add_argument('--arch-file', type=str, required=True, help='Architecture csv file')
    parser.add_argument('--input-pipeline', type=str, default='cifar10', help='input pipeline to test.')

    args, _ = parser.parse_known_args()

    representation_name = 'dvolver.representations.' + args.representation

    try:
        print('Loading representation:', representation_name)
        representation = importlib.import_module(representation_name)

    except ImportError:
        print('Failed to find representation:', representation_name)
        exit()

    # load specific arguments for current representation
    representation.add_argument(parser, TrainMode.FULL)
    args = parser.parse_args()

    input_pipeline_name = 'input_pipeline.' + args.input_pipeline

    try:
        print('Loading input pipeline:', input_pipeline_name)
        input_pipeline = importlib.import_module(input_pipeline_name)

    except ImportError:
        print('Failed to find input pipeline:', input_pipeline_name)
        exit()

    args.arch_file = find_reference_file(args.representation, args.arch_file)

    print('Command line arguments:')
    for arg in sorted(vars(args)):
        print('\t', arg+':', getattr(args, arg))

    main(args)
