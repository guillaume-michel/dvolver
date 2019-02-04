import numpy as np
import os
import glob
import random

import tensorflow as tf

from .search_method import *

def checkpoint_name(gen):
    return 'checkpoint.'+str(gen).zfill(7)+'.npy'


def get_checkpoint_paths(path):
    """
    Return an ordered list of checkpoint paths
    """

    def split_name(p):
        f, file_extension = os.path.splitext(p)
        _, file_extension = os.path.splitext(f)

        gen = int(file_extension[1:])

        return (gen, p)


    paths = tf.gfile.Glob(path+'/checkpoint.*[0-9].npy')

    return sorted(map(split_name, paths))


def get_latest_checkpoint(path):
    """
    return the tuple (latest_gen, latest_checkpoint_path) in the given path
    """
    paths = get_checkpoint_paths(path)

    if len(paths) == 0:
        return None

    return paths[-1]


def get_checkpoint_base_path(path):
    """
    Returns the base path for the checkpoints
    """
    return path + '/checkpoints'


def saveToFile(filename, o):
    """
    save numpy object o to file filename
    """
    with tf.gfile.GFile(filename, 'wb') as f:
        np.save(f, o)


def readFromFile(filename):
    """
    load numpy object from file filename
    """
    with tf.gfile.GFile(filename, 'rb') as f:
        return np.load(f)


def save_checkpoint(checkpoint_path,
                    search_method,
                    rndstate,
                    lastArchIndex,
                    generation,
                    samples,
                    history,
                    hypervolumes,
                    hof_hypervolumes,
                    logbook,
                    population,
                    hof,
                    cache):

    # Fill the dictionary using the dict(key=value[, ...]) constructor
    cp = dict(search_method=search_method,
              rndstate=random.getstate(),
              lastArchIndex=lastArchIndex,
              generation=generation,
              samples=samples,
              history=history,
              hypervolumes=hypervolumes,
              hof_hypervolumes=hof_hypervolumes,
              population=population,
              hof=hof,
              population_count=len(population),
              cache=cache)

    if logbook:
        cp['logbook'] = logbook

    # create parent directory if needed
    checkpoint_dir = os.path.dirname(checkpoint_path)

    if not tf.gfile.Exists(checkpoint_dir):
        tf.gfile.MakeDirs(checkpoint_dir)

    saveToFile(checkpoint_path, cp)


def restore_checkpoint(filename):
    return readFromFile(filename).item()


def restore_latest_checkpoint(log_dir):
    checkpoint_base_path = get_checkpoint_base_path(log_dir)
    latest_checkpoint = get_latest_checkpoint(checkpoint_base_path)

    if latest_checkpoint:
        return restore_checkpoint(latest_checkpoint[1])

    return None


def read_external_cache_file(filename):

    if not filename or not tf.gfile.Exists(filename):
        return {}

    return readFromFile(filename).item()


def merge_caches(cache1, cache2):
    """merge cache1 and cache2. cache2 entries override entries in cache1"""
    return {**cache1, **cache2}
