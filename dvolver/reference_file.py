import pandas as pd
import tensorflow as tf

from .individual import create_individual

def read_reference_file(representation, filename):
    """
    Read CSV file with 1 line per reference point
    first column should have header: name
    Returns: list of reference individuals with archIndex = reference's name or None
    """

    if not filename or not tf.gfile.Exists(filename) or tf.gfile.IsDirectory(filename):
        return None

    with tf.gfile.GFile(filename, 'r') as f:
        df = pd.read_csv(f)

    names = df.name.values.tolist()
    archs = df.loc[:, df.columns != 'name'].values.tolist()

    archs = [representation.parse_arch(arch) for arch in archs]

    return [create_individual(arch, archIndex=name) for name, arch in zip(names, archs)]


def find_reference_file(representation, filename):
    """
    Returns path for reference file or None if it does not exists
    """
    for fname in [filename,
                  'data/'+representation+'/'+filename]:
        if tf.gfile.Exists(fname) and not tf.gfile.IsDirectory(fname):
            return fname

    return None
