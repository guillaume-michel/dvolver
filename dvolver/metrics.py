import tensorflow as tf

def get_float_operations(inference, shape):

    g = tf.Graph()
    run_meta = tf.RunMetadata()
    with g.as_default():

        output = inference(tf.random_normal(shape),
                           tf.estimator.ModeKeys.EVAL)

        opts = tf.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.profiler.profile(g, run_meta=run_meta, cmd='op', options=opts)

        return flops.total_float_ops


def get_parameters_count(inference, shape):

    g = tf.Graph()
    run_meta = tf.RunMetadata()
    with g.as_default():

        output = inference(tf.random_normal(shape),
                           tf.estimator.ModeKeys.EVAL)

        opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
        params_count = tf.profiler.profile(g, run_meta=run_meta, cmd='op', options=opts)

        return params_count.total_parameters

