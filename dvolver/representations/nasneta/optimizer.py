import tensorflow as tf
import sys
sys.path.append("../../../")
from deployment_util import TowerOptimizer

def get_optimizer_fn(model_params, params):

    def optimizer_fn():

        learning_rate = model_params.learning_rate
        momentum = model_params.momentum
        train_batch_size = model_params.train_batch_size
        num_epochs_per_decay = model_params.num_epochs_per_decay
        num_train_samples = model_params.num_train_samples

        decay_steps = int(num_train_samples / train_batch_size * num_epochs_per_decay)
        num_samples_per_epoch = model_params.num_train_samples

        global_step = tf.train.get_global_step()


        if model_params.learning_rate_decay_type == 'triangular':
            stepsize = int(num_samples_per_epoch/train_batch_size*num_epochs_per_decay/2)
            cycle = tf.floor((1 + global_step / (2 * stepsize)))
            x = tf.abs(tf.cast((global_step/stepsize - 2 * cycle + 1),tf.float32))
            min_lr = model_params.triangular_min_learning_rate
            max_lr = model_params.triangular_max_learning_rate
            learning_rate = min_lr + (max_lr - min_lr) * tf.maximum(0.0, 1.0 - x)



        elif model_params.learning_rate_decay_type == 'resnet_training':
            stepsize = int(num_samples_per_epoch/train_batch_size)*num_epochs_per_decay
            learning_rate = tf.cond(global_step > stepsize,
                                    lambda: 0.1*learning_rate,
                                    lambda: learning_rate)
            learning_rate = tf.cond(global_step > 2*stepsize,
                                    lambda: 0.1*learning_rate,
                                    lambda: learning_rate)

        elif model_params.learning_rate_decay_type == 'fastai_imagenet':
            epoch = int(num_samples_per_epoch/train_batch_size)

            learning_rate = tf.cond(global_step < 4*num_samples_per_epoch,
                                    lambda: model_params.learning_rate/(4-epoch),
                                    lambda: tf.cond(global_step < 28*num_samples_per_epoch,
                                                    lambda: model_params.learning_rate/1,
                                                    lambda: tf.cond(global_step < 47*num_samples_per_epoch,
                                                                    lambda: model_params.learning_rate/10,
                                                                    lambda: tf.cond(global_step < 57*num_samples_per_epoch,
                                                                                    lambda: model_params.learning_rate/100,
                                                                                    lambda: model_params.learning_rate/1000)
                                                                    )
                                                    )
                                    )


        elif model_params.learning_rate_decay_type == 'triangular2':

            stepsize = int(num_samples_per_epoch/train_batch_size)*num_epochs_per_decay
            cycle = tf.floor((1 + global_step / (2 * stepsize)))
            x = tf.abs(tf.cast((global_step/stepsize - 2 * cycle + 1),tf.float32))
            min_lr = model_params.triangular_min_learning_rate
            max_lr = model_params.triangular_max_learning_rate
            learning_rate = min_lr + (max_lr - min_lr) * tf.maximum(0.0, 1.0 - x)

            # x = tf.cond(global_step > 3*stepsize, lambda: 1.0, lambda: x)
            # learning_rate = tf.cond(global_step > 2*stepsize, lambda: 1e-9 + min_lr * tf.maximum(0.0, 1.0 - x), lambda: learning_rate)

            final_min_lr = model_params.final_min_learning_rate

            learning_rate = tf.cond(global_step > 2*stepsize,
                                    lambda: final_min_lr + (min_lr - final_min_lr) * tf.maximum(0.0, x),
                                    lambda: learning_rate)

            learning_rate = tf.cond(global_step > 3*stepsize,
                                    lambda: final_min_lr,
                                    lambda: learning_rate)

        elif model_params.learning_rate_decay_type == 'triangular2_LR_momentum':

            stepsize = int(num_samples_per_epoch/train_batch_size)*num_epochs_per_decay
            cycle = tf.floor((1 + global_step / (2 * stepsize)))
            x = tf.abs(tf.cast((global_step/stepsize - 2 * cycle + 1),tf.float32))
            min_lr = model_params.triangular_min_learning_rate
            max_lr = model_params.triangular_max_learning_rate
            learning_rate = min_lr + (max_lr - min_lr) * tf.maximum(0.0, 1.0 - x)

            final_min_lr = model_params.final_min_learning_rate

            learning_rate = tf.cond(global_step > 2*stepsize,
                                    lambda: final_min_lr + (min_lr - final_min_lr) * tf.maximum(0.0, x),
                                    lambda: learning_rate)
            learning_rate = tf.cond(global_step > 3*stepsize,
                                    lambda: final_min_lr,
                                    lambda: learning_rate)

            max_mm = model_params.triangular_max_momentum
            min_mm = model_params.triangular_min_momentum

            momentum = min_mm + (max_mm - min_mm) * tf.maximum(0.0, x)

            momentum = tf.cond(global_step > 2*stepsize,
                               lambda: max_mm,
                               lambda: momentum)

        elif model_params.learning_rate_decay_type == 'cosine':
            learning_rate = tf.train.cosine_decay(learning_rate,
                                                  global_step,
                                                  decay_steps,
                                                  name='cosine_decay_learning_rate')

        elif model_params.learning_rate_decay_type == 'exponential':
            learning_rate = tf.train.exponential_decay(learning_rate,
                                                       global_step,
                                                       decay_steps,
                                                       model_params.learning_rate_decay_factor,
                                                       staircase=True,
                                                       name='exponential_decay_learning_rate')
        else:
            raise ValueError('Unsupported learning rate decay type: ' + model_params.learning_rate_decay_type)

        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.scalar('momentum', momentum)


        if model_params.optimizer == 'momentum':
            return tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                              momentum=momentum)

        elif model_params.optimizer == 'rmsprop':
            return tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                             decay=model_params.rmsprop_decay,
                                             momentum=momentum,
                                             epsilon=model_params.epsilon)

        else:
            raise ValueError('Unsupported optimizer: ' + optimizer)

    # decorate optimizers by tower optimizer for multi gpu training,
    # which doesn't affect single gpu training
    return lambda : TowerOptimizer(optimizer_fn())
