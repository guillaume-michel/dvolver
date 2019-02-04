import os
import time
import threading
import importlib

import numpy as np
import tensorflow as tf

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import session_run_hook
from deployment_util import replicate_model_fn, replicate_input_fn

from dvolver import *

def get_num_gpus():
  """
  return the number of GPU defined in CUDA_VISIBLE_DEVICES
  """
  gpus = os.environ.get('CUDA_VISIBLE_DEVICES')

  if gpus == None:
    return 0

  return len(list(filter(lambda x: int(x)>=0,
                         gpus.split(','))))


class ExamplesPerSecondHook(session_run_hook.SessionRunHook):
  """Hook to print out examples per second.
    Total time is tracked and then divided by the total number of steps
    to get the average step time and then batch_size is used to determine
    the running average of examples per second. The examples per second for the
    most recent interval is also logged.
  """

  def __init__(self,
               batch_size,
               every_n_steps=100,
               every_n_secs=None):
    """Initializer for ExamplesPerSecondHook.
      Args:
      batch_size: Total batch size used to calculate examples/second from
      global time.
      every_n_steps: Log stats every n steps.
      every_n_secs: Log stats every n seconds.
    """
    if (every_n_steps is None) == (every_n_secs is None):
      raise ValueError('exactly one of every_n_steps and every_n_secs should be provided.')

    self._timer = basic_session_run_hooks.SecondOrStepTimer(every_steps=every_n_steps,
                                                            every_secs=every_n_secs)

    self._step_train_time = 0
    self._total_steps = 0
    self._batch_size = batch_size

  def begin(self):
    self._global_step_tensor = tf.train.get_global_step()

    if self._global_step_tensor is None:
      raise RuntimeError('Global step should be created to use StepCounterHook.')

  def before_run(self, run_context):  # pylint: disable=unused-argument
    return basic_session_run_hooks.SessionRunArgs(self._global_step_tensor)

  def after_run(self, run_context, run_values):
    _ = run_context

    global_step = run_values.results

    if self._timer.should_trigger_for_step(global_step):
      elapsed_time, elapsed_steps = self._timer.update_last_triggered_step(global_step)

      if elapsed_time is not None:
        steps_per_sec = elapsed_steps / elapsed_time
        self._step_train_time += elapsed_time
        self._total_steps += elapsed_steps

        average_examples_per_sec = self._batch_size * (self._total_steps / self._step_train_time)
        current_examples_per_sec = steps_per_sec * self._batch_size

        # Average examples/sec followed by current examples/sec
        #print('Average examples/sec: {} ({}), step = {}'.format(int(average_examples_per_sec),
        #                                                        int(current_examples_per_sec),
        #                                                        self._total_steps))
        logging.info('%s: %g (%g), step = %g', 'Average examples/sec',
                     average_examples_per_sec, current_examples_per_sec,
                     self._total_steps)



def make_objective(args, train_list, test_list, fresh_train=True):

    # some tuning to be a good citizen
    session_config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=args.log_device_placement,
                                    intra_op_parallelism_threads=0,
                                    gpu_options=tf.GPUOptions(force_gpu_compatible=True,
                                                              allow_growth=True))

    # Set model params
    model_params = args

    representation_name = model_params.representation_name

    try:
        print('Loading representation:', representation_name)
        representation = importlib.import_module(representation_name)

    except ImportError:
        print('Failed to find representation:', representation_name)
        exit()

    input_pipeline_name = model_params.input_pipeline_name

    try:
        print('Loading input pipeline:', input_pipeline_name)
        input_pipeline = importlib.import_module(input_pipeline_name)

    except ImportError:
        print('Failed to find input pipeline:', input_pipeline_name)
        exit()

    def objective(params):

        archIndex = params.archIndex
        gen = params.gen

        model_dir = args.job_dir+'/'+get_dir_name_for_train_mode(model_params.train_mode)+'/' +'gen_'+(str(gen).zfill(4))+'/'+(str(archIndex).zfill(7))

        if fresh_train and tf.gfile.Exists(model_dir):
          # make sure the directory to store model does not exists
          tf.gfile.DeleteRecursively(model_dir)

        num_gpus = get_num_gpus()
        distribution = num_gpus >= 2
        enable_summary = True

        if distribution:
          print('Multi GPU training with ', num_gpus, 'GPUs')
        else:
          print('Local training')

        # see RunConfig options
        if args.save_checkpoints_secs:
          print("save_checkpoint_steps will be ignored")
          run_config = tf.estimator.RunConfig(model_dir=model_dir,
                                              session_config=session_config,
                                              log_step_count_steps=args.log_step_count_steps,
                                              save_summary_steps=args.save_summary_steps,
                                              save_checkpoints_secs=args.save_checkpoints_secs,
                                              keep_checkpoint_every_n_hours=args.keep_checkpoint_every_n_hours,
                                              keep_checkpoint_max=args.keep_checkpoint_max)
        else:
          run_config = tf.estimator.RunConfig(model_dir=model_dir,
                                              session_config=session_config,
                                              log_step_count_steps=args.log_step_count_steps,
                                              save_summary_steps=args.save_summary_steps,
                                              save_checkpoints_steps=args.save_checkpoints_steps,
                                              keep_checkpoint_every_n_hours=args.keep_checkpoint_every_n_hours,
                                              keep_checkpoint_max=args.keep_checkpoint_max)

        # create a function for the inference graph and a model_fn for the complete training process
        inference, model_fn = representation.get_model_fn(params, model_params, enable_summary)

        # decorate model_fn for multi_gpu training
        model_fn = replicate_model_fn(model_fn)

        examples_sec_hook = ExamplesPerSecondHook(batch_size=args.train_batch_size,
                                                  every_n_steps=run_config.log_step_count_steps)

        train_hooks = []
        train_hooks += [examples_sec_hook] # WARNING: early_stop_hook should be after saver_hook

        # Instantiate Estimator
        classifier = tf.estimator.Estimator(model_fn=model_fn,
                                            config=run_config,
                                            params=model_params)

        input_fn = lambda: input_pipeline.get_input_fn(train_list,
                                                       batch_size=args.train_batch_size,
                                                       num_threads=args.preproc_threads,
                                                       perform_shuffle=True,
                                                       perform_augmentation=True,
                                                       per_image_standardization=True,
                                                       enable_cutout=model_params.enable_cutout,
                                                       repeat_count=-1)
        # decorate input_fn for prefetching to GPUs
        # input_fn = replicate_input_fn(input_fn)

        train_spec = tf.estimator.TrainSpec(input_fn,
                                            max_steps=args.max_steps,
                                            hooks=train_hooks)

        eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_pipeline.get_input_fn(test_list,
                                                                                       batch_size=args.eval_batch_size,
                                                                                       num_threads=args.preproc_threads,
                                                                                       perform_shuffle=False,
                                                                                       perform_augmentation=False,
                                                                                       per_image_standardization=True,
                                                                                       repeat_count=1), # only goes through validation set once
                                          steps=None, # eval until the end of validation dataset
                                          start_delay_secs=0,
                                          throttle_secs=model_params.throttle_secs)

        if TrainMode(model_params.train_mode) == TrainMode.SEARCH:
          classifier.train(input_fn=train_spec.input_fn,
                           hooks=train_spec.hooks,
                           max_steps=train_spec.max_steps)
        elif TrainMode(model_params.train_mode) == TrainMode.FULL:
          tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
        else:
          raise ValueError('Unsupported train_mode:', model_params.train_mode)

        # eval loop on all checkpoints in model_dir
        if TrainMode(model_params.train_mode) == TrainMode.FULL:
          checkpoints_paths = [filename.replace('.index','')
                               for filename in tf.gfile.Glob(os.path.join(run_config.model_dir,
                                                           "*ckpt-*.index"))]
          checkpoints_paths = sorted(checkpoints_paths,
                                     key=lambda x: int(os.path.basename(x).replace("model.ckpt-","")))

          for path in checkpoints_paths:
            results = classifier.evaluate(input_fn=eval_spec.input_fn, checkpoint_path=path, name="final_eval")
        else:
          results = classifier.evaluate(input_fn=eval_spec.input_fn)
        flops = get_float_operations(inference, input_pipeline.get_default_inference_shape(batch_size=1, data_format=args.data_format))

        # arbitrary value
        # 1GHz FMA no vectorization
        ref_flops = 2e9

        speed = ref_flops/flops

        accuracy = float(results['accuracy'])

        return (speed, accuracy)

    return objective
