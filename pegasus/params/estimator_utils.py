# Copyright 2020 The PEGASUS Authors..
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Library for working with tf estimators."""

import collections
import re

from absl import logging
import numpy as np
from pegasus.ops import public_parsing_ops
from pegasus.eval.rouge_tensors import evaluate_r1, evaluate_r2, evaluate_rl
from tensor2tensor.utils import adafactor
import tensorflow as tf

from tensorflow.contrib import summary as contrib_summary
from tensorflow.contrib import tpu as contrib_tpu
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer


def create_estimator(master,  # local tensorflow server
                     model_dir,  # directory pointing to model checkpoints (e.g.ckpt/pegasus_ckpt/cnn_dailymail)
                     use_tpu,  # false by default
                     iterations_per_loop,  # 1000 by default
                     num_shards,  # 1 by default
                     model_params,  # should be the name of the model we want (defined above - e.g. cnn_dailymail_transformer)
                     include_features_in_predictions=True,
                     decode_keys=(),
                     train_init_checkpoint=None,   # the pre-trained model checkpoint (e.g. ckpt/pegasus_ckpt/model.ckpt-1500000)
                     train_warmup_steps=10000,  # number of steps to warm up, 10000 by default
                     save_checkpoints_steps=1000,  # number of steps to save ckpt, 1000 by default
                     keep_checkpoint_max=5):  # number of recent ckpt to keep (older deleted), 5 by default
  """Returns an tensorflow estimator."""

  # This is the runtime config for tensorflow estimators
  run_config = tpu_config.RunConfig(
      master=master,  # local tensorflow server
      model_dir=model_dir,  # directory pointing to model checkpoints (e.g.ckpt/pegasus_ckpt/cnn_dailymail)
      session_config=tf.ConfigProto(
          allow_soft_placement=True, log_device_placement=False),  # some session config???
      tpu_config=tpu_config.TPUConfig(iterations_per_loop),  # another estimator config - set to 1000
      save_checkpoints_steps=save_checkpoints_steps,  # number of steps to save ckpt, 1000 by default
      keep_checkpoint_max=keep_checkpoint_max)  # number of recent ckpt to keep (older deleted), 5 by default

  # It will return the tensorflow estimator created as we would want it to be
  return tpu_estimator.TPUEstimator(
      # This is to instantiate the estimator model function with the following params:
      # use_tpu = False
      # model_params = name of model (e.g. cnn_dailymail_transformer) - points to pegasus/public_params.py
      # model_dir = directory pointing to model checkpoints (e.g.ckpt/pegasus_ckpt/cnn_dailymail)
      # include_features_preds = True
      # decode_keys = ()
      # train_init_checkpoint = ckpt/pegasus_ckpt/model.ckpt-1500000
      # train_warmup_steps = 10000
      model_fn=_estimator_model_fn(use_tpu, model_params, model_dir,
                                   include_features_in_predictions, decode_keys,
                                   train_init_checkpoint, train_warmup_steps),
      use_tpu=use_tpu,  # false
      train_batch_size=model_params.batch_size * num_shards,  # batch_size * 1 by default
      eval_batch_size=model_params.batch_size * num_shards,  # batch_size * 1 by default
      predict_batch_size=model_params.batch_size * num_shards,  # batch_size * 1 by default
      config=run_config)  # the runtime config as defined above


def _estimator_model_fn(use_tpu, model_params, model_dir,
                        include_features_in_predictions, decode_keys,
                        train_init_checkpoint, train_warmup_steps):
  """Returns an estimator model function."""

  # The model input function points to infeed.get_input_fn() which during training
  # defines the mode to be TRAIN
  def model_fn(features, labels, mode, config, params):
    """Estimator model function."""

    # Not sure why it does this?
    del labels
    del config
    del params

    tf.get_variable_scope().set_initializer(
        tf.variance_scaling_initializer(
            1.0, mode="fan_avg", distribution="uniform"))

    # PREDICTION (e.g. evaluate)
    if mode == tf.estimator.ModeKeys.PREDICT:
      predictions = model_params.estimator_prediction_fn(features)

      if include_features_in_predictions:
        predictions.update(features)

      if decode_keys:
        # Decode the raw ids into strings in prediction.
        def decode_host_call(tensor_dict):
          for key in decode_keys:
            predictions[key] = public_parsing_ops.decode(
                tensor_dict[key], model_params.vocab_filename,
                model_params.encoder_type)
          return tensor_dict

        contrib_tpu.outside_compilation(decode_host_call, predictions)
      return tpu_estimator.TPUEstimatorSpec(mode=mode, predictions=predictions)

    # TRAINING
    # I think that this is where the model will calculate the loss and the outputs
    # at the END of the training steps? I would think it would do it for the number
    # of training steps as a for loop?
    training = mode == tf.estimator.ModeKeys.TRAIN
    # use_tpu is false by default so this skips
    if use_tpu and model_params.use_bfloat16:
      with contrib_tpu.bfloat16_scope():
        loss, outputs = model_params.model()(features, training)
    else:
      XENT_loss, outputs = model_params.model()(features, training)
      # outputs = model_params.model()(features, training)

    # TPU requires outputs all have batch dimension and doesn't handle scalar.
    # Tile all scalars to 1 dimension vector.
    outputs = _tile_scalar_to_batch_size(outputs, model_params.batch_size)

    if mode == tf.estimator.ModeKeys.TRAIN:
      init_lr = model_params.learning_rate
      global_step = tf.train.get_global_step()
      lr = init_lr / 0.01 * tf.rsqrt(
          tf.maximum(tf.to_float(global_step), 10000))
      if train_init_checkpoint:
        lr = tf.minimum(
            tf.to_float(global_step + 1) / train_warmup_steps * init_lr, lr)

      optimizer = adafactor.AdafactorOptimizer(
          learning_rate=lr,
          decay_rate=adafactor.adafactor_decay_rate_pow(0.8),
          beta1=0.0)
      if use_tpu:
        optimizer = tpu_optimizer.CrossShardOptimizer(optimizer)

      ##########################################################################################

      # Normalise logits to log-prob, and compute Gumbel samples with location
      logit_probs = tf.math.softmax(outputs["logits"])  # should not be x <= 0
      clipped_logit_probs = tf.clip_by_value(logit_probs, 1e-8, 1.0)
      logp = tf.log(clipped_logit_probs)

      # ARGMAX
      argmax_logp_index = tf.math.argmax(logp, axis=2)  # Returns indexes where logp is max
      # SOFTMAX - 'soft' labels of the Gumbel samples, and their one-hot labels
      u = tf.random_uniform(shape=outputs["one_hot_targets"].get_shape().as_list(), minval=0,
                            maxval=1, dtype=tf.float32)
      z = tf.math.add(-tf.log(-tf.log(u)), logp)
      y_soft = tf.math.softmax(tf.div(z, 0.1))  # tau = 0.1
      sample_y = tf.math.argmax(y_soft, axis=2)  # argmax along the vocab dimension

      # Decode text for ROUGE
      decode_target_text_tensor = public_parsing_ops.decode(outputs["targets"],
                                                            model_params.vocab_filename,
                                                            model_params.encoder_type)
      decode_target_text = decode_target_text_tensor[0]  # returned tensor in bytes format

      decode_preds_text_tensor_hard = public_parsing_ops.decode(argmax_logp_index,
                                                                model_params.vocab_filename,
                                                                model_params.encoder_type)
      decode_preds_text_hard = decode_preds_text_tensor_hard[0]  # returned tensor in bytes format

      # do not want to propagate the gradient through the ROUGE hook
      decode_target_text = tf.stop_gradient(decode_target_text)
      decode_preds_text_hard = tf.stop_gradient(decode_preds_text_hard)

      # calculate ROUGE-1
      r1_score_hard = tf.py_function(evaluate_r1, (decode_target_text, decode_preds_text_hard),
                                     tf.float32)

      # Implement REINFORCE loss w/ ARGMAX
      # Create index tensors to stack and get corresponding probabilities from logp
      # argmax_logp_new = tf.reshape(argmax_logp_index, [argmax_logp_index.get_shape().as_list()[
      # 1]])
      sequence_index = tf.constant(np.arange(0, 32))  # DYNAMIC: seq_len, not 32
      batch_index = tf.constant(np.zeros(sequence_index.get_shape().as_list()[0]), dtype=tf.int64)

      # index_tensor_hard = tf.stack([batch_index, sequence_index, argmax_logp_new], axis=1)
      # argmax_logp = tf.gather_nd(logp, index_tensor_hard)  # finds log probs using hard indexing

      # Implement REINFORCE loss w/ SOFTMAX
      decode_preds_text_tensor_soft = public_parsing_ops.decode(sample_y,
                                                                model_params.vocab_filename,
                                                                model_params.encoder_type)
      decode_preds_text_soft = decode_preds_text_tensor_soft[0]
      decode_preds_text_soft = tf.stop_gradient(decode_preds_text_soft)

      r1_score_soft = tf.py_function(evaluate_r1, (decode_target_text, decode_preds_text_soft),
                                     tf.float32)

      sample_y_new = tf.reshape(sample_y, [sample_y.get_shape().as_list()[1]])
      index_tensor_soft = tf.stack([batch_index, sequence_index, sample_y_new], axis=1)
      softmax_logp = tf.gather_nd(logp, index_tensor_soft)  # finds log probs using soft indexing

      # New LOSS calculation
      # weight the logp by ROUGE score, sum values, and invert sign (of logp)
      # soft_reinforce_loss = tf.reduce_sum(tf.multiply(r1_score_soft, -softmax_logp))
      # hard_reinforce_loss = tf.reduce_sum(tf.multiply(r1_score_hard, -argmax_logp))

      # combined_loss = tf.math.add(tf.multiply(tf.constant(0.8, dtype=tf.float32), XENT_loss),
      #                             tf.multiply(tf.constant(0.2, dtype=tf.float32), reinforce_loss))

      # Socher (2017) - r(y_hat) - r(y_s) needs to be swapped as we have loss, not reward
      loss_difference = tf.subtract(r1_score_soft, r1_score_hard)
      reinforce_baseline = tf.reduce_sum(tf.multiply(loss_difference, softmax_logp))

      # RELAX loss

      ##########################################################################################

      # Accessing the gradient of loss
      list_of_gradient_variable_pairs = optimizer.compute_gradients(reinforce_baseline)
      train_op = optimizer.apply_gradients(list_of_gradient_variable_pairs,
                                           global_step=global_step)

      # train_op = optimizer.minimize(loss, global_step=global_step)

      tf.logging.set_verbosity(tf.logging.INFO)
      # Debugging steps - add into logging hook directly if needed
      # tf.debugging.check_numerics(sum_logp, "DEBUG: sum_logp has a NaN")

      logging_hook = tf.train.LoggingTensorHook({"loss": reinforce_baseline,  # or loss
                                                 # "hard_reinforce_loss": hard_reinforce_loss,
                                                 # "soft_reinforce_loss": soft_reinforce_loss,
                                                 "XENT_loss": XENT_loss,
                                                 "target_text": decode_target_text,
                                                 "soft_preds_text": decode_preds_text_soft,
                                                 "hard_preds_text": decode_preds_text_hard,
                                                 "hard_rouge_score": r1_score_hard,
                                                 "soft_rouge_score": r1_score_soft
                                                 }, every_n_iter=5)

      # This is the configured estimator function that is returned to train the model
      return tpu_estimator.TPUEstimatorSpec(
          mode=mode,
          loss=reinforce_baseline,  # change loss here
          train_op=train_op,
          training_hooks=[logging_hook],
          scaffold_fn=_load_vars_from_checkpoint(use_tpu,
                                                 train_init_checkpoint),
          host_call=add_scalars_to_summary(model_dir, {"learning_rate": lr}))

    # EVALUATION (evaluating the performance)
    if mode == tf.estimator.ModeKeys.EVAL:
      eval_metrics = model_params.estimator_eval_metrics_fn(features, outputs)
      return tpu_estimator.TPUEstimatorSpec(
          mode=mode, loss=loss, eval_metrics=eval_metrics)

  return model_fn


def _tile_scalar_to_batch_size(tensor_dict, batch_size):
  """Tile scalar tensors in the dictionary to have batch dimension."""
  # bool(tf.constant(1).shape) = True.
  # this is inconsistent with python default length test. Disable pylint.
  scalar_keys = [k for k, v in tensor_dict.items() if len(v.shape) == 0]  # pylint: disable=g-explicit-length-test
  tiled_dict = {}
  for k, v in tensor_dict.items():
    if k in scalar_keys:
      logging.info("Expand scalar to vector: %s", k)
      v = tf.tile(tf.reshape(v, [1]), [batch_size])
    tiled_dict[k] = v
  return tiled_dict


def add_scalars_to_summary(summary_dir, scalar_tensors_dict):
  """Creates a host_call function that writes summaries on TPU."""

  #  All tensors outfed from TPU should preserve batch size dimension.
  scalar_tensors_dict = {
      k: tf.reshape(v, [1]) for k, v in scalar_tensors_dict.items()
  }

  def host_call_fn(**kwargs):
    writer = contrib_summary.create_file_writer(summary_dir, max_queue=1000)
    always_record = contrib_summary.always_record_summaries()
    with writer.as_default(), always_record:
      for name, scalar in kwargs.items():
        contrib_summary.scalar(name, tf.reduce_mean(scalar))
      return contrib_summary.all_summary_ops()

  return host_call_fn, scalar_tensors_dict


def _load_vars_from_checkpoint(use_tpu, init_checkpoint):
  """load variables from initial checkpoints.

  Args:
    use_tpu: bool whether to use tpu.
    init_checkpoint: path of checkpoint containing variables to be initialized.

  Returns:
    scaffold_fn: The scaffold_fn used by tpu estimator spec. If use_tpu=False,
    this is set to None.
  """
  if not init_checkpoint:
    return None
  tvars = tf.trainable_variables()
  (assignment_map,
   initialized_variable_names) = get_assignment_map_from_checkpoint(
       tvars, init_checkpoint)
  if not initialized_variable_names:
    raise ValueError("No matching variables in init_checkpoint. "
                     "Double check the naming in both models.")
  scaffold_fn = None
  if use_tpu:

    def tpu_scaffold():
      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
      return tf.train.Scaffold()

    scaffold_fn = tpu_scaffold
  else:
    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

  logging.info("**** Trainable Variables ****")
  for var in tvars:
    init_string = ""
    if var.name in initialized_variable_names:
      init_string = ", *INIT_FROM_CKPT*"
    logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)
  return scaffold_fn


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
  """Compute the union of the current variables and checkpoint variables."""
  assignment_map = {}
  initialized_variable_names = {}

  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    name_to_variable[name] = var

  init_vars = tf.train.list_variables(init_checkpoint)

  assignment_map = collections.OrderedDict()
  for x in init_vars:
    (name, var) = (x[0], x[1])
    if name not in name_to_variable:
      continue
    assignment_map[name] = name
    initialized_variable_names[name] = 1
    initialized_variable_names[name + ":0"] = 1

  return (assignment_map, initialized_variable_names)
