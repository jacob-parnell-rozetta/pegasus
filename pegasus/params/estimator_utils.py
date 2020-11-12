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
from pegasus.models.control_variate import ffn_baseline, control_variate, Q_func
from pegasus.methods.decoder_sampling import iid_sampling, beam_sampling, non_beam_sampling
from pegasus.methods.reinforce import rouge_decoding, iid_log_probs
from pegasus.methods.risk import risk_loss
from pegasus.methods.relax import create_variables, create_cv_target, create_variables_from_samples
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
      predictions, _, _ = model_params.estimator_prediction_fn(features)

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
    training = mode == tf.estimator.ModeKeys.TRAIN
    # use_tpu is false by default so this skips
    if use_tpu and model_params.use_bfloat16:
      with contrib_tpu.bfloat16_scope():
        loss, outputs = model_params.model()(features, training)
    else:
      XENT_loss, outputs = model_params.model()(features, training)

    # TPU requires outputs all have batch dimension and doesn't handle scalar.
    # Tile all scalars to 1 dimension vector.
    outputs = _tile_scalar_to_batch_size(outputs, model_params.batch_size)

    # Create optimizer and define learning rate
    if mode == tf.estimator.ModeKeys.TRAIN:
      init_lr = model_params.learning_rate
      global_step = tf.train.get_global_step()
      lr = init_lr / 0.01 * tf.rsqrt(
          tf.maximum(tf.to_float(global_step), 10000))
      if train_init_checkpoint:
          lr = tf.minimum(tf.to_float(global_step + 1) / train_warmup_steps * init_lr, lr)

      optimizer = adafactor.AdafactorOptimizer(
          learning_rate=lr,
          decay_rate=adafactor.adafactor_decay_rate_pow(0.8),
          beta1=0.0)
      if use_tpu:
          optimizer = tpu_optimizer.CrossShardOptimizer(optimizer)

      ###############################################################################################################
      ##### VARIABLES ###############################################################################################
      # Create index tensors to stack and get corresponding probabilities from logp
      # max_seq_len = outputs["targets"].get_shape().as_list()[1]
      # sequence_index = tf.constant(np.arange(0, max_seq_len))
      # batch_index = tf.constant(np.zeros(sequence_index.get_shape().as_list()[0]), dtype=tf.int64)

      ##### I.I.D SAMPLING ##########################################################################################
      # Normalise logits to log-prob, and compute Gumbel samples with location
      # logit_probs = tf.math.softmax(outputs["logits"])  # should not be x <= 0
      # clipped_logit_probs = tf.clip_by_value(logit_probs, 1e-8, 1.0)
      # logp = tf.log(clipped_logit_probs)

      # TO RETURN MORE THAN ONE DECODER SAMPLE
      # argmax_logp_index, soft_logp_index, topk_out, z = iid_sampling(logp, max_seq_len, greedy=True, soft=True,
      #                                                                topk=True, k=2)
      # topk_probs, topk_indices = topk_out

      ##### DECODER SAMPLING ########################################################################################
      # greedy_beam_params = {"top_k": 0, "top_p": 0.0, "temperature": 0.0}
      # random_beam_params = {"top_k": 0, "top_p": 0.0, "temperature": 1.0}
      # topk_beam_params = {"top_k": 10000, "top_p": 0.0, "temperature": 0.0}
      # topp_beam_params = {"top_k": 0, "top_p": 0.9, "temperature": 0.0}

      # PREDS: IDs, LOGP: transformed logits, SCORE: scalar score using RISK trick, LOGP: [BxTxV] beam logp
      # {ids, logp_BxT, sent_score, logp_BxTxV}
      # Note: the logp_BxTxV are analogous to z -> should be used for RELAX, preds are the BxT of these -> b=H(z), and
      # logp are the corresponding values (score is normalised to sentence score)
      # greedy_dict = non_beam_sampling(model_params, features, max_seq_len,
      #                                 beam_params=greedy_beam_params, sentence_score=False)
      # random_dict = non_beam_sampling(model_params, features, max_seq_len,
      #                                 beam_params=random_beam_params, sentence_score=False)
      # topk_dict = non_beam_sampling(model_params, features, max_seq_len,
      #                               beam_params=topk_beam_params, sentence_score=False)
      # topp_dict = non_beam_sampling(model_params, features, max_seq_len,
      #                               beam_params=topp_beam_params, sentence_score=False)

      ##### BEAM SEARCH #############################################################################################
      # greedy_beam_params = {"_beam": 2, "top_k": 0, "top_p": 0.0, "temperature": 0.0}
      # random_beam_params = {"_beam": 2, "top_k": 0, "top_p": 0.0, "temperature": 1.0}
      # topk_beam_params = {"_beam": 2, "top_k": 10000, "top_p": 0.0, "temperature": 1.0}
      # topp_beam_params = {"_beam": 2, "top_k": 0, "top_p": 0.9, "temperature": 1.0}

      # PREDS: IDs, SCORE: scalar sentence score returned as sum of logp from beam search
      # {ids1, sent_score1, ...} and {beam_1logp, beam_2logp, beam_1logp}
      # greedy_dict, greedy_logp_dict = beam_sampling(model_params, features, max_seq_len,
      #                                               beam_params=greedy_beam_params)
      # random_dict, random_logp_dict = beam_sampling(model_params, features, max_seq_len,
      #                                               beam_params=random_beam_params)
      # topk_dict, topk_logp_dict = beam_sampling(model_params, features, max_seq_len,
      #                                           beam_params=topk_beam_params)
      # topp_dict, topp_logp_dict = beam_sampling(model_params, features, max_seq_len,
      #                                           beam_params=topp_beam_params)

      ##### RELAX VARIABLES #########################################################################################
      # FROM TRADITIONAL DECODER
      # z_tilde, logp_b = create_variables(z, logp, batch_index, sequence_index, clipped_logit_probs)
      # FROM DECODER SAMPLING PROCESS
      # z_tilde, logp_b = create_variables_from_samples(random_dict["logp_BxTxV"],
      #                                                 random_dict["logp_BxT"], batch_index, sequence_index)
      # FROM BEAM SEARCH SAMPLING
      # z_tilde, logp_b = create_variables_from_samples(random_logp_dict["beam_1logp"],
      #                                                 random_dict["ids1"], batch_index, sequence_index)

      ##### TEXT AND ROUGE ##########################################################################################
      # target_text = rouge_decoding(outputs["targets"], model_params)  # TARGET SAMPLES
      # argmax_pred_text = rouge_decoding(greedy_dict["logp_BxT"], model_params)  # ARGMAX SAMPLES
      # soft_pred_text = rouge_decoding(random_dict["logp_BxT"], model_params)  # SOFTMAX SAMPLES
      # additional_pred_text = rouge_decoding(topk_dict["logp_BxT"], model_params)  # ADDITIONAL SAMPLES

      # CALCULATE ROUGE LOSS: ROUGE score -> ROUGE loss = -ROUGE score
      # NOTE: for ROUGE variant, change value (0: precision, 1: recall, 2: f1)
      # rouge_loss_argmax = -tf.py_function(evaluate_rl, (target_text, argmax_pred_text, 2), tf.float32)
      # rouge_loss_soft = -tf.py_function(evaluate_rl, (target_text, soft_pred_text, 2), tf.float32)
      # rouge_loss_extra = -tf.py_function(evaluate_rl, (target_text, additional_pred_text, 2), tf.float32)

      ##### REINFORCE LOSS ##########################################################################################
      # FIND CORRESPONDING LOG_PROBS OF THE I.I.D SAMPLED TOKENS
      # ARGMAX -> logp(argmax(y))
      # argmax_logp = iid_log_probs(argmax_logp_index, batch_index, sequence_index, logp)
      # SOFTMAX -> logp(sample_y)
      # softmax_logp = iid_log_probs(sample_y, batch_index, sequence_index, logp)
      # ADDITIONAL
      # additional_logp = iid_log_probs(topk_indices_2, batch_index, sequence_index, logp)

      # CHANGE BELOW IF USING DECODER SAMPLED TOKENS/SCORES
      # weight the logp by ROUGE score (neg ROUGE_loss), sum values
      # reinforce_loss = tf.reduce_sum(tf.multiply(r1_score_hard, argmax_logp))

      ##### REINFORCE w/ BASELINE ###################################################################################
      # Socher (2017)
      # improve the probs of the SOFT labels (soft - hard)*soft_logp
      # improve the probs of the HARD labels (hard - soft)*hard_logp

      # BASELINE: CONTROL VARIATE
      # ffn_output = control_variate(source, targets)

      # loss_difference = tf.subtract(rouge_loss_soft, rouge_loss_argmax)
      # reinforce_baseline = tf.reduce_sum(tf.multiply(loss_difference, softmax_logp))

      ##### REINFORCE w/ THRESHOLD ##################################################################################
      # we take output of ROUGE score as ROUGE_loss = -ROUGE score
      # intermediate_loss = tf.reduce_sum(tf.multiply(tf.subtract(0.3, -rouge_loss_argmax), argmax_logp))

      ##### EXPECTED RISK MINIMISATION ##############################################################################
      # L_risk = risk_loss(max_seq_len, rouge_losses=[rouge_loss_argmax, rouge_loss_soft, rouge_loss_extra],
      #                    logits=[argmax_logp, softmax_logp, additional_logp], n=3)

      ##### MIXED LOSS ##############################################################################################
      # combined_loss = tf.math.add(tf.multiply(tf.constant(0.3, dtype=tf.float32), XENT_loss),
      #                             tf.multiply(tf.constant(0.7, dtype=tf.float32), L_risk))

      # OR conditional loss switch
      # constraint = tf.random_uniform(shape=(), minval=0, maxval=1, dtype=tf.float32)
      # combined_loss = tf.cond(constraint > 0.8, lambda: hard_reinforce_loss, lambda: XENT_loss)

      ##### RELAX CONTROL VARIATE ###################################################################################
      # z_target, zt_target = create_cv_target(outputs, batch_index, sequence_index, z, z_tilde)

      ##### RELAX LOSS ##############################################################################################
      # RELAX Q_func
      # with tf.variable_scope("Q_func"):
      #     c_z = Q_func(z, z_target)

      # with tf.variable_scope("Q_func", reuse=True):
      #     c_z_tilde = Q_func(z_tilde, zt_target)

      # Formulate RELAX as a loss function
      # f_y = rouge_loss_soft  # negative for loss (defined above)
      # c_z_tilde1 = tf.stop_gradient(tf.identity(c_z_tilde))  # clone, detach, stop grad
      # L_relax = tf.reduce_sum(((f_y - c_z_tilde1)*logp_b) - c_z_tilde + c_z)

      # OR construct gradient estimator
      # theta = [tv for tv in tf.trainable_variables() if "Q_func" not in tv.name]
      # d_logp_d_theta = tf.gradients(logp_b, theta)[0]  # logp
      # d_c_z_tilde_d_theta = tf.gradients(c_z_tilde, theta)[0]
      # d_c_z_d_theta = tf.gradients(c_z, theta)[0]

      # TODO: c_z_tilde must be scalar to calculate this, otherwise size error is an issue
      # relax = tf.reduce_sum(f_y - c_z_tilde)*d_logp_d_theta - d_c_z_tilde_d_theta + d_c_z_d_theta
      # relax = tf.gradients(L_relax, theta)[0]  # will this work?

      # Calculate the first optimization step with loss
      # list_of_gradient_variable_pairs = optimizer.compute_gradients(L_relax)
      # train_op = optimizer.apply_gradients(list_of_gradient_variable_pairs, global_step=global_step)

      # Variance reduction objective
      # relax_grads = [i[0] for i in list_of_gradient_variable_pairs]  # TODO: extraction does not work
      # variance_loss = tf.reduce_mean(tf.square(relax), name="variance_loss")

      # initialise adafactor again for variance optimiser
      # var_opt = adafactor.AdafactorOptimizer(
      #           learning_rate=lr,
      #           decay_rate=adafactor.adafactor_decay_rate_pow(0.8),
      #           beta1=0.0)

      # est_params = [eta, log_temperature]  # TODO: REBAR implementation

      # Adds the parameters of the FFNN
      # nn_params = [tv for tv in tf.trainable_variables() if "Q_func" in tv.name]
      # est_params = nn_params
      # est_params = est_params + nn_params  # TODO: REBAR implementation

      # Additional optimization step
      # var_gradvars = var_opt.compute_gradients(variance_loss, var_list=est_params)
      # var_train_op = var_opt.apply_gradients(var_gradvars)

      # This may allow for both train ops to be passed in the return statement below?
      # with tf.control_dependencies([train_op, var_train_op]):
      #     train_op = tf.no_op()

      ###############################################################################################################
      # Calculate gradients
      # If freezing layers, only optimise wrt certain layers (find names) - speeds up, worsens performance
      # last_params = [tv for tv in tf.trainable_variables() if "decoder/LayerNorm/" in tv.name]
      # list_of_gradient_variable_pairs = optimizer.compute_gradients(combined_loss, var_list=last_params)

      list_of_gradient_variable_pairs = optimizer.compute_gradients(XENT_loss)
      train_op = optimizer.apply_gradients(list_of_gradient_variable_pairs, global_step=global_step)

      tf.logging.set_verbosity(tf.logging.INFO)
      # Debugging steps - add into logging hook directly if needed
      # tf.debugging.check_numerics(sum_logp, "DEBUG: sum_logp has a NaN")

      logging_hook = tf.train.LoggingTensorHook({"loss": XENT_loss,
                                                 # "variance_loss": variance_loss,
                                                 "learning_rate": lr,
                                                 "global_step": global_step,
                                                 }, every_n_iter=5)

      # This is the configured estimator function that is returned to train the model
      return tpu_estimator.TPUEstimatorSpec(
          mode=mode,
          loss=XENT_loss,
          train_op=train_op,
          training_hooks=[logging_hook],
          scaffold_fn=_load_vars_from_checkpoint(use_tpu,
                                                 train_init_checkpoint),
          host_call=add_scalars_to_summary(model_dir, {"learning_rate": lr,
                                                       # "rouge_loss_hard": rouge_loss_argmax,
                                                       # "rouge_loss_soft": rouge_loss_soft,
                                                       # "rouge_loss_extra": rouge_loss_extra,
                                                       # "reinforce_loss": soft_reinforce_baseline,
                                                       # "XENT_loss": XENT_loss,
                                                       }))

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
