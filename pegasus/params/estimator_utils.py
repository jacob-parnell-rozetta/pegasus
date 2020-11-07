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
      predictions, _ = model_params.estimator_prediction_fn(features)

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

      # ARGMAX OF LOG_PROB
      # argmax_logp_index = tf.math.argmax(logp, axis=2)  # Returns indexes where logp is max

      # TOP-K SAMPLES FROM LOG_PROB
      # topk_probs, topk_indices = tf.math.top_k(logp, k=2)

      # topk_probs_2 = tf.slice(topk_probs, [0, 0, 1], [1, max_seq_len, 1])
      # topk_probs_2 = tf.squeeze(topk_probs_2, 2)
      # topk_indices_2 = tf.slice(topk_indices, [0, 0, 1], [1, max_seq_len, 1])
      # topk_indices_2 = tf.squeeze(topk_indices_2, 2)

      # SOFT SAMPLES OF LOG_PROB - 'soft' labels of the Gumbel samples, and their one-hot labels
      # u = tf.random_uniform(shape=outputs["one_hot_targets"].get_shape().as_list(),
      #                       minval=1e-8,
      #                       maxval=1,
      #                       dtype=tf.float32)
      # z = tf.math.add(-tf.log(-tf.log(u)), logp)

      # use y_soft and sample_y for REINFORCE -> RELAX uses b = H(z)
      # y_soft = tf.math.softmax(tf.div(z, 0.1))  # this is Gumbel-Softmax; low temp -> approaches argmax
      # sample_y = tf.math.argmax(y_soft, axis=2)

      ##### DECODER SAMPLING ########################################################################################
      # RANDOMLY SAMPLE INDIVIDUAL TOKENS FROM DECODER DISTRIBUTION
      # random_preds, random_logits = model_params.model().predict(features, max_seq_len, beam_size=1, top_k=0,
      #                                                            top_p=0.0, temperature=1.0)
      # random_preds = random_preds["outputs"]  # gets the IDs
      # random_logp = tf.squeeze(tf.log(tf.clip_by_value(tf.math.softmax(random_logits), 1e-8, 1.0)), 0)  # to log_p

      # RANDOMLY SAMPLE INDIVIDUAL TOKENS FROM DECODER'S TOP-K DISTRIBUTION
      # topk_preds, topk_logits = model_params.model().predict(features, max_seq_len, beam_size=1, top_k=1000,
      #                                                        top_p=0.0, temperature=1.0)
      # topk_preds = topk_preds["outputs"]  # gets the IDs
      # topk_logp = tf.squeeze(tf.log(tf.clip_by_value(tf.math.softmax(topk_logits), 1e-8, 1.0)), 0)  # to log_p
      # topk_logp_sent = tf.exp((1 / max_seq_len) * tf.reduce_sum(topk_logp))  # sentence score 0-1

      # GREEDY SAMPLE INDIVIDUAL TOKENS FROM DECODER DISTRIBUTION
      # greedy_preds, greedy_logits = model_params.model().predict(features, max_seq_len, beam_size=1, top_k=0,
      #                                                            top_p=0.0, temperature=0.0)
      # greedy_preds = greedy_preds["outputs"]
      # greedy_logp = tf.squeeze(tf.log(tf.clip_by_value(tf.math.softmax(greedy_logits), 1e-8, 1.0)), 0)
      # greedy_logp_sent = tf.exp((1/max_seq_len) * tf.reduce_sum(greedy_logp))  # sentence score 0-1

      # RANDOMLY SAMPLE INDIVIDUAL TOKENS FROM DECODER'S TOP-P DISTRIBUTION
      # topp_preds, topp_logits = model_params.model().predict(features, max_seq_len, beam_size=1, top_k=0,
      #                                                        top_p=0.9, temperature=1.0)
      # topp_preds = topp_preds["outputs"]
      # topp_logp = tf.squeeze(tf.log(tf.clip_by_value(tf.math.softmax(topp_logits), 1e-8, 1.0)), 0)
      # topp_logp_sent = tf.exp((1/max_seq_len) * tf.reduce_sum(topp_logp))  # sentence score 0-1

      ##### BEAM SEARCH #############################################################################################
      # TODO: CURRENT IMPLEMENTATION ONLY ALLOWS SENTENCE SCORES, NEED TO CONFIGURE RETURN OF LOGITS FOR TOKEN_LEVEL
      #  ADD ADDITIONAL PREDS/SCORE TERMS FOR BEAM_SEARCH >= 3 (memory constrained)
      # _beam = 2
      # RANDOMLY SAMPLE INDIVIDUAL TOKENS USING BEAM SEARCH
      # random_preds_dict, random_scores = model_params.model().predict(features, max_seq_len, beam_size=_beam,
      #                                                                 top_k=0, top_p=0.0, temperature=1.0)
      # random_preds = random_preds_dict["outputs"][0]  # gets the IDs
      # random_preds2 = random_preds_dict["outputs"][1]  # gets the IDs of second best
      # random_score = random_scores[:, 0]  # sentence score (sum of log_prob) for first
      # random_score2 = random_scores[:, 1]  # sentence score (sum of log_prob) for second

      # RANDOMLY SAMPLE INDIVIDUAL TOKENS USING BEAM SEARCH FROM TOP-K DISTRIBUTION
      # topk_preds_dict, topk_scores = model_params.model().predict(features, max_seq_len, beam_size=_beam,
      #                                                             top_k=10000, top_p=0.0, temperature=1.0)
      # topk_preds = topk_preds_dict["outputs"][0]  # gets the IDs
      # topk_preds2 = topk_preds_dict["outputs"][1]  # gets the IDs of second best
      # topk_score = topk_scores[:, 0]  # sentence score (sum of log_prob) for first
      # topk_score2 = topk_scores[:, 1]  # sentence score (sum of log_prob) for second

      # GREEDY SAMPLE INDIVIDUAL TOKENS USING BEAM SEARCH
      # greedy_preds_dict, greedy_scores = model_params.model().predict(features, max_seq_len, beam_size=_beam,
      #                                                                 top_k=0, top_p=0.0, temperature=1.0)
      # greedy_preds = greedy_preds_dict["outputs"][0]  # gets the IDs
      # greedy_preds2 = greedy_preds_dict["outputs"][1]  # gets the IDs of second best
      # greedy_score = greedy_scores[:, 0]  # sentence score (sum of log_prob) for first
      # greedy_score2 = greedy_scores[:, 1]  # sentence score (sum of log_prob) for second

      # RANDOMLY SAMPLE INDIVIDUAL TOKENS USING BEAM SEARCH FROM TOP-P DISTRIBUTION
      # topp_preds_dict, topp_scores = model_params.model().predict(features, max_seq_len, beam_size=_beam,
      #                                                             top_k=0, top_p=0.9, temperature=1.0)
      # topp_preds = topp_preds_dict["outputs"][0]  # gets the IDs
      # topp_preds2 = topp_preds_dict["outputs"][1]  # gets the IDs of second best
      # topp_score = topp_scores[:, 0]  # sentence score (sum of log_prob) for first
      # topp_score2 = topp_scores[:, 1]  # sentence score (sum of log_prob) for second

      ##### RELAX VARIABLES #########################################################################################
      # v = tf.random_uniform(shape=outputs["one_hot_targets"].get_shape().as_list(),
      #                       minval=1e-8,
      #                       maxval=1,
      #                       dtype=tf.float32)
      # b = tf.stop_gradient(tf.math.argmax(z, axis=2))  # this is Gumbel-Max (used for RELAX)

      # create index tensor where b is the argmax, to use as indexer for substitution
      # b_new = tf.squeeze(b, 0)
      # index_tensor_b = tf.expand_dims(tf.stack([batch_index, sequence_index, b_new], axis=1), 0)

      # v_b = tf.gather_nd(v, index_tensor_b)  # values of v where b are the argmax indexes
      # update = -tf.log(-tf.log(v_b))  # for i == b

      # create z_tilde as for the case where i != b
      # z_tilde = -tf.log(-tf.div(tf.log(v), clipped_logit_probs) - tf.expand_dims(tf.log(v_b), 2))
      # z_tilde = tf.tensor_scatter_nd_update(z_tilde, index_tensor_b, update)

      # logp_b = tf.gather_nd(logp, index_tensor_b)  # used in loss func

      ##### TEXT AND ROUGE ##########################################################################################
      # TARGET SAMPLES
      # decode_target_text_tensor = public_parsing_ops.decode(outputs["targets"], model_params.vocab_filename,
      #                                                       model_params.encoder_type)
      # decode_target_text = tf.stop_gradient(decode_target_text_tensor[0])

      # ARGMAX SAMPLES
      # decode_preds_text_tensor_hard = public_parsing_ops.decode(argmax_logp_index, model_params.vocab_filename,
      #                                                           model_params.encoder_type)
      # decode_preds_text_hard = tf.stop_gradient(decode_preds_text_tensor_hard[0])

      # NOTE: for ROUGE variant, change value (0: precision, 1: recall, 2: f1)
      # calculate ROUGE score (argmax) -> ROUGE loss = -ROUGE score
      # r1_score_hard = -tf.py_function(evaluate_rl, (decode_target_text, decode_preds_text_hard, 2), tf.float32)

      # SOFTMAX SAMPLES
      # decode_preds_text_tensor_soft = public_parsing_ops.decode(b, model_params.vocab_filename,
      #                                                           model_params.encoder_type)
      # decode_preds_text_soft = tf.stop_gradient(decode_preds_text_tensor_soft[0])

      # NOTE: for ROUGE variant, change value (0: precision, 1: recall, 2: f1)
      # calculate ROUGE loss (softmax) -> ROUGE loss = -ROUGE score
      # r1_score_soft = -tf.py_function(evaluate_rl, (decode_target_text, decode_preds_text_soft, 2), tf.float32)

      # 2ND ARGMAX SAMPLES
      # decode_preds_text_tensor_hard2 = public_parsing_ops.decode(topk_indices_2, model_params.vocab_filename,
      #                                                            model_params.encoder_type)
      # decode_preds_text_hard2 = tf.stop_gradient(decode_preds_text_tensor_hard2[0])

      # r1_score_hard2 = -tf.py_function(evaluate_rl, (decode_target_text, decode_preds_text_hard2, 2), tf.float32)

      ##### REINFORCE LOSS ##########################################################################################
      # FIND CORRESPONDING LOG_PROBS OF THE I.I.D SAMPLED TOKENS
      # ARGMAX -> logp(argmax(y))
      # argmax_logp_new = tf.reshape(argmax_logp_index, [argmax_logp_index.get_shape().as_list()[1]])
      # index_tensor_hard = tf.stack([batch_index, sequence_index, argmax_logp_new], axis=1)
      # argmax_logp = tf.gather_nd(logp, index_tensor_hard)  # finds log probs using hard indexing

      # 2ND ARGMAX
      # top_k_logp_new = tf.cast(tf.reshape(topk_indices_2, [topk_indices_2.get_shape().as_list()[1]]), tf.int64)
      # index_tensor_hard2 = tf.stack([batch_index, sequence_index, top_k_logp_new], axis=1)
      # top_k_logp = tf.gather_nd(logp, index_tensor_hard2)  # finds log probs using hard indexing

      # SOFTMAX -> logp(sample_y)
      # sampled_vals_new = tf.reshape(sample_y, [sample_y.get_shape().as_list()[1]])
      # index_tensor_soft = tf.stack([batch_index, sequence_index, sampled_vals_new], axis=1)
      # softmax_logp = tf.gather_nd(logp, index_tensor_soft)  # finds log probs using soft indexing

      # CHANGE BELOW IF USING DECODER SAMPLED TOKENS/SCORES
      # weight the logp by ROUGE score (neg ROUGE_loss), sum values
      # reinforce_loss = tf.reduce_sum(tf.multiply(r1_score_hard, argmax_logp))

      ##### REINFORCE w/ BASELINE ###################################################################################
      # Socher (2017)
      # improve the probs of the SOFT labels (soft - hard)*soft_logp
      # improve the probs of the HARD labels (hard - soft)*hard_logp

      # using control variate as baseline
      # ffn_output = control_variate(source, targets)

      # loss_difference = tf.subtract(r1_score_soft, r1_score_hard)
      # reinforce_baseline = tf.reduce_sum(tf.multiply(soft_loss_difference, softmax_logp))

      ##### REINFORCE w/ THRESHOLD ##################################################################################
      # we take output of ROUGE score as ROUGE_loss = -ROUGE score
      # intermediate_loss = tf.reduce_sum(tf.multiply(tf.subtract(0.3, -r1_score_hard), argmax_logp))

      ##### EXPECTED RISK MINIMISATION ##############################################################################
      # TODO: functionalize
      # L_risk = -r(u,y)*p(u|x,theta) -> U(x) is a set of candidate translations
      # Calculate f_u for as many sequences
      # f_u_soft = tf.exp(tf.div(1.0, max_seq_len) * tf.reduce_sum(softmax_logp))
      # f_u_hard = tf.exp(tf.div(1.0, max_seq_len) * tf.reduce_sum(argmax_logp))
      # f_u_hard2 = tf.exp(tf.div(1.0, max_seq_len) * tf.reduce_sum(top_k_logp))

      # For beam samples as we currently have a score as sum of log prob
      # f_u_soft = tf.exp((1.0 / max_seq_len) * random_score)
      # f_u_hard = tf.exp((1.0 / max_seq_len) * greedy_score)
      # f_u_hard2 = tf.exp((1.0 / max_seq_len) * greedy_score2)

      # Calculate p_u for as many sequences
      # p_u_soft = f_u_soft / tf.reduce_sum([f_u_hard, f_u_hard2, f_u_soft])
      # p_u_hard = f_u_hard / tf.reduce_sum([f_u_hard, f_u_hard2, f_u_soft])
      # p_u_hard2 = f_u_hard2 / tf.reduce_sum([f_u_hard, f_u_hard2, f_u_soft])

      # Calculate each risk loss
      # L_risk_hard = tf.reduce_sum(tf.multiply(r1_score_hard, p_u_hard))
      # L_risk_hard2 = tf.reduce_sum(tf.multiply(r1_score_hard2, p_u_hard2))
      # L_risk_soft = tf.reduce_sum(tf.multiply(r1_score_soft, p_u_soft))

      # Overall Risk loss
      # L_risk = tf.reduce_sum([L_risk_hard, L_risk_hard2, L_risk_soft])

      ##### MIXED LOSS ##############################################################################################
      # combined_loss = tf.math.add(tf.multiply(tf.constant(0.3, dtype=tf.float32), XENT_loss),
      #                             tf.multiply(tf.constant(0.7, dtype=tf.float32), L_risk))

      # OR conditional loss switch
      # constraint = tf.random_uniform(shape=(), minval=0, maxval=1, dtype=tf.float32)
      # combined_loss = tf.cond(constraint > 0.8, lambda: hard_reinforce_loss, lambda: XENT_loss)

      ##### RELAX CONTROL VARIATE ###################################################################################
      # Here we need to convert the IDs from the target, to the probabilities for ROUGE mimic
      # TODO: probabilities from z or from logp?
      # target_id_cv = tf.reshape(outputs['targets'], [outputs['targets'].get_shape().as_list()[1]])
      # index_tensor_target = tf.stack([batch_index, sequence_index, target_id_cv], axis=1)

      # finds log probs using targets indexing
      # tgt_probs_cv_z = tf.expand_dims(tf.expand_dims(tf.gather_nd(z, index_tensor_target),0),2)
      # tgt_probs_cv_ztilde = tf.expand_dims(tf.expand_dims(tf.gather_nd(z_tilde, index_tensor_target),0),2)

      # z_target = tf.broadcast_to(tgt_probs_cv_z, z.get_shape().as_list())
      # zt_target = tf.broadcast_to(tgt_probs_cv_ztilde, z_tilde.get_shape().as_list())

      ##### RELAX LOSS ##############################################################################################
      # RELAX Q_func
      # with tf.variable_scope("Q_func"):
      #     c_z = Q_func(z, z_target)

      # with tf.variable_scope("Q_func", reuse=True):
      #     c_z_tilde = Q_func(z_tilde, zt_target)

      # Formulate RELAX as a loss function
      # f_y = r1_score_soft  # negative for loss (defined above)
      # c_z_tilde1 = tf.stop_gradient(tf.identity(c_z_tilde))  # clone, detach, stop grad
      # L_relax = tf.reduce_sum(((f_y - c_z_tilde1)*logp_b) - c_z_tilde + c_z)

      # OR construct gradient estimator
      # theta = [tv for tv in tf.trainable_variables() if "Q_func" not in tv.name]
      # d_logp_d_theta = tf.gradients(logp_b, theta)[0]  # logp
      # d_c_z_tilde_d_theta = tf.gradients(c_z_tilde, theta)[0]
      # d_c_z_d_theta = tf.gradients(c_z, theta)[0]

      # TODO: [1, 32] * [96103, 1024]
      # relax = tf.reduce_sum(f_y - c_z_tilde)*d_logp_d_theta - d_c_z_tilde_d_theta + d_c_z_d_theta

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
                                                 # "target_text": decode_target_text,
                                                 # "soft_preds_text": decode_preds_text_soft,
                                                 # "hard_preds_text": decode_preds_text_hard,
                                                 # "hard2_preds_text": decode_preds_text_hard2,
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
                                                       # "rouge_loss_hard": r1_score_hard,
                                                       # "rouge_loss_soft": r1_score_soft,
                                                       # "rouge_loss_hard2": r1_score_hard2,
                                                       # "reinforce_loss": soft_reinforce_baseline,
                                                       # "XENT_loss": XENT_loss,
                                                       # "c_z": c_z, "c_z_tilde": c_z_tilde
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
