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

"""Beam search.
This beam search implementation is designed for TPU usage only and prefers
flexibility over efficiency. Transformer attention caching is not enabled yet.
Mostly follows implementation in T2T. Several difference to pure beamsearch:
1. has finished and alive seqs, use 2 * beam_size to grow alive seqs,
   which makes beam_size=1 doesn't equal greedy.
2. prefers finished seq over alive seqs.
3. prefers lower indices when equal probability (though unlikely).
4. with custom length normalization and constraint.
Notations:
  B: batch_size, M: beam_size, T: max_decode_len, V: vocab_size, U: undefined
"""
#
# pylint: disable=invalid-name

import tensorflow as tf


def length_normalization(start, alpha, min_len, max_len, out_of_range_penalty):
  r"""Create length normalization function.
  Combines length penalty from https://arxiv.org/abs/1609.08144,
  and length constraint from https://www.aclweb.org/anthology/W18-2706.pdf.
  scores = \sum_j log(P_j) / ((start + lengths)/(1 + start))**alpha
          + out_of_range_penalty * (length > max_len or length < min_len)
  Args:
    start: int, length normalization start offset.
    alpha: float, [0, 1.0],  length normalization power.
    min_len: int, minimum decode length.
    max_len: int, maximum decode lengths.
    out_of_range_penalty: float, penalty for lengths outside min len and max
      len. Use a negative number that penalize out of range decodes, does hard
      constraint if set to -inf.
  Returns:
    fn(log_probs_BxM, length)->scores_BxM: a function to normalize sum log
    probabilities of sequence with current decoding lengths.
  """

  def length_norm_fn(log_probs_BxM, length_int):
    """Normalize sum log probabilities given a sequence length."""
    dtype = log_probs_BxM.dtype
    norm_flt = tf.pow(((start + tf.cast(length_int, dtype)) / (1. + start)),
                      alpha)
    log_probs_BxM /= norm_flt
    too_short_bool = tf.less(length_int, min_len)
    too_long_bool = tf.logical_and(tf.greater(length_int, max_len), max_len > 0)
    out_of_range_bool = tf.logical_or(too_long_bool, too_short_bool)
    log_probs_BxM += out_of_range_penalty * tf.cast(out_of_range_bool, dtype)
    return log_probs_BxM

  return length_norm_fn


def beam_search(symbols_to_logits_fn,
                init_seq_BxT,
                initial_cache_BxU,
                vocab_size,
                beam_size,
                length_norm_fn,
                eos_id=1,
                sampling=False):
  """Beam search.
  Args:
    symbols_to_logits_fn: fn(seq_BxT, cache_BxU, i) -> (logits_BxV, cache_BxU)
    init_seq_BxT: initial sequence ids.
    initial_cache_BxU: dictionary of tensors with shape BxU.
    vocab_size: vocabulary size.
    beam_size: beam size.
    length_norm_fn: length normalization function.
    eos_id: end of sequence.
    sampling: for training.
  Returns:
    Tuple of (beams_BxMxT, scores_BxM). Beam searched sequences and scores.
  """
  B, T = init_seq_BxT.shape
  M, V = beam_size, vocab_size
  dtype = tf.float32
  int_dtype = init_seq_BxT.dtype

  def _loop_body(i, alive_seq_BxMxT, alive_log_probs_BxM, alive_cache_BxMxU,
                 finished_seq_BxMxT, finished_scores_BxM,
                 init_finished_logitsM1_BxTxV, init_finished_logitsM2_BxTxV, init_finished_logitsM3_BxTxV):
    """Beam search loop body."""
    # Decode one step with beam
    logits_BMxV, cache_BMxU = symbols_to_logits_fn(
        _flatten_beam_dim(alive_seq_BxMxT),
        tf.nest.map_structure(_flatten_beam_dim, alive_cache_BxMxU), i)
    logits_BxMxV = _unflatten_beam_dim(logits_BMxV, M)
    new_cache_BxMxU = tf.nest.map_structure(lambda t: _unflatten_beam_dim(t, M),
                                            cache_BMxU)
    logitsM1_BxTxV, logitsM2_BxTxV, logitsM3_BxTxV = _separate(logits_BxMxV,
                                                               [init_finished_logitsM1_BxTxV,
                                                                init_finished_logitsM2_BxTxV,
                                                                init_finished_logitsM3_BxTxV], B, T, V, M, i)

    # select top 2 * beam_size and fill alive and finished.
    log_probs_BxMxV = logits_BxMxV - tf.reduce_logsumexp(
        logits_BxMxV, axis=2, keepdims=True)
    log_probs_BxMxV += tf.expand_dims(alive_log_probs_BxM, axis=2)
    log_probs_BxMV = tf.reshape(log_probs_BxMxV, [B, -1])
    new_log_probs_Bx2M, topk_indices_Bx2M = tf.nn.top_k(log_probs_BxMV, k=2 * M)
    topk_beam_Bx2M = topk_indices_Bx2M // V
    topk_seq_Bx2MxT, new_cache_Bx2MxU = _gather_nested(
        [alive_seq_BxMxT, new_cache_BxMxU], topk_beam_Bx2M)
    topk_ids_Bx2M = topk_indices_Bx2M % V
    new_seq_Bx2MxT = _update_i(topk_seq_Bx2MxT, topk_ids_Bx2M, i)
    new_finished_flags_Bx2M = tf.cast(
        tf.reduce_any(tf.equal(new_seq_Bx2MxT, eos_id), axis=-1), dtype)

    # get new alive
    _, topk_alive_indices_BxM = tf.nn.top_k(
        new_log_probs_Bx2M + new_finished_flags_Bx2M * dtype.min, k=M)
    (alive_seq_BxMxT, alive_log_probs_BxM, alive_cache_BxMxU) = _gather_nested(
        [new_seq_Bx2MxT, new_log_probs_Bx2M, new_cache_Bx2MxU],
        topk_alive_indices_BxM)

    # get new finished
    new_scores_Bx2M = length_norm_fn(new_log_probs_Bx2M, i + 1)
    new_scores_Bx2M += (1 - new_finished_flags_Bx2M) * dtype.min
    finished_seq_Bx3MxT = tf.concat([finished_seq_BxMxT, new_seq_Bx2MxT],
                                    axis=1)
    finished_scores_Bx3M = tf.concat([finished_scores_BxM, new_scores_Bx2M],
                                     axis=1)
    _, topk_finished_indices_BxM = tf.nn.top_k(finished_scores_Bx3M, k=M)
    (finished_seq_BxMxT, finished_scores_BxM) = _gather_nested(
        [finished_seq_Bx3MxT, finished_scores_Bx3M], topk_finished_indices_BxM)

    return [
        i + 1, alive_seq_BxMxT, alive_log_probs_BxM, alive_cache_BxMxU,
        finished_seq_BxMxT, finished_scores_BxM,
        logitsM1_BxTxV, logitsM2_BxTxV, logitsM3_BxTxV
    ]

  # initialize.
  init_i = tf.constant(0, dtype=int_dtype)
  init_alive_seq_BxMxT = _expand_to_beam_size(init_seq_BxT, M)
  log_probs_1xM = tf.constant([[0.] + [dtype.min] * (M - 1)], dtype=dtype)
  init_alive_log_probs_BxM = tf.tile(log_probs_1xM, [B, 1])
  init_alive_cache_BxMxU = tf.nest.map_structure(
      lambda t: _expand_to_beam_size(t, M), initial_cache_BxU)
  init_finished_seq_BxMxT = tf.zeros(tf.shape(init_alive_seq_BxMxT), int_dtype)
  init_finished_scores_BxM = tf.zeros([B, M], dtype=dtype) + dtype.min

  init_finished_logitsM1_BxTxV = tf.zeros([B, T, V], dtype=dtype)
  init_finished_logitsM2_BxTxV = tf.zeros([B, T, V], dtype=dtype)
  init_finished_logitsM3_BxTxV = tf.zeros([B, T, V], dtype=dtype)

  # run loop.
  (_, final_alive_seq_BxMxT, final_alive_scores_BxM, _,
   final_finished_seq_BxMxT, final_finished_scores_BxM,
   logitsM1_BxTxV, logitsM2_BxTxV, logitsM3_BxTxV) = tf.while_loop(
       lambda *args: True,  # Always do T iterations
       _loop_body,
       loop_vars=[
           init_i, init_alive_seq_BxMxT, init_alive_log_probs_BxM,
           init_alive_cache_BxMxU, init_finished_seq_BxMxT,
           init_finished_scores_BxM,
           init_finished_logitsM1_BxTxV, init_finished_logitsM2_BxTxV, init_finished_logitsM3_BxTxV
       ],
       parallel_iterations=1,
       back_prop=False,
       maximum_iterations=T,
   )

  # process finished.
  final_finished_flag_BxMx1 = tf.reduce_any(
      tf.equal(final_finished_seq_BxMxT, eos_id), axis=-1, keepdims=True)
  final_seq_BxMxT = tf.where(
      tf.tile(final_finished_flag_BxMx1, [1, 1, T]), final_finished_seq_BxMxT,
      final_alive_seq_BxMxT)
  final_scores_BxM = tf.where(
      tf.squeeze(final_finished_flag_BxMx1, axis=-1), final_finished_scores_BxM,
      final_alive_scores_BxM)
  return final_seq_BxMxT, final_scores_BxM, {"beam_1_logits": logitsM1_BxTxV,
                                             "beam_2_logits": logitsM2_BxTxV,
                                             "beam_3_logits": logitsM3_BxTxV}


def _update_i(tensor_BxNxT, updates_BxN, i):
  B, N, T = tensor_BxNxT.shape
  tensor_BNxT = tf.reshape(tensor_BxNxT, [-1, T])
  updates_BN = tf.reshape(updates_BxN, [-1])
  batch_BN = tf.range(B * N, dtype=tf.int64)
  i_BN = tf.fill([B * N], tf.cast(i, tf.int64))
  ind_BNx2 = tf.stack([batch_BN, i_BN], axis=-1)
  tensor_BNxT = tf.tensor_scatter_nd_update(tensor_BNxT, ind_BNx2, updates_BN)
  return tf.reshape(tensor_BNxT, [B, N, T])


def _expand_to_beam_size(tensor_BxU, beam_size):
  tensor_Bx1xU = tf.expand_dims(tensor_BxU, axis=1)
  tile_dims = [1] * tensor_Bx1xU.shape.ndims
  tile_dims[1] = beam_size
  tensor_BxMxU = tf.tile(tensor_Bx1xU, tile_dims)
  return tensor_BxMxU


def _flatten_beam_dim(tensor_BxMxU):
  shape = tensor_BxMxU.shape.as_list()
  tensor_BMxU = tf.reshape(tensor_BxMxU, [shape[0] * shape[1]] + shape[2:])
  return tensor_BMxU


def _unflatten_beam_dim(tensor_BMxU, M):
  shape = tensor_BMxU.shape.as_list()
  tensor_BxMxU = tf.reshape(tensor_BMxU, [shape[0] // M, M] + shape[1:])
  return tensor_BxMxU


def _gather_nested(nested_BxMxU, indices_BxN):

  def _gather_beam(tensor_BxMxU):
    tensor_BxNxU = tf.gather(tensor_BxMxU, indices_BxN, batch_dims=1, axis=1)
    return tensor_BxNxU

  return tf.nest.map_structure(_gather_beam, nested_BxMxU)


def _inplace_update_i(tensor_BxL, updates_B, i):
  """Inplace update a tensor. B: batch_size, L: tensor length.
  Copied from pegasus.decoding.py"""
  batch_size = tensor_BxL.shape[0]
  indices_Bx2 = tf.stack([
      tf.range(batch_size, dtype=tf.int64),
      tf.fill([batch_size], tf.cast(i, tf.int64))
  ],
                         axis=-1)
  return tf.tensor_scatter_nd_update(tensor_BxL, indices_Bx2, updates_B)


def _separate(logp_BxMxV, logpMi_BxTxV, B, T, V, M, i):
    if M == 2:
        logpM1_BxV = tf.reshape(logp_BxMxV[0, 0], (B, V))
        logpM2_BxV = tf.reshape(logp_BxMxV[0, 1], (B, V))

        logpM1_BxTxV = _inplace_update_i(logpMi_BxTxV[0], logpM1_BxV, i)
        logpM2_BxTxV = _inplace_update_i(logpMi_BxTxV[1], logpM2_BxV, i)

        # return empty tensor for 3rd beam
        return logpM1_BxTxV, logpM2_BxTxV, tf.zeros([B, T, V])

    elif M == 3:
        logpM1_BxV = tf.reshape(logp_BxMxV[0, 0], (B, V))
        logpM2_BxV = tf.reshape(logp_BxMxV[0, 1], (B, V))
        logpM3_BxV = tf.reshape(logp_BxMxV[0, 2], (B, V))

        logpM1_BxTxV = _inplace_update_i(logpMi_BxTxV[0], logpM1_BxV, i)
        logpM2_BxTxV = _inplace_update_i(logpMi_BxTxV[1], logpM2_BxV, i)
        logpM3_BxTxV = _inplace_update_i(logpMi_BxTxV[2], logpM3_BxV, i)

        return logpM1_BxTxV, logpM2_BxTxV, logpM3_BxTxV