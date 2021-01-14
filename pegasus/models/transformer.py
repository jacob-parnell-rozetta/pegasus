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

"""Standard Transformer models.

Models contain embedding, encoding, and loss functions, and expect text ids as
inputs. All models have same format as below:
  model = TransformerModel(...)
  loss, output = model(features, training)
Features and outputs are dictionary of tensors. Features usually inlucdes inputs
and targets ids.
"""
#
# pylint: disable=invalid-name
# pylint: disable=g-long-lambda

from pegasus.layers import attention
from pegasus.layers import decoding
from pegasus.layers import embedding
from pegasus.layers import timing
from pegasus.layers import transformer_block
from pegasus.models import base
from pegasus.methods.decoder_sampling import iid_process_logits
import tensorflow as tf
from tensorflow.contrib import layers as contrib_layers
from absl import logging
import numpy as np


class TransformerEncoderDecoderModel(base.BaseModel):
  """Transformer encoder+decoder.

  Notations:
    B: batch_size, I: max_input_len, T: max_target/decode_len, D: hidden_size
    V: vocab_size
  """

  def __init__(self, vocab_size, hidden_size, filter_size, num_heads,
               num_encoder_layers, num_decoder_layers, label_smoothing,
               dropout):
    self._dtype = tf.float32
    self._embedding_layer = embedding.Embedding(vocab_size, hidden_size,
                                                "weights", self._dtype)
    block_fn = lambda: transformer_block.TransformerBlock(
        hidden_size, filter_size, num_heads, dropout)
    self._encoder_layers = [block_fn() for _ in range(num_encoder_layers)]
    self._decoder_layers = [block_fn() for _ in range(num_decoder_layers)]
    self._dropout_fn = lambda x, training: tf.compat.v2.nn.dropout(
        x, dropout, noise_shape=[x.shape[0], 1, x.shape[2]]) if training else x
    self._vocab_size = vocab_size
    self._num_heads = num_heads
    self._label_smoothing = label_smoothing
    self._decoder_scope_name = "decoder"

  def _encode(self, features, training):
    inputs_BxI = features["inputs"]
    inputs_bias_Bx1xI = attention.ids_to_bias(inputs_BxI, self._dtype)
    states_BxIxD = self._embedding_layer(inputs_BxI, True)
    states_BxIxD = self._dropout_fn(
        timing.add_time_signal(states_BxIxD), training)
    with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
      states_BxIxD = transformer_block.stack(self._encoder_layers, training,
                                             states_BxIxD, inputs_bias_Bx1xI,
                                             None, None)
      states_BxIxD = contrib_layers.layer_norm(states_BxIxD, begin_norm_axis=2)
    return {"memory": states_BxIxD, "memory_bias": inputs_bias_Bx1xI}

  def __call__(self, features, training):
    """Create model.

    Args:
      features: dictionary of tensors including "inputs" [batch, input_len] and
        "targets" [batch, output_len]
      training: bool of whether the mode is training.

    Returns:
     Tuple of (loss, outputs): Loss is a scalar. Output is a dictionary of
       tensors, containing model's output logits.
    """
    if "inputs" not in features or "targets" not in features:
      raise ValueError("Require inputs and targets keys in features.")

    context = self._encode(features, training)
    self._context = context
    targets_BxT = features["targets"]
    bias_1xTxT = attention.upper_triangle_bias(
        tf.shape(targets_BxT)[1], self._dtype)
    states_BxTxD = self._embedding_layer(targets_BxT, True)
    states_BxTxD = tf.pad(states_BxTxD, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
    states_BxTxD = timing.add_time_signal(states_BxTxD)
    states_BxTxD = self._dropout_fn(states_BxTxD, training)
    with tf.variable_scope(self._decoder_scope_name, reuse=tf.AUTO_REUSE):
      states_BxTxD = transformer_block.stack(self._decoder_layers, training,
                                             states_BxTxD, bias_1xTxT,
                                             context["memory"],
                                             context["memory_bias"])
      states_BxTxD = contrib_layers.layer_norm(states_BxTxD, begin_norm_axis=2)
    logits_BxTxV = self._embedding_layer(states_BxTxD, False)
    targets_mask_BxT = tf.cast(tf.greater(targets_BxT, 0), self._dtype)

    XENT_loss = tf.losses.softmax_cross_entropy(
        tf.one_hot(targets_BxT, self._vocab_size),
        logits_BxTxV,
        label_smoothing=self._label_smoothing,
        weights=targets_mask_BxT)

    # want the one hot targets for sampling
    one_hot_targets = tf.one_hot(targets_BxT, self._vocab_size)

    return XENT_loss, {"logits": logits_BxTxV, "targets": targets_BxT, "one_hot_targets": one_hot_targets,
                       "hidden_states": states_BxTxD, "context_memory": context["memory"], "context_bias": context[
                        "memory_bias"]}

  def predict(self, features, max_decode_len, beam_size, **beam_kwargs):
    """Predict."""
    cache = self._encode(features, False)
    B, _, D = cache["memory"].shape
    T, V, H = max_decode_len, self._vocab_size, self._num_heads

    bias_1xTxT = attention.upper_triangle_bias(T, self._dtype)
    for i in range(len(self._decoder_layers)):
      cache[str(i)] = {
          "k": tf.zeros([B, H, T, D // H], self._dtype),
          "v": tf.zeros([B, H, T, D // H], self._dtype)
      }

    def symbols_to_logits_fn(dec_BxT, context, i):
      """Decode loop."""
      dec_Bx1 = tf.slice(dec_BxT, [0, tf.maximum(tf.cast(0, i.dtype), i - 1)],
                         [dec_BxT.shape[0], 1])
      bias_1x1xT = tf.slice(bias_1xTxT, [0, i, 0], [1, 1, T])
      dec_Bx1xD = self._embedding_layer(dec_Bx1, True)
      dec_Bx1xD *= tf.cast(tf.greater(i, 0), self._dtype)
      dec_Bx1xD = timing.add_time_signal(dec_Bx1xD, start_index=i)
      with tf.variable_scope(self._decoder_scope_name, reuse=tf.AUTO_REUSE):
        dec_Bx1xD = transformer_block.stack(self._decoder_layers, False,
                                            dec_Bx1xD, bias_1x1xT,
                                            context["memory"],
                                            context["memory_bias"], context, i)
        dec_Bx1xD = contrib_layers.layer_norm(dec_Bx1xD, begin_norm_axis=2)
      logits_Bx1xV = self._embedding_layer(dec_Bx1xD, False)
      logits_BxV = tf.squeeze(logits_Bx1xV, axis=1)
      return logits_BxV

    decodes_BxT, output_scores, logits_BxTxV = decoding.left2right_decode(symbols_to_logits_fn, cache, B, T,
                                                                          V, beam_size, **beam_kwargs)
    return {"outputs": decodes_BxT}, output_scores, logits_BxTxV

  def double_sampling(self, features, training, batchsize, seqlen, mixed=False):
      if "inputs" not in features or "targets" not in features:
          raise ValueError("Require inputs and targets keys in features.")

      # First "loop" - uses ground truth to supplement
      context = self._encode(features, training)
      self._context = context
      targets_BxT = features["targets"]
      bias_1xTxT = attention.upper_triangle_bias(
          tf.shape(targets_BxT)[1], self._dtype)
      states_BxTxD = self._embedding_layer(targets_BxT, True)
      states_BxTxD = tf.pad(states_BxTxD, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
      states_BxTxD = timing.add_time_signal(states_BxTxD)
      states_BxTxD = self._dropout_fn(states_BxTxD, training)
      with tf.variable_scope(self._decoder_scope_name, reuse=tf.AUTO_REUSE):
          states_BxTxD = transformer_block.stack(self._decoder_layers, training,
                                                 states_BxTxD, bias_1xTxT,
                                                 context["memory"],
                                                 context["memory_bias"])
          states_BxTxD = contrib_layers.layer_norm(states_BxTxD, begin_norm_axis=2)
      logits_BxTxV = self._embedding_layer(states_BxTxD, False)
      targets_mask_BxT = tf.cast(tf.greater(targets_BxT, 0), self._dtype)

      # argmax the logits to get teacher-forcing sequence
      # ensure this does not have any EOS apart from the end token, before passing into next loop.
      new_input = tf.reshape(tf.math.argmax(logits_BxTxV, axis=2), [batchsize, seqlen])

      # nucleus or top-k processing
      # new_input = iid_process_logits(logits_BxTxV, seqlen, batchsize, logits_BxTxV.get_shape().as_list()[-1],
      #                                top_k=0, top_p=0.9, temperature=1.0)

      # replace repeated EOS tokens with second-argmax
      # def tensor_loop(i, max_decode_len, logits, new_input, unused_targets_BxT):
      #     def f2(logits_BxTxV, new_input):
      #         topk_probs, topk_indices = tf.math.top_k(logits_BxTxV[0, i], k=2)
      #         topk_inds2 = tf.slice(topk_indices, [1], [1, ])
      #         return tf.tensor_scatter_nd_update(new_input, [[0, i]], tf.cast(topk_inds2, tf.int64))

      #     def f3(i, new_input):
      #         new_input2 = new_input[0].numpy().tolist()
      #         return True if new_input2[0][i] == 1 else False

      #     new_input = tf.cond(tf.py_function(f3, (i, [new_input]), tf.bool), lambda: f2(logits_BxTxV, new_input),
      #                         lambda: new_input)
      #     return i + 1, max_decode_len, logits, new_input, unused_targets_BxT

      # def finish_cond_ref(i, max_decode_len, unused_logits, unused_new_input, targets_BxT):
          # add here condition to return reference summary length

      #     def f4(i, targets, max_len):
      #         targets2 = targets[0].numpy().tolist()
      #         if targets2[i] == 0:  # padded token
      #           return i
      #         else:  # if not 1, still needs a number to refer to
      #           return max_len

      #     ref_len = tf.py_function(f4, (i, targets_BxT, max_decode_len), tf.int32)
      #     return i < ref_len  # T/F -> will change depending on padded token presence

      # def finish_cond_max(i, max_decode_len, unused_logits, unused_new_input, unused_targets_BxT):
      #     return i < max_decode_len

      # _, _, _, new_input, _ = tf.while_loop(finish_cond_max, tensor_loop,
      #                                       [0, seqlen, logits_BxTxV, new_input, targets_BxT])

      # find target length -> py.func() as it has to be outside graph
      def f5(targets_BxT):
          try:
              exist = targets_BxT[0].numpy().tolist().index(0)
          except ValueError:
              exist = targets_BxT.get_shape().as_list()[-1]

          return tf.Variable(exist, shape=()).read_value()

      cut_off = tf.py_function(f5, [targets_BxT], tf.int32)

      # implement cut_off for new_input
      new_input2 = tf.slice(new_input, [0, 0], [1, cut_off])
      # reshaping from unknown to known gives it tensor shape for second loop
      new_input = tf.reshape(tf.pad(new_input2, [[0, 0], [0, new_input.get_shape().as_list()[-1] - cut_off]],
                                    "CONSTANT"), [batchsize, seqlen])

      print(new_input)  # should be [BxT], and "cut-off" to target length with same padding

      # Second "loop" - uses predicted sequence as input
      context_2 = self._encode(features, training)
      # targets_BxT = features["targets"]
      bias_1xTxT_2 = attention.upper_triangle_bias(
          tf.shape(new_input)[1], self._dtype)
      states_BxTxD_2 = self._embedding_layer(new_input, True)
      states_BxTxD_2 = tf.pad(states_BxTxD_2, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
      states_BxTxD_2 = timing.add_time_signal(states_BxTxD_2)
      states_BxTxD_2 = self._dropout_fn(states_BxTxD_2, training)
      with tf.variable_scope(self._decoder_scope_name, reuse=tf.AUTO_REUSE):
          states_BxTxD_2 = transformer_block.stack(self._decoder_layers, training,
                                                   states_BxTxD_2, bias_1xTxT_2,
                                                   context_2["memory"],
                                                   context_2["memory_bias"])
          states_BxTxD_2 = contrib_layers.layer_norm(states_BxTxD_2, begin_norm_axis=2)
      logits_BxTxV_2 = self._embedding_layer(states_BxTxD_2, False)
      targets_mask_BxT_2 = tf.cast(tf.greater(new_input, 0), self._dtype)

      # mixed parallel scheduled sampling
      if mixed:
        bool_mask = np.random.choice([True, False], [batchsize, seqlen, logits_BxTxV.get_shape().as_list()[2]],
                                     p=[0.75, 0.25])
        mixed_logits = tf.where(bool_mask, logits_BxTxV, logits_BxTxV_2)
      else:
        mixed_logits = None

      XENT_loss = tf.losses.softmax_cross_entropy(
          tf.one_hot(new_input, self._vocab_size),
          logits_BxTxV_2,
          label_smoothing=self._label_smoothing,
          weights=targets_mask_BxT_2)

      # want the one hot targets for sampling
      one_hot_targets = tf.one_hot(new_input, self._vocab_size)
      return XENT_loss, {"sampled_BxT": new_input,
                         "logits1": logits_BxTxV,
                         "logits2": logits_BxTxV_2,
                         "targets": targets_BxT,
                         "mixed_logits": mixed_logits}
