import tensorflow as tf
from pegasus.layers.decoding import process_logits, inplace_update_i


def iid_sampling(logp, max_seq_len, greedy=True, soft=False, topk=False, k=2):
    """
    I.I.D sampling from the decoder, rather than using beam search.
    :param logp: logits returned from the decoder
    :param max_seq_len: maximum sequence length for the decoder
    :param greedy: argmax samples
    :param soft: soft samples
    :param topk: top-k samples (tuple of probs and indexes)
    :param k: the top-k values
    :return: the IDs for each sampled sequence, and z for RELAX
    """
    if greedy:
        argmax_logp_index = tf.math.argmax(logp, axis=2)  # returns indices where logp is max
    else:
        argmax_logp_index = None

    if soft:
        u = tf.random_uniform(shape=logp.get_shape().as_list(),
                              minval=1e-8,
                              maxval=1,
                              dtype=tf.float32)
        z = tf.math.add(-tf.log(-tf.log(u)), logp)  # Return for RELAX?

        # use y_soft and sample_y for REINFORCE -> RELAX uses b = H(z)
        y_soft = tf.math.softmax(tf.div(z, 0.1))  # this is Gumbel-Softmax; low temp -> approaches argmax
        sample_y = tf.math.argmax(y_soft, axis=2)
    else:
        z = None
        sample_y = None

    if topk:
        topk_probs, topk_indices = tf.math.top_k(logp, k=k)

        # finds the probabilities
        topk_probs_2 = tf.slice(topk_probs, [0, 0, 1], [1, max_seq_len, 1])
        topk_probs_2 = tf.squeeze(topk_probs_2, 2)
        # finds the indexes
        topk_indices_2 = tf.slice(topk_indices, [0, 0, 1], [1, max_seq_len, 1])
        topk_indices_2 = tf.squeeze(topk_indices_2, 2)
        topk_out = (topk_probs_2, topk_indices_2)
    else:
        topk_out = None

    return argmax_logp_index, sample_y, topk_out, z


def non_beam_sampling(model_params, features, max_seq_len, beam_params, sentence_score=False):
    """
    Samples the decoder using various different sampling methods, defined in pegasus.layers.decoding.
    :param model_params: parameters for the PEGASUS model
    :param features: inputs and targets dict
    :param max_seq_len: the maximum sequence length for given dataset
    :param beam_params: parameters for sampling method, beam size defaults to 1
    :param sentence_score: boolean, flag to determine if we should use logits/sentence score for further tests
    :return: IDs returned by beam search, the logp, possibly the sentence score (scalar) for pred seq, and also
             the [BxN] logits stacked from each decoding loop into a [BxTxV] tensor.
    """
    # SAMPLE TOKENS FROM DECODER (NOT USING BEAM SEARCH)
    preds_dict, _, preds_logits_BxTxV = model_params.model().predict(features, max_seq_len,
                                                                     beam_size=1,
                                                                     top_k=beam_params["top_k"],
                                                                     top_p=beam_params["top_p"],
                                                                     temperature=beam_params["temperature"],
                                                                     training=True)
    preds = preds_dict["outputs"][0]  # gets the IDs -> by default are argmax(logits) or H(z)

    # convert logits to logp and extract logp values
    logp_BxTxV = tf.log(tf.clip_by_value(tf.math.softmax(preds_logits_BxTxV, axis=2), 1e-8, 1.0))
    preds_logp_BxT = tf.reshape(tf.reduce_max(logp_BxTxV, axis=2), [model_params.batch_size, max_seq_len])

    if sentence_score:
        score = tf.exp((1 / max_seq_len) * tf.reduce_sum(preds_logp_BxT, axis=1))  # sentence score 0-1
    else:
        score = None
    return {"ids": tf.reshape(preds, [model_params.batch_size, max_seq_len]),
            "logp_BxT": preds_logp_BxT, "sent_score": score,
            "logp_BxTxV": logp_BxTxV, "logits_BxTxV": preds_logits_BxTxV}


def beam_sampling(model_params, features, max_seq_len, batch_index, sequence_index, beam_params):
    """
    Uses Beam Search to sample the decoder using various different sampling methods, defined in
    #     pegasus.layers.decoding
    :param model_params: parameters for the PEGASUS model
    :param features: inputs and targets dict
    :param max_seq_len: the maximum sequence length for given dataset
    :param batch_index: batch index for indexing
    :param sequence_index: corresponding token
    :param beam_params: parameters for sampling method, beam size should be no bigger than 3 (memory)
    :return: IDs returned by beam search, and the respective sum(logp) score for that sequence, soon: logp_BxMxTxV
    """
    # SAMPLE TOKENS USING BEAM SEARCH
    preds_dict, preds_scores, beam_dict = model_params.model().predict(features, max_seq_len,
                                                                       beam_size=beam_params["_beam"],
                                                                       top_k=beam_params["top_k"],
                                                                       top_p=beam_params["top_p"],
                                                                       temperature=beam_params["temperature"],
                                                                       training=True)
    preds = preds_dict["outputs"][0]  # gets the IDs
    preds_score = preds_scores[:, 0]  # sentence score (sum of log_prob) for first
    logp1_BxTxV = tf.log(tf.clip_by_value(tf.math.softmax(beam_dict["beam_1_logits"], axis=2), 1e-8, 1.0))

    index_tensor1 = tf.stack([batch_index, sequence_index, tf.reshape(preds, [max_seq_len])], axis=1)
    logp1_BxT = tf.gather_nd(logp1_BxTxV, index_tensor1)  # extract logps at ids

    preds2, preds_score2, preds3, preds_score3 = None, None, None, None
    logp2_BxT, logp3_BxT = None, None
    logp2_BxTxV, logp3_BxTxV = None, None

    if beam_params["_beam"] == 2:
        preds2 = preds_dict["outputs"][1]  # gets the IDs of second best
        preds_score2 = preds_scores[:, 1]  # sentence score (sum of log_prob) for second
        logp2_BxTxV = tf.log(tf.clip_by_value(tf.math.softmax(beam_dict["beam_2_logits"], axis=2), 1e-8, 1.0))
        index_tensor2 = tf.stack([batch_index, sequence_index, tf.reshape(preds2, [max_seq_len])], axis=1)

        logp2_BxT = tf.gather_nd(logp2_BxTxV, index_tensor2)  # extract logps at ids
        logp3_BxT = None
        logp3_BxTxV = None

    elif beam_params["_beam"] == 3:
        preds2 = preds_dict["outputs"][1]  # gets the IDs of second best
        preds_score2 = preds_scores[:, 1]  # sentence score (sum of log_prob) for second
        preds3 = preds_dict["outputs"][2]  # gets the IDs of third best
        preds_score3 = preds_scores[:, 2]  # sentence score (sum of log_prob) for third

        logp2_BxTxV = tf.log(tf.clip_by_value(tf.math.softmax(beam_dict["beam_2_logits"], axis=2), 1e-8, 1.0))
        index_tensor2 = tf.stack([batch_index, sequence_index, tf.reshape(preds2, [max_seq_len])], axis=1)
        logp2_BxT = tf.gather_nd(logp2_BxTxV, index_tensor2)  # extract logps at ids
        logp3_BxTxV = tf.log(tf.clip_by_value(tf.math.softmax(beam_dict["beam_3_logits"], axis=2), 1e-8, 1.0))
        index_tensor3 = tf.stack([batch_index, sequence_index, tf.reshape(preds3, [max_seq_len])], axis=1)
        logp3_BxT = tf.gather_nd(logp3_BxTxV, index_tensor3)  # extract logps at ids

    return {"ids1": preds, "sent_score1": preds_score, "logp1": logp1_BxT, "logp1_BxTxV": logp1_BxTxV,
            "logits1_BxTxV": beam_dict["beam_1_logits"],
            "ids2": preds2, "sent_score2": preds_score2, "logp2": logp2_BxT, "logp2_BxTxV": logp2_BxTxV,
            "ids3": preds3, "sent_score3": preds_score3, "logp3": logp3_BxT, "logp3_BxTxV": logp3_BxTxV}


def iid_process_logits(logits_BxTxV, max_decode_len, batchsize, vocab_size, top_k=0, top_p=0.0, temperature=0.0):
    # loop over logits along T axis, and process similar to decoding
    def logits_loop(i, decode_BxT, logits_BxTxV):
        logits_BxV = tf.reshape(logits_BxTxV[0][i], [batchsize, vocab_size])
        logits_BxV = process_logits(logits_BxV, top_k, top_p, temperature)
        sampled_BxT = inplace_update_i(decode_BxT, tf.argmax(logits_BxV, -1), i)
        return i + 1, sampled_BxT, logits_BxTxV

    def loop_cond(i, decode_BxT, logits_BxTxV):
        return i < max_decode_len

    init_decode_BxT = tf.zeros([batchsize, max_decode_len], tf.int64)
    _, sampled_BxT, _ = tf.while_loop(loop_cond, logits_loop,
                                      [tf.constant(0, tf.int64), init_decode_BxT, logits_BxTxV])

    return tf.reshape(sampled_BxT, [batchsize, max_decode_len])
