import tensorflow as tf


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


def beam_sampling(model_params, features, max_seq_len, beam_params):
    """
    Uses Beam Search to sample the decoder using various different sampling methods, defined in pegasus.layers.decoding
    :param model_params: parameters for the PEGASUS model
    :param features: inputs and targets dict
    :param max_seq_len: the maximum sequence length for given dataset
    :param beam_params: parameters for sampling method, beam size should be no bigger than 3 (memory)
    :return: IDs returned by beam search, and the respective sum(logp) score for that sequence
    """
    # TODO: need to implement returning logits in the decoding.py/beam_search.py file. Currently only returns a
    #  sentence score for the sampled sequence
    # SAMPLE TOKENS USING BEAM SEARCH
    preds_dict, preds_scores = model_params.model().predict(features, max_seq_len,
                                                            beam_size=beam_params["_beam"],
                                                            top_k=beam_params["top_k"],
                                                            top_p=beam_params["top_p"],
                                                            temperature=beam_params["temperature"])
    preds = preds_dict["outputs"][0]  # gets the IDs
    preds_score = preds_scores[:, 0]  # sentence score (sum of log_prob) for first
    preds2, preds_score2, preds3, preds_score3 = None, None, None, None

    if beam_params["_beam"] == 2:
        preds2 = preds_dict["outputs"][1]  # gets the IDs of second best
        preds_score2 = preds_scores[:, 1]  # sentence score (sum of log_prob) for second
    elif beam_params["_beam"] == 3:
        preds3 = preds_dict["outputs"][2]  # gets the IDs of third best
        preds_score3 = preds_scores[:, 2]  # sentence score (sum of log_prob) for third

    return preds, preds_score, preds2, preds_score2, preds3, preds_score3


def non_beam_sampling(model_params, features, max_seq_len, beam_params, sentence_score=False):
    """
    Samples the decoder using various different sampling methods, defined in pegasus.layers.decoding - not using
    beam search.
    :param model_params: parameters for the PEGASUS model
    :param features: inputs and targets dict
    :param max_seq_len: the maximum sequence length for given dataset
    :param beam_params: parameters for sampling method, beam size defaults to 1
    :param sentence_score: boolean, flag to determine if we should use logits/sentence score for further tests
    :return: IDs returned by beam search, the logits, possibly the sentence score (scalar) for pred seq, and also
             the [BxN] logits stacked from each decoding loop into a [BxTxV] tensor.
    """
    # SAMPLE TOKENS FROM DECODER (NOT USING BEAM SEARCH)
    preds_dict, preds_scores, preds_logits_BxTxV = model_params.model().predict(features, max_seq_len,
                                                                                beam_size=1,
                                                                                top_k=beam_params["top_k"],
                                                                                top_p=beam_params["top_p"],
                                                                                temperature=beam_params["temperature"])
    preds = preds_dict["outputs"][0]  # gets the IDs
    # not using beam search allows us to return logits instead of scalar sentence scores
    logp = tf.squeeze(tf.log(tf.clip_by_value(tf.math.softmax(preds_scores), 1e-8, 1.0)), 0)  # to log_p

    if sentence_score:
        score = tf.exp((1 / max_seq_len) * tf.reduce_sum(logp))  # sentence score 0-1
    else:
        score = None
    return preds, logp, score, preds_logits_BxTxV
