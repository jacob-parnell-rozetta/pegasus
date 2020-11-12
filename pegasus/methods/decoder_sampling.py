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


def beam_sampling(model_params, features, max_seq_len, batch_index, sequence_index, beam_params):
    """
    Uses Beam Search to sample the decoder using various different sampling methods, defined in pegasus.layers.decoding
    :param model_params: parameters for the PEGASUS model
    :param features: inputs and targets dict
    :param max_seq_len: the maximum sequence length for given dataset
    :param batch_index: batch index for indexing
    :param sequence_index: corresponding token
    :param beam_params: parameters for sampling method, beam size should be no bigger than 3 (memory)
    :return: IDs returned by beam search, and the respective sum(logp) score for that sequence, soon: logp_BxMxTxV
    """
    # SAMPLE TOKENS USING BEAM SEARCH
    preds_dict, preds_scores, beam_logp_dict = model_params.model().predict(features, max_seq_len,
                                                                            beam_size=beam_params["_beam"],
                                                                            top_k=beam_params["top_k"],
                                                                            top_p=beam_params["top_p"],
                                                                            temperature=beam_params["temperature"],
                                                                            sampling=True)
    preds = preds_dict["outputs"][0]  # gets the IDs
    preds_score = preds_scores[:, 0]  # sentence score (sum of log_prob) for first
    logp1_BxTxV = beam_logp_dict["beam_1logp"]  # [B,T,V] tensor we need to index with IDs
    index_tensor1 = tf.stack([batch_index, sequence_index, logp1_BxTxV], axis=1)
    logp1_BxT = tf.gather_nd(logp1_BxTxV, index_tensor1)  # extract logps at ids

    preds2, preds_score2, preds3, preds_score3 = None, None, None, None
    logp2_BxT, logp3_BxT = None, None

    if beam_params["_beam"] == 2:
        preds2 = preds_dict["outputs"][1]  # gets the IDs of second best
        preds_score2 = preds_scores[:, 1]  # sentence score (sum of log_prob) for second
        logp2_BxTxV = beam_logp_dict["beam_2logp"]  # [B,T,V] tensor we need to index with IDs
        index_tensor2 = tf.stack([batch_index, sequence_index, logp2_BxTxV], axis=1)

        logp2_BxT = tf.gather_nd(logp2_BxTxV, index_tensor2)  # extract logps at ids
        logp3_BxT = None

    elif beam_params["_beam"] == 3:
        preds2 = preds_dict["outputs"][1]  # gets the IDs of second best
        preds_score2 = preds_scores[:, 1]  # sentence score (sum of log_prob) for second
        preds3 = preds_dict["outputs"][2]  # gets the IDs of third best
        preds_score3 = preds_scores[:, 2]  # sentence score (sum of log_prob) for third

        logp2_BxTxV = beam_logp_dict["beam_2logp"]  # [B,T,V] tensor we need to index with IDs
        index_tensor2 = tf.stack([batch_index, sequence_index, logp2_BxTxV], axis=1)
        logp2_BxT = tf.gather_nd(logp2_BxTxV, index_tensor2)  # extract logps at ids
        logp3_BxTxV = beam_logp_dict["beam_3logp"]  # [B,T,V] tensor we need to index with IDs
        index_tensor3 = tf.stack([batch_index, sequence_index, logp3_BxTxV], axis=1)
        logp3_BxT = tf.gather_nd(logp3_BxTxV, index_tensor3)  # extract logps at ids

    return {"ids1": preds, "sent_score1": preds_score, "logp1": logp1_BxT,
            "ids2": preds2, "sent_score2": preds_score2, "logp2": logp2_BxT,
            "ids3": preds3, "sent_score3": preds_score3, "logp3": logp3_BxT}, beam_logp_dict


def non_beam_sampling(model_params, features, max_seq_len, beam_params, sentence_score=False):
    """
    Samples the decoder using various different sampling methods, defined in pegasus.layers.decoding - not using
    beam search.
    :param model_params: parameters for the PEGASUS model
    :param features: inputs and targets dict
    :param max_seq_len: the maximum sequence length for given dataset
    :param beam_params: parameters for sampling method, beam size defaults to 1
    :param sentence_score: boolean, flag to determine if we should use logits/sentence score for further tests
    :return: IDs returned by beam search, the logp, possibly the sentence score (scalar) for pred seq, and also
             the [BxN] logits stacked from each decoding loop into a [BxTxV] tensor.
    """
    # SAMPLE TOKENS FROM DECODER (NOT USING BEAM SEARCH)
    preds_dict, preds_logp_BxT, preds_logp_BxTxV = model_params.model().predict(features, max_seq_len,
                                                                                beam_size=1,
                                                                                top_k=beam_params["top_k"],
                                                                                top_p=beam_params["top_p"],
                                                                                temperature=beam_params["temperature"],
                                                                                sampling=True)
    preds = preds_dict["outputs"][0]  # gets the IDs

    if sentence_score:
        score = tf.exp((1 / max_seq_len) * tf.reduce_sum(preds_logp_BxT))  # sentence score 0-1
    else:
        score = None
    return {"ids": tf.reshape(preds, [model_params.batch_size, max_seq_len]),
            "logp_BxT": preds_logp_BxT, "sent_score": score, "logp_BxTxV": preds_logp_BxTxV}
