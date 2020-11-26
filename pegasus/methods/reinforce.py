import tensorflow as tf
from pegasus.ops import public_parsing_ops


def rouge_decoding(ids, model_params):
    """
    To convert the IDs to text using the vocab and SP encoder.
    :param ids: [B,T] tensor of ids in vocab
    :param model_params: the defined model parameters
    :return: decoded text removed from graph (stop grad)
    """
    decode_text_tensor = public_parsing_ops.decode(ids, model_params.vocab_filename, model_params.encoder_type)
    decode_text = tf.stop_gradient(decode_text_tensor[0])
    return decode_text


def rouge_token(ref, pred, pen=0, norm=0):
    ref = ref.numpy().tolist()
    pred = pred.numpy().tolist()
    token_level_score = [ref.count(x) for x in pred]
    if pen == 1:
        token_level_score = [x - 0.5 for x in token_level_score]

    if norm == 1:
        token_level_score = [x / len(ref) for x in token_level_score]

    return token_level_score


def iid_log_probs(ids, batch_index, sequence_index, logp):
    """
    Stacks the ids into a matrix that allows you to extract the corresponding logp
    from the iid samples from the decoder.
    :param ids: [B,T] tensor of ids in vocab
    :param batch_index: [B,T] tensor of the batch size repeated for seq len
    :param sequence_index: [B,T] tensor of range(0, seq len)
    :param logp: the [B,T,V] tensor of logp from the decoder (softmax of logits)
    :return: the corresponding log probabilities for the sampled IDs
    """
    logp_new = tf.reshape(ids, [ids.get_shape().as_list()[1]])
    index_tensor = tf.stack([batch_index, sequence_index, logp_new], axis=1)
    logp_out = tf.gather_nd(logp, index_tensor)  # finds log probs using indexing
    return logp_out
