import tensorflow as tf


def create_variables(z, logp, batch_index, sequence_index, clipped_logit_probs):
    """
    Create the variables for RELAX control variate
    :param z: soft Gumbel samples - from iid sampling, or beam sampling: [B,T,V] tensor
    :param logp: the [B,T,V] tensor of logp from the decoder (softmax of logits)
    :param batch_index: [B,T] tensor of the batch size repeated for seq len
    :param sequence_index: [B,T] tensor of range(0, seq len)
    :param clipped_logit_probs: same as logp, but not in log format -> clipped logits
    :return: z_tilde, and logp(b) for equation
    """
    v = tf.random_uniform(shape=logp.get_shape().as_list(),
                          minval=1e-8,
                          maxval=1,
                          dtype=tf.float32)
    b = tf.stop_gradient(tf.math.argmax(z, axis=2))  # this is Gumbel-Max (used for RELAX)

    # create index tensor where b is the argmax, to use as indexer for substitution
    b_new = tf.squeeze(b, 0)
    index_tensor_b = tf.expand_dims(tf.stack([batch_index, sequence_index, b_new], axis=1), 0)

    v_b = tf.gather_nd(v, index_tensor_b)  # values of v where b are the argmax indexes
    update = -tf.log(-tf.log(v_b))  # for i == b

    # create z_tilde as for the case where i != b
    z_tilde = -tf.log(-tf.div(tf.log(v), clipped_logit_probs) - tf.expand_dims(tf.log(v_b), 2))
    z_tilde = tf.tensor_scatter_nd_update(z_tilde, index_tensor_b, update)

    logp_b = tf.gather_nd(logp, index_tensor_b)  # used in loss func
    return z_tilde, logp_b


def create_variables_from_samples(sample_z, sample_b, batch_index, sequence_index):
    """
    Create the variables for RELAX control variate
    :param sample_z: [B,T,V] tensor containing sampled processed logp created by stacking logp during
                    decoding loop of sampling process
    :param sample_b: the [B,T] tensor containing the H(z) indices (Gumbel-Max)
    :param batch_index: [B,T] tensor of the batch size repeated for seq len
    :param sequence_index: [B,T] tensor of range(0, seq len)
    :return: z_tilde, and logp(b) for equation
    """
    v = tf.random_uniform(shape=sample_z.get_shape().as_list(),
                          minval=1e-8,
                          maxval=1,
                          dtype=tf.float32)

    # create index tensor where b is the argmax, to use as indexer for substitution
    b_new = tf.squeeze(sample_b, 0)
    index_tensor_b = tf.expand_dims(tf.stack([batch_index, sequence_index, b_new], axis=1), 0)

    v_b = tf.gather_nd(v, index_tensor_b)  # values of v where b are the argmax indexes
    update = -tf.log(-tf.log(v_b))  # for i == b

    # create z_tilde as for the case where i != b
    clipped_logit_probs = tf.clip_by_value(tf.math.softmax(sample_z), 1e-8, 1.0)
    z_tilde = -tf.log(-tf.div(tf.log(v), clipped_logit_probs) - tf.expand_dims(tf.log(v_b), 2))
    z_tilde = tf.tensor_scatter_nd_update(z_tilde, index_tensor_b, update)

    logp_b = tf.gather_nd(sample_z, index_tensor_b)  # used in loss func
    return z_tilde, logp_b


def create_cv_target(outputs, batch_index, sequence_index, z, z_tilde):
    """
    Converts the target IDs to probabilities from [B,T,V] tensor of z defined in above function.
    :param outputs: output from final layer of transformer
    :param batch_index: [B,T] tensor of the batch size repeated for seq len
    :param sequence_index: [B,T] tensor of range(0, seq len)
    :param z: soft Gumbel samples - from iid sampling, or beam sampling: [B,T,V] tensor
    :param z_tilde: [B,T,V] tensor parameterized by b=H(z)
    :return: target probabilities for z and z_tilde
    """
    # Here we need to convert the IDs from the target, to the probabilities for ROUGE mimic
    target_id_cv = tf.reshape(outputs['targets'], [outputs['targets'].get_shape().as_list()[1]])
    index_tensor_target = tf.stack([batch_index, sequence_index, target_id_cv], axis=1)

    # finds log probs using targets indexing
    tgt_probs_cv_z = tf.expand_dims(tf.expand_dims(tf.gather_nd(z, index_tensor_target), 0), 2)
    tgt_probs_cv_ztilde = tf.expand_dims(tf.expand_dims(tf.gather_nd(z_tilde, index_tensor_target), 0), 2)

    z_target = tf.broadcast_to(tgt_probs_cv_z, z.get_shape().as_list())
    zt_target = tf.broadcast_to(tgt_probs_cv_ztilde, z_tilde.get_shape().as_list())

    return z_target, zt_target
