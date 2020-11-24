import tensorflow as tf


def risk_loss(batch_size, max_seq_len, rouge_losses=None, logps=None, n=2):
    """
    Calculates the expected risk minimisation loss, given by:
    L_risk = -r(u,y)*p(u|x,theta) -> U(x) is a set of candidate translations
    :param batch_size: batch size for dataset - normally 1
    :param max_seq_len: the maximum sequence length for a given dataset
    :param rouge_losses: the rouge losses that will be used for this loss
    :param logps: the logp that are derived from the beam search algorithm. If beam_size > 1, the returned logp
                    value from the beam will be a score anyway (will not affect the equation)
    :param n: the number of samples to pass into the loss
    :return: Scalar value denoting the risk loss
    """
    # reshape to BxT
    logps[0] = tf.reshape(logps[0], [batch_size, max_seq_len])
    logps[1] = tf.reshape(logps[1], [batch_size, max_seq_len])

    if n == 3:
        logps[2] = tf.reshape(logps[2], [batch_size, max_seq_len])

    if n == 2:
        # Calculate f_u for as many sequences
        f_u_1 = tf.exp(tf.reduce_mean(logps[0], axis=1))
        f_u_2 = tf.exp(tf.reduce_mean(logps[1], axis=1))

        # Calculate p_u for as many sequences
        p_u_1 = f_u_1 / tf.reduce_sum([f_u_1, f_u_2])
        p_u_2 = f_u_2 / tf.reduce_sum([f_u_1, f_u_2])

        # Calculate each risk loss
        L_risk_1 = tf.reduce_sum(tf.multiply(rouge_losses[0], p_u_1))
        L_risk_2 = tf.reduce_sum(tf.multiply(rouge_losses[1], p_u_2))

        # Overall Risk loss
        L_risk = tf.reduce_sum([L_risk_1, L_risk_2])
        return L_risk

    elif n == 3:
        # Calculate f_u for as many sequences
        f_u_1 = tf.exp(tf.reduce_mean(logps[0], axis=1))
        f_u_2 = tf.exp(tf.reduce_mean(logps[1], axis=1))
        f_u_3 = tf.exp(tf.reduce_mean(logps[2], axis=1))

        # Calculate p_u for as many sequences
        p_u_1 = f_u_1 / tf.reduce_sum([f_u_1, f_u_2, f_u_3])
        p_u_2 = f_u_2 / tf.reduce_sum([f_u_1, f_u_2, f_u_3])
        p_u_3 = f_u_3 / tf.reduce_sum([f_u_1, f_u_2, f_u_3])

        # Calculate each risk loss
        L_risk_1 = tf.reduce_sum(tf.multiply(rouge_losses[0], p_u_1))
        L_risk_2 = tf.reduce_sum(tf.multiply(rouge_losses[1], p_u_2))
        L_risk_3 = tf.reduce_sum(tf.multiply(rouge_losses[2], p_u_3))

        # Overall Risk loss
        L_risk = tf.reduce_sum([L_risk_1, L_risk_2, L_risk_3])
        return L_risk
