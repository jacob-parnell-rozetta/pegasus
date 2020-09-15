import tensorflow as tf
from tensorflow.keras import layers

"""
def ffn_model(inputs, ground_truth):
    # TODO: pass the inputs AND ground_truth with two encoders, and an output layer that generates a single score
    ffn_input_size = inputs.get_shape().as_list()[2]  # max input length [x,y,z]
    ffn_target_size = ground_truth.get_shape().as_list()[2]  # hidden state size 1024 [B, I, D]
    hidden1_size = 128
    ffn_output_size = 1  # scalar value to subtract from rouge loss

    # x = tf.placeholder(tf.float32, [None, ffn_input_size], name='data')
    # y = tf.placeholder(tf.float32, [None, ffn_target_size], name='targets')

    # Names used to extract trainable variables in RELAX variance reduction
    weights = {"w1": tf.Variable(tf.random_normal([ffn_input_size, hidden1_size]) * 0.01, name="control_variate_w1"),
               "w2": tf.Variable(tf.random_normal([hidden1_size, ffn_output_size]) * 0.01, name="control_variate_w2")}

    biases = {"b1": tf.Variable(tf.zeros([hidden1_size]), name="control_variate_b1"),
              "b2": tf.Variable(tf.zeros([ffn_output_size]), name="control_variate_b2")}

    def shallow_network_1(x_input, w, b):
        layer_1 = tf.add(tf.matmul(x_input, w['w1']), b['b1'])
        layer_1 = tf.nn.relu(layer_1)
        output_layer = tf.add(tf.matmul(layer_1, w['w2']), b['b2'])
        return output_layer

    def shallow_network_2(y_input, w, b):
        layer_1 = tf.add(tf.matmul(y_input, w['w1']), b['b1'])
        layer_1 = tf.nn.relu(layer_1)
        output_layer = tf.add(tf.matmul(layer_1, w['w2']), b['b2'])
        return output_layer

    with tf.variable_scope("control_variate", reuse=tf.AUTO_REUSE):
        ffn_output_1 = shallow_network_1(inputs, weights, biases)  # baseline scorer for inputs
        ffn_output_2 = shallow_network_2(inputs, weights, biases)  # baseline scorer for inputs

        combined = tf.nn.relu(tf.concat([ffn_output_1, ffn_output_2]))  # concatenate
        # FC layer
        # regression prediction
    return ffn_output"""


def ffn_model(inputs, ground_truth):
    inputA = layers.Input(shape=(inputs.get_shape().as_list()[2],))  # max input length [x,y,z]
    inputB = layers.Input(shape=(ground_truth.get_shape().as_list()[2],))  # hidden state size 1024 [B, I, D]

    # inputs
    x = layers.Dense(128, activation='relu')(inputA)
    x = tf.keras.Model(inputs=inputA, outputs=x)

    # ground truth
    y = layers.Dense(128, activation='relu')(inputB)
    y = tf.keras.Model(inputs=inputB, outputs=y)

    combined = layers.concatenate([x.output, y.output])

    # FC layer and regression prediction
    z = layers.Dense(2, activation='relu')(combined)
    z = layers.Dense(1, activation='linear')(z)

    ffn_output = tf.keras.Model(inputs=[x.input, y.input], outputs=z)

    return ffn_output
