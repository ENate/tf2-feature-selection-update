import argparse
import tensorflow as tf
from tensorflow import keras
from adapted_lm_algorithm import ParseArgs


def make_argparse():
    """An argparse function"""
    parsed = argparse.ArgumentParser()
    parsed.add_argument('--time_delta_n', help='Delta time change', default=0, type=float)
    arguments = parsed.parse_args()
    return arguments

def function_call_argparse():
    def_values = make_argparse()
    print(def_values.time_delta_n)

# Passing input dimension to initialize weights before building network model
class Linear(keras.layers.Layer):
    def __init__(self, input_dim=2, units=2):
        super(Linear, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, units), dtype="float32"),
            trainable=True,
        )
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(units,), dtype="float32"), trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


# Unknown number of input dimensions
    def __init__(self, units=32):
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


if __name__ == '__main__':
    arg_p = ParseArgs([3, 2])
    # print(arg_p.parse_command_line_argument_function().mlp_hid_structure)
    x = tf.ones((2, 2))
    # linear_layer = Linear(2, 4)
    # y = linear_layer(x)
    # print(y)

    # At instantiation, we don't know on what inputs this is going to get called
    linear_layer = Linear(10)
    print(linear_layer)
    # The layer's weights are created dynamically the first time the layer is called
    # y = linear_layer(x)