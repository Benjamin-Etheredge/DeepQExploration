from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
np.random.seed(4)
tf.random.set_seed(4)

class Q_Prime_Layer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        self.gamma = tf.constant(0.99)
        super(Q_Prime_Layer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        super(Q_Prime_Layer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        assert isinstance(x, list)
        state_action_values, next_state_action_values, reward, is_done = x
        #print("q call")
        #print(state_action_values.shape)
        #print(next_state_action_values.shape)
        ##print(reward.shape)
        #print(is_done.shape)
        q_prime = 0.
        # TODO should axis be zero?
        squeezed_done = tf.squeeze(is_done, axis=[1]) # must specify axis due to inference sometimes having a batch size of 1
        action_values = K.max(next_state_action_values, axis=1)
        zeroes = tf.zeros((1,))
        new_values = tf.where(squeezed_done, zeroes, action_values)
        #new_values = tf.where(is_done, tf.zeros(is_done.shape[0]), K.max(next_state_action_values, axis=1))
        #return new_values
        squeezed_reward = tf.squeeze(reward, axis=[1]) # must specify axis due to inference sometimes having a batch size of 1
        adjusted_q_prime = (new_values * self.gamma) + squeezed_reward
        return adjusted_q_prime


class MyLayer(tf.keras.layers.Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        assert isinstance(x, list)
        state_action_values, action, q_prime = x
        #squeezed_action = tf.squeeze(action, axis=[1])
        #return [state_action_values
        #exp_q = tf.expand_dims(q_prime, -1)
        cols = tf.squeeze(action, axis=[1])
        rows = tf.range(tf.shape(action)[0])
        indicies = tf.stack([rows, cols], axis=-1)

        return tf.tensor_scatter_nd_update(state_action_values, indicies, q_prime)
        #return tf.tensor_scatter_nd_update(state_action_values, action, q_prime)
        #return tf.tensor_scatter_nd_update(state_action_values, squeezed_action, q_prime)
        # TODO does printing slow down?
        #print(state_action_values.shape)
        #print(action.shape)
        #print(q_prime.shape)

        #return tf.tensor_scatter_nd_update(state_action_values, action, q_prime)
        #return tf.tensor_scatter_nd_update(action, state_action_values, q_prime)


    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [(shape_a[0], self.output_dim), shape_b[:-1]]


def mean(array):
    return keras.backend.mean(array, axis=-1, keepdims=True)