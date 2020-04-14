from tensorflow.keras.layers import Layer, Lambda, Subtract, Add
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
np.random.seed(4)
tf.random.set_seed(4)

class QPrimeLayer(Layer):

    def __init__(self, **kwargs):
        self.gamma = tf.constant(0.99)
        super(QPrimeLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        super(QPrimeLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        assert isinstance(x, list)
        next_state_action_values, reward, is_done = x
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

class DoubleQPrimeLayer(Layer):

    def __init__(self, **kwargs):
        self.gamma = tf.constant(0.99)
        super(DoubleQPrimeLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        super(DoubleQPrimeLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        assert isinstance(x, list)
        next_state_actual_values, next_state_action_values, reward, is_done = x
        #print("q call")
        #print(state_action_values.shape)
        #print(next_state_action_values.shape)
        ##print(reward.shape)
        #print(is_done.shape)
        q_prime = 0.
        # TODO should axis be zero?
        squeezed_done = tf.squeeze(is_done, axis=[1]) # must specify axis due to inference sometimes having a batch size of 1
        action_values_idxs = K.argmax(next_state_action_values, axis=1)

        cols = tf.cast(action_values_idxs, dtype=tf.int32)
        rows = tf.range(tf.shape(action_values_idxs)[0])
        indicies = tf.stack([rows, cols], axis=-1)

        action_values = tf.gather_nd(next_state_actual_values, indices=indicies)



        zeroes = tf.zeros((1,))
        new_values = tf.where(squeezed_done, zeroes, action_values)
        #new_values = tf.where(is_done, tf.zeros(is_done.shape[0]), K.max(next_state_action_values, axis=1))
        #return new_values
        squeezed_reward = tf.squeeze(reward, axis=[1]) # must specify axis due to inference sometimes having a batch size of 1
        adjusted_q_prime = (new_values * self.gamma) + squeezed_reward
        return adjusted_q_prime


class ClippedDoubleQPrimeLayer(Layer):

    def __init__(self, **kwargs):
        self.gamma = tf.constant(0.99)
        super(ClippedDoubleQPrimeLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        super(ClippedDoubleQPrimeLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        assert isinstance(x, list)
        next_state_actual_values, next_state_action_values, reward, is_done = x
        q_prime = 0.
        squeezed_done = tf.squeeze(is_done, axis=[1]) # must specify axis due to inference sometimes having a batch size of 1
        action_values_idxs = K.argmax(next_state_action_values, axis=1)
        target_action_values = K.max(next_state_action_values, axis=1)

        cols = tf.cast(action_values_idxs, dtype=tf.int32)
        rows = tf.range(tf.shape(action_values_idxs)[0])
        indicies = tf.stack([rows, cols], axis=-1)

        action_values = tf.gather_nd(next_state_actual_values, indices=indicies)
        target_action_values = K.max(next_state_action_values, axis=1)
        all_action_values = tf.stack([action_values, target_action_values], axis=-1)
        minimum_action_values = K.max(all_action_values, axis=1)



        zeroes = tf.zeros((1,))
        new_values = tf.where(squeezed_done, zeroes, minimum_action_values)
        #new_values = tf.where(is_done, tf.zeros(is_done.shape[0]), K.max(next_state_action_values, axis=1))
        #return new_values
        squeezed_reward = tf.squeeze(reward, axis=[1])  # TODO must specify axis due to inference sometimes having a batch size of 1
        adjusted_q_prime = (new_values * self.gamma) + squeezed_reward
        return adjusted_q_prime


class BellmanLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(BellmanLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        super(BellmanLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        assert isinstance(x, list)
        state_action_values, action, q_prime = x
        #squeezed_action = tf.squeeze(action, axis=[1])
        #return [state_action_values
        #exp_q = tf.expand_dims(q_prime, -1)
        action = tf.cast(action, dtype=tf.int32)  # lots of issues trying to make uint8
        cols = tf.squeeze(action, axis=[1])
        rows = tf.range(tf.shape(action)[0], dtype=action.dtype)
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

class DuelingCombiningLayer(Layer):

    def __init__(self, **kwargs):
        super(DuelingCombiningLayer, self).__init__(**kwargs)

    def build(self, input_shape):
       #assert isinstance(input_shape, list)
        super(DuelingCombiningLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        assert isinstance(x, list)
        advantage_values, values = x

        advantage_average = K.mean(advantage_values, axis=1, keepdims=True)
        advantage = Subtract()([advantage_values, advantage_average])
        action_values = Add()([advantage, values])
        return action_values

class TestAdvantageLayer(Layer):

    def __init__(self, **kwargs):
        super(DuelingCombiningLayer, self).__init__(**kwargs)

    def build(self, input_shape):
       #assert isinstance(input_shape, list)
        super(DuelingCombiningLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        assert isinstance(x, list)
        advantage_values, values = x

        advantage_average = K.mean(advantage_values, axis=1)
        advantage = Subtract()([advantage_values, advantage_average])
        action_values = Add()([advantage_average, values])
        return action_values


def mean(array):
    return keras.backend.mean(array, axis=-1, keepdims=True)