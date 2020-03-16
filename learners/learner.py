import numpy as np
#from tensorflow_core._api.v2.compat import v1 as tf
#from tensorflow_core.python.keras.api._v1 import keras
from tensorflow import keras

#from tensorflow.keras.mixed_precision import experimental as mixed_precision
#policy = mixed_precision.Policy('mixed_float16')
#mixed_precision.set_policy(policy)
#print('Compute dtype: %s' % policy.compute_dtype)
#print('Variable dtype: %s' % policy.variable_dtype)


from buffer import ReplayBuffer
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
#os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

#import tensorflow.compat.v1 as tf
#tf.disable_eager_execution()
#tf.disable_eager_execution()
import tensorflow as tf
#tf.compat.v1.disable_eager_execution()


#import tensorflow.compat.v1.keras.backend as K
import tensorflow.keras.backend as K
#from tensorflow.compat.v1.keras.layers import Dense, Conv2D, Lambda, Flatten
#from tensorflow.compat.v1.keras import Input
from tensorflow.keras.layers import Dense, Conv2D, Lambda, Flatten
from tensorflow.keras import Input
#dtype = 'float16'
#K.set_floatx(dtype)
#K.set_epsilon(1e-4)
#from learners.learner import tf_config

#tf.disable_eager_execution()  # disable eager for performance boost
#tf.set_random_seed(4)
#from tensorflow.python.framework.ops import disable_eager_execution
#disable_eager_execution()
#import  tensorflow.compat.v2.k

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

#writer = tf.summary.create_file_writer("logs")
#config = tf.ConfigProto()
# TODO investigate making tf dataset to get boost from eager
#config.gpu_options.allow_growth = True
#config.gpu_options.allow_growth=True
#sess = tf.Session(config=config)
#tf_config.gpu_options.allow_growth=True
#train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
#train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
#test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)

class DeepQ:
    #@profile
    def __init__(self,
                 name,
                 q_prime_function,
                 build_model_function):

        self.name = name
        self.q_prime_function = q_prime_function
        self.build_model_function = build_model_function

        self.gamma = None

        self.model = None
        self.target_model = None
        #log_dir = f"logs/{self.name}" + datetime.now().strftime("%Y%m%d-%H%M%S")
        #self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        #self.update_count = 0
        #self.summary_writer = tf.summary.create_file_writer(log_dir)

    #@profile
    def build_model(self, input_dimension, output_dimension,
                    nodes_per_layer: int = 128,  # TODO difference between 128 vs 256
                    layer_count: int = 1,
                    gamma: float = 0.99,
                    learning_rate: float = 0.001, *args, **kwargs):
        self.gamma = gamma
        self.model, self.target_model, self.action_selector, self.train = self.build_model_function(input_dimension, output_dimension,
                                                                nodes_per_layer, layer_count, learning_rate, *args, **kwargs)
        #tf.summary.
        #self.model.name = "Live_Network"
        #with tf.device('/cpu:0'):
        #self.target_model = self.build_model_function(input_dimension, output_dimension,
                                             #nodes_per_layer, layer_count, learning_rate, *args, **kwargs)
        #self.target_model.name = "Target_Network"
        #self.tensorboard_callback.set_model(self.model)
        self.update_target_model()

    #@profile
    def get_name(self):
        return self.name

    #@profile
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    #@profile
    def log(self):
        pass

    #@profile
    def get_next_action(self, state):
        # TODO this is terrible.... refactor. I just want it to run right now
        np_state = np.array(state)
        #print(np_state.shape)
        if len(np_state.shape) > 1:
            np_state = np_state[:, np.newaxis, :, :]

        return self.action_selector.predict_on_batch([np_state[0], np_state[1], np_state[2], np_state[3]])[0]

        # TODO test without target
        # statePrimes = sample.nextStates
        # Get the action values for state primes NOTE TARGET MODEL
        # target_prime_action_values = self.targetModel.predict(statePrimes)
        # return target_prime_action_values

    #@profile
    def update(self, sample: ReplayBuffer):
        # TODO refactor
        #TODO combine model predections
        #states = np.array(sample.states)
        #from PIL import Image
        #im = Image.fromarray(states[0, :, :, :3])
        #im.save("img.jpeg")
        #im = Image.fromarray(states[4, :, :, :3])
        #im.save("img2.jpeg")

        states, actions, next_states, rewards, is_dones = sample.training_items()
        #losses = self.train.train_on_batch(sample.training_items)
        #losses = self.train.train_on_batch(*sample.training_items())

        #next_states = np.array(sample.next_states)
        #action_values = self.model.predict_on_batch(np.concatenate((states, next_states), axis=0))
        #current_all_action_values, current_all_prime_action_values = np.split(action_values, 2)
        losses = self.train.train_on_batch([states[:, 0, :], states[:, 1, :], states[:, 2, :], states[:, 3, :],
                                            np.array(actions),
                                            next_states[:, 0, :], next_states[:, 1, :], next_states[:, 2, :], next_states[:, 3, :],
                                            np.array(rewards), np.array(is_dones)])

        '''
        current_all_action_values = self.model.predict_on_batch(states)  # TODO invistaigate explictly make array due to TF eager
        current_all_prime_action_values = self.model.predict_on_batch(next_states)
        target_all_prime_action_values = self.target_model.predict_on_batch(next_states)

        for idx, (action, reward, is_done) in enumerate(sample.training_items):
            q_prime = 0
            if not is_done:
                # TODO refactor
                q_prime = self.q_prime_function(target_all_prime_action_values[idx],
                                                current_all_prime_action_values[idx])
                # q_prime = self.qPrime(target_all_prime_action_values[idx], current_all_action_values[idx])

            actual_value = (self.gamma * q_prime) + reward
            current_all_action_values[idx][action] = actual_value
            #idx += 1

        # TODO refactor
        losses = self.model.train_on_batch(x=states, y=current_all_action_values)
        '''
        return losses


class DeepQFactory:
    # Different Q-prime computating functions
    @staticmethod
    def vanilla_q_prime(target_prime_action_values, prime_action_values) -> float:
        #return np.max(target_prime_action_values)
        return max(target_prime_action_values)

    @staticmethod
    def double_deepq_q_prime(target_prime_action_values, prime_action_values) -> float:
        max_action = np.argmax(prime_action_values)
        return target_prime_action_values[max_action]

    @staticmethod
    def clipped_double_deep_q_q_prime(target_prime_action_values, prime_action_values) -> float:
        max_action_1 = np.argmax(target_prime_action_values)
        max_action_2 = np.argmax(prime_action_values)
        return min(prime_action_values[max_action_1], target_prime_action_values[max_action_2])

    # Different Model Factory Methods
    @staticmethod
    def create_vanilla_deep_q(*args, **kwargs) -> DeepQ:
        return DeepQ(name="Vanilla_DeepQ", q_prime_function=DeepQFactory.vanilla_q_prime,
                     build_model_function=DeepQFactory.vanilla_build_model, *args, **kwargs)

    @staticmethod
    def create_double_deep_q(*args, **kwargs) -> DeepQ:
        """
        Reduce the overestimations by breaking up action seleciton and action evaluation
        """
        return DeepQ(name="Double_DeepQ", q_prime_function=DeepQFactory.double_deepq_q_prime,
                     build_model_function=DeepQFactory.vanilla_build_model, *args, **kwargs)

    @staticmethod
    def create_clipped_double_deep_q(*args, **kwargs) -> DeepQ:
        """
        Reduce the overestimations by breaking up action seleciton and action evaluation
        """
        return DeepQ(name="Clipped_Double_DeepQ", q_prime_function=DeepQFactory.clipped_double_deep_q_q_prime,
                     build_model_function=DeepQFactory.vanilla_build_model, *args, **kwargs)

    @staticmethod
    def create_duel_deep_q(*args, **kwargs) -> DeepQ:
        return DeepQ(name="Duel_DeepQ", q_prime_function=DeepQFactory.vanilla_q_prime,
                     build_model_function=DeepQFactory.dueling_build_model, *args, **kwargs)

    @staticmethod
    def create_double_duel_deep_q(*args, **kwargs) -> DeepQ:
        return DeepQ(name="Double_Duel_DeepQ", q_prime_function=DeepQFactory.double_deepq_q_prime,
                     build_model_function=DeepQFactory.dueling_build_model, *args, **kwargs)

    @staticmethod
    def create_clipped_double_duel_deep_q(*args, **kwargs) -> DeepQ:
        return DeepQ(name="Clipped_Double_Duel_DeepQ",
                     q_prime_function=DeepQFactory.clipped_double_deep_q_q_prime,
                     build_model_function=DeepQFactory.dueling_build_model, *args, **kwargs)

    @staticmethod
    def create_atari_clipped_double_duel_deep_q(*args, **kwargs) -> DeepQ:
        return DeepQ(name="Atari_Testing_Arch",
                     #q_prime_function=DeepQFactory.vanilla_q_prime,
                     q_prime_function=DeepQFactory.double_deepq_q_prime,
                     build_model_function=DeepQFactory.vanilla_conv_build_model_raw, *args, **kwargs)

    #@static create_atari()

    # Different Model Construction Methods.
    @staticmethod
    def vanilla_build_model(input_dimension, output_dimension, nodes_per_layer, hidden_layer_count, learning_rate):
        inputs = keras.Input(shape=(input_dimension,))
        hidden_layer = inputs
        for _ in range(hidden_layer_count):
            hidden_layer = keras.layers.Dense(nodes_per_layer, activation='relu')(hidden_layer)
            # TODO explore batchnorm in RL.
            #hidden_layer = keras.layers.BatchNormalization()(hidden_layer)
        predictions = keras.layers.Dense(output_dimension, activation='linear')(hidden_layer)
        model = keras.Model(inputs=inputs, outputs=predictions)
        # TODO do more testing on MSE vs Huber
        #model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate, epsilon=1.5e-4), loss=tf.keras.losses.Huber())
        model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate), loss=tf.keras.losses.Huber())
        return model

    @staticmethod
    def dueling_build_model(input_dimension, output_dimension, nodes_per_layer, hidden_layer_count, learning_rate):
        inputs = keras.Input(shape=(input_dimension,))

        # Build Advantage layer
        advantage_hidden_layer = inputs
        for _ in range(hidden_layer_count):
            advantage_hidden_layer = keras.layers.Dense(nodes_per_layer, activation='relu')(advantage_hidden_layer)
        predictions_advantage = keras.layers.Dense(output_dimension, activation='linear')(advantage_hidden_layer)

        # Build Value layer
        value_hidden_layer = inputs
        for _ in range(hidden_layer_count):
            value_hidden_layer = keras.layers.Dense(nodes_per_layer, activation='relu')(value_hidden_layer)
        predictions_value = keras.layers.Dense(1, activation='linear')(value_hidden_layer)

        # Combine layers
        advantage_average = keras.layers.Lambda(mean)(predictions_advantage)

        advantage = keras.layers.Subtract()([predictions_advantage, advantage_average])

        predictions = keras.layers.Add()([advantage, predictions_value])

        model = keras.Model(inputs=inputs, outputs=predictions)
        model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate), loss=tf.keras.losses.Huber())
        return model

    import tensorflow.keras.backend as K
    def double_custom_loss(layer):
        def loss(y):
            pass

    # Different Model Construction Methods.
    @staticmethod
    def vanilla_conv_build_model_raw(input_dimensions, output_dimension, nodes_per_layer, hidden_layer_count, learning_rate,
                                 conv_nodes, kernel_size, conv_stride):

        input_dimensions = tuple(((int(round(input_dimensions[0]/2))), int(round((input_dimensions[1]/2)))))
        state_frames = [Input(shape=input_dimensions, name=f"state_frame_{idx}")  for idx in range(4)]
        states = Lambda(lambda x: tf.stack(x, axis=-1))(state_frames)

        next_state_frames = [Input(shape=input_dimensions, name=f"next_state_frame_{i}") for i in range(4)]
        next_states = Lambda(lambda x: tf.stack(x, axis=-1))(next_state_frames)

        action = Input(shape=(1,), dtype=tf.int32, name='action') 
        is_done = Input(shape=(1,), dtype=tf.bool, name='is_done') 
        reward = Input(shape=(1,), name='reward') 

        scaled_layer_states = keras.layers.Lambda(lambda x: x / 255.0)(states)
        scaled_layer_next_states = keras.layers.Lambda(lambda x: x / 255.0)(next_states)
        networks = []
        for idx, scaled_layer in enumerate([scaled_layer_states, scaled_layer_next_states]):
            hidden_layer = scaled_layer

            for conv_count, kernel, stride in zip(conv_nodes, kernel_size, conv_stride):
                hidden_layer = Conv2D(filters=conv_count,
                                                   kernel_size=kernel,
                                                   strides=stride,
                                                   activation='relu',
                                                   use_bias=False,
                                                   data_format='channels_last')(hidden_layer)

            hidden_layer = Flatten()(hidden_layer)
            for _ in range(hidden_layer_count):
                hidden_layer = keras.layers.Dense(nodes_per_layer, activation='relu')(hidden_layer)

            predictions = keras.layers.Dense(output_dimension, activation='linear', name=f'action_values_{idx}')(hidden_layer)
            networks.append(predictions)

        #TODO switch back optimizers and huber
        #model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate, epsilon=1.5e-4), loss=tf.keras.losses.Huber())

        model_action_values, target_model_action_values = networks

        #with tf.device('/cpu:0'):
        best_action = Lambda(lambda x: K.argmax(x, axis=1), name='best_action')(model_action_values)
        action_selector = Model(inputs=state_frames, outputs=best_action)

        def custom_loss(values, correct_values):
            def loss(y_true, y_pred):
                return K.mean(K.square(correct_values - values), axis=-1)
            return loss

        def custom_huber_loss(y_pred, y_true):
            def huber_loss(_1, _2, clip_delta=1.0):
                error = y_true - y_pred
                cond  = tf.keras.backend.abs(error) < clip_delta

                squared_loss = 0.5 * tf.keras.backend.square(error)
                linear_loss  = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)

                return tf.where(cond, squared_loss, linear_loss)
            return huber_loss

        # TODO double Q
        #target_action_value = keras.Model(inputs=states, outputs=best_action)

        model = keras.Model(inputs=state_frames, outputs=model_action_values)

        target = keras.Model(inputs=next_state_frames, outputs=target_model_action_values)
        # TODO
        q_prime_value = Q_Prime_Layer(None)([model_action_values, target_model_action_values, reward, is_done])
        test = MyLayer(output_dimension)([model_action_values, action, q_prime_value])
        trainable = keras.Model(inputs=[*state_frames, action, *next_state_frames, reward, is_done], outputs=model_action_values)

        #trainable.compile(optimizer=keras.optimizers.Adam(lr=learning_rate), loss=custom_loss(model_action_values, test))
        trainable.compile(optimizer=keras.optimizers.Adam(lr=learning_rate), loss=custom_huber_loss(model_action_values, test))

        #model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate), loss={'values_0': tf.keras.losses.Huber()})
        return model, target, action_selector, trainable


class Q_Prime_Layer(tf.keras.layers.Layer):

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