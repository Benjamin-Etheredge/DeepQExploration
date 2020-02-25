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
import tensorflow as tf

#import tensorflow.compat.v1.keras.backend as K
import tensorflow.keras.backend as K
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
        if len(state.shape) > 1:
            state = state[np.newaxis, :, :, :]

        #action_values = self.model.predict_on_batch(state)[0]
        #return np.argmax(action_values)
        return self.action_selector.predict_on_batch(state)[0]

        # TODO test without target
        # statePrimes = sample.nextStates
        # Get the action values for state primes NOTE TARGET MODEL
        # target_prime_action_values = self.targetModel.predict(statePrimes)
        # return target_prime_action_values

    #@profile
    def update(self, sample: ReplayBuffer):
        # TODO refactor
        #TODO combine model predections
        states = np.array(sample.states)
        #from PIL import Image
        #im = Image.fromarray(states[0, :, :, :3])
        #im.save("img.jpeg")
        #im = Image.fromarray(states[4, :, :, :3])
        #im.save("img2.jpeg")
        next_states = np.array(sample.next_states)
        #action_values = self.model.predict_on_batch(np.concatenate((states, next_states), axis=0))
        #current_all_action_values, current_all_prime_action_values = np.split(action_values, 2)
        losses = self.train.train_on_batch([states, np.array(sample.actions), next_states, np.array(sample.rewards), np.array(sample.is_dones)])

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
        return DeepQ(name="Atari_Clipped_Double_Duel_DeepQ",
                     #q_prime_function=DeepQFactory.vanilla_q_prime,
                     q_prime_function=DeepQFactory.double_deepq_q_prime,
                     build_model_function=DeepQFactory.vanilla_conv_build_model, *args, **kwargs)

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
    def vanilla_conv_build_model(input_dimensions, output_dimension, nodes_per_layer, hidden_layer_count, learning_rate,
                                 conv_nodes, kernel_size, conv_stride):

        print('building model')
        input_dimensions = (int(round(input_dimensions[0]/2))), int(round((input_dimensions[1]/2))), input_dimensions[2]
        states = keras.Input(shape=tuple(input_dimensions))  # we'll be using the past 4 frames
        next_states = keras.Input(shape=tuple(input_dimensions))  # we'll be using the past 4 frames
        action = keras.Input(shape=(1,), dtype=tf.int32)  # we'll be using the past 4 frames
        is_done = keras.Input(shape=(1,), dtype=tf.bool)  # we'll be using the past 4 frames
        reward = keras.Input(shape=(1,))  # we'll be using the past 4 frames

        scaled_layer_states = keras.layers.Lambda(lambda x: x / 255.0)(states)
        scaled_layer_next_states = keras.layers.Lambda(lambda x: x / 255.0)(next_states)
        networks = []
        for idx, scaled_layer in enumerate([scaled_layer_states, scaled_layer_next_states]):
            hidden_layer = scaled_layer

            for conv_count, kernel, stride in zip(conv_nodes, kernel_size, conv_stride):
                hidden_layer = keras.layers.Conv2D(filters=conv_count,
                                                   kernel_size=kernel,
                                                   strides=stride,
                                                   activation='relu',
                                                   use_bias=False,
                                                   data_format='channels_last')(hidden_layer)

            hidden_layer = keras.layers.Flatten()(hidden_layer)
            for _ in range(hidden_layer_count):
                hidden_layer = keras.layers.Dense(nodes_per_layer, activation='relu')(hidden_layer)

            predictions = keras.layers.Dense(output_dimension, activation='linear', name=f'values_{idx}')(hidden_layer)
            networks.append(predictions)

        #TODO switch back optimizers and huber
        #model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate, epsilon=1.5e-4), loss=tf.keras.losses.Huber())

        #with tf.device('/cpu:0'):
        best_action = keras.layers.Lambda(lambda x: K.argmax(x, axis=1), name='best_action')(networks[0])
        action_selector = keras.Model(inputs=states, outputs=best_action)

        #target_best_action = keras.layers.Lambda(lambda x: K.argmax(x, axis=1))(networks[0])
        #action_value = keras.layers.Lambda(lambda x: K.max(x, axis=1))(networks[1])
        #action_value = keras.layers.maximum()(networks[1])

        def custom_loss(values, correct_values):
            #q_prime_value = K.max(correct_values)
            #temp = np.array(q_prime_value)
            #temp[a]
            def loss(y_true, y_pred):
                #0.99 *
                return K.mean(K.square(correct_values - values), axis=-1)
                #return K.
            return loss

        # TODO double Q
        #target_action_value = keras.Model(inputs=inputs, outputs=best_action)

        model = keras.Model(inputs=states, outputs=networks[0])
        target = keras.Model(inputs=next_states, outputs=networks[1])
        q_prime_value = Q_Prime_Layer(1)([networks[0], networks[1], reward, is_done])
        test = MyLayer(output_dimension)([networks[0], action, q_prime_value])
        trainable = keras.Model(inputs=[states, action, next_states, reward, is_done], outputs=networks[0])

        trainable.compile(optimizer=keras.optimizers.Adam(lr=learning_rate), loss=custom_loss(networks[0], test))

        #model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate), loss={'values_0': tf.keras.losses.Huber()})
        return model, target, action_selector, trainable


class Q_Prime_Layer(tf.keras.layers.Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        self.gamma = tf.constant(0.97)
        super(Q_Prime_Layer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        super(Q_Prime_Layer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        assert isinstance(x, list)
        state_action_values, next_state_action_values, reward, is_done = x
        q_prime = 0.
        #jif not is_done:
            #q_prime = tf.gather(state_action_values, [K.argmax(next_state_action_values, axis=1)])
            #j3q_prime = [K.max(next_state_action_values, axis=1)]
        #tf.where(is_done, K.max(next_state_action_values, axis=1), 0.)

        new_values = tf.where(is_done, tf.zeros(is_done.shape[-1]), K.max(next_state_action_values, axis=0))
        return (new_values * self.gamma) + reward

        #tf.where(is)

        #q_prime = (q_prime * ) + reward
        #return [q_prime]


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
        print(state_action_values.shape)
        print(action.shape)
        print(q_prime.shape)
        #action_idxs = tf.concat([tf.range(0, action.shape[0]), action], axis=0)
        #q_prime_idxs = tf.concat([tf.range(0, q_prime.shape[0], ), q_prime], axis=0)

        #return tf.tensor_scatter_nd_update(state_action_values, action, q_prime_idxs)
        return tf.tensor_scatter_nd_update(state_action_values, action, q_prime)
        #delta = tf.scatter_nd(action, q_prime, [64, 780])
        #return tf.SparseTensor(action, q_prime, state_action_values.shape)
        #return tf.assign(state_action_values[:, [action], q_prime])

        #return tf.sparse.to_dense(delta)

        #return [state_action_values]
        action_values = tf.gather(state_action_values, action)

        #return [tf.where(state_action_values !=  action_values, state_action_values, q_prime)]
        return [state_action_values]

        """
        maskValues = tf.tile([0.0], [tf.shape(state_action_values)[0]])  # one 0 for each element in "indices"
        mask = tf.SparseTensor([action], maskValues, tf.shape(state_action_values, out_type=tf.int64))
        maskedInput = tf.multiply([action], tf.sparse_tensor_to_dense(mask,
                                                                    default_value=1.0))  # set values in coordinates in "indices" to 0's, leave everything else intact

        # replace elements in "indices" with "values"
        delta = tf.SparseTensor([action], [q_prime], tf.shape(state_action_values, out_type=tf.int64))
        outputs = tf.add(maskedInput, tf.sparse_tensor_to_dense(delta))
        return outputs
        """

        '''
        mask = np.array([idx == action for idx in range(4)])

        idx_remove = tf.where(mask==True)[:,-1]
        idx_keep = tf.where(mask==False)[:,-1]

        values_remove = tf.tile([q_prime], [tf.shape(idx_remove)[0]])
        values_keep = tf.gather(state_action_values[0], idx_keep)

        # to create a sparse vector we still need 2d indices like [ [0,1], [0,2], [0,10] ]
        # create vectors of 0's that we'll later stack with the actual indices
        zeros_remove = tf.zeros_like(idx_remove)
        zeros_keep = tf.zeros_like(idx_keep)

        idx_remove = tf.stack([zeros_remove, idx_remove], axis=1)
        idx_keep = tf.stack([zeros_keep, idx_keep], axis=1)

        # now we can create a sparse matrix
        logits_remove = tf.SparseTensor(idx_remove, values_remove, tf.shape(state_action_values, out_type=tf.int64))
        logits_keep = tf.SparseTensor(idx_keep, values_keep, tf.shape(state_action_values, out_type=tf.int64))

        # add together the two matrices (need to convert them to dense first)
        filtered_logits = tf.add(
            tf.sparse.to_dense(logits_remove, default_value=0.),
            tf.sparse.to_dense(logits_keep, default_value=0.)
        )

        return [filtered_logits]
        '''



        #test = np.array(state_action_values)
        #test[action] = q_prime
        return [state_action_values]

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [(shape_a[0], self.output_dim), shape_b[:-1]]


def mean(array):
    return keras.backend.mean(array, axis=-1, keepdims=True)