import numpy as np
np.random.seed(4)
import random
random.seed(4)

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
    #@jit
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    #@profile
    def log(self):
        pass

    #@profile
    #@jit
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
    #@jit
    def update(self, sample: buffer.VoidBuffer):
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
        losses = self.train.train_on_batch([*states, actions, *next_states, rewards, is_dones])

        #losses = self.train.train_on_batch(states, np.array(actions), next_states[:, 0, :], next_states[:, 1, :], next_states[:, 2, :], next_states[:, 3, :],
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
        return losses, None

