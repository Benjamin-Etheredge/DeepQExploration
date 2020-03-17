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
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

#from numba import jit


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

#from tensorflow.keras.mixed_precision import experimental as mixed_precision
#policy = mixed_precision.Policy('mixed_float16')
#policy = mixed_precision.set_policy(policy)

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

