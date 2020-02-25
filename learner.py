#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
#os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

#import tensorflow.compat.v1 as tf

#import tensorflow.compat.v1.keras.backend as K
#dtype = 'float16'
#K.set_floatx(dtype)
#K.set_epsilon(1e-4)
#from learners.learner import tf_config

#tf.disable_eager_execution()  # disable eager for performance boost
#tf.set_random_seed(4)
#from tensorflow.python.framework.ops import disable_eager_execution
#disable_eager_execution()
#import  tensorflow.compat.v2.k

##writer = tf.summary.FileWriter("log")
#writer = tf.summary.create_file_writer("logs")
#config = tf.ConfigProto()
# TODO investigate making tf dataset to get boost from eager
#config.gpu_options.allow_growth = True
#tf_config.gpu_options.allow_growth=True
#train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
#train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
#test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)


