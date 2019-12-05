import util
import os
import tensorflow as tf
# print(os.environ)
os.environ["GPU"] = "0"
print(os.environ["GPU"])
config = util.initialize_from_env()
print(tf.test.gpu_device_name())
print(config)

