import tensorflow as tf
from tensorflow.python.keras.callbacks import Callback

class CustomTensorBoard (Callback):
	def __init__(self, log_dir):
		self.log_dir = log_dir
		self.sess = K.get_session()
		super(CustomTensorBoard, self).__init__()


	def on_train_begin(self, logs={}):
		tf.summary.scalar('stddev', self.model.loss)
		tf.summary.scalar('stddev', self.model.metrics[0])
		merged = tf.summary.merge_all()
		train_writer = tf.summary.FileWriter(self.log_dir, sess.graph)



	def on_batch_end(self, batch, logs={}):
		pass
        
