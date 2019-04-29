import tensorflow as tf
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras._impl.keras import backend as K

class CustomTensorBoard (Callback):
	def __init__(self, log_dir, global_iter=0):
		super(CustomTensorBoard, self).__init__()
		self.log_dir = log_dir
		self.global_iter = global_iter


	def set_model(self, model):
		super(CustomTensorBoard, self).set_model(model)
		tf.summary.scalar('loss', self.model.total_loss)
		tf.summary.scalar('max_iou', self.model.metrics_tensors[0])
		self.merged = tf.summary.merge_all()
		self.writer = tf.summary.FileWriter(self.log_dir, K.get_session().graph)
		#self.global_iter = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="global_iter")[0]
	

	def on_batch_end(self, batch, logs={}):
		for name, value in logs.items():
			if name in ['batch', 'size']:
				continue
			summary = tf.summary.Summary()
			summary_value = summary.value.add()
			summary_value.simple_value = value.item()
			summary_value.tag = name
			self.writer.add_summary(summary, self.global_iter)
		self.writer.flush()
		self.global_iter+=1
		#K.eval(self.global_iter.assign(self.global_iter+1))
		

	def on_train_end(self, logs=None):
		self.writer.close()

	

