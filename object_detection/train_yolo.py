import tensorflow as tf
from custom_loss import yolo_loss
import bbox

#<======================_LOAD_INPUT_DATA_======================>


#<======================_SET_CALLBACKS_======================>
# tensorboard --logdir ./log_dir
#tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='./log_dir/model2_1', write_graph=True)
#callbacks = [tbCallBack]


#<======================_LOAD_CLEAR_MODEL_======================>
with open('./saved_model/model1.json', 'rt', encoding='utf-8') as fileobj:
	json_model = fileobj.read()
model = tf.keras.models.model_from_json(json_model)
model.compile(
	optimizer='adam',							#tf.train -> optimizers
	loss=yolo_loss,			#tf.keras.losses
	metrics=['accuracy'])			 			#tf.keras.metrics


#<======================_LOAD_FULL_MODEL_======================>
#model = tf.keras.models.load_model('./full_model/model1_10ep.h5')


#<======================_TRAIN_MODEL_======================>
model.fit(
	f1,
	l1,
	batch_size=100,
	epochs=10,
	validation_data=test,
	verbose=2,
	)


#<======================_SAVE_WEIGHTS_MODEL_======================>
model.save('full_model/model1_custom_error_10ep.h5')
model.save_weights('weight/model1_custom_error_10ep')


