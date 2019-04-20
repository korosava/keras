import tensorflow as tf
from custom_loss import yolo_loss
import bbox
from input_data import transform_to_conv

#<======================_LOAD_INPUT_DATA_======================>
data = bbox.train_bbox_data()
data = transform_to_conv(data)
train, test = data
f1, l1 = train
f2, l2 = test


#<======================_SET_CALLBACKS_======================>
# tensorboard --logdir ./log_dir
tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='./log_dir/model_coords', write_graph=True)
callbacks = [tbCallBack]


#<======================_LOAD_CLEAR_MODEL_======================>
with open('./saved_model/model_coords.json', 'rt', encoding='utf-8') as fileobj:
	json_model = fileobj.read()
model = tf.keras.models.model_from_json(json_model)
model.compile(
	optimizer='adam',							#tf.train -> optimizers
	loss='mse',			#tf.keras.losses
	metrics=['accuracy'])			 			#tf.keras.metrics


#<======================_LOAD_FULL_MODEL_======================>
#model = tf.keras.models.load_model('./full_model/model1_10ep.h5')


#<======================_TRAIN_MODEL_======================>
model.fit(
	f1,
	l1,
	batch_size=100,
	epochs=50,
	validation_data=test,
	verbose=2,
	callbacks = callbacks
	)


#<======================_SAVE_WEIGHTS_MODEL_======================>
model.save('full_model/model_coords_10ep.h5')
model.save_weights('weight/model_coords_10ep')


