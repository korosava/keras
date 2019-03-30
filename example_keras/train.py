from input_data import load_data, transform_to_dense
import tensorflow as tf


#<======================_LOAD_INPUT_DATA_======================>
data = load_data()
data = transform_to_dense(data)
train, test = data
f1, l1 = train
f2, l2 = test


#<======================_SET_CALLBACKS_======================>
# tensorboard --logdir ./log_dir
tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='./log_dir/model1', write_graph=True)
stopCallBack = tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss')
callbacks = [tbCallBack, stopCallBack]


#<======================_LOAD_CLEAR_MODEL_======================>
with open('./saved_model/model1.json', 'rt', encoding='utf-8') as fileobj:
	json_model = fileobj.read()
model = tf.keras.models.model_from_json(json_model)
model.compile(
	optimizer='adam',							#tf.train -> optimizers
	loss='categorical_crossentropy', 			#tf.keras.losses
	metrics=['accuracy'])			 			#tf.keras.metrics

#<======================_TRAIN_MODEL_======================>
model.fit(
	f1,
	l1,
	batch_size=100,
	epochs=1,
	validation_data=test,
	verbose=2,
	callbacks=callbacks,
	)


#<======================_SAVE_WEIGHTS_MODEL_======================>
model.save('full_model/model1_1ep.h5')
model.save_weights('weight/model1_1ep')




