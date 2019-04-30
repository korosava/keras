import tensorflow as tf
from custom_loss import yolo_loss
from custom_metrics import metric_iou
from input_data import yolo_input_pippeline
from custom_tensorBoard import CustomTensorBoard as Ctb


#<==============================_DISABLE_WARNINGS_==============================>
tf.logging.set_verbosity(tf.logging.ERROR)


#<==============================_LOAD_INPUT_DATA_==============================>
train = yolo_input_pippeline(
	num_imgs=1000,  
	img_size=28, 
	cell_size=7, 
	min_object_size=3, 
	max_object_size=7, 
	num_objects=1,
	num_bboxes=1,
	channels=1,
	train=True)
imgs, bboxes = train


#<==============================_SET_CALLBACKS_==============================>
# tensorboard --logdir ./log_dir
#next_global_iter = (num_imgs/batch_size)*epochs
tbCallBack = Ctb(log_dir='./log_dir/modelyolo_1_test', global_iter=50)
callbacks = [tbCallBack,]


#<==============================_LOAD_CLEAR_MODEL_==============================>
with open('./saved_model/modelyolo_1.json', 'rt', encoding='utf-8') as fileobj:
	json_model = fileobj.read()
model = tf.keras.models.model_from_json(json_model)
model.compile(
	optimizer='adam',							#tf.train -> optimizers
	loss=yolo_loss,		#tf.keras.losses
	metrics=[metric_iou]
	)			 			#tf.keras.metrics


#<======================_WEIGHTS_LOAD_======================>
model.load_weights('./weight/model_yolo_1_test')


#<==============================_TRAIN_MODEL_==============================>
model.fit(
	imgs,
	bboxes,
	batch_size=100,
	epochs=2,
	verbose=2,
	callbacks = callbacks
	)


#<======================_SAVE_WEIGHTS_&_MODEL_======================>
model.save('full_model/model_yolo_1_test.h5')
model.save_weights('weight/model_yolo_1_test')
