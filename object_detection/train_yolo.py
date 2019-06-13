import tensorflow as tf
from custom_loss import yolo_loss
from custom_metrics import new_max_iou, new_mean_iou
from input_data import yolo_input_pippeline2
from custom_tensorBoard import CustomTensorBoard as Ctb


#<==============================_SET_CONSTATS_==============================>
batch_size = 53


#<==============================_DISABLE_WARNINGS_==============================>
tf.logging.set_verbosity(tf.logging.ERROR)


#<==============================_LOAD_INPUT_DATA_==============================>
# зробити метод для перемішки датасету
# сортувати bboxes за перемішаними зображеннями
train = yolo_input_pippeline2(
	num_cells=4,
	num_objects=1,
	num_bboxes=2,
	return_offsets=1,
	data_dir='data/train_new')

imgs, bboxes, offsets = train
yolo_loss.offsets = offsets
yolo_loss.batch_size = batch_size
yolo_loss.iter = 0

new_max_iou.offsets = offsets
new_max_iou.batch_size = batch_size
new_max_iou.iter = 0

new_mean_iou.offsets = offsets
new_mean_iou.batch_size = batch_size
new_mean_iou.iter = 0


#<==============================_SET_CALLBACKS_==============================>
# tensorboard --logdir ./log_dir
# next_global_iter = (num_imgs/batch_size)*epochs
tbCallBack = Ctb(log_dir='log_dir/model_yolo_card_dp2_test7_200ep', global_iter=0)
callbacks = [tbCallBack]


#<==============================_LOAD_CLEAR_MODEL_==============================>
with open('./saved_model/modelyolo_card_drop02.json', 'rt', encoding='utf-8') as fileobj:
	json_model = fileobj.read()
model = tf.keras.models.model_from_json(json_model)
model.compile(
	optimizer='adam',
	loss=yolo_loss,
	metrics=[new_mean_iou, new_max_iou]
	)			 							


#<======================_WEIGHTS_LOAD_======================>
#model.load_weights('weight/model_yolo_card_drop02_adam_newData_200ep')


#<==============================_TRAIN_MODEL_==============================>
model.fit(
	imgs,
	bboxes,
	batch_size=batch_size,
	epochs=200,
	verbose=2,
	callbacks=callbacks
	)


#<======================_SAVE_WEIGHTS_&_MODEL_======================>
#model.save('full_model/model_yolo_card_drop02_adagrad1_50ep.h5')
model.save_weights('weight/model_yolo_card_dp2_test7_200ep')


'''
train = yolo_input_pippeline(
	num_imgs=5000,  
	img_size=28, 
	cell_size=7, 
	min_object_size=3, 
	max_object_size=7, 
	num_objects=1,
	num_bboxes=2,
	channels=1,
	train=True)
'''