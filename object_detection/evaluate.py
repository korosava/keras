import tensorflow as tf
from custom_loss import yolo_loss
from custom_metrics import new_max_iou, new_mean_iou
from input_data import yolo_input_pippeline2



#<==============================_SET_CONSTATS_==============================>
batch_size = 53

#<==============================_DISABLE_WARNINGS_==============================>
tf.logging.set_verbosity(tf.logging.ERROR)


#<======================_INPUT_DATA_======================>
test = yolo_input_pippeline2(
	num_cells=4,
	num_objects=1,
	num_bboxes=2,
	return_offsets=1,
	data_dir='data/train_new')
imgs, bboxes, offsets = test

yolo_loss.iter = 0
yolo_loss.batch_size = batch_size
yolo_loss.offsets = offsets

new_max_iou.offsets = offsets
new_max_iou.batch_size = batch_size
new_max_iou.iter = 0

new_mean_iou.offsets = offsets
new_mean_iou.batch_size = batch_size
new_mean_iou.iter = 0


#<======================_LOAD_CLEAR_MODEL_======================>
with open('./saved_model/modelyolo_card_drop02.json', 'rt', encoding='utf-8') as fileobj:
	json_model = fileobj.read()
model = tf.keras.models.model_from_json(json_model)
model.compile(
	optimizer='adam',				
	loss=yolo_loss,
	metrics=[new_max_iou, new_mean_iou])		


#<======================_WEIGHTS_LOAD_======================>
model.load_weights('weight/model_yolo_card_dp2_test6_100ep')


#<======================_EVALUATE_MODEL_======================>
evaluate = model.evaluate(imgs, bboxes, verbose=0, batch_size=batch_size)
print('lose: {}\nnew_max_iou: {}\n new_mean_iou: {}'.format(evaluate[0], evaluate[1], evaluate[2]))



'''
test = yolo_input_pippeline(
	num_imgs=200,  
	img_size=28, 
	cell_size=7, 
	min_object_size=3, 
	max_object_size=7, 
	num_objects=1,
	num_bboxes=2,
	channels=1,
	train=True)
'''
