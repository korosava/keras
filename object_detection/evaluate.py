import tensorflow as tf
from input_data import transform_to_conv
from custom_loss import yolo_loss
from input_data import yolo_input_pippeline
from custom_metrics import my_accuracy

#<======================_INPUT_DATA_======================>
test = yolo_input_pippeline(
	num_imgs=50,  
	img_size=28, 
	cell_size=7, 
	min_object_size=3, 
	max_object_size=7, 
	num_objects=1,
	num_bboxes=1,
	channels=1,
	train=True)
imgs, bboxes = test
'''
#<======================_LOAD_FULL_MODEL_======================>
model = tf.keras.models.load_model('./full_model/model_yolo_1ep.h5')

'''
#<======================_LOAD_CLEAR_MODEL_======================>
with open('./saved_model/modelyolo_1.json', 'rt', encoding='utf-8') as fileobj:
	json_model = fileobj.read()
model = tf.keras.models.model_from_json(json_model)
model.compile(
	optimizer='adam',				
	loss=yolo_loss,
	metrics=[my_accuracy])		

#<======================_WEIGHTS_LOAD_======================>
model.load_weights('./weight/model_yolo_10ep')


#<======================_EVALUATE_MODEL_======================>
evaluate = model.evaluate(imgs, bboxes, verbose=0)
print('lose: {}\nmy_accuracy: {}'.format(evaluate[0], evaluate[1]))