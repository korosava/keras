import tensorflow as tf
import matplotlib.pyplot as plt
from custom_loss1 import yolo_loss
from input_data import yolo_input_pippeline
from custom_metrics import my_accuracy
import numpy as np
import bbox


#<======================_INPUT_DATA_======================>
test = yolo_input_pippeline(
	num_imgs=2,  
	img_size=28, 
	cell_size=7, 
	min_object_size=3, 
	max_object_size=7, 
	num_objects=1,
	num_bboxes=1,
	channels=1,
	train=False)
imgs, bboxes, offsets = test

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


#<======================_MODEL_PREDICT_======================>
bboxes = model.predict(imgs)


bboxes = np.reshape(bboxes, [-1, 4, 4, 5])
bboxes, confidences = bbox.loss_to_labels(bboxes, offsets, 4, 1, 28)
imgs = bbox.restore_imgs(imgs)

#print('\nconfidences:\n{}\n\n'.format(confidences))
imgs = np.reshape(imgs, [-1, 28, 28])

print('\nbboex:\n',bboxes)
bbox.build_bbox(imgs, bboxes, img_size=28)

# ПОМИЛКА, IMG БІЛИЙ