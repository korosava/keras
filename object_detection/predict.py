import tensorflow as tf
from custom_loss import yolo_loss
from input_data import transform_to_conv
import bbox
import matplotlib.pyplot as plt
import numpy as np


#<======================_INPUT_DATA_======================>
train, test, offsets = bbox.use_bbox_data(
	num_imgs_train=500,
	num_imgs_test=50, 
	img_size=28, 
	cell_size=4, 
	min_object_size=7, 
	max_object_size=14, 
	num_objects=1, 
	train=0)

train, test = transform_to_conv((train,test))
features, labels = test
offsets = offsets[1]	# test offsets
first, last = 0, 5	# imgs to show

#<======================_LOAD_FULL_MODEL_======================>
model = tf.keras.models.load_model('./full_model/model_coords_10ep.h5')


#<======================_MODEL_PREDICT_======================>
bboxes = model.predict(features)

features, bboxes = bbox.transform_from_conv(features, bboxes)
imgs = bbox.restore_imgs(features)
bboxes = bbox.restore_bbox(bboxes, offsets, img_size=28, cell_size=4)

bbox.build_bbox(imgs[first:last], bboxes, img_size=28)