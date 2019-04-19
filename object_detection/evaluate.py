import tensorflow as tf
from custom_loss import yolo_loss
from input_data import transform_to_conv
import bbox

#<======================_INPUT_DATA_======================>
data = bbox.train_bbox_data()
data = transform_to_conv(data)
train, test = data
f1, l1 = train
f2, l2 = test


#<======================_LOAD_FULL_MODEL_======================>
model = tf.keras.models.load_model('./full_model/model_yolo_10ep.h5')

'''
#<======================_LOAD_CLEAR_MODEL_======================>
with open('./saved_model/model1.json', 'rt', encoding='utf-8') as fileobj:
	json_model = fileobj.read()
model = tf.keras.models.model_from_json(json_model)
model.compile(
	optimizer='adam',							#tf.train -> optimizers
	loss='categorical_crossentropy', 			#tf.keras.losses
	metrics=['accuracy'])			 			#tf.keras.metrics
'''
#<======================_WEIGHTS_LOAD_======================>
#model.load_weights('./weight/model1_10ep')


#<======================_EVALUATE_MODEL_======================>
evaluate = model.evaluate(f2, l2, verbose=0)
print('lose: {}\naccuracy: {}'.format(evaluate[0], evaluate[1]))