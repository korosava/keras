import tensorflow as tf
import custom_loss
from tensorflow.python.keras._impl.keras import backend as K


def metric_iou(y_true, y_pred):
	#print('SHAPE: ', y_true.shape, y_pred.shape)
	#print('\n\n\n\n')
	y_true1 = K.reshape(y_true, [-1,4,4,5])
	y_pred1 = K.reshape(y_pred, [-1,4,4,5])

	iou = custom_loss.iouFinder(y_true1, y_pred1) # вектор
	#iou = K.reshape(iou, [-1,4,4,1])
	#y_true = K.concatenate((y_true1[:,:,:,0:4], iou), axis = 3)
	#acuracy = y_true[:,:,:,4]*y_pred1[:,:,:,4]
	#acuracy = K.reshape(acuracy, [-1,16])
	#greater_zero_iou = K.map_fn(greater_zero, iou, dtype='float32')
	#acuracy = K.sum(iou) / K.sum(greater_zero_iou) # sum(iou) / num(iou > 0)
	#print('\ngreater_zero Shape:\n{}\n\n'.format(greater_zero_iou.shape))
	max_iou = K.max(iou)
	return max_iou

def greater_zero(x):
	return K.cast((tf.cond(x>0, lambda:1, lambda:0)), 'float32')