import tensorflow as tf
import custom_loss
from tensorflow.python.keras._impl.keras import backend as K
import bbox


def new_max_iou(y_true, y_pred):
	y_true = K.reshape(y_true, [-1,4,4,2,5])
	y_pred = K.reshape(y_pred, [-1,4,4,2,5])

	chunk = new_max_iou.iter * new_max_iou.batch_size
	offset = new_max_iou.offsets[chunk:chunk+new_max_iou.batch_size]
	iou = custom_loss.new_iouFinder(y_true, y_pred, offset)
	iou = bbox.iou_to_loss(iou, offset, 1, 2, 4)

	new_max_iou.iter += 1
	if new_max_iou.iter*new_max_iou.batch_size >= len(new_max_iou.offsets):
		new_max_iou.iter = 0
	
	iou = K.reshape(iou, [-1])
	max_iou = K.max(iou)
	return max_iou
	

def new_mean_iou (y_true, y_pred):
	y_true = K.reshape(y_true, [-1,4,4,2,5])
	y_pred = K.reshape(y_pred, [-1,4,4,2,5])
	
	chunk = new_mean_iou.iter * new_mean_iou.batch_size
	offset = new_mean_iou.offsets[chunk:chunk+new_mean_iou.batch_size]
	iou = custom_loss.new_iouFinder(y_true, y_pred, offset)
	iou = bbox.iou_to_loss(iou, offset, 1, 2, 4)
	
	new_mean_iou.iter += 1
	if new_mean_iou.iter*new_mean_iou.batch_size >= len(new_mean_iou.offsets):
		new_mean_iou.iter = 0

	iou = K.reshape(iou, [-1])

	greater_zero_iou = K.map_fn(greater_zero, iou, dtype='float32')
	mean_iou = K.sum(iou) / K.sum(greater_zero_iou) # sum(iou) / num(iou > 0)
	return mean_iou


def greater_zero(x):
	return K.cast((tf.cond(x>0, lambda:1, lambda:0)), 'float32')


def make_biger_if_zero(x):
	return K.cast((tf.cond(x>0, lambda:x, lambda:K.cast(101,'float32'))), 'float32')






'''
#iou = K.reshape(iou, [-1,4,4,1])
#y_true = K.concatenate((y_true1[:,:,:,0:4], iou), axis = 3)
#acuracy = y_true[:,:,:,4]*y_pred1[:,:,:,4]
#acuracy = K.reshape(acuracy, [-1,16])


# ЗАВЖДИ 0
# вибирати тільки з тих, шо > 0
def min_iou(y_true, y_pred):
	y_true = K.reshape(y_true, [-1,4,4,2,5])
	y_pred = K.reshape(y_pred, [-1,4,4,2,5])
	iou = custom_loss.iouFinder(y_true, y_pred) # вектор
	iou = K.reshape(iou, [-1,4,4,2])

	kij = custom_loss.kijFinder(y_pred)
	ki = custom_loss.kiFinder(y_true)
	iou *= ki
	iou *= kij

	iou = K.reshape(iou, [-1])
	norm_iou = K.map_fn(make_biger_if_zero, iou, dtype='float32')
	min_iou = K.min(norm_iou)
	return min_iou


def max_iou(y_true, y_pred):
	#print('SHAPE: ', y_true.shape, y_pred.shape)
	#print('\n\n\n\n')
	y_true = K.reshape(y_true, [-1,4,4,2,5])
	y_pred = K.reshape(y_pred, [-1,4,4,2,5])

	iou = custom_loss.iouFinder(y_true, y_pred) # вектор
	iou = K.reshape(iou, [-1,4,4,2])


	kij = custom_loss.kijFinder(y_pred)
	ki = custom_loss.kiFinder(y_true)
	iou *= ki
	iou *= kij

	iou = K.reshape(iou, [-1])
	max_iou = K.max(iou)
	return max_iou


def mean_iou (y_true, y_pred):
	y_true = K.reshape(y_true, [-1,4,4,2,5])
	y_pred = K.reshape(y_pred, [-1,4,4,2,5])
	iou = custom_loss.iouFinder(y_true, y_pred) # вектор
	iou = K.reshape(iou, [-1,4,4,2])

	kij = custom_loss.kijFinder(y_pred)
	ki = custom_loss.kiFinder(y_true)
	iou *= ki
	iou *= kij
	iou = K.reshape(iou, [-1])


	greater_zero_iou = K.map_fn(greater_zero, iou, dtype='float32')
	mean_iou = K.sum(iou) / K.sum(greater_zero_iou) # sum(iou) / num(iou > 0)
	return mean_iou
'''