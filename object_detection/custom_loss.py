import tensorflow as tf
from tensorflow.python.keras._impl.keras import backend as K
import numpy as np
import bbox


#shape: S*S*B*5+3
#B: (x, y, w, h, confidence)
# бере batch елементів pred, true з їх масивів у fit, застосовує помилку


def yolo_loss(y_true, y_pred):
	#print('\n\n\n\n')
	y_true1 = K.reshape(y_true, [-1,4,4,5])
	y_pred1 = K.reshape(y_pred, [-1,4,4,5])
	iou = iouFinder(y_true1, y_pred1)
	iou = K.reshape(iou, [-1,4,4,1])
	y_true = K.concatenate((y_true1[:,:,:,0:4], iou), axis = 3)
	y_true = K.reshape(y_true, [-1, 80])
	return K.mean(K.square(y_pred-y_true))


#S,S,B[conf_max]
def kijFinder(y_pred):
	shape = y_pred.shape
	bbox1 = np.reshape(y_pred[:,:,4], [shape[0], shape[1],1])
	bbox2 = np.reshape(y_pred[:,:,9], [shape[0], shape[1],1])
	bboxes_conf = np.concatenate((bbox1,bbox2), axis=2)
	res = K.argmax(bboxes_conf, axis=2)
	return res


#S,S,B[coords!=0]
def kiFinder(y_true):
	x_zero = K.equal(y_true[:,:,0], 0)	# x=0 -> 1
	y_zero = K.equal(y_true[:,:,1], 0)	# y=0 -> 1
	xy_zero =  tf.logical_and(x_zero, y_zero)	# 1,1 -> 1
	return xy_zero


def iouFinder(y_true, y_pred):
	coord1 = y_true[:,:,:,0:4]
	coord2 = y_pred[:,:,:,0:4]
	coords = K.concatenate((coord1,coord2), axis=3)
	coords = K.reshape(coords, [-1,8])####
	res = K.map_fn(iou, coords, dtype='float32')
	return res


def iou(coords):
	x1=coords[0]; y1=coords[1]; w1=coords[2]; h1=coords[3]
	x2=coords[4]; y2=coords[5]; w2=coords[6]; h2=coords[7]
	
	w_I = K.minimum(x1 + w1, x2 + w2) - K.maximum(x1, x2)
	h_I = K.minimum(y1 + h1, y2 + h2) - K.maximum(y1, y2)
	
	def i_dev_u(): return K.cast((w_I*h_I) / (w1*h1+w2*h2 - w_I*h_I), 'float32')
	def zero1(): return K.cast(0.0, 'float32')
	# нема перетину -> res=0
	# є перетин -> res = I/U
	###
	
	res = tf.cond(tf.logical_or(w_I<=0,h_I<=0), zero1, i_dev_u)
	return res



if __name__ == '__main__':
	a = np.ones((2,2,2,7))
	b = np.zeros((2,2,2,7))
	a[:,:,:,0]=1.5; a[:,:,:,1]=2.3; a[:,:,:,2]=3.1; a[:,:,:,3]=4
	b[:,:,:,0]=1; b[:,:,:,1]=2; b[:,:,:,2]=3; b[:,:,:,3]=4
	res = iouFinder(a, b)
	
	print(K.eval(res), res.shape, sep='\n')















'''
1) обчислити змінні, необхідні для помилки:
Kij, Ki, iou,


kij:
	max confidence серед bbox одної клітинки
	max(y[i,j,:,4]) -> 1
	скласти масив argmax індексів для всіх клітин
	вектор коефів Kij для кожної клітини
	- якшо макс індекс - 1, ні - 0


ki:
	не 0 (x,y) координати серед bbox 1ї клітини
	y[i,j,0,:] != 0 -> 1


iou:
	вхід: y1,y2 ---> (x,y,w,h)
	w_I = min(x1 + w1, x2 + w2) - max(x1, x2)
    h_I = min(y1 + h1, y2 + h2) - max(y1, y2)
	if w_I <= 0 or h_I <= 0:  # no overlap
        return 0.
    I = w_I * h_I

    U = w1 * h1 + w2 * h2 - I

    return I / U
'''