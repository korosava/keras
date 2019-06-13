import tensorflow as tf
import numpy as np
import bbox
from tensorflow.python.keras._impl.keras import backend as K
from tensorflow.python.keras.utils import to_categorical

#shape: S*S*B*5+3
#B: (x, y, w, h, confidence)
# бере batch елементів pred, true з їх масивів у fit, застосовує помилку


def yolo_loss(y_true, y_pred):
	l_coord = 5
	l_size = 5
	l_conf = 0.1	# 0.1   0.5
	l_noobj = 0.01	# 0.01  0.1
	
	y_true = K.reshape(y_true, [-1,4,4,2,5])
	y_pred = K.reshape(y_pred, [-1,4,4,2,5])

	chunk = yolo_loss.iter * yolo_loss.batch_size
	offset = yolo_loss.offsets[chunk:chunk+yolo_loss.batch_size]
	new_iou = new_iouFinder(y_true, y_pred, offset)

	yolo_loss.iter += 1
	if yolo_loss.iter*yolo_loss.batch_size >= len(yolo_loss.offsets):
		yolo_loss.iter = 0
	
	new_iou = bbox.iou_to_loss(new_iou, offset, 1, 2, 4)

	#new_iou = iouFinder(y_true,y_pred)
	#new_iou = K.reshape(new_iou, [-1,4,4,2,1])

	y_true = K.concatenate((y_true[:,:,:,:,0:4], new_iou), axis = 4)
	kij = kijFinder(y_pred)
	ki = kiFinder(y_true)
	kij_not = K.cast(tf.logical_not(K.cast(kij, 'bool')), 'float32')

	loss_coord = (
	l_coord *
		K.mean(
			K.sum(
				K.sum(
					K.sum(
						ki *
							kij *
								K.sum(
									K.square(
										y_pred[:,:,:,:,0:2]-y_true[:,:,:,:,0:2]), axis=4), axis=3), axis=2), axis=1), axis=0))
	loss_size = (
	l_size *
		K.mean(
			K.sum(
				K.sum(
					K.sum(
						ki *
							kij *
								K.sum(
									K.square(
										K.sqrt(y_pred[:,:,:,:,2:4]) - K.sqrt(y_true[:,:,:,:,2:4])), axis=4), axis=3), axis=2), axis=1), axis=0))
	loss_confidence_1 = (
	l_conf *
		K.mean(
			K.sum(
				K.sum(
					K.sum(
						ki *
							kij *
								K.square(
									y_pred[:,:,:,:,4]-y_true[:,:,:,:,4]), axis=3), axis=2), axis=1), axis=0))
	loss_confidence_2 = (
	l_noobj *
		K.mean(
			K.sum(
				K.sum(
					K.sum(
						ki *
							kij_not *
								K.square(
									y_pred[:,:,:,:,4]-y_true[:,:,:,:,4]), axis=3), axis=2), axis=1), axis=0))

	return loss_coord + loss_size + loss_confidence_1+loss_confidence_2


# batch,S,S,B[conf_max]
def kijFinder(y_pred):
	max_bbox_iou_index = K.argmax(y_pred[:,:,:,:,4], axis=3)
	res = K.one_hot(max_bbox_iou_index, num_classes=2)

	return res


# batch,S,S,B[coords!=0]
def kiFinder(y_true):
	x_zero = K.equal(y_true[:,:,:,:,0], 0)	# x=0 -> 1
	y_zero = K.equal(y_true[:,:,:,:,1], 0)	# y=0 -> 1
	xy_zero =  tf.logical_not(tf.logical_and(x_zero, y_zero))	# 1,1 -> 0

	return K.cast(xy_zero, 'float32')


def iouFinder(y_true, y_pred):
	coord1 = y_true[:,:,:,:,0:4]
	coord2 = y_pred[:,:,:,:,0:4]
	coords = K.concatenate((coord1,coord2), axis=4)
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



'''
Обчислення IOU для одного з передбачених bbox та правдивого bbox:
[x, y, w, h]1  =  [0.46875, 0.1875, 0.515625, 0.234375] (px)	
[x, y, w, h]t = [0.5, 0.1875, 0.5234375, 0.265625] (px)
w_i = min(0.46875+0.515625, 0.5+0.5234375) – max(0.46875, 0.5) = 0.99-0.5 = 0.49
h_i = min(0.1875+0.234375, 0.1875+0.265625) – max(0.1875, 0.1875) = 0.4215 – 0.1875 = 0.234
I = 0.49*0.234 = 0.114
U = (0.5156*0.2343 + 0.5234*0.2656) – 0.114 = 0.1458 
IOU = 0.114/0.1458 ~ 0.78

треба переробити на нормальне, в PX
'''
def new_iouFinder(y_true, y_pred, offset):
	# [img, bbox, obj, 4]
	y_pred = bbox.loss_to_labels_tensor(y_pred, offset, num_cells=4, num_bboxes=2, img_size=128)
	y_true = bbox.loss_to_labels_tensor(y_true, offset, num_cells=4, num_bboxes=2, img_size=128)
	#[img*bbox, 4]
	coords = K.concatenate((y_true, y_pred), axis=1)
	res = K.map_fn(iou, coords, dtype='float32')
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