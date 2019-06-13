import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from copy import deepcopy
from tensorflow.python.keras._impl.keras import backend as K
import os
from os.path import join
import tensorflow as tf


#<==================================_BBOX_&_IMAGE_CREATING_==================================>
# Create images with random rectangles and bounding boxes. 
def create_rect(num_imgs, img_size, min_object_size, max_object_size, num_objects, channels):

	bboxes = np.zeros((num_imgs, num_objects, 4))
	imgs = np.zeros((num_imgs, img_size, img_size, channels))  # set background to 0

	for i_img in range(num_imgs):
		for i_object in range(num_objects):
			for i_ch in range(channels):
				w, h = np.random.randint(min_object_size, max_object_size, size=2)
				x = np.random.randint(0, img_size - w) # лівий нижній край (x,y)
				y = np.random.randint(0, img_size - h)
				imgs[i_img, x:x+w, y:y+h, i_ch] = 200  # set rectangle to 1
		bboxes[i_img, i_object] = [x, y, w, h]
	return (imgs, bboxes)


# повертає масив [batch, 4]
def bbox_from_file(bboxes_dir, num_objects):
	labels = []
	file_names = os.listdir(bboxes_dir)
	for file_name in file_names:
		with open(join(bboxes_dir, file_name), 'rt') as file:
			read = 0
			for line in file:
				if read:
					line = line[0:-1]
					arr = line.split(" ")
					labels.append(arr)
				else:
					read=1 # пропуск 1 стрічки
	labels = convert_bbox_format(np.asarray(labels, dtype='float32'))
	return labels.reshape([-1, num_objects, 4])


# (лівий верх), (правий низ) -> (лівий верх), (ширина висота); (0 координат зліва зверху)
def convert_bbox_format(bboxes):
	new_bboxes = np.zeros([len(bboxes), 1, 4])
	for i in range(len(bboxes)):
		w = bboxes[i][2] - bboxes[i][0]
		h = bboxes[i][3] - bboxes[i][1]
		x = bboxes[i][0]
		y = bboxes[i][1]
		new_bboxes[i] = [x, y, w, h]
	return new_bboxes


# масив посортованих за номером name.number.jpeg зображень
def img_from_file(imgs_dir):
	imgs = []
	for file_name in os.listdir(imgs_dir):
		img = plt.imread(os.path.join(imgs_dir, file_name))
		imgs.append(img)
	return np.asarray(imgs, 'int')


# shuffles imgs and it's bboxes
def shuffle_data(imgs, bboxes):
	seed = np.random.randint(low=1000)
	np.random.seed(seed)
	np.random.shuffle(imgs)
	np.random.seed(seed)
	np.random.shuffle(bboxes)
	return (imgs, bboxes)


#<==================================_NORMALIZING_==================================>
def normalize_img(imgs):
	return imgs/256+0.001

def restore_imgs(imgs):
	return np.asarray(((imgs-0.001)*256), dtype='int')

#[img_i,bbox_i,coords]
# нормалізація bbox:
# (x,y) -> центр, відносно розмірів клітини
# (w,h) -> ширина, висота, відностно розмірів зображення
# imgs має націло ділитись на cell size
# offsets - [img_num,obj_num,2]
# 2: x_offset, y_offset
def normalize_bbox(bboxes, img_size, cell_size):
	assert img_size%cell_size == 0, 'imgs МАЄ НАЦІЛО ДІЛИТИСЬ НА cell_size'
	# лівий верхній край -> центр (0 координат зліва зверху)
	bboxes[:,:,0]+=bboxes[:,:,2]/2 # x+w/2
	bboxes[:,:,1]+=bboxes[:,:,3]/2 # y+h/2

	bboxes[:,:,2:4]/=img_size # (w,h) відносто розмірів зображення

	offsets = np.int_(bboxes[:,:,0:2]/cell_size) #(x,y) відносно кожної клітинки (int до меншого)
	bboxes[:,:,0:2] -= cell_size*offsets
	bboxes[:,:,0:2] /= cell_size
	return (bboxes, offsets)

# денормалізація bbox
def restore_bbox(bboxes, offsets, img_size, cell_size):
	assert img_size%cell_size == 0, 'imgs МАЄ НАЦІЛО ДІЛИТИСЬ НА cell_size'

	bboxes[:,:,0:2] *= cell_size
	bboxes[:,:,0:2] += cell_size*offsets #(x,y) відносно кожної клітинки
	bboxes[:,:,2:4] *= img_size # (w,h) відносто розмірів зображення

	# центр -> лівий верхній край (0 координат зліва зверху)
	bboxes[:,:,0]-=bboxes[:,:,2]/2 # x-w/2
	bboxes[:,:,1]-=bboxes[:,:,3]/2 # y-h/2
	bboxes = np.int_(bboxes)
	return bboxes




'''
# денормалізація bbox (по 1 через map)
def restore_bbox_tensor(bboxes, offsets, img_size, cell_size):
	assert img_size%cell_size == 0, 'imgs МАЄ НАЦІЛО ДІЛИТИСЬ НА cell_size'
	res = [0 for i in range(bboxes.shape[0])]
	
	# ЗРОБИТИ ЧЕРЕЗ КОНКАТЕНАЦІЮ ТЕНЗОРІВ
	res[2] = bboxes[2]*img_size
	res[3] = bboxes[3]*img_size
	
	res[0] = bboxes[0]*cell_size + cell_size*offsets[0] - bboxes[2]/2
	res[1] = bboxes[1]*cell_size + cell_size*offsets[1] - bboxes[3]/2

	res = K.cast(res, 'int32')
	return bboxes
'''

# денормалізація bbox (по 1 через map)
def restore_bbox_tensor(bboxes, offsets, img_size, cell_size):
	assert img_size%cell_size == 0, 'imgs МАЄ НАЦІЛО ДІЛИТИСЬ НА cell_size'

	res_wh = bboxes[2:4]*img_size
	res_x = bboxes[0]*cell_size + cell_size*offsets[0] - res_wh[0]/2
	res_y = bboxes[1]*cell_size + cell_size*offsets[1] - res_wh[1]/2
	full_res = tf.stack([res_x, res_y, res_wh[0], res_wh[1]])

	return full_res





#<==================================_DATASET_CREATING_==================================>
# returns (features, labels)
# features: imgs
# labels: (bboxes, offsets)
# bboxes: (x,y,w,h)
# offsets: offsets of (x,y) to restore normalized bbox
def create_bbox_data(num_imgs, img_size, cell_size, min_object_size, max_object_size, num_objects):
	imgs,bboxes = create_rect(num_imgs, img_size, min_object_size, max_object_size, num_objects)
	imgs = normalize_img(imgs)
	packed_bboxes = normalize_bbox(bboxes,img_size,cell_size)
	return(imgs, packed_bboxes)

# train data_set (without offsets)
# COORDS, NOT YOLO
def use_bbox_data(num_imgs_train, num_imgs_test, img_size, cell_size, min_object_size, max_object_size, num_objects, train=True):
	f1, lo1 = create_bbox_data(num_imgs_train, img_size, cell_size, min_object_size, max_object_size, num_objects)
	f2, lo2 = create_bbox_data(num_imgs_test, img_size, cell_size, min_object_size, max_object_size, num_objects)
	l1, offset1 = lo1
	l2, offset2 = lo2
	l1 = np.reshape(l1, [num_imgs_train,4])
	l2 = np.reshape(l2, [num_imgs_test,4])
	if train:
		res = (f1,l1), (f2,l2)
	else:
		res = (f1,l1), (f2,l2), (offset1,offset2)
	return res

#[img_i, bbox_i, coords]
def transform_from_conv(imgs, bboxes):
	imgs = imgs.reshape(-1,28,28)
	bboxes = bboxes.reshape(-1,1,4)
	return imgs,bboxes


#<==================================_BBOXES_TO_LOSS_==================================>
# predictions vector -> loss input shape
def predict_to_loss(
	bboxes,
	num_cells,
	num_bboxes):

	bboxes = K.reshape(bboxes,[-1,num_cells,num_cells,num_bboxes*4])
	return bboxes

# generated bboxes -> loss input shape
def labels_to_loss(
	bboxes,
	num_cells,
	num_bboxes,
	img_size,
	num_imgs):

	cell_size = int(img_size/num_cells)
	norm_bboxes, offsets = normalize_bbox(bboxes, img_size, cell_size)
	labels = np.zeros([num_imgs,num_cells,num_cells,num_bboxes,4])

	for img in range(num_imgs):
		for obj in range(len(offsets[0])):
			for bbox in range(num_bboxes):
				x_offset,y_offset = offsets[img,obj]
				labels[img,x_offset,y_offset,bbox] = bboxes[img,obj] # !!!

	confidences = np.ones([num_imgs,num_cells,num_cells,num_bboxes,1]) # !!!
	
	labels = np.concatenate((labels, confidences), axis=4)
	labels = np.reshape(labels,[-1,num_cells*num_cells*num_bboxes*5])
	return labels, offsets


def loss_to_labels(
	labels,
	offsets,
	num_cells,
	num_bboxes,
	img_size):
	num_imgs = len(offsets)
	cell_size = int(img_size/num_cells)
	num_objects = len(offsets[0])

	labels = np.reshape(labels, [-1,num_cells,num_cells,num_bboxes,5])
	confidences = labels[:,:,:,:,4]
	confidences = np.reshape(confidences, [-1, num_cells*num_cells*num_bboxes])
	labels = labels[:,:,:,:,0:4]

	bboxes_batch = np.zeros([num_imgs, num_bboxes, num_objects, 4])
	for img in range(num_imgs):
		for obj in range(num_objects):
			for bbox in range(num_bboxes):
				x_offset,y_offset = offsets[img,obj]
				bboxes_batch[img,bbox,obj] = labels[img,x_offset,y_offset,bbox] # !!!

	for bbox in range(num_bboxes):
		bboxes_batch[:,bbox,:] = restore_bbox(bboxes_batch[:, bbox, :], offsets, img_size, cell_size)
	
	return bboxes_batch, confidences




# перетворення тензорів, повернення рохмірів bbox до сбсолютних ів відносних
def loss_to_labels_tensor(
	labels,
	offsets,
	num_cells,
	num_bboxes,
	img_size):
	num_imgs = len(offsets)
	cell_size = int(img_size/num_cells)
	num_objects = len(offsets[0])

	labels = K.reshape(labels, [-1, num_cells, num_cells, num_bboxes, 5])
	confidences = labels[:,:,:,:,4]
	confidences = K.reshape(confidences, [-1, num_cells*num_cells*num_bboxes])
	labels = labels[:,:,:,:,0:4]

	bboxes_batch = [] # [-1, bboxes]
	for img in range(num_imgs):
		for obj in range(num_objects):
			for bbox in range(num_bboxes):
				x_offset,y_offset = offsets[img,obj]
				bboxes_batch.append(labels[img,x_offset,y_offset,bbox])

	offsets = K.reshape(offsets, [-1, num_bboxes])
	offsets = K.cast(offsets, 'float32')

	for i,n in zip(range(int(len(bboxes_batch)/num_bboxes)), range(len(bboxes_batch))):
		for j in range(num_bboxes):
			bboxes_batch[n] = restore_bbox_tensor(bboxes_batch[i], offsets[i], img_size, cell_size) # однакові offsets для всіх bbox

	# відновлені bbox
	return K.cast(bboxes_batch, 'int32')

# перетворення iou до формату помилки
def iou_to_loss(iou, offsets, num_objects, num_bboxes, num_cells):
	num_imgs = len(offsets)
	iou = K.reshape(iou, [-1, 2])
	#[imgs, cells, cells, bboxes, 1]
	res = [[[[[0 for m in range(1)] 
	 			 for l in range(num_bboxes)] 
	 			 for k in range(num_cells)] 
	 			 for j in range(num_cells)] 
	 			 for i in range(num_imgs)] 

	for img in range(num_imgs):
		for obj in range(num_objects):
			for bbox in range(num_bboxes):
				x_offset,y_offset = offsets[img,obj]
				res[img][x_offset][y_offset][bbox][0] = iou[img, bbox]
	return res




#<==================================_BBOXES_&_IMAGE_PLOTTING_==================================>
def build_bbox(imgs, bboxes, img_size):
	for i in range(len(imgs)):
		matplotlib.pyplot.figure()
		plt.imshow(imgs[i].T, cmap='Greys', interpolation='none', origin='lower', extent=[0, img_size, 0, img_size], vmin=0, vmax=255)
		for bbox in bboxes[i]:
			plt.gca().add_patch(matplotlib.patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], ec='r', fc='none'))
		
		plt.show()

def build_bboxes(imgs, bboxes_batch):
	imgs_shape = imgs.shape
	bboxes_shape = bboxes_batch.shape
	for i in range(imgs_shape[0]):
		matplotlib.pyplot.figure()
		plt.imshow(imgs[i], origin='lower')
		for bbox in range(bboxes_shape[1]):
			for obj in range(bboxes_shape[2]):
				x,y,w,h = bboxes_batch[i,bbox,obj,:]
				plt.gca().add_patch(matplotlib.patches.Rectangle((x, y), w, h, ec='r', fc='none'))
		print('\n\n',bboxes_batch[i])
		plt.show()

def buid_labs_preds(imgs, labels, preds):
	imgs_shape = imgs.shape
	bboxes_shape = labels.shape
	for i in range(imgs_shape[0]):
		matplotlib.pyplot.figure()
		plt.imshow(imgs[i], origin='lower')
		for bbox in range(bboxes_shape[1]):
			for obj in range(bboxes_shape[2]):
				x,y,w,h = labels[i,bbox,obj,:]
				plt.gca().add_patch(matplotlib.patches.Rectangle((x, y), w, h, ec='g', fc='none'))
				x1,y1,w1,h1 = preds[i,bbox,obj,:]
				plt.gca().add_patch(matplotlib.patches.Rectangle((x1, y1), w1, h1, ec='r', fc='none'))
		print('\n\n\n',labels[i], '\n', preds[i])
		plt.show()

#<==================================_TESTING_==================================>
if __name__ == '__main__':
	imgs, packed_bboxes = create_bbox_data(num_imgs=5, img_size=28, cell_size=4, min_object_size=3, max_object_size=8, num_objects=2)
	print('\n\n',packed_bboxes[1].shape,'\n\n')
	bboxes = restore_bbox(*packed_bboxes,img_size=28,cell_size=4)
	imgs = restore_imgs(imgs)
	build_bbox(imgs, bboxes, img_size=28)

