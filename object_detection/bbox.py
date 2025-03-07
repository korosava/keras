import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from copy import deepcopy
from tensorflow.python.keras._impl.keras import backend as K


#<==================================_BBOX_&_IMAGE_CREATING_==================================>
# Create images with random rectangles and bounding boxes. 
def create_rect(num_imgs, img_size, min_object_size, max_object_size, num_objects, channels):

	bboxes = np.zeros((num_imgs, num_objects, 4))
	imgs = np.zeros((num_imgs, img_size, img_size,channels))  # set background to 0

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
def bbox_from_file(data_file):
	labels = []
	with open(data_file, 'rt') as file:
		for line in file:
			line = line[0:-2]
			arr = line.split(" ")
			labels.append(arr)
	return np.asarray(labels, 'int32')


#<==================================_NORMALIZING_==================================>
def normalize_img(imgs):
	return imgs/256+0.001

def restore_imgs(imgs):
	return (imgs-0.001)*256

#[img_i,bbox_i,coords]
# нормалізація bbox:
# (x,y) -> центр, відносно розмірів клітини
# (w,h) -> ширина, висота, відностно розмірів зображення
# imgs має націло ділитись на cell size
# offsets - [img_num,obj_num,2]
# 2: x_offset, y_offset
def normalize_bbox(bboxes, img_size, cell_size):
	assert img_size%cell_size == 0, 'imgs МАЄ НАЦІЛО ДІЛИТИСЬ НА cell_size'
	# лівий нижній край -> центр
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
	# центр -> лівий нижній край
	bboxes[:,:,2:4]*=img_size # (w,h) відносто розмірів зображення
	bboxes[:,:,0]-=bboxes[:,:,2]/2 # x-w/2
	bboxes[:,:,1]-=bboxes[:,:,3]/2 # y-h/2
	bboxes = np.int_(bboxes)
	return bboxes


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
		plt.imshow(imgs[i].T, cmap='Greys', interpolation='none', origin='lower', extent=[0, imgs_shape[1], 0, imgs_shape[1]], vmin=0, vmax=255)
		for bbox in range(bboxes_shape[1]):
			for obj in range(bboxes_shape[2]):
				x,y,w,h = bboxes_batch[i,bbox,obj,:]
				plt.gca().add_patch(matplotlib.patches.Rectangle((x, y), w, h, ec='r', fc='none'))
		plt.show()


#<==================================_TESTING_==================================>
if __name__ == '__main__':
	imgs, packed_bboxes = create_bbox_data(num_imgs=5, img_size=28, cell_size=4, min_object_size=3, max_object_size=8, num_objects=2)
	print('\n\n',packed_bboxes[1].shape,'\n\n')
	bboxes = restore_bbox(*packed_bboxes,img_size=28,cell_size=4)
	imgs = restore_imgs(imgs)
	build_bbox(imgs, bboxes, img_size=28)

