import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from copy import deepcopy


# Create images with random rectangles and bounding boxes. 
def create_rect(num_imgs, img_size, min_object_size, max_object_size, num_objects):

	bboxes = np.zeros((num_imgs, num_objects, 4))
	imgs = np.zeros((num_imgs, img_size, img_size))  # set background to 0

	for i_img in range(num_imgs):
	    for i_object in range(num_objects):
	    	w, h = np.random.randint(min_object_size, max_object_size, size=2)
	    	x = np.random.randint(0, img_size - w) # лівий нижній край (x,y)
	    	y = np.random.randint(0, img_size - h)
	    	imgs[i_img, x:x+w, y:y+h] = 200  # set rectangle to 1
	    	bboxes[i_img, i_object] = [x, y, w, h, 1]
	return (imgs, bboxes)


def create_grid_predicts(
	bboxes,
	num_imgs,
	num_cells,
	num_bboxes):

	bboxes = np.reshape(bboxes,[num_imgs,num_cells,num_cells,num_bboxes*5])
	return bboxes


def build_bbox(imgs, bboxes, img_size):
	for i in range(len(imgs)):
		matplotlib.pyplot.figure()
		plt.imshow(imgs[i].T, cmap='Greys', interpolation='none', origin='lower', extent=[0, img_size, 0, img_size], vmin=0, vmax=255)
		for bbox in bboxes[i]:
			plt.gca().add_patch(matplotlib.patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], ec='r', fc='none'))
		
		plt.show()


def normalize_img(imgs):
	return imgs/256+0.001


def restore_imgs(imgs):
	return (imgs-0.001)*256


#[img_i,bbox_i,coords]
# нормалізація bbox:
# (x,y) -> центр, відносно розмірів клітини
# (w,h) -> ширина, висота, відностно розмірів зображення
# imgs має націло ділитись на cell size 
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


if __name__ == '__main__':
	imgs, packed_bboxes = create_bbox_data(num_imgs=5, img_size=28, cell_size=4, min_object_size=3, max_object_size=8, num_objects=2)
	bboxes = restore_bbox(*packed_bboxes,img_size=28,cell_size=4)
	imgs = restore_imgs(imgs)
	build_bbox(imgs, bboxes, img_size=28)
