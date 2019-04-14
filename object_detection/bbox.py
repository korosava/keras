import numpy as np
import matplotlib.pyplot as plt
import matplotlib




# Create images with random rectangles and bounding boxes. 
def create_rect(num_imgs=50000, img_size=8, min_object_size=1, max_object_size=4, num_objects=1):

	bboxes = np.zeros((num_imgs, num_objects, max_object_size))
	imgs = np.zeros((num_imgs, img_size, img_size))  # set background to 0

	for i_img in range(num_imgs):
	    for i_object in range(num_objects):
	    	w, h = np.random.randint(min_object_size, max_object_size, size=2)
	    	x = np.random.randint(0, img_size - w)
	    	y = np.random.randint(0, img_size - h)
	    	imgs[i_img, x:x+w, y:y+h] = 150  # set rectangle to 1
	    	bboxes[i_img, i_object] = [x, y, w, h]
	return (imgs, bboxes)


def build_bbox(imgs, bboxes, img_size=8):
	for i in range(1):
		matplotlib.pyplot.figure()
		plt.imshow(imgs[i].T, cmap='Purples', interpolation='none', origin='lower', extent=[0, img_size, 0, img_size], vmin=0,vmax=255)
		for bbox in bboxes[i]:
			plt.gca().add_patch(matplotlib.patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], ec='r', fc='none'))
		
	plt.show()
	

if __name__ == '__main__':
	imgs, bboxes = create_rect()
	print(imgs.shape, bboxes.shape, sep='\n')
	build_bbox(imgs, bboxes)
