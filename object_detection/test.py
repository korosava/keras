import numpy as np
import bbox
from custom_metrics import my_accuracy
from input_data import yolo_input_pippeline
from tensorflow.python.keras._impl.keras import backend as K

'''
a = np.zeros([2*3*3*5]) 
for i in range (0,90,5):
	a[i:i+10] = i
print(a,'\n\n')
#a = np.reshape(a, [-1,3,3,10])
a = np.reshape(a,[-1,3,3,5])
print(a)
a = np.reshape(a,[-1,3*3*5])
print(a)
'''
train = yolo_input_pippeline(
	num_imgs=500,  
	img_size=28, 
	cell_size=7, 
	min_object_size=3, 
	max_object_size=7, 
	num_objects=1,
	num_bboxes=1,
	channels=1,
	train=True)
imgs, bboxes = train

accuraccy = my_accuracy(bboxes, bboxes)
print(K.eval(accuraccy))