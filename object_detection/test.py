import numpy as np
import bbox
from custom_loss import kijFinder
from input_data import yolo_input_pippeline
from tensorflow.python.keras._impl.keras import backend as K
from tensorflow.python.keras.utils import to_categorical
import matplotlib.pyplot as plt

'''
a = np.zeros([2*3*3*2*5]) 
for i in range (0,180,5):
	a[i:i+5] = i

a = np.reshape(a, [2,3,3,2,5])
a[:,:,2,0,:] = 100
print(a, '\n\n')

_max = kijFinder(a)
print(_max, '\n\n')
'''
'''
data_file = './data/code_labels.txt'
labels = bbox.bbox_from_file(data_file, num_objects = 1)
for arr in labels:
	print(arr)
print(labels.shape)
print(arr.dtype)
'''

a = [1,2,3]

np.random.seed(12)
np.random.shuffle(a)
print(a)

np.random.seed(12)
np.random.shuffle(a)
print(a)

np.random.seed(12)
np.random.shuffle(a)
print(a)