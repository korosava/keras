import numpy as np
import bbox
from custom_loss import kijFinder
from input_data import yolo_input_pippeline
from tensorflow.python.keras._impl.keras import backend as K
from tensorflow.python.keras.utils import to_categorical
import matplotlib.pyplot as plt
import os
import dataGen as dg

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

#print(os.listdir('data/test/code'))

# міняє імена файлів і директорій на в data_dir на їх номер
# (якшо такого імені не існує)
def rename_data(data_dir):
	file_names = os.listdir(data_dir)
	for i in range(len(file_names)):
		file_path = os.path.join(data_dir, file_names[i])
		file_tokens = file_names[i].split('.')
		new_name = os.path.join(data_dir, str(i)+'.'+file_tokens[-1])
		os.rename(file_path, new_name)

data_dir = r'E:\programming\python\study\tutorials\keras\img_labeling\BBox-Label-Tool\Images\train'
rename_data(data_dir)