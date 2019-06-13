import numpy as np
import bbox
from custom_loss import kijFinder,iou
from input_data import yolo_input_pippeline
from tensorflow.python.keras._impl.keras import backend as K
from tensorflow.python.keras.utils import to_categorical
import matplotlib.pyplot as plt
import os
import dataGen as dg

'''
____KI_TEST____
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


'''
____IOU TEST____
tr =  [48, 38, 67, 34]
y  =  [51, 39, 60, 29] # 0.764
x  =  [47, 38, 66, 30] # 0.84

x1 = [0.46875, 0.1875, 0.515625, 0.234375]
tr1 = [0.5, 0.1875, 0.5234375, 0.265625] # 0.78

x = np.asarray(x, 'float32')
x = np.reshape(x, [1,1,4])
x1, off = bbox.normalize_bbox(x, 128, 32)

x1 = np.reshape(x1, [-1,4])
off = np.reshape(off, [-1,2])
print(x1,'\n\n', off)
x = bbox.restore_bbox_tensor(x1[0], off[0], 128, 32)
print(K.eval(x))
'''
'''
a = iou(x1)
print(K.eval(a))
print(x[:,:,0])
'''
































#print(os.listdir('data/test/code'))
# міняє імена файлів і директорій в data_dir на їх номер + offset
# (якшо такого імені не існує)
def rename_data(data_dir, offset):
	file_names = os.listdir(data_dir)
	for i in range(len(file_names)):
		file_path = os.path.join(data_dir, file_names[i])
		file_tokens = file_names[i].split('.')
		new_name = os.path.join(data_dir, str(offset+i)+'.'+file_tokens[-1])
		os.rename(file_path, new_name)

data_dir = r'E:\programming\python\study\tutorials\keras\img_labeling\BBox-Label-Tool\Images\train'
data_dir = r'E:\programming\python\study\neural\keras\object_detection\data\train_new\imgs'

rename_data(data_dir, 0)

