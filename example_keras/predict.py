import tensorflow as tf
from input_data import load_data, transform_to_dense
import matplotlib.pyplot as plt
import numpy as np


#<======================_INPUT_DATA_======================>
data = load_data()
data = transform_to_dense(data)
train, test = data
f1, l1 = train
f2, l2 = test
first,last = 0,5


#<======================_LOAD_FULL_MODEL_======================>
model = tf.keras.models.load_model('./full_model/model1.h5')
#model.load_weights('./weight/model1')


#<======================_MODEL_PREDICT_======================>
predict = model.predict(f2[first:last], batch_size=1)
print(predict.argmax(axis=1))
print(l2[first:last].argmax(axis=1))

f2 = np.reshape(f2,(-1,28,28))


# <==============================_SHOW_==============================>
for i in range(first,last):
	show_imgs = plt.imshow(f2[i], cmap='gray')
	plt.show()