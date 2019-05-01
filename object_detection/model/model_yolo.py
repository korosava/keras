from tensorflow import keras as ks
import tensorflow as tf
from tensorflow.python.keras._impl.keras import backend as K

#<======================_STATIC_DATA_======================>
cell = 4
bbox = 2
coords = 5
classes = 0

#<======================_CREATE_MODEL_======================>
# послідовна модель, шар за шаром
model = ks.Sequential()
model.add(ks.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(28,28,1), padding='same', kernel_initializer='he_normal'))
model.add(ks.layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(ks.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))#14

model.add(ks.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(ks.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(ks.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))#7

model.add(ks.layers.Flatten())

model.add(ks.layers.Dense(1096, activation='selu', kernel_initializer='he_normal'))
model.add(ks.layers.Dropout(rate=0.5))
model.add(ks.layers.Dense(cell*cell*bbox*(coords+classes)))


#<======================_SAVE_CLEAR_MODEL_======================>
json_model = model.to_json()
with open('../saved_model/modelyolo_4.json', 'wt', encoding='utf-8') as fileobj:
	fileobj.write(json_model)