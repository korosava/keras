from tensorflow import keras as ks

#<======================_STATIC_DATA_======================>
cell = 4
bbox = 1
coords = 4
classes = 0

#<======================_CREATE_MODEL_======================>
# послідовна модель, шар за шаром
model = ks.Sequential()
model.add(ks.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(28,28,1), padding='same'))
model.add(ks.layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same'))
model.add(ks.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(ks.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'))
model.add(ks.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'))
model.add(ks.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(ks.layers.Flatten())

model.add(ks.layers.Dense(1096, activation='relu'))
model.add(ks.layers.Dropout(rate=0.5))
model.add(ks.layers.Dense(cell*cell*bbox*(coords+classes)))


#<======================_SAVE_CLEAR_MODEL_======================>
json_model = model.to_json()
with open('../saved_model/modelyolo_1.json', 'wt', encoding='utf-8') as fileobj:
	fileobj.write(json_model)
