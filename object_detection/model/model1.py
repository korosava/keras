from tensorflow import keras as ks


#<======================_CREATE_MODEL_======================>
# послідовна модель, шар за шаром
model = ks.Sequential()
model.add(ks.layers.Dense(64, activation='relu', input_shape=(784,)))
model.add(ks.layers.Dropout(rate=0.2))
model.add(ks.layers.Dense(10))


#<======================_SAVE_CLEAR_MODEL_======================>
json_model = model.to_json()
with open('../saved_model/model1.json', 'wt', encoding='utf-8') as fileobj:
	fileobj.write(json_model)


