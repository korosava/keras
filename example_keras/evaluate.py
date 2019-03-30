import tensorflow as tf
from input_data import load_data, transform_to_dense


#<======================_INPUT_DATA_======================>
data = load_data()
data = transform_to_dense(data)
train, test = data
f1, l1 = train
f2, l2 = test


#<======================_LOAD_FULL_MODEL_======================>
model = tf.keras.models.load_model('./full_model/model1.h5')


#<======================_EVALUATE_MODEL_======================>
evaluate = model.evaluate(f2, l2, verbose=0)
print('lose: {}\naccuracy: {}'.format(evaluate[0], evaluate[1]))