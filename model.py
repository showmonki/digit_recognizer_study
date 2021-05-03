import numpy as np
np.random.seed(1)  # or use import tensorflow as tf, tf.random.set_seed(1)
from tensorflow import keras
from tensorflow.keras import layers

class base_model():

	def __init__(self):
		self.input_shape = 784
		self.output_shape = 10
		self.model = self.load_model()

	def load_model(self):
		model = keras.Sequential(name='base_model')
		model.add(layers.Dense(50, activation='relu', input_shape=(self.input_shape,)))
		model.add(layers.Dense(100, activation='relu'))
		model.add(layers.Dropout(0.3))
		# model.add(layers.Dense(256, activation='relu'))
		# model.add(layers.Dropout(0.3))
		model.add(layers.Dense(50, activation='relu'))
		model.add(layers.Dense(self.output_shape, activation='softmax'))
		model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["acc"])
		return model


class cnn_model():

	def __init__(self):
		self.input_shape = (28,28,1) # 1 channel therefore 1
		self.output_shape = 10
		self.model = self.load_model()

	def load_model(self):
		model = keras.Sequential(name='base_model')
		model.add(layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=self.input_shape))
		model.add(layers.Conv2D(64, kernel_size=(3,3), activation='relu'))
		model.add(layers.MaxPool2D(pool_size=(2,2)))
		model.add(layers.Dropout(0.3))

		model.add(layers.Conv2D(64, kernel_size=(3,3), activation='relu'))
		model.add(layers.MaxPool2D(pool_size=(2,2)))
		model.add(layers.Dropout(0.3))

		model.add(layers.Flatten())
		model.add(layers.Dense(256, activation='relu'))
		model.add(layers.Dropout(0.3))
		model.add(layers.Dense(self.output_shape, activation='softmax'))

		model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["acc"])
		return model

if __name__ =='__main__':
	import pandas as pd
	from tensorflow.keras.utils import to_categorical
	import utils
	data_path = '../../digit-recognizer/'
	train_path = data_path + 'train.csv'
	test_path = data_path + 'test.csv'

	train_df = pd.read_csv(train_path)
	test_df = pd.read_csv(test_path)

	train_df['img_array'] = train_df.iloc[:, 1:].apply(lambda x: utils.convert_2d(x), axis=1)
	test_df['img_array'] = test_df.iloc[:, :].apply(lambda x: utils.convert_2d(x), axis=1)
	X = train_df.iloc[:, 1:-1]
	X_cnn = np.reshape(X.values,(len(train_df),28,28))
	X_cnn = np.expand_dims(X_cnn,-1)
	# X_cnn = train_df.iloc[:, -1]
	y = to_categorical(train_df['label'])
	model = base_model().model
	model.fit(X, y, epochs=5, validation_split=0.2)

	model.summary()

	cnn = cnn_model().model
	cnn.fit(X_cnn, y, epochs=10, validation_split=0.2)
	cnn.summary()

	# y_array = model.predict(X_test)
	# y_test = np.argmax(y_array, axis=1)
	X_test = test_df.iloc[:, :-1]
	X_test_cnn = np.expand_dims(np.reshape(X_test.values, (len(test_df),28,28)),-1)
	X_test_cnn = X_test_cnn.astype(np.float32)
	y_test_base = model.predict_classes(X_test)
	y_test_cnn = cnn.predict_classes(X_test_cnn)
	test_df['base'] = y_test_base
	test_df['cnn'] = y_test_cnn
	test_df['check'] = y_test_base == y_test_cnn
	test_df[~test_df['check']].head().index

	test_df['Label'] = y_test_cnn
	test_df['ImageId'] = test_df.index + 1
	test_df[['ImageId','Label']].to_csv(data_path+'submission.csv',index=False)
	# scores = model.evaluate(X, y)
	# print("test score is {0}, and acc is {1}".format(scores[0], scores[1]))
