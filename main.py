import pandas as pd
from tensorflow.keras.utils import to_categorical
import utils
from model import base_model, cnn_model
import numpy as np
np.random.seed(1)


def load_df(train_path, test_path):
	train_df = pd.read_csv(train_path)
	test_df = pd.read_csv(test_path)

	train_df['img_array'] = train_df.iloc[:, 1:].apply(lambda x: utils.convert_2d(x), axis=1)
	test_df['img_array'] = test_df.iloc[:, :].apply(lambda x: utils.convert_2d(x), axis=1)
	return train_df, test_df

def main(train_path, test_path):
	train_df, test_df = load_df(train_path, test_path)
	X = train_df.iloc[:, 1:-1]
	X_cnn = np.reshape(X.values, (len(train_df), 28, 28))
	X_cnn = np.expand_dims(X_cnn, -1)
	# X_cnn = train_df.iloc[:, -1]
	y = to_categorical(train_df['label'])
	model = base_model().model
	model.fit(X, y, epochs=5, validation_split=0.2)
	# model.summary()

	cnn = cnn_model().model
	cnn.fit(X_cnn, y, epochs=10, validation_split=0.2)
	# cnn.summary()

	# y_array = model.predict(X_test)
	# y_test = np.argmax(y_array, axis=1)
	X_test = test_df.iloc[:, :-1]
	X_test_cnn = np.expand_dims(np.reshape(X_test.values, (len(test_df), 28, 28)), -1)
	X_test_cnn = X_test_cnn.astype(np.float32)
	y_test_base = model.predict_classes(X_test)
	y_test_cnn = cnn.predict_classes(X_test_cnn)
	test_df['base'] = y_test_base
	test_df['cnn'] = y_test_cnn
	return test_df


if __name__ == '__main__':
	data_path = '../../digit-recognizer/'
	train_path = data_path + 'train.csv'
	test_path = data_path + 'test.csv'

	test_df = main(train_path, test_path)
	print(test_df[~test_df['check']].head().index)
