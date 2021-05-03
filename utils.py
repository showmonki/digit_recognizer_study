import numpy as np
import matplotlib.pyplot as plt


def convert_2d(x):
	return np.reshape(x.values, (28, 28))


def print_img(img, label):
	plt.imshow(img)
	if label:
		plt.title('The label is %s' % label)


def model_compare(img, result1, result2):
	plt.imshow(img)
	plt.title('Base model predict: {0}, CNN predict: {1}'.format(result1, result2))


