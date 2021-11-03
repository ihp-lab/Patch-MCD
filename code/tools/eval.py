import itertools
import numpy as np

from sklearn.metrics import accuracy_score, f1_score

def get_labels(predictions, threshold_list):
	for i in range(0, predictions.shape[1]):
		predictions[:, i] = (predictions[:, i] > threshold_list[i])
	return predictions

def get_eval_metrics(groundtruth, prediction, threshold_list, opt):
	groundtruth = list(itertools.chain.from_iterable(groundtruth))
	prediction = list(itertools.chain.from_iterable(prediction))

	groundtruth = np.array(groundtruth)
	prediction = np.array(prediction)
	prediction_clf = get_labels(prediction, threshold_list)

	acc_list = []
	f1_list = []

	for i in range(0, opt.num_class):
		_groundtruth = groundtruth[:,i]
		_prediction = prediction[:,i]
		_prediction_clf = prediction_clf[:,i]

		acc_list.append(accuracy_score(_prediction_clf, _groundtruth))
		f1_list.append(f1_score(_prediction_clf, _groundtruth))

	return acc_list, f1_list
