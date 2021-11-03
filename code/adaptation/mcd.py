import torch.nn as nn
import torch

def classifier_discrepancy(predictions1: torch.Tensor, predictions2: torch.Tensor) -> torch.Tensor:
	r"""The `Classifier Discrepancy` in `Maximum Classiﬁer Discrepancy for Unsupervised Domain Adaptation <https://arxiv.org/abs/1712.02560>`_.
	The classfier discrepancy between predictions :math:`p_1` and :math:`p_2` can be described as:

	.. math::
		d(p_1, p_2) = \dfrac{1}{K} \sum_{k=1}^K | p_{1k} - p_{2k} |,

	where K is number of classes.

	Parameters:
		- **predictions1** (tensor): Classifier predictions :math:`p_1`. Expected to contain raw, normalized scores for each class
		- **predictions2** (tensor): Classifier predictions :math:`p_2`
	"""
	return torch.mean(torch.abs(predictions1 - predictions2))

def entropy(predictions: torch.Tensor) -> torch.Tensor:
	r"""Entropy of N predictions :math:`(p_1, p_2, ..., p_N)`.
	The definition is:

	.. math::
		d(p_1, p_2, ..., p_N) = -\dfrac{1}{K} \sum_{k=1}^K \log \left( \dfrac{1}{N} \sum_{i=1}^N p_{ik} \right)

	where K is number of classes.

	Parameters:
		- **predictions** (tensor): Classifier predictions. Expected to contain raw, normalized scores for each class
	"""
	num_class = predictions.shape[1]
	result = 0
	for i in range(num_class):
		prediction_1 = predictions[:, i]
		prediction_0 = 1 - prediction_1
		prediction = torch.cat((torch.unsqueeze(prediction_0, 1), torch.unsqueeze(prediction_1, 1)), dim=1)
		result += -torch.mean(torch.log(torch.mean(prediction, 0) + 1e-6))
	return result
