"""
Samples elements randomly for imbalanced dataset
"""

import torch
import torch.utils.data
import torchvision

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):

	def __init__(self, dataset, opt, indices=None, num_samples=None):
		# if indices is not provided, all elements in the dataset will be considered
		self.indices = list(range(len(dataset))) \
			if indices is None else indices

		# if num_samples is not provided, draw `len(indices)` samples in each iteration
		self.num_samples = len(self.indices) \
			if num_samples is None else num_samples

		# distribution of classes in the dataset
		label_to_count = [[0, 0]] * opt.num_class
		for idx in self.indices:
			label_list = self._get_label(dataset, idx)
			for i in range(opt.num_class):
				label_to_count[i][label_list[i]] += 1

		# weight for each sample
		weights = [0] * len(self.indices)
		for i in range(len(self.indices)):
			idx = self.indices[i]
			label_list = self._get_label(dataset, idx)
			for j in range(opt.num_class):
				weights[i] += 1.0 / label_to_count[j][label_list[j]]

		self.weights = torch.DoubleTensor(weights)

	def __iter__(self):
		return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

	def __len__(self):
		return self.num_samples

	def _get_label(self, dataset, idx):
		return dataset.__getlabel__(idx)
