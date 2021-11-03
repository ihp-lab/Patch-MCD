"""
Loads data
"""

import os
import PIL
import torch
import pandas
import numpy as np
import tools.sampler as sampler

from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

class NonTemporalDataset(Dataset):
	def __init__(self, dataset_file_path, file_name_list, loader_type, opt):
		dataset_file = pandas.read_csv(dataset_file_path, index_col=0)
		dataset_subset = dataset_file.loc[file_name_list]

		self.file_name_list = file_name_list
		self.source_domain = opt.source_domain
		self.target_domain = opt.target_domain
		self.images_path = dataset_subset.image_path.to_dict()
		self.anchor_images_path = dataset_subset.anchor_image_path.to_dict()
		self.speakers = dataset_subset.speaker.to_dict()

		if self.target_domain == 'DISFA':
			self.au1s = dataset_subset.au1.to_dict()
			self.au2s = dataset_subset.au2.to_dict()
			self.au4s = dataset_subset.au4.to_dict()
			self.au6s = dataset_subset.au6.to_dict()
			self.au12s = dataset_subset.au12.to_dict()
		elif self.target_domain == 'GFT':
			self.au1s = dataset_subset.au1.to_dict()
			self.au2s = dataset_subset.au2.to_dict()
			self.au4s = dataset_subset.au4.to_dict()
			self.au6s = dataset_subset.au6.to_dict()
			self.au10s = dataset_subset.au10.to_dict()
			self.au12s = dataset_subset.au12.to_dict()
			self.au14s = dataset_subset.au14.to_dict()
			self.au15s = dataset_subset.au15.to_dict()
			self.au23s = dataset_subset.au23.to_dict()
			self.au24s = dataset_subset.au24.to_dict()
		else:
			self.au1s = dataset_subset.au1.to_dict()
			self.au2s = dataset_subset.au2.to_dict()
			self.au4s = dataset_subset.au4.to_dict()
			self.au6s = dataset_subset.au6.to_dict()
			self.au7s = dataset_subset.au7.to_dict()
			self.au10s = dataset_subset.au10.to_dict()
			self.au12s = dataset_subset.au12.to_dict()
			self.au14s = dataset_subset.au14.to_dict()
			self.au15s = dataset_subset.au15.to_dict()
			self.au17s = dataset_subset.au17.to_dict()
			self.au23s = dataset_subset.au23.to_dict()
			self.au24s = dataset_subset.au24.to_dict()

		if loader_type == 'train':
			transform_list = [	transforms.CenterCrop(140),
								transforms.Resize(128),
								transforms.ToTensor(),
								transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
		else:
			transform_list = [	transforms.CenterCrop(140),
								transforms.Resize(128),
								transforms.ToTensor(),
								transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]

		self.transform = transforms.Compose(transform_list)

	def __getitem__(self, idx):
		file_name = self.file_name_list[idx]

		image_path = self.images_path[file_name]
		image = PIL.Image.open(image_path)
		image = self.transform(image).float()

		anchor_image_path = self.anchor_images_path[file_name]
		anchor_image = PIL.Image.open(anchor_image_path)
		anchor_image = self.transform(anchor_image).float()

		speaker = self.speakers[file_name]

		if self.target_domain == 'DISFA':
			au1 = self.au1s[file_name]
			au2 = self.au2s[file_name]
			au4 = self.au4s[file_name]
			au6 = self.au6s[file_name]
			au12 = self.au12s[file_name]

			return image, anchor_image, speaker, [au1, au2, au4, au6, au12]

		elif self.target_domain == 'GFT':
			au1 = self.au1s[file_name]
			au2 = self.au2s[file_name]
			au4 = self.au4s[file_name]
			au6 = self.au6s[file_name]
			au10 = self.au10s[file_name]
			au12 = self.au12s[file_name]
			au14 = self.au14s[file_name]
			au15 = self.au15s[file_name]
			au23 = self.au23s[file_name]
			au24 = self.au24s[file_name]

			return image, anchor_image, speaker, [au1, au2, au4, au6, au10, au12, au14, au15, au23, au24]

		else:
			au1 = self.au1s[file_name]
			au2 = self.au2s[file_name]
			au4 = self.au4s[file_name]
			au6 = self.au6s[file_name]
			au7 = self.au7s[file_name]
			au10 = self.au10s[file_name]
			au12 = self.au12s[file_name]
			au14 = self.au14s[file_name]
			au15 = self.au15s[file_name]
			au17 = self.au17s[file_name]
			au23 = self.au23s[file_name]
			au24 = self.au24s[file_name]

			return image, anchor_image, speaker, [au1, au2, au4, au6, au7, au10, au12, au14, au15, au17, au23, au24]

	def __len__(self):

		return len(self.file_name_list)

	def __getlabel__(self, idx):
		file_name = self.file_name_list[idx]

		if self.target_domain == 'DISFA':
			au1 = self.au1s[file_name]
			au2 = self.au2s[file_name]
			au4 = self.au4s[file_name]
			au6 = self.au6s[file_name]
			au12 = self.au12s[file_name]

			return [au1, au2, au4, au6, au12]

		elif self.target_domain == 'GFT':
			au1 = self.au1s[file_name]
			au2 = self.au2s[file_name]
			au4 = self.au4s[file_name]
			au6 = self.au6s[file_name]
			au10 = self.au10s[file_name]
			au12 = self.au12s[file_name]
			au14 = self.au14s[file_name]
			au15 = self.au15s[file_name]
			au23 = self.au23s[file_name]
			au24 = self.au24s[file_name]

			return [au1, au2, au4, au6, au10, au12, au14, au15, au23, au24]

		else:
			au1 = self.au1s[file_name]
			au2 = self.au2s[file_name]
			au4 = self.au4s[file_name]
			au6 = self.au6s[file_name]
			au7 = self.au7s[file_name]
			au10 = self.au10s[file_name]
			au12 = self.au12s[file_name]
			au14 = self.au14s[file_name]
			au15 = self.au15s[file_name]
			au17 = self.au17s[file_name]
			au23 = self.au23s[file_name]
			au24 = self.au24s[file_name]

			return [au1, au2, au4, au6, au7, au10, au12, au14, au15, au17, au23, au24]

def collate_non_fn_temporal_dataset(data):
	images, anchor_images, speakers, labels = zip(*data)

	images = torch.stack(images)
	anchor_images = torch.stack(anchor_images)
	speakers = torch.LongTensor(speakers)
	labels = torch.FloatTensor(labels)

	return images, anchor_images, speakers, labels

def get_non_temporal_dataset(	dataset_file_path, file_name_list, batch_size, balance,
								shuffle, workers_num, collate_fn, loader_type, opt):
	dataset = NonTemporalDataset(	dataset_file_path=dataset_file_path,
									file_name_list=file_name_list,
									loader_type=loader_type,
									opt=opt)

	if balance:
		dataloader = DataLoader(dataset=dataset,
								batch_size=batch_size,
								sampler=sampler.ImbalancedDatasetSampler(dataset, opt),
								num_workers=workers_num,
								collate_fn=collate_fn,
								pin_memory=False)
	else:
		dataloader = DataLoader(dataset=dataset,
								batch_size=batch_size,
								shuffle=shuffle,
								num_workers=workers_num,
								collate_fn=collate_fn,
								pin_memory=False)

	return dataloader

def get_loaders_non_temporal_dataset(dataset_file_path, file_name_list, loader_type, opt):
	if loader_type == 'train':
		dataloader = get_non_temporal_dataset(	dataset_file_path=dataset_file_path,
												file_name_list=file_name_list,
												batch_size=opt.batch_size,
												balance=True,
												shuffle=True,
												workers_num=opt.workers_num,
												collate_fn=collate_non_fn_temporal_dataset, 
												loader_type=loader_type, 
												opt=opt)
	else:
		dataloader = get_non_temporal_dataset(	dataset_file_path=dataset_file_path,
												file_name_list=file_name_list,
												batch_size=opt.batch_size,
												balance=False,
												shuffle=False,
												workers_num=opt.workers_num,
												collate_fn=collate_non_fn_temporal_dataset,
												loader_type=loader_type, 
												opt=opt)

	return dataloader

def get_dataloader(dataset_file_path, loader_type, opt):
	# Data
	data = pandas.read_csv(dataset_file_path)
	file_name_list = data['file_name_list'].tolist()

	dataloader = get_loaders_non_temporal_dataset(	dataset_file_path,
													file_name_list,
													loader_type, opt)

	return dataloader
