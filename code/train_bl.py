"""
Trains and validates models
"""

import os
import csv
import loss
import torch
import models
import warnings
import argparse
import numpy as np

from tools.utils import set_seed
from datasets import get_dataloader
from tools.eval import get_eval_metrics

from test_models import test

warnings.filterwarnings('always')

def main():
	parser = argparse.ArgumentParser()

	# Seed
	parser.add_argument('--seed', type=int, default=0)

	# Names, paths, logs
	parser.add_argument('--logger_path', default='checkpoints/bl', help='relative path to log')
	parser.add_argument('--source_domain', default='BP4D')
	parser.add_argument('--target_domain', default='GFT')
	parser.add_argument('--verbose', type=bool, default=True, help='True or False')
	parser.add_argument('--save_checkpoint', type=bool, default=False, help='True or False')
	parser.add_argument('--save_model', type=bool, default=True, help='True or False')

	# Data parameters
	parser.add_argument('--workers_num', type=int, default=4, help='number of workers for data loading')
	parser.add_argument('--folder', default='')

	# Training and optimization
	parser.add_argument('--epochs_num', type=int, default=30, help='number of training epochs')
	parser.add_argument('--batch_size', type=int, default=64, help='size of a mini-batch')
	parser.add_argument('--learning_rate', type=float, default=2e-4, help='initial learning rate')

	# Model parameters
	parser.add_argument('--threshold', type=float, default=0.5)
	parser.add_argument('--num_class', type=int, default=0)
	parser.add_argument('--dropout_rate', type=float, default=0.1, help='0.1')

	parser.add_argument('--pixel_weight', type=float, default=30, help='30')
	parser.add_argument('--smooth_weight', type=float, default=0.16, help='0.16')
	parser.add_argument('--dissim_weight', type=float, default=1, help='1')
	parser.add_argument('--au_weight', type=float, default=30, help='30')

	# GPU
	parser.add_argument('--gpu_num', default='cuda:0', help='GPU device')

	opt = parser.parse_args()

	if opt.verbose:
		print('Training and validating models')
		for arg in vars(opt):
			print(arg + ' = ' + str(getattr(opt, arg)))

	set_seed(opt.seed)

	if opt.target_domain == 'DISFA':
		aus = [1, 2, 4, 6, 12]
	elif opt.target_domain == 'GFT':
		aus = [1, 2, 4, 6, 10, 12, 14, 15, 23, 24]
	else:
		aus = [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24]

	opt.num_class = len(aus)

	f1_list = []
	for folder in range(1, 4):
		test_f1, _ = train_one_folder(opt, str(folder))
		f1_list.append(test_f1)

	print('average performance')
	f1_list = np.array(f1_list)
	for i in range(0, opt.num_class):
		_f1_list = np.array(f1_list[:,i])
		print(	'au{}'.format(aus[i]),
				'f1 mean: {0:.3f}'.format(np.mean(_f1_list)),
				'f1 std: {0:.3f}'.format(np.std(_f1_list)))

	print(np.mean(f1_list))

def train_one_folder(opt, folder):
	# Use specific GPU
	device = torch.device(opt.gpu_num)

	opt.folder = folder

	# Dataloaders
	train_dataset_file_path = os.path.join('../dataset_all_labels', opt.source_domain, opt.folder, 'train.csv')
	train_loader = get_dataloader(train_dataset_file_path, 'train', opt)

	test_dataset_file_path = os.path.join('../dataset_all_labels', opt.source_domain, opt.folder, 'test.csv')
	test_loader = get_dataloader(test_dataset_file_path, 'test', opt)

	# Model, optimizer and loss function
	model = models.AU_Detection(opt)
	# model.apply(models.init_weights)
	for param in model.parameters():
		param.requires_grad = True
	model.to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)

	threshold_list = [opt.threshold] * opt.num_class

	criterion = loss.Total_Loss(opt)
 
	best_eval_metrics = 0.
	best_f1 = [0]*opt.num_class
	single_f1 = [0]*opt.num_class

	# Train and validate
	for epoch in range(opt.epochs_num):
		if opt.verbose:
			print('epoch: {}/{}'.format(epoch + 1, opt.epochs_num))

		train_loss, train_acc, train_f1 = train(train_loader, model,
												optimizer, criterion,
												device, threshold_list, opt)
		mean_train_acc = sum(train_acc)/len(train_acc)
		mean_train_f1 = sum(train_f1)/len(train_f1)
		test_loss, test_acc, test_f1 = test(test_loader, model,
											criterion, device,
											threshold_list, opt)
		mean_test_acc = sum(test_acc)/len(test_acc)
		mean_test_f1 = sum(test_f1)/len(test_f1)

		if opt.verbose:
			print(	'\ntrain_loss: {0:.3f}'.format(train_loss),
					'mean_train_acc: {0:.3f}'.format(mean_train_acc),
					'mean_train_f1: {0:.3f}'.format(mean_train_f1),
					'\ntest_loss: {0:.3f}'.format(test_loss),
					'mean_test_acc: {0:.3f}'.format(mean_test_acc),
					'mean_test_f1: {0:.3f}'.format(mean_test_f1))

		state = {	'epoch': epoch+1,
					'model': model.state_dict(),
					'opt': opt}
		os.makedirs(os.path.join(opt.logger_path, opt.source_domain, opt.folder), exist_ok=True)

		if opt.save_checkpoint:
			model_file_name = os.path.join(opt.logger_path, opt.source_domain, opt.folder, 'checkpoint.pth.tar')
			torch.save(state, model_file_name)

		for i in range(opt.num_class):
			if single_f1[i] < test_f1[i]:
				single_f1[i] = test_f1[i]

		if mean_test_f1 > best_eval_metrics:
			best_eval_metrics = mean_test_f1
			best_f1 = test_f1

			if opt.save_model:
				model_file_name = os.path.join(opt.logger_path, opt.source_domain, opt.folder, 'model.pth.tar')
				torch.save(state, model_file_name)

	return best_f1, single_f1

def train(train_loader, model, optimizer, criterion, device, threshold_list, opt):
	model.train()

	running_loss = 0.

	groundtruth = []
	prediction = []

	for i, train_data in enumerate(train_loader):
		images, anchor_images, speakers, labels = train_data

		images = images.to(device)
		anchor_images = anchor_images.to(device)
		speakers = speakers.to(device)
		labels = labels.to(device)

		predictions, reconstructed_images, flows = model(images, anchor_images)

		loss = criterion(images, reconstructed_images, predictions, labels, flows)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		running_loss += loss.item()

		groundtruth.append(labels.tolist())
		predictions = torch.sigmoid(predictions)
		prediction.append(predictions.tolist())

		if opt.verbose and i > 0 and int(len(train_loader) / 10) > 0 and i % (int(len(train_loader) / 10)) == 0:
			print('.', flush=True, end='')

	train_loss = running_loss / len(train_loader)

	train_acc, train_f1 = get_eval_metrics(groundtruth, prediction, threshold_list, opt)

	return train_loss, train_acc, train_f1

if __name__ == '__main__':
	main()
