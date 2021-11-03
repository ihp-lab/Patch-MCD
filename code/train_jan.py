"""
Trains and validates models
"""

import os
import csv
import loss
import copy
import torch
import models
import warnings
import argparse
import numpy as np

from tools.utils import set_seed
from datasets import get_dataloader
from tools.lr_scheduler import StepwiseLR
from tools.eval import get_eval_metrics

from adaptation.jan import JointMultipleKernelMaximumMeanDiscrepancy, Theta
from modules.kernels import GaussianKernel

from test_models import test

warnings.filterwarnings('always')

def main():
	parser = argparse.ArgumentParser()

	# Seed
	parser.add_argument('--seed', type=int, default=0)

	# Names, paths, logs
	parser.add_argument('--logger_path', default='checkpoints/jan', help='relative path to log')
	parser.add_argument('--source_domain', default='BP4D')
	parser.add_argument('--target_domain', default='GFT')
	parser.add_argument('--verbose', type=bool, default=False, help='True or False')
	parser.add_argument('--save_checkpoint', type=bool, default=False, help='True or False')
	parser.add_argument('--save_model', type=bool, default=False, help='True or False')

	# Data parameters
	parser.add_argument('--workers_num', type=int, default=4, help='number of workers for data loading')
	parser.add_argument('--folder', default='')

	# Training and optimization
	parser.add_argument('--epochs_num', type=int, default=10, help='number of training epochs')
	parser.add_argument('--batch_size', type=int, default=64, help='size of a mini-batch')
	parser.add_argument('--learning_rate', type=float, default=1e-4, help='1e-3,3e-4,1e-4,3e-5,1e-5')

	parser.add_argument('--non-linear', default=False, action='store_true',
						help='whether not use the linear version')
	parser.add_argument('--quadratic-program', default=False, action='store_true',
						help='whether use quadratic program to solve beta')

	# Model parameters
	parser.add_argument('--threshold', type=float, default=0.5)
	parser.add_argument('--num_class', type=int, default=5)
	parser.add_argument('--dropout_rate', type=float, default=0.1, help='0.1')

	parser.add_argument('--pixel_weight', type=float, default=30, help='30')
	parser.add_argument('--smooth_weight', type=float, default=0.16, help='0.16')
	parser.add_argument('--dissim_weight', type=float, default=1, help='1')
	parser.add_argument('--au_weight', type=float, default=30, help='30')

	parser.add_argument('--features_dim', type=int, default=512*4*4)
	parser.add_argument('--domain_weight', type=float, default=1, help='0.3,1,3')

	parser.add_argument('--linear', default=False, action='store_true',
						help='whether use the linear version')
	parser.add_argument('--adversarial', default=False, action='store_true',
						help='whether use adversarial theta')

	# GPU
	parser.add_argument('--gpu_num', default='cuda:0', help='GPU device')

	opt = parser.parse_args()

	if opt.verbose:
		print('Training and validating models')
		for arg in vars(opt):
			print(arg + ' = ' + str(getattr(opt, arg)))

	set_seed(opt.seed)

	half_batch = opt.batch_size // 2
	opt.batch_size = half_batch

	if opt.target_domain == 'DISFA':
		aus = [1, 2, 4, 6, 12]
	elif opt.target_domain == 'GFT':
		aus = [1, 2, 4, 6, 10, 12, 14, 15, 23, 24]
	else:
		aus = [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24]

	opt.num_class = len(aus)

	mean_val_f1, test_f1 = domain_adaptation(opt)

	print(opt.source_domain, ' to ', opt.target_domain)
	for i in range(0, opt.num_class):
		print(	'au{}'.format(aus[i]),
				'f1: {0:.3f}'.format(test_f1[i]))
	
	print('val', mean_val_f1)
	print('test', np.mean(test_f1))

def domain_adaptation(opt):
	# Use specific GPU
	device = torch.device(opt.gpu_num)

	# Dataloaders
	val_dataset_file_path = os.path.join('../dataset_all_labels', opt.target_domain, 'val.csv')
	val_loader = get_dataloader(val_dataset_file_path, 'val', opt)

	test_dataset_file_path = os.path.join('../dataset_all_labels', opt.target_domain, 'test.csv')
	test_loader = get_dataloader(test_dataset_file_path, 'test', opt)

	# Model, optimizer and loss function
	checkpoint = torch.load(os.path.join(	'checkpoints/bl', opt.source_domain, '1', 'model.pth.tar'),
											map_location=device)

	model = models.AU_Detection(opt)
	model.load_state_dict(checkpoint['model'])
	for param in model.parameters():
		param.requires_grad = True
	model.to(device)

	if opt.adversarial:
		thetas = [Theta(dim).to(device) for dim in (opt.features_dim, opt.num_class)]
	else:
		thetas = None

	domain_criterion = JointMultipleKernelMaximumMeanDiscrepancy(
		kernels=(
			[GaussianKernel(alpha=2 ** k) for k in range(-3, 2)],
			(GaussianKernel(sigma=0.92, track_running_stats=False),)
		),
		linear=opt.linear, thetas=thetas
	).to(device)

	parameters = model.get_parameters()
	if thetas is not None:
		parameters += [{"params": theta.parameters(), 'lr_mult': 0.1} for theta in thetas]

	optimizer = torch.optim.Adam(parameters, lr=opt.learning_rate)
	lr_scheduler = StepwiseLR(optimizer, init_lr=opt.learning_rate, gamma=0.0003, decay_rate=0.75)

	threshold_list = [opt.threshold] * opt.num_class

	main_criterion = loss.Total_Loss(opt)

	best_eval_metrics = 0.
	best_model = model

	# Train and validate
	for epoch in range(opt.epochs_num):
		if opt.verbose:
			print('epoch: {}/{}'.format(epoch + 1, opt.epochs_num))

		batch_iterator, n_batches = get_batch_iterator(opt)

		domain_loss, train_loss, train_acc, train_f1 = train(	batch_iterator, n_batches,
																model, optimizer,
																lr_scheduler,
																main_criterion,
																domain_criterion, device,
																threshold_list, opt)
		mean_train_acc = sum(train_acc)/len(train_acc)
		mean_train_f1 = sum(train_f1)/len(train_f1)
		val_loss, val_acc, val_f1 = test(val_loader, model, main_criterion, device, threshold_list, opt)
		mean_val_acc = sum(val_acc)/len(val_acc)
		mean_val_f1 = sum(val_f1)/len(val_f1)

		if opt.verbose:
			print(	'\ndomain_loss: {0:.5f}'.format(domain_loss),
					'\ntrain_loss: {0:.5f}'.format(train_loss),
					'mean_train_acc: {0:.3f}'.format(mean_train_acc),
					'mean_train_f1: {0:.3f}'.format(mean_train_f1),
					'\nval_loss: {0:.5f}'.format(val_loss),
					'mean_val_acc: {0:.3f}'.format(mean_val_acc),
					'mean_val_f1: {0:.3f}'.format(mean_val_f1))

		state = {	'epoch': epoch+1,
					'model': model.state_dict(),
					'opt': opt}
		os.makedirs(os.path.join(opt.logger_path, opt.source_domain, opt.target_domain), exist_ok=True)

		if opt.save_checkpoint:
			model_file_name = os.path.join(opt.logger_path, opt.source_domain, opt.target_domain, 'checkpoint.pth.tar')
			torch.save(state, model_file_name)

		if mean_val_f1 > best_eval_metrics:
			best_eval_metrics = mean_val_f1
			best_model = copy.deepcopy(model)

			if opt.save_model:
				model_file_name = os.path.join(opt.logger_path, opt.source_domain, opt.target_domain, 'model.pth.tar')
				torch.save(state, model_file_name)

	test_loss, test_acc, test_f1 = test(test_loader, best_model, main_criterion, device, threshold_list, opt)

	return best_eval_metrics, test_f1

def get_batch_iterator(opt):
	source_dataset_file_path = os.path.join('../dataset_all_labels', opt.source_domain, 'all.csv')
	source_loader = get_dataloader(source_dataset_file_path, 'train', opt)

	target_dataset_file_path = os.path.join('../dataset_all_labels', opt.target_domain, 'train.csv')
	target_loader = get_dataloader(target_dataset_file_path, 'train', opt)

	batches = zip(source_loader, target_loader)
	n_batches = min(len(source_loader), len(target_loader))

	return batches, n_batches - 1

def train(batches, n_batches, model, optimizer, lr_scheduler, main_criterion, domain_criterion, device, threshold_list, opt):
	model.train()
	domain_criterion.train()

	total_domain_loss = 0.
	total_label_loss = 0.

	groundtruth = []
	prediction = []

	for i, train_data in enumerate(batches):
		lr_scheduler.step()

		(source_images, source_anchor_images, _, source_labels), \
		(target_images, target_anchor_images, _, _) = train_data

		source_images = source_images.to(device)
		source_anchor_images = source_anchor_images.to(device)
		source_labels = source_labels[:,:opt.num_class].to(device)

		target_images = target_images.to(device)
		target_anchor_images = target_anchor_images.to(device)

		source_flow, source_reconstructed_images = model.OF_Net(source_images, source_anchor_images)
		target_flow, _ = model.OF_Net(target_images, target_anchor_images)

		y_s = model.AU_Net(source_flow)
		y_t = model.AU_Net(target_flow)

		f_s = model.AU_Net.forward_1(source_flow)
		f_t = model.AU_Net.forward_1(target_flow)

		domain_loss = domain_criterion(	(f_s, torch.sigmoid(y_s)),
										(f_t, torch.sigmoid(y_t)))

		label_loss = main_criterion(source_images, source_reconstructed_images,
									y_s, source_labels, source_flow)

		loss = opt.au_weight * opt.domain_weight * domain_loss + label_loss

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		total_domain_loss += domain_loss.item()
		total_label_loss += label_loss.item()

		groundtruth.append(source_labels.tolist())
		y_s = torch.sigmoid(y_s)
		prediction.append(y_s.tolist())

		if opt.verbose and i > 0 and i % int(n_batches / 10) == 0:
			print('.', flush=True, end='')

		if i >= n_batches - 1:
			break

	label_acc, label_f1 = get_eval_metrics(groundtruth, prediction, threshold_list, opt)

	domain_loss = total_domain_loss / n_batches
	label_loss = total_label_loss / n_batches

	return domain_loss, label_loss, label_acc, label_f1

if __name__ == '__main__':
	main()
