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
import datasets
import argparse
import numpy as np

from tools.utils import set_seed
from datasets import get_dataloader
from tools.eval import get_eval_metrics
from tools.transformations import crop_patches

from adaptation.mcd import entropy, classifier_discrepancy

warnings.filterwarnings('always')

def main():
	parser = argparse.ArgumentParser()

	# Seed
	parser.add_argument('--seed', type=int, default=0)

	# Names, paths, logs
	parser.add_argument('--logger_path', default='checkpoints/patch_mcd', help='relative path to log')
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
	parser.add_argument('--learning_rate', type=float, default=1e-4, help='1e-3,3e-4,1e-4,3e-5')

	# Model parameters
	parser.add_argument('--threshold', type=float, default=0.5)
	parser.add_argument('--num_class', type=int, default=5)
	parser.add_argument('--dropout_rate', type=float, default=0.1, help='0.1')

	parser.add_argument('--pixel_weight', type=float, default=30, help='30')
	parser.add_argument('--smooth_weight', type=float, default=0.16, help='0.16')
	parser.add_argument('--dissim_weight', type=float, default=1, help='1')
	parser.add_argument('--au_weight', type=float, default=30, help='30')

	parser.add_argument('--domain_weight', type=float, default=1, help='0.3,1,3')
	parser.add_argument('--num-k', type=int, default=4, help='how many steps to repeat the generator update')
	
	parser.add_argument('--patch_weight', type=float, default=1, help='0.3,1,3')
	parser.add_argument('--sigma', type=float, default=0.03, help='0.03')

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

	patch_net = models.PatchNet(opt)
	patch_net.apply(models.init_weights)
	for param in patch_net.parameters():
		param.requires_grad = True
	patch_net.to(device)

	G = model.OF_Net

	F1 = models.AU_Net(opt).to(device)
	F2 = models.AU_Net(opt).to(device)

	models.init_weights(F1)
	for param in F1.parameters():
		param.requires_grad = True
	model.to(device)

	models.init_weights(F2)
	for param in F2.parameters():
		param.requires_grad = True
	model.to(device)

	optimizer_g = torch.optim.Adam(G.get_parameters()+patch_net.get_parameters(), lr=opt.learning_rate)
	optimizer_f = torch.optim.Adam(F1.get_parameters()+F2.get_parameters(), lr=opt.learning_rate)

	threshold_list = [opt.threshold] * opt.num_class

	main_criterion = loss.Total_Loss(opt)
	patch_criterion = torch.nn.CrossEntropyLoss()

	best_eval_metrics = 0.

	best_G = G
	best_F1 = F1
	best_F2 = F2

	# Train and validate
	for epoch in range(opt.epochs_num):
		if opt.verbose:
			print('epoch: {}/{}'.format(epoch + 1, opt.epochs_num))

		batch_iterator, n_batches = get_batch_iterator(opt)

		train(	batch_iterator, n_batches, G, patch_net, F1, F2, optimizer_g, optimizer_f,
				main_criterion, patch_criterion, device, threshold_list, opt)
		val_loss_1, val_acc_1, val_f1_1, val_loss_2, val_acc_2, val_f1_2 \
			= test(val_loader, G, F1, F2, main_criterion, device, threshold_list, opt)

		mean_val_acc_1 = sum(val_acc_1)/len(val_acc_1)
		mean_val_f1_1 = sum(val_f1_1)/len(val_f1_1)

		mean_val_acc_2 = sum(val_acc_2)/len(val_acc_2)
		mean_val_f1_2 = sum(val_f1_2)/len(val_f1_2)

		if opt.verbose:
			print(	'\nval_loss_1: {0:.5f}'.format(val_loss_1),
					'mean_val_acc_1: {0:.3f}'.format(mean_val_acc_1),
					'mean_val_f1_1: {0:.3f}'.format(mean_val_f1_1))

			print(	'\nval_loss_2: {0:.5f}'.format(val_loss_2),
					'mean_val_acc_2: {0:.3f}'.format(mean_val_acc_2),
					'mean_val_f1_2: {0:.3f}'.format(mean_val_f1_2))

		state = {	'epoch': epoch+1,
					'G': G.state_dict(),
					'F1': F1.state_dict(),
					'F2': F2.state_dict(),
					'opt': opt}
		os.makedirs(os.path.join(opt.logger_path, opt.source_domain, opt.target_domain), exist_ok=True)

		if opt.save_checkpoint:
			model_file_name = os.path.join(opt.logger_path, opt.source_domain, opt.target_domain, 'checkpoint.pth.tar')
			torch.save(state, model_file_name)

		if max(mean_val_f1_1, mean_val_f1_2) > best_eval_metrics:
			best_eval_metrics = max(mean_val_f1_1, mean_val_f1_2)
			best_G = copy.deepcopy(G)
			best_F1 = copy.deepcopy(F1)
			best_F2 = copy.deepcopy(F2)

			if opt.save_model:
				model_file_name = os.path.join(opt.logger_path, opt.source_domain, opt.target_domain, 'model.pth.tar')
				torch.save(state, model_file_name)

	_, _, test_f1_1, _, _, test_f1_2 = test(test_loader, best_G, best_F1, best_F2, main_criterion,
											device, threshold_list, opt)
	
	mean_test_f1_1 = sum(test_f1_1)/len(test_f1_1)
	mean_test_f1_2 = sum(test_f1_2)/len(test_f1_2)

	if mean_test_f1_1 > mean_test_f1_2:
		test_f1 = test_f1_1
	else:
		test_f1 = test_f1_2

	return best_eval_metrics, test_f1

def get_batch_iterator(opt):
	source_dataset_file_path = os.path.join('../dataset_all_labels', opt.source_domain, 'all.csv')
	source_loader = get_dataloader(source_dataset_file_path, 'train', opt)

	target_dataset_file_path = os.path.join('../dataset_all_labels', opt.target_domain, 'train.csv')
	target_loader = get_dataloader(target_dataset_file_path, 'train', opt)

	batches = zip(source_loader, target_loader)
	n_batches = min(len(source_loader), len(target_loader))

	return batches, n_batches - 1

def train(	batches, n_batches, G, patch_net, F1, F2, optimizer_g, optimizer_f,
			criterion, patch_criterion, device, threshold_list, opt):
	G.train()
	patch_net.train()
	F1.train()
	F2.train()

	for i, train_data in enumerate(batches):
		(x_s, source_anchor_images, _, labels_s), \
		(x_t, target_anchor_images, _, _) = train_data

		x_s = x_s.to(device)
		source_anchor_images = source_anchor_images.to(device)
		labels_s = labels_s.to(device)

		x_t = x_t.to(device)
		target_anchor_images = target_anchor_images.to(device)

		x = torch.cat((x_s, x_t), dim=0)
		anchor_images = torch.cat((source_anchor_images, target_anchor_images), dim=0)
		assert x.requires_grad is False
		assert anchor_images.requires_grad is False

		# Step A train all networks to minimize loss on source domain
		optimizer_g.zero_grad()
		optimizer_f.zero_grad()

		g, reconstructed_images = G(x, anchor_images)
		g_s, g_t = g.chunk(2, dim=0)
		source_reconstructed_images, target_reconstructed_images = reconstructed_images.chunk(2, dim=0)

		y_1 = F1(g)
		y_2 = F2(g)
		y1_s, y1_t = y_1.chunk(2, dim=0)
		y2_s, y2_t = y_2.chunk(2, dim=0)
		y1_t, y2_t = torch.sigmoid(y1_t), torch.sigmoid(y2_t)

		patches = torch.cat((g_s, g_t), dim=0)
		patch_labels = torch.randint(0, 16, (g_s.shape[0]+g_t.shape[0],)).long().to(device)

		patches = crop_patches(patches, patch_labels, opt.sigma, device).to(device)
		pred_labels = patch_net(patches)

		patch_loss = patch_criterion(pred_labels, patch_labels)

		loss_1 = criterion(x_s, source_reconstructed_images, y1_s, labels_s, g_s)
		loss_2 = criterion(x_s, source_reconstructed_images, y2_s, labels_s, g_s)

		loss = loss_1 + loss_2 + 0.01 * (entropy(y1_t) + entropy(y2_t)) + opt.au_weight * opt.patch_weight * patch_loss

		loss.backward()
		optimizer_g.step()
		optimizer_f.step()

		# Step B train classifier to maximize discrepancy
		optimizer_f.zero_grad()

		g, reconstructed_images = G(x, anchor_images)
		g_s, g_t = g.chunk(2, dim=0)
		source_reconstructed_images, target_reconstructed_images = reconstructed_images.chunk(2, dim=0)

		y_1 = F1(g)
		y_2 = F2(g)
		y1_s, y1_t = y_1.chunk(2, dim=0)
		y2_s, y2_t = y_2.chunk(2, dim=0)
		y1_t, y2_t = torch.sigmoid(y1_t), torch.sigmoid(y2_t)

		loss_1 = criterion(x_s, source_reconstructed_images, y1_s, labels_s, g_s)
		loss_2 = criterion(x_s, source_reconstructed_images, y2_s, labels_s, g_s)

		loss = loss_1 + loss_2 + 0.01 * (entropy(y1_t) + entropy(y2_t)) \
				- classifier_discrepancy(y1_t, y2_t) * opt.domain_weight
		loss.backward()
		optimizer_f.step()

		# Step C train genrator to minimize discrepancy
		for k in range(opt.num_k):
			optimizer_g.zero_grad()

			g, reconstructed_images = G(x, anchor_images)
			g_s, g_t = g.chunk(2, dim=0)
			source_reconstructed_images, target_reconstructed_images = reconstructed_images.chunk(2, dim=0)

			y_1 = F1(g)
			y_2 = F2(g)
			y1_s, y1_t = y_1.chunk(2, dim=0)
			y2_s, y2_t = y_2.chunk(2, dim=0)
			y1_t, y2_t = torch.sigmoid(y1_t), torch.sigmoid(y2_t)

			patches = torch.cat((g_s, g_t), dim=0)
			patch_labels = torch.randint(0, 16, (g_s.shape[0]+g_t.shape[0],)).long().to(device)

			patches = crop_patches(patches, patch_labels, opt.sigma, device).to(device)
			pred_labels = patch_net(patches)

			patch_loss = patch_criterion(pred_labels, patch_labels)

			mcd_loss = classifier_discrepancy(y1_t, y2_t) * opt.domain_weight + opt.au_weight * opt.patch_weight * patch_loss
			mcd_loss.backward()
			optimizer_g.step()

		if opt.verbose and i > 0 and i % int(n_batches / 10) == 0:
			print('.', flush=True, end='')

		if i >= n_batches - 1:
			break

def test(test_loader, G, F1, F2, criterion, device, threshold_list, opt):
	G.eval()
	F1.eval()
	F2.eval()

	running_loss_1 = 0.
	running_loss_2 = 0.

	with torch.no_grad():
		groundtruth = []
		prediction_1 = []
		prediction_2 = []

		for i, test_data in enumerate(test_loader):
			images, anchor_images, _, labels = test_data

			images = images.to(device)
			anchor_images = anchor_images.to(device)
			labels = labels[:,:opt.num_class].to(device)

			flows, reconstructed_images = G(images, anchor_images)

			predictions_1 = F1(flows)
			predictions_2 = F2(flows)

			loss_1 = criterion(images, reconstructed_images, predictions_1, labels, flows)
			running_loss_1 += loss_1.item()

			loss_2 = criterion(images, reconstructed_images, predictions_2, labels, flows)
			running_loss_2 += loss_2.item()

			groundtruth.append(labels.tolist())
			predictions_1 = torch.sigmoid(predictions_1)
			prediction_1.append(predictions_1.tolist())
			predictions_2 = torch.sigmoid(predictions_2)
			prediction_2.append(predictions_2.tolist())

		test_loss_1 = running_loss_1 / len(test_loader)
		test_loss_2 = running_loss_2 / len(test_loader)

		test_acc_1, test_f1_1 = get_eval_metrics(groundtruth, prediction_1, threshold_list, opt)
		test_acc_2, test_f1_2 = get_eval_metrics(groundtruth, prediction_2, threshold_list, opt)

		return test_loss_1, test_acc_1, test_f1_1, test_loss_2, test_acc_2, test_f1_2

if __name__ == '__main__':
	main()
