import os
import csv
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tools.utils import SSIM

def charbonnier(x, alpha=0.25, epsilon=1e-9):
	return torch.pow(torch.pow(x, 2) + epsilon**2, alpha)

class Total_Loss(nn.Module):
	def __init__(self, opt):
		super(Total_Loss, self).__init__()

		self.opt = opt
		self.pixel = Pixel_Loss(opt)
		self.smooth = Smooth_Loss(opt)
		self.dissim = Dissim_Loss(opt)
		self.au = AU_Loss(opt)

	def forward(self, images, reconstructed_images, predictions, labels, flows):
		pixel_loss = self.opt.pixel_weight * self.pixel(images, reconstructed_images)
		smooth_loss = self.opt.smooth_weight * self.smooth(flows)
		dissim_loss = self.opt.dissim_weight * self.dissim(images, reconstructed_images)
		au_loss = self.opt.au_weight * self.au(predictions, labels)

		# print(pixel_loss.item(), smooth_loss.item(), dissim_loss.item(), au_loss.item())

		return pixel_loss + smooth_loss + dissim_loss + au_loss

class Pixel_Loss(nn.Module):
	def __init__(self, opt):
		super(Pixel_Loss, self).__init__()

	def forward(self, images, reconstructed_images):
		h, w = reconstructed_images.shape[2:]
		images = F.interpolate(images, (h, w), mode='bilinear', align_corners=False)
		error = charbonnier(reconstructed_images - images)
		error = torch.sum(error, dim=1) / 3
		loss = torch.mean(error)

		return loss

class Smooth_Loss(nn.Module):
	def __init__(self, opt):
		super(Smooth_Loss, self).__init__()

	def forward(self, flows):
		b, c, h, w = flows.shape
		v_translated = torch.cat((flows[:, :, 1:, :], torch.zeros(b, c, 1, w, device=flows.device)), dim=-2)
		h_translated = torch.cat((flows[:, :, :, 1:], torch.zeros(b, c, h, 1, device=flows.device)), dim=-1)
		error = charbonnier(flows - v_translated) + charbonnier(flows - h_translated)
		error = torch.sum(error, dim=1) / 2
		loss = torch.mean(error)

		return loss

class Dissim_Loss(nn.Module):
	def __init__(self, opt):
		super(Dissim_Loss, self).__init__()

		self.device = torch.device(opt.gpu_num)
		self.ssim = SSIM()

	def forward(self, images_1, images_2):
		loss = -self.ssim(images_1, images_2)

		return loss

class AU_Loss(nn.Module):
	def __init__(self, opt):
		super(AU_Loss, self).__init__()

		self.device = torch.device(opt.gpu_num)
		# self.weight = self.get_loss_weight(opt)
		self.au = nn.BCEWithLogitsLoss()
		# self.au = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([self.weight]).to(self.device))

	def get_loss_weight(self, opt):
		dataset_file_path = os.path.join('../dataset', opt.source_domain, str(opt.folder), 'train.csv')

		weight_list = []
		groundtruth = []
		with open(dataset_file_path, newline='') as csvfile:
			reader = csv.reader(csvfile, delimiter=',', quotechar='|')
			title = next(reader)
			for row in reader:
				groundtruth.append([int(i) for i in row[-7:]])

		groundtruth = np.array(groundtruth)
		for i in range(0, opt.num_class):
			weight = float(list(groundtruth[:,i]).count(0)) / list(groundtruth[:,i]).count(1)
			weight_list.append(weight)

		return weight_list

	def forward(self, predictions, labels):
		loss = self.au(predictions, labels)

		return loss
