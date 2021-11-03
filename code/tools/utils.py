import torch
import random
import numpy as np
import torch.nn.functional as F

from math import exp
from torch.autograd import Variable

def set_seed(seed):
	# Reproducibility
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

	random.seed(seed)
	np.random.seed(seed)

def set_requires_grad(model, requires_grad=True):
	for param in model.parameters():
		param.requires_grad = requires_grad

def flow_warp(x, flow, padding_mode='zeros'):
	"""Warp an image or feature map with optical flow
	Args:
		x (Tensor): size (n, c, h, w)
		flow (Tensor): size (n, 2, h, w), values range from -1 to 1 (relevant to image width or height)
		padding_mode (str): 'zeros' or 'border'

	Returns:
		Tensor: warped image or feature map
	"""
	assert x.size()[-2:] == flow.size()[-2:]
	n, _, h, w = x.size()
	x_ = torch.arange(w).view(1, -1).expand(h, -1)
	y_ = torch.arange(h).view(-1, 1).expand(-1, w)
	grid = torch.stack([x_, y_], dim=0).float().cuda(flow.get_device())
	grid = grid.unsqueeze(0).expand(n, -1, -1, -1)
	grid[:, 0, :, :] = 2 * grid[:, 0, :, :] / (w - 1) - 1
	grid[:, 1, :, :] = 2 * grid[:, 1, :, :] / (h - 1) - 1
	grid = grid + 2 * flow
	grid = grid.permute(0, 2, 3, 1)
	return F.grid_sample(x, grid, padding_mode=padding_mode)

def gaussian(window_size, sigma):
	gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
	return gauss/gauss.sum()

def create_window(window_size, channel):
	_1D_window = gaussian(window_size, 1.5).unsqueeze(1)
	_2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
	window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
	return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
	mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
	mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

	mu1_sq = mu1.pow(2)
	mu2_sq = mu2.pow(2)
	mu1_mu2 = mu1*mu2

	sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
	sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
	sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

	C1 = 0.01**2
	C2 = 0.03**2

	ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

	if size_average:
		return ssim_map.mean()
	else:
		return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
	def __init__(self, window_size = 11, size_average = True):
		super(SSIM, self).__init__()
		self.window_size = window_size
		self.size_average = size_average
		self.channel = 3
		self.window = create_window(window_size, self.channel)

	def forward(self, img1, img2):
		(_, channel, _, _) = img1.size()

		if channel == self.channel and self.window.data.type() == img1.data.type():
			window = self.window
		else:
			window = create_window(self.window_size, channel)
			
			if img1.is_cuda:
				window = window.cuda(img1.get_device())
			window = window.type_as(img1)
			
			self.window = window
			self.channel = channel

		return _ssim(img1, img2, window, self.window_size, channel, self.size_average)
