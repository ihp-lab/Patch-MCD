"""
Defines models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.utils import flow_warp
from torch.autograd import Function
from typing import Tuple, Optional, List, Dict

def init_weights(m):
	if type(m) == nn.Linear or type(m) == nn.Conv2d:
		torch.nn.init.xavier_uniform_(m.weight)
		if m.bias is not None:
			m.bias.data.fill_(0.01)

class Flatten(nn.Module):
	def forward(self, input):
		
		return input.view(input.size(0), -1)

class AU_Detection(nn.Module):
	def __init__(self, opt):
		super(AU_Detection, self).__init__()

		self.OF_Net = OF_Net(opt)
		self.AU_Net = AU_Net(opt)

	def forward(self, image, anchor_image):
		flow, reconstructed_image = self.OF_Net(image, anchor_image)
		au_label = self.AU_Net(flow)

		return au_label, reconstructed_image, flow

	def get_parameters(self) -> List[Dict]:
		"""A parameter list which decides optimization hyper-parameters,
			such as the relative learning rate of each layer
		"""
		params = [
			{"params": self.OF_Net.parameters(), "lr_mult": 1.},
			{"params": self.AU_Net.parameters(), "lr_mult": 1.}
		]
		return params

class OF_Net(nn.Module):
	def __init__(self, opt):
		super(OF_Net, self).__init__()

		self.dropout_rate = opt.dropout_rate
		self.dropout = nn.Dropout(self.dropout_rate)

		self.layer1 = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(64),
			self.dropout,
			nn.ReLU())
		self.layer2 = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(128),
			self.dropout,
			nn.ReLU())
		self.layer3 = nn.Sequential(
			nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(256),
			self.dropout,
			nn.ReLU())
		self.layer4 = nn.Sequential(
			nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(512),
			self.dropout,
			nn.ReLU())
		self.layer5 = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(512),
			self.dropout,
			nn.ReLU())
		self.layer6 = nn.Sequential(
			nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
			nn.BatchNorm2d(512),
			self.dropout,
			nn.ReLU())
		self.layer7 = nn.Sequential(
			nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
			nn.BatchNorm2d(256),
			self.dropout,
			nn.ReLU())
		self.layer8 = nn.Sequential(
			nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
			nn.BatchNorm2d(128),
			self.dropout,
			nn.ReLU())
		self.layer9 = nn.Sequential(
			nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
			nn.BatchNorm2d(64),
			self.dropout,
			nn.ReLU())
		self.layer10 = nn.Sequential(
			nn.ConvTranspose2d(64, 2, kernel_size=3, stride=1, padding=1),
			nn.Tanh())

	def forward(self, image, anchor_image):
		x = self.layer1(image)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.layer5(x)
		x = self.layer6(x)
		x = self.layer7(x)
		x = self.layer8(x)
		x = self.layer9(x)
		flow = self.layer10(x)

		reconstructed_image = flow_warp(anchor_image, flow)

		return flow, reconstructed_image

	def get_parameters(self) -> List[Dict]:
		"""A parameter list which decides optimization hyper-parameters,
			such as the relative learning rate of each layer
		"""
		params = [
			{"params": self.parameters(), "lr_mult": 1.}
		]
		return params

class AU_Net(nn.Module):
	def __init__(self, opt):
		super(AU_Net, self).__init__()

		self.dropout_rate = opt.dropout_rate
		self.num_class = opt.num_class
		self.flatten = Flatten()
		self.dropout = nn.Dropout(self.dropout_rate)
		self.relu = nn.ReLU()

		self.layer1 = nn.Sequential(
			nn.Conv2d(2, 64, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(64),
			self.dropout,
			nn.ReLU())
		self.layer2 = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(128),
			self.dropout,
			nn.ReLU())
		self.layer3 = nn.Sequential(
			nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(256),
			self.dropout,
			nn.ReLU())
		self.layer4 = nn.Sequential(
			nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(512),
			self.dropout,
			nn.ReLU())
		self.layer5 = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(512),
			self.dropout,
			nn.ReLU())
		self.layer6 = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(512),
			self.dropout,
			nn.ReLU())

		self.fc1 = nn.Linear(512*4*4, 1024)
		self.fc2 = nn.Linear(1024, self.num_class)

	def forward_1(self, x):
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.layer5(x)
		x = self.flatten(x)

		return x

	def forward_2(self, x):
		x = self.relu(self.fc1(x))
		x = self.fc2(x)

		return x

	def forward(self, x):
		x = self.forward_1(x)
		x = self.forward_2(x)

		return x

	def get_parameters(self) -> List[Dict]:
		"""A parameter list which decides optimization hyper-parameters,
			such as the relative learning rate of each layer
		"""
		params = [
			{"params": self.parameters(), "lr_mult": 1.}
		]
		return params

class PatchNet(nn.Module):
	def __init__(self, opt):
		super(PatchNet, self).__init__()

		self.dropout_rate = opt.dropout_rate
		self.num_class = 16
		self.flatten = Flatten()
		self.dropout = nn.Dropout(self.dropout_rate)
		self.relu = nn.ReLU()

		self.layer1 = nn.Sequential(
			nn.Conv2d(2, 64, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(64),
			self.dropout,
			nn.ReLU())
		self.layer2 = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(128),
			self.dropout,
			nn.ReLU())
		self.layer3 = nn.Sequential(
			nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(256),
			self.dropout,
			nn.ReLU())
		self.layer4 = nn.Sequential(
			nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(512),
			self.dropout,
			nn.ReLU())
		self.layer5 = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(512),
			self.dropout,
			nn.ReLU())
		self.layer6 = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(512),
			self.dropout,
			nn.ReLU())

		self.fc1 = nn.Linear(512, 128)
		self.fc2 = nn.Linear(128, self.num_class)

	def forward_1(self, x):
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.layer5(x)
		x = self.flatten(x)

		return x

	def forward_2(self, x):
		x = self.relu(self.fc1(x))
		x = self.fc2(x)

		return x

	def forward(self, x):
		x = self.forward_1(x)
		x = self.forward_2(x)

		return x

	def get_parameters(self) -> List[Dict]:
		"""A parameter list which decides optimization hyper-parameters,
			such as the relative learning rate of each layer
		"""
		params = [
			{"params": self.parameters(), "lr_mult": 1.}
		]
		return params