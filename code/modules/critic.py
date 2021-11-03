import torch.nn as nn
import torch

class Critic(nn.Module):
	def __init__(self, in_feature: int, hidden_size: int):
		super(Critic, self).__init__()
		self.layer1 = nn.Linear(in_feature, hidden_size)
		self.relu1 = nn.ReLU()
		self.layer2 = nn.Linear(hidden_size, hidden_size)
		self.relu2 = nn.ReLU()
		self.layer3 = nn.Linear(hidden_size, 1)

	def forward(self, x):
		x = self.relu1(self.layer1(x))
		x = self.relu2(self.layer2(x))
		y = self.layer3(x)
		return y
