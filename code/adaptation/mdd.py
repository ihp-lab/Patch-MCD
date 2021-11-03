from typing import Optional, List, Dict
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from tools.eval import get_labels 
from models import AU_Net, init_weights
from modules.grl import WarmStartGradientReverseLayer

class MarginDisparityDiscrepancy(nn.Module):
	r"""The margin disparity discrepancy (MDD) is proposed to measure the distribution discrepancy in domain adaptation.

	The :math:`y^s` and :math:`y^t` are logits output by the main classifier on the source and target domain respectively.
	The :math:`y_{adv}^s` and :math:`y_{adv}^t` are logits output by the adversarial classifier.
	They are expected to contain raw, unnormalized scores for each class.

	The definition can be described as:

	.. math::
		\mathcal{D}_{\gamma}(\hat{\mathcal{S}}, \hat{\mathcal{T}}) =
		\gamma \mathbb{E}_{y^s, y_{adv}^s \sim\hat{\mathcal{S}}} \log\left(\frac{\exp(y_{adv}^s[h_{y^s}])}{\sum_j \exp(y_{adv}^s[j])}\right) +
		\mathbb{E}_{y^t, y_{adv}^t \sim\hat{\mathcal{T}}} \log\left(1-\frac{\exp(y_{adv}^t[h_{y^t}])}{\sum_j \exp(y_{adv}^t[j])}\right),

	where :math:`\gamma` is a margin hyper-parameter and :math:`h_y` refers to the predicted label when the logits output is :math:`y`.
	You can see more details in `Bridging Theory and Algorithm for Domain Adaptation <https://arxiv.org/abs/1904.05801>`_.

	Parameters:
		- **margin** (float): margin :math:`\gamma`. Default: 4
		- **reduction** (string, optional): Specifies the reduction to apply to the output:
		  ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
		  ``'mean'``: the sum of the output will be divided by the number of
		  elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``

	Inputs: y_s, y_s_adv, y_t, y_t_adv
		- **y_s**: logits output :math:`y^s` by the main classifier on the source domain
		- **y_s_adv**: logits output :math:`y^s` by the adversarial classifier on the source domain
		- **y_t**: logits output :math:`y^t` by the main classifier on the target domain
		- **y_t_adv**: logits output :math:`y_{adv}^t` by the adversarial classifier on the target domain

	Shape:
		- Inputs: :math:`(minibatch, C)` where C = number of classes, or :math:`(minibatch, C, d_1, d_2, ..., d_K)`
		  with :math:`K \geq 1` in the case of `K`-dimensional loss.
		- Output: scalar. If :attr:`reduction` is ``'none'``, then the same size as the target: :math:`(minibatch)`, or
		  :math:`(minibatch, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case of K-dimensional loss.

	Examples::
		>>> num_classes = 2
		>>> batch_size = 10
		>>> loss = MarginDisparityDiscrepancy(margin=4.)
		>>> # logits output from source domain and target domain
		>>> y_s, y_t = torch.randn(batch_size, num_classes), torch.randn(batch_size, num_classes)
		>>> # adversarial logits output from source domain and target domain
		>>> y_s_adv, y_t_adv = torch.randn(batch_size, num_classes), torch.randn(batch_size, num_classes)
		>>> output = loss(y_s, y_s_adv, y_t, y_t_adv)
	"""

	def __init__(self, num_class, threshold_list, device, margin: Optional[int] = 4):
		super(MarginDisparityDiscrepancy, self).__init__()
		self.margin = margin
		self.num_class = num_class
		self.threshold_list = threshold_list
		self.device = device
		self.loss = nn.BCEWithLogitsLoss()

	def forward(self, y_s: torch.Tensor, y_s_adv: torch.Tensor, y_t: torch.Tensor, y_t_adv: torch.Tensor) -> torch.Tensor:
		prediction_s = np.array(torch.sigmoid(y_s).tolist())
		prediction_s = get_labels(prediction_s, self.threshold_list)
		prediction_s = torch.FloatTensor(prediction_s).to(self.device)

		prediction_t = np.array(torch.sigmoid(y_t).tolist())
		prediction_t = get_labels(prediction_t, self.threshold_list)
		prediction_t = torch.LongTensor(prediction_t).to(self.device)

		result = self.margin * self.loss(y_s_adv, prediction_s)
		prop_1 = torch.sigmoid(y_t)
		prop_0 = 1. - prop_1

		prop_0 = prop_0.view(prop_0.shape[0], prop_0.shape[1], -1)
		prop_1 = prop_1.view(prop_1.shape[0], prop_1.shape[1], -1)
		prop = torch.cat([prop_0, prop_1], dim=2)

		for i in range(self.num_class):
			result += F.nll_loss(shift_log(1.-prop[:,i,:]), prediction_t[:,i])

		return result


def shift_log(x: torch.Tensor, offset: Optional[float] = 1e-6) -> torch.Tensor:
	r"""
	First shift, then calculate log, which can be described as:

	.. math::
		y = \max(\log(x+\text{offset}), 0)

	Used to avoid the gradient explosion problem in log(x) function when x=0.

	Parameters:
		- **x**: input tensor
		- **offset**: offset size. Default: 1e-6

	.. note::
		Input tensor falls in [0., 1.] and the output tensor falls in [-log(offset), 0]
	"""
	return torch.log(torch.clamp(x + offset, max=1.))


class AU_Classifier(nn.Module):
	r"""Classifier for MDD.
	Parameters:
		- **backbone** (class:`nn.Module` object): Any backbone to extract 1-d features from data
		- **num_classes** (int): Number of classes
		- **bottleneck_dim** (int, optional): Feature dimension of the bottleneck layer. Default: 1024
		- **width** (int, optional): Feature dimension of the classifier head. Default: 1024

	.. note::
		Classifier for MDD has one backbone, one bottleneck, while two classifier heads.
		The first classifier head is used for final predictions.
		The adversarial classifier head is only used when calculating MarginDisparityDiscrepancy.

	.. note::
		Remember to call function `step()` after function `forward()` **during training phase**! For instance,

		>>> # x is inputs, classifier is an ImageClassifier
		>>> outputs, outputs_adv = classifier(x)
		>>> classifier.step()

	Inputs:
		- **x** (Tensor): input data

	Outputs: (outputs, outputs_adv)
		- **outputs**: logits outputs by the main classifier
		- **outputs_adv**: logits outputs by the adversarial classifier

	Shapes:
		- x: :math:`(minibatch, *)`, same shape as the input of the `backbone`.
		- outputs, outputs_adv: :math:`(minibatch, C)`, where C means the number of classes.

	"""

	def __init__(self, opt: int):
		super(AU_Classifier, self).__init__()
		self.opt = opt
		self.grl_layer = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000., auto_step=False)

		self.AU_Net = AU_Net(self.opt)
		self.adv_AU_Net = AU_Net(self.opt)

		self.AU_Net.apply(init_weights)
		self.adv_AU_Net.apply(init_weights)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		outputs = self.AU_Net(x)
		x_adv = self.grl_layer(x)
		outputs_adv = self.adv_AU_Net(x_adv)
		return outputs, outputs_adv

	def step(self):
		"""Call step() each iteration during training.
		Will increase :math:`\lambda` in GRL layer.
		"""
		self.grl_layer.step()

	def get_parameters(self) -> List[Dict]:
		"""
		:return: A parameters list which decides optimization hyper-parameters,
			such as the relative learning rate of each layer
		"""
		params = [
			{"params": self.AU_Net.parameters(), "lr_mult": 1.},
			{"params": self.adv_AU_Net.parameters(), "lr_mult": 1.}
		]
		return params
