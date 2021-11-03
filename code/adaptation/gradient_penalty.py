import torch
from torch.autograd import grad

def gradient_penalty(critic, h_s, h_t, device):
	# based on: https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py#L116
	alpha = torch.rand(h_s.size(0), 1).to(device)
	differences = h_t - h_s
	interpolates = h_s + (alpha * differences)
	interpolates = torch.stack([interpolates, h_s, h_t]).requires_grad_()

	preds = critic(interpolates)
	gradients = grad(preds, interpolates,
						grad_outputs=torch.ones_like(preds),
						retain_graph=True, create_graph=True)[0]
	gradient_norm = gradients.norm(2, dim=1)
	gradient_penalty = ((gradient_norm - 1)**2).mean()
	return gradient_penalty
