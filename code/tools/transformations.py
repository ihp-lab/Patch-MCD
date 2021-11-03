import cv2
import torch
import random
import numpy as np

from torchvision import transforms

def crop_patches(patches, patch_labels, sigma, device):
	length = patches.shape[0]

	new_patches = torch.zeros((length, 2, 32, 32)).to(device)

	for i in range(length):
		patch = patches[i]
		label = patch_labels[i].item()
		
		new_patch = crop_patch(patch, label)
		new_patch = transform(new_patch, sigma, device)
		new_patches[i] = new_patch

	return new_patches

def transform(patch, sigma, device):
	# gaussian noise
	p = random.uniform(0, 1)
	if p < 0.5:
		patch = patch + sigma * torch.randn(2, 32, 32).to(device)
	# cutout
	p = random.uniform(0, 1)
	if p < 0.5:
		h = random.randint(0, 16)
		w = random.randint(0, 16)
		patch[:,h:h+16,w:w+16] = torch.zeros((2, 16, 16)).to(device)
	# flip
	p = random.uniform(0, 1)
	if p < 0.5:
		patch = torch.flip(patch, [2])
	# rotate
	p = random.uniform(0, 1)
	if p < 0.25:
		patch = torch.rot90(patch, 1, [1, 2])
	elif p < 0.5:
		patch = torch.rot90(patch, 2, [1, 2])
	elif p < 0.75:
		patch = torch.rot90(patch, 3, [1, 2])
	return patch

def crop_patch(patch, label):
	if label == 0:
		new_patch = patch[:,0:32,0:32]
	elif label == 1:
		new_patch = patch[:,32:64,0:32]
	elif label == 2:
		new_patch = patch[:,64:96,0:32]
	elif label == 3:
		new_patch = patch[:,96:128,0:32]
	elif label == 4:
		new_patch = patch[:,0:32,32:64]
	elif label == 5:
		new_patch = patch[:,32:64,32:64]
	elif label == 6:
		new_patch = patch[:,64:96,32:64]
	elif label == 7:
		new_patch = patch[:,96:128,32:64]
	elif label == 8:
		new_patch = patch[:,0:32,64:96]
	elif label == 9:
		new_patch = patch[:,32:64,64:96]
	elif label == 10:
		new_patch = patch[:,64:96,64:96]
	elif label == 11:
		new_patch = patch[:,96:128,64:96]
	elif label == 12:
		new_patch = patch[:,0:32,96:128]
	elif label == 13:
		new_patch = patch[:,32:64,96:128]
	elif label == 14:
		new_patch = patch[:,64:96,96:128]
	else:
		new_patch = patch[:,96:128,96:128]
	return new_patch
