'''
This file has the custom dataset loading classes defined
'''
import os
import sys
import numpy as np 
import torch 
import random
import pdb
from torch.utils.data import Dataset
from torchvision import transforms


class Single_Image_Dataset(Dataset):
	def __init__(self, image, erase=True):
		self.image = image
		self.erase = erase     # to turn on/off image mising patch
		if erase:
			self.erase_transform = transforms.RandomErasing(p=1.0, scale=(0.002, 0.002), ratio=(1, 1), value='random', inplace=False)

	def __getitem__(self, idx):
		missing_image = self.image[idx]
		if self.erase:
			missing_image = self.erase_transform(missing_image)
		return missing_image, self.image[idx]

	def __len__(self):
		return len(self.image)

class INR_Single_Image_Dataset(Dataset):
	def __init__(self, grid, image):
		self.grid = grid
		self.image = image

	def __getitem__(self, idx):
		return self.grid[idx], self.image[idx]

	def __len__(self):
		return len(self.image)

class INR_Multi_Image_Dataset(Dataset):
	def __init__(self, grid, image):
		self.grid = grid
		self.image = image

		grid_min = np.min(grid[0])
		grid_max = np.max(grid[0])
		image_min = np.min(image[0])
		image_max = np.max(image[0])

		for i in range(len(grid)):

			cur_grid_min = np.min(grid[i])
			cur_grid_max = np.max(grid[i])
			cur_image_min = np.min(image[i])
			cur_image_max = np.max(image[i])

			if cur_grid_min < grid_min:
				grid_min = cur_grid_min
			if cur_grid_max > grid_max:
				grid_max = cur_grid_max
			if cur_image_min < image_min:
				image_min = cur_image_min
			if cur_image_max > image_max:
				image_max = cur_image_max

		print("Grid min max:", grid_min, grid_max)
		print('Image min max:', image_min, image_max)


	def __getitem__(self, idx):
		image = transforms.ToTensor()(np.array(self.image[idx]))
		grid = transforms.ToTensor()(np.array(self.grid[idx]))*2 - 1
		return grid, image

	def __len__(self):
		return len(self.image)

class INR_Multi_Image_Dataset_v3(Dataset):
	def __init__(self, grid, image):
		self.grid = grid
		self.image = image

	def __getitem__(self, idx):
		image = transforms.ToTensor()(np.array(self.image[idx]))
		grid = self.grid[idx]
		return grid, image

	def __len__(self):
		return len(self.image)