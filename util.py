'''
Thie file is for all sharable utility function
'''
import torch 
import numpy as np 
import os, sys
import pdb
import pickle
import random
import matplotlib.pyplot as plt
import pandas as pd
from torchvision.utils import save_image
import math
from PIL import Image
import scipy.io

import numpy as np
import scipy.sparse
import numpy as np
from scipy.io import loadmat


def get_sine_mgrid(sidelen, num_sine, noise_channel=True, dim=2):
	'''
	Generates input to INR wavelet model with 2/3 channels.
	Uses positional encoding for both sine and cosine
	Optional noise channel
	'''
	tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
	mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
	
	x_grid = mgrid[:, :, 0]
	y_grid = mgrid[:, :, 1]
	
	all_x = []
	all_y = []
	for i in range(num_sine):
		x_value = torch.sin((2**i)*math.pi*x_grid)
		y_value = torch.sin((2**i)*math.pi*y_grid)
		all_x.append(x_value)
		all_y.append(y_value)       
		
	all_x = torch.stack(all_x)
	all_y = torch.stack(all_y)
	mgrid = torch.cat((all_x, all_y), dim=0)    
	
	if noise_channel:
		noise = torch.randn((sidelen, sidelen)).unsqueeze(0)
		mgrid = torch.cat((mgrid, noise), dim=0)
	else:
		mgrid = mgrid
	return mgrid

def get_cos_sine_mgrid_patch(coords, num_sine, noise_channel=False):
	'''
	'''
	mgrid = torch.stack(torch.meshgrid(*coords), dim=-1)
	
	x_grid = mgrid[:, :, 0]
	y_grid = mgrid[:, :, 1]
	
	all_x = []
	all_y = []
	for i in range(num_sine):
		# add sine value
		x_value = torch.sin((2**i)*math.pi*x_grid)
		y_value = torch.sin((2**i)*math.pi*y_grid)
		all_x.append(x_value)
		all_y.append(y_value)
		# add cosine value
		x_value = torch.cos((2**i)*math.pi*x_grid)
		y_value = torch.cos((2**i)*math.pi*y_grid)
		all_x.append(x_value)
		all_y.append(y_value)       
		
	all_x = torch.stack(all_x)
	all_y = torch.stack(all_y)
	mgrid = torch.cat((all_x, all_y), dim=0)    
	
	# not supporting noise addition for now
	# if noise_channel:
		# noise = torch.randn((sidelen, sidelen)).unsqueeze(0)
		# mgrid = torch.cat((mgrid, noise), dim=0)
	# else:
		# mgrid = mgrid
	return mgrid

def get_cos_sine_mgrid(sidelen: tuple, num_sine, noise_channel=True, dim=2):
	'''
	Generates input to INR wavelet model with 2/3 channels.
	Uses positional encoding for both sine and cosine
	Optional noise channel
	'''
	tensors = (torch.linspace(-1, 1, steps=sidelen[0]), torch.linspace(-1, 1, steps=sidelen[1]))
	mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
	
	x_grid = mgrid[:, :, 1]
	y_grid = mgrid[:, :, 0]
	
	all_x = []
	all_y = []
	for i in range(num_sine):
		# add sine value
		x_value = torch.sin((2**i)*math.pi*x_grid)
		y_value = torch.sin((2**i)*math.pi*y_grid)
		all_x.append(x_value)
		all_y.append(y_value)
		# add cosine value
		x_value = torch.cos((2**i)*math.pi*x_grid)
		y_value = torch.cos((2**i)*math.pi*y_grid)
		all_x.append(x_value)
		all_y.append(y_value)       
		
	all_x = torch.stack(all_x)
	all_y = torch.stack(all_y)
	mgrid = torch.cat((all_x, all_y), dim=0)    
	
	if noise_channel:
		noise = torch.randn((sidelen[0], sidelen[1])).unsqueeze(0)
		mgrid = torch.cat((mgrid, noise), dim=0)
	else:
		mgrid = mgrid
	return mgrid

def derivative_get_cos_sine_mgrid(sidelen, num_sine, noise_channel=True, dim=2):
	'''
	Generates input to INR wavelet model with 2/3 channels.
	Uses positional encoding for both sine and cosine
	Optional noise channel
	'''
	tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
	mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
	
	x_grid = mgrid[:, :, 1]
	y_grid = mgrid[:, :, 0]
	
	all_x = []
	all_y = []
	for i in range(num_sine):
		# add sine value
		x_value = (2**i)*math.pi*torch.cos((2**i)*math.pi*x_grid)
		y_value = (2**i)*math.pi*torch.cos((2**i)*math.pi*y_grid)
		all_x.append(x_value)
		all_y.append(y_value)
		# add cosine value
		x_value = -1*(2**i)*math.pi*torch.sin((2**i)*math.pi*x_grid)
		y_value = -1*(2**i)*math.pi*torch.sin((2**i)*math.pi*y_grid)
		all_x.append(x_value)
		all_y.append(y_value)       
		
	all_x = torch.stack(all_x)
	all_y = torch.stack(all_y)
	mgrid = torch.cat((all_x, all_y), dim=0)    
	
	if noise_channel:
		noise = torch.randn((sidelen, sidelen)).unsqueeze(0)
		mgrid = torch.cat((mgrid, noise), dim=0)
	else:
		mgrid = mgrid
	return mgrid

def get_cos_sine_mgrid_3d(sidelen, num_sine, noise_channel=True, dim=2):
	'''
	Generates input to 3DINR wavelet model with 2/3 channels.
	Uses positional encoding for both sine and cosine
	Optional noise channel
	'''
	tensors = (torch.linspace(-1, 1, steps=sidelen[0]), torch.linspace(-1, 1, steps=sidelen[1]), torch.linspace(-1, 1, steps=sidelen[2]))
	mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)

	x_grid = mgrid[:, :, :, 0]
	y_grid = mgrid[:, :, :, 1]
	z_grid = mgrid[:, :, :, 2]

	all_x = []
	all_y = []
	all_z = []

	for i in range(num_sine):
		# add sine value
		x_value = torch.sin((2**i)*math.pi*x_grid)
		y_value = torch.sin((2**i)*math.pi*y_grid)
		z_value = torch.sin((2**i)*math.pi*z_grid)
		all_x.append(x_value)
		all_y.append(y_value)
		all_z.append(z_value)
		# add cosine value
		x_value = torch.cos((2**i)*math.pi*x_grid)
		y_value = torch.cos((2**i)*math.pi*y_grid)
		z_value = torch.cos((2**i)*math.pi*z_grid)
		all_x.append(x_value)
		all_y.append(y_value)
		all_z.append(z_value)       
	
	all_x = torch.stack(all_x)
	all_y = torch.stack(all_y)
	all_z = torch.stack(all_z)
	mgrid = torch.cat((all_x, all_y, all_z), dim=0)    

	if noise_channel:
		noise = torch.randn((sidelen, sidelen)).unsqueeze(0)
		mgrid = torch.cat((mgrid, noise), dim=0)
	else:
		mgrid = mgrid
	return mgrid

def get_mgrid(sidelen, noise_channel=True, dim=2):
	'''
	Generates input to INR wavelet model with 2/3 channels.
	The mandatory two channels are for x-coordinate and y-coordinate projected in the range of -1 to 1
	The third channels is the optiional noise channel
	'''
	tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
	mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
	if noise_channel:
		noise = torch.randn((sidelen, sidelen)).unsqueeze(-1)
		mgrid = torch.cat((mgrid, noise), dim=-1).transpose(0,2).transpose(1,2)
	else:
		mgrid = mgrid.transpose(0,2).transpose(1,2)
	return mgrid

def create_multiple_data(min_size, max_size, noise_channel):
	image_list = []
	grid_list = []
	for image_size in range(min_size, max_size, 2):
		im = Image.open('../cameraman.jpeg').resize((image_size, image_size))
		pic = np.asarray(im)
		grid_image = get_mgrid(image_size, noise_channel=noise_channel, dim=2)
		image_list.append(pic)
		grid_list.append(grid_image)
	return grid_list, image_list

def create_multiple_data_v2(min_size, max_size, noise_channel):
	image_list = []
	grid_list = []
	im = Image.open('../cameraman.jpeg')
	grid = get_mgrid(im.size[0], noise_channel=noise_channel, dim=2)
	save_image(grid, '../grid.png')
	print("saved image")
	for image_size in range(min_size, max_size, 2):
		im = Image.open('../cameraman.jpeg').resize((image_size, image_size))
		grid_image = Image.open('../grid.png').resize((image_size, image_size))
		pic = np.asarray(im)
		grid_image= np.asarray(grid_image)
		image_list.append(pic)
		grid_list.append(grid_image)
	return grid_list, image_list

def create_multiple_data_v3(img_file, min_size, max_size, noise_channel, num_sine):
	image_list = []
	grid_list = []
	for image_size in range(min_size, max_size, 2):
		im = Image.open(img_file).resize((image_size, image_size))
		pic = np.asarray(im)
		grid_image = get_cos_sine_mgrid(sidelen=(image_size, image_size), num_sine=num_sine, noise_channel=noise_channel, dim=2)
		image_list.append(pic)
		grid_list.append(grid_image)
	return grid_list, image_list

def manual_set_seed(seed):
	print("Setting all seeds to: ", seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

def plot_and_save(exp, all_losses):
	loss_file = exp+'/all_losses.pkl'
	with open(loss_file, 'wb') as f:
		pickle.dump(all_losses, f)

	plt.plot(all_losses['train_total_loss'], label='Training')
	plt.legend()
	plt.xlabel("Epoch")
	plt.ylabel("Total loss")
	plt.savefig(exp+'/TotalLoss.png')
	plt.cla()
	plt.clf()
	plt.close()