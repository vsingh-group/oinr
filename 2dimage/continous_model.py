'''
Implementation of O-INR using continuous convolution
This is necessary to train using different resolutions of the 
same image for tasks such as image super-resolution, etc.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

import sys
sys.path.append('..')

import ckconv


kernel_cfg = {
	'type': 'SIREN', #'MLP' 
	'no_hidden': 2,
	'no_layers': 1,
	'nonlinearity': 'GELU',   
	# 'norm': 'BatchNorm2d',
	'norm': 'InstanceNorm2d',
	'omega_0': 42,
	'bias': True,
	'size': 3,
	'chang_initialize': True,
	'init_spatial_value': 0.75,
}

conv_cfg = {
	'use_fft': True,
	'bias': True,
	'padding': True,
	'stride': 1,
	'causal': False,
}

class OINR_cont_2D(nn.Module):
	def __init__(self, inchannel, outchannel):
		super(OINR_cont_2D, self).__init__()

		self.inchannel = inchannel
		self.outchannel = outchannel
		print("model inchannel:", inchannel)
		print("model outchannel:", outchannel)

		self.approx_conv1 = ckconv.nn.CKConv(inchannel, 64, 2, kernel_cfg, conv_cfg)
		self.approx_conv2 = ckconv.nn.CKConv(64, 128, 2, kernel_cfg, conv_cfg)
		self.approx_conv3 = ckconv.nn.CKConv(128, 128, 2, kernel_cfg, conv_cfg)
		self.approx_conv4 = ckconv.nn.CKConv(128, 64, 2, kernel_cfg, conv_cfg)
		self.approx_conv5 = ckconv.nn.CKConv(64, outchannel, 2, kernel_cfg, conv_cfg)

	def forward(self, x):
		x = self.approx_conv1(x)
		x = torch.sin(x)
		x = self.approx_conv2(x)
		x = torch.sin(x)
		x = self.approx_conv3(x)
		x = torch.sin(x)
		x = self.approx_conv4(x)
		x = torch.sin(x)
		x = self.approx_conv5(x)
		return x