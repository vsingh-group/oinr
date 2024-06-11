'''
Model file of O-INR where operator is CZ
Code from https://github.com/sourav-roni/BCR-DE
'''

import torch
import numpy as np
import pdb
import random
import os, sys
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import math
from einops import rearrange
from pytorch_wavelets import DWT1DForward, DWT1DInverse
from tqdm import tqdm

class PartiallyUnsharedConv1d(nn.Module):
	'''
	Proposed Partially Unshared Convolution (PUC) layer.
	A faster version with slight difference in end point computation 
	is present int the class cheapPartiallyUnsharedConv1d below
	'''
	def __init__(self, in_channels, out_channels, output_size, kernel_size, stride, one_side_pad_length, num_sparse_LC, dim_d, dim_k, conv_bias, level, nk_LC):
		super(PartiallyUnsharedConv1d, self).__init__()
		self.num_sparse_LC = num_sparse_LC
		self.weight = nn.Parameter(
			0.02*torch.randn(dim_d, dim_k, out_channels, in_channels, num_sparse_LC, 1, kernel_size)
		)
		torch.nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

		self.conv_bias = conv_bias
		if self.conv_bias:
			self.conv_bias_param = nn.Parameter(
				0.02*torch.randn(dim_d, dim_k, out_channels, num_sparse_LC, 1)
			)
		
		self.kernel_size = kernel_size
		self.stride = stride
		self.one_side_pad_length = one_side_pad_length
		self.padder = nn.ConstantPad1d(self.one_side_pad_length, 0)

		self.output_size = output_size
		self.padded_size = output_size + 2*one_side_pad_length
		self.rough_segment_length = int(self.padded_size/self.num_sparse_LC)
		
		self.repeat_lengths = []
		total_repeated = 0
		for i in range(self.num_sparse_LC):
			if i==self.num_sparse_LC-1:
				to_cover = self.output_size - total_repeated
			else:
				to_cover = self.rough_segment_length - 2*one_side_pad_length
			self.repeat_lengths.append(to_cover)
			total_repeated += to_cover

	def forward(self, x):
		n, d, k, t, l = x.size()     # ~ [bs, dim_d, dim_k, 2, len]
		x = self.padder(x)           # ~ [bs, dim_d, dim_k, 2, len']
		new_pad_len = x.shape[-1]
		kl = self.kernel_size
		dl = self.stride             # This is for all puposes 1, otherwise there will be change in dimension
		x = x.unfold(-1, kl, dl)
		x = x.contiguous()           # ~ [bs, dim_d, dim_k, 2, len, kernel_size]

		all_weights_repeated = []
		all_conv_bias_repeated = []
		for i in range(self.num_sparse_LC):
			current_patch = self.weight[:, :, :, :, i, :, :]
			all_weights_repeated += [current_patch] * self.repeat_lengths[i]
			if self.conv_bias:
				current_conv_bias = self.conv_bias_param[:, :, :, i, :]
				all_conv_bias_repeated += [current_conv_bias] * self.repeat_lengths[i]
		weight = torch.cat(all_weights_repeated, dim=-2)
		if self.conv_bias:
			conv_bias_param = torch.cat(all_conv_bias_repeated, dim=-1)

		out = torch.einsum("dkoilf,bdkilf->bdkol", weight, x)                # ~ [bs, dim_d, dim_k, 2, len]
		if self.conv_bias:
			out = out + conv_bias_param
		return out

class cheapPartiallyUnsharedConv1d(nn.Module):
	'''
	This variant of Partially Unshared Convolution (PUC) layer is a bit different from PartiallyUnsharedConv1d
	The difference is in how the patches are laid over the sequence
	and in the end point computation. However this is much faster and is the recommended one
	'''
	def __init__(self, in_channels, out_channels, output_size, kernel_size, stride, one_side_pad_length, num_sparse_LC, dim_d, dim_k, conv_bias, level, nk_LC):
		super(cheapPartiallyUnsharedConv1d, self).__init__()
		self.num_sparse_LC = num_sparse_LC
		self.weight = nn.Parameter(
			0.02*torch.randn(dim_d, dim_k, out_channels, in_channels, num_sparse_LC, 1, kernel_size)
		)
		torch.nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

		self.conv_bias = conv_bias
		if self.conv_bias:
			self.conv_bias_param = nn.Parameter(
				0.02*torch.randn(dim_d, dim_k, out_channels, num_sparse_LC, 1)
			)
		
		self.kernel_size = kernel_size
		self.stride = stride
		self.one_side_pad_length = one_side_pad_length
		self.padder = nn.ConstantPad1d(self.one_side_pad_length, 0)


	def forward(self, x):
		n, d, k, t, l = x.size()     # ~ [bs, dim_d, dim_k, 2, len]
		x = self.padder(x)           # ~ [bs, dim_d, dim_k, 2, len']
		kl = self.kernel_size
		dl = self.stride             # This is for all puposes 1, otherwise there will be change in dimension
		x = x.unfold(-1, kl, dl)
		x = x.contiguous()           # ~ [bs, dim_d, dim_k, 2, len, kernel_size]

		dim_d, dim_k, out_channels, in_channels, num_sparse_LC, _, kernel_size = self.weight.shape       # input and output channels are both 1 for all practical purpose
		weight = self.weight.repeat(1, 1, 1, 1, 1, int(math.floor(l / num_sparse_LC)), 1)                 # ~ [dim_d, dim_k, out_channels, in_channels, num_sparse_LC, length_of_each_LC, kernel_size]
		weight = weight.reshape(dim_d, dim_k, out_channels, in_channels, -1, kernel_size)                # ~ [dim_d, dim_k, out_channels, in_channels, num_sparse_Lc*length_of_each_LC, kernel_size]
		remainder = x.shape[-2] - weight.shape[-2]
		last = self.weight[:, :, :, :, -1, :, :]
		weight = torch.cat([weight] + [last] * remainder, dim = -2)

		if self.conv_bias:
			conv_bias_param = self.conv_bias_param.repeat(1, 1, 1, 1, int(math.floor(l / num_sparse_LC)))
			conv_bias_param = conv_bias_param.reshape(dim_d, dim_k, out_channels, -1)
			last_conv_bias = self.conv_bias_param[:, :, :, -1, :]
			conv_bias_param = torch.cat([conv_bias_param] + [last_conv_bias]*remainder, dim=-1)

		# Sum in in_channel and kernel_size dims
		out = torch.einsum("dkoilf,bdkilf->bdkol", weight, x)                # ~ [bs, dim_d, dim_k, 2, len]
		if self.conv_bias:
			out = out + conv_bias_param
		return out

class myDense(nn.Module):
	'''
	Dense layer using einsum, for multiple dimension
	'''
	def __init__(self, dim_d, dim_k, dense_dim, bias=False):
		super(myDense, self).__init__()
		self.dLayer = nn.Parameter(
			0.02*torch.randn(dim_d, dim_k, dense_dim, dense_dim)
		)
		torch.nn.init.kaiming_normal_(self.dLayer, mode='fan_out', nonlinearity='relu')
		self.bias = bias
		if bias:
			self.d_bias = nn.Parameter(torch.randn(dim_d, dim_k, self.dLayer.shape[-2]))

	def forward(self, x):
		transform_x = torch.einsum('dktq,bdkq->bdkt', self.dLayer, x)
		if self.bias:
			transform_x = transform_x + self.d_bias
		return transform_x

class CDE_BCR(nn.Module):
	'''
	Main model for BCR_DE
	'''
	def __init__(self, wave, D, D_out, d, k, original_length, num_classes, nonlinearity, n_levels, K_dense, K_LC, nb, num_sparse_LC, use_cheap_sparse_LC, interpol, conv_bias, predict=False, masked_modelling=False):
		super(CDE_BCR, self).__init__()
		print("Efficient model")
		self.wave = wave
		self.dim_D = D
		self.dim_D_out = D_out
		self.dim_d = d
		self.dim_k = k
		self.original_length = original_length
		self.n_levels = n_levels
		self.K_dense = K_dense
		self.K_LC = K_LC
		self.nb = nb
		self.num_classes = num_classes
		self.num_sparse_LC = num_sparse_LC
		self.interpol = interpol
		self.conv_bias = conv_bias

		self.forward_wavelet = DWT1DForward(wave=self.wave, J=1, mode='periodization')
		self.inverse_wavelet = DWT1DInverse(wave=self.wave, mode='periodization')

		if nonlinearity == 'relu':
			self.nl_act = nn.ReLU()
		elif nonlinearity == 'tanh':
			self.nl_act = nn.Tanh()
		elif nonlinearity == 'LeakyReLU':
			self.nl_act = nn.LeakyReLU()
		elif nonlinearity == 'ELU':
			self.nl_act = nn.ELU()
		elif nonlinearity == 'PReLU':
			self.nl_act = nn.PReLU()
		elif nonlinearity == 'tanh':
			self.nl_act = nn.Tanh()
		elif nonlinearity == 'Tanhshrink':
			self.nl_act = nn.Tanhshrink()
		elif nonlinearity == 'ReLU6':
			self.nl_act = nn.ReLU6()
		elif nonlinearity == 'GELU':
			self.nl_act = nn.GELU()
		elif nonlinearity == 'SiLU':
			self.nl_act = nn.SiLU()
		elif nonlinearity == 'Softshrink':
			self.nl_act = nn.Softshrink()
		else:
			print("Invalid activation function")
			exit(0)

		self.g_layer = nn.Linear(self.dim_D, self.dim_d, bias=False)
		# self.h_layer = nn.Linear(self.dim_d, self.dim_D * self.dim_k, bias=False)

  		# Forward pass to get dimension of dense layer
		x = torch.tensor(np.random.rand(4, 1, self.original_length)).float()
		self.dense_dim, self.output_sizes = self.fake_pass_get_dim(x)
		self.output_sizes.reverse()
		print("Ouput sizes: ", self.output_sizes)

		self.dk_pair_dense_weight = nn.ModuleList()
		for k in range(self.K_dense):
			dl = myDense(self.dim_d, self.dim_k, self.dense_dim, bias=False)
			self.dk_pair_dense_weight.append(dl)

		self.dk_pair_LC_einsum = nn.ModuleList()
		for i in range(self.n_levels):
			level_LCs = nn.ModuleList()
			for j in range(0, self.K_LC):
				if use_cheap_sparse_LC:
					LC_layer = cheapPartiallyUnsharedConv1d(in_channels=2*1, out_channels=2*1, output_size=self.output_sizes[-i-1], kernel_size=self.nb, stride=1, 
											one_side_pad_length=math.floor(nb/2), num_sparse_LC=self.num_sparse_LC, dim_d = self.dim_d, dim_k = self.dim_k, conv_bias=True, 
											level=i, nk_LC=j)
				else:
					LC_layer = PartiallyUnsharedConv1d(in_channels=2*1, out_channels=2*1, output_size=self.output_sizes[-i-1], kernel_size=self.nb, stride=1, 
												one_side_pad_length=math.floor(nb/2), num_sparse_LC=self.num_sparse_LC, dim_d = self.dim_d, dim_k = self.dim_k, conv_bias=True, 
												level=i, nk_LC=j)
				level_LCs.append(LC_layer)
			self.dk_pair_LC_einsum.append(level_LCs)

		self.reverse_g_layer = nn.Linear(self.dim_d, self.dim_D_out, bias=False)
		if predict:
			print("Adding final prediction layer")
			self.predict = True
			self.prediction_layer = nn.Sequential(
						nn.Linear(self.dim_D_out, 20, bias=True),
						nn.ReLU(),
						nn.Linear(20, self.num_classes, bias=True),
						)
		else:
			self.predict = False
		self.masked_modelling = masked_modelling

		
	def forward(self, seq):
		batch_size = seq.shape[0]
		sequence_length = seq.shape[1]
		z = self.nl_act(self.g_layer(seq))
		v = z
		v = v.transpose(1,2)

		current_approx = v
		all_detail = []
		all_approx = []
		for l in range(self.n_levels):
			current_approx, current_detail = self.forward_wavelet(current_approx)  # [bs, chna, len]
			all_detail.append(current_detail[0])
			all_approx.append(current_approx)

		last_approx = all_approx[-1]
		dth_corase_approx = []

		current_approx = last_approx[:, None, :, :].repeat(1, self.dim_d, 1, 1)
		for k in range(self.K_dense):
			current_approx = self.nl_act(self.dk_pair_dense_weight[k](current_approx))

		dth_corase_approx = current_approx
		dth_current_approx = dth_corase_approx

		if self.masked_modelling:
			masked_modelling_approx = None
			masked_modelling_detail = []

		for l in reversed(range(self.n_levels)):
			prev_detail_l = all_detail[l]
			prev_approx_l = all_approx[l]

			chi_l = torch.stack([prev_detail_l, prev_approx_l], dim = 2).unsqueeze(1).repeat(1, self.dim_d, 1, 1, 1) # ~ [bs, dim_d, dim_k, 2, len]

			for k in range(self.K_LC):
				chi_l = self.nl_act(self.dk_pair_LC_einsum[l][k](chi_l)).float()

			current_approx = dth_current_approx
			current_approx = self.shape_correction(chi_l, current_approx)
			padded_current_approx = torch.stack([torch.zeros_like(current_approx), current_approx], dim = -2)
			X_l = padded_current_approx + chi_l

			bs, dd, kd, _, length = X_l.shape
			X_l_detail = X_l[:, :, :, 0, :].reshape(bs * dd * kd, 1, length)
			X_l_approx = X_l[:, :, :, 1, :].reshape(bs * dd * kd, 1, length)
			if self.masked_modelling:
				if l==self.n_levels-1:
					masked_modelling_approx = X_l_approx
				masked_modelling_detail.append(X_l_detail)
			current_next_approx = self.inverse_wavelet((X_l_approx, [X_l_detail]))
			current_next_approx = current_next_approx.reshape(bs, dd, kd, -1)

			dth_current_approx = current_next_approx

		dth_current_approx = torch.sum(dth_current_approx,dim=2)
		current_approx = dth_current_approx  # ~ [bs, dim_D, len]

		U = self.reverse_g_layer(current_approx.transpose(1,2))

		if self.masked_modelling:
			return U, masked_modelling_approx, masked_modelling_detail

		if self.predict:
			last_observable = U[:, -1, :]
			prediction = self.prediction_layer(last_observable)
			return U, prediction
		else:
			return U

	def shape_correction(self, chi_l, current_approx):
		if chi_l.shape[-1] == current_approx.shape[-1]:
			return current_approx
		else:
			# There is size mismatch, so use padding zeros
			left_diff = chi_l.shape[-1] - current_approx.shape[-1]
			m = nn.ConstantPad1d((left_diff,0), 0)
			current_approx = m(current_approx)
			return current_approx

	def fake_pass_get_dim(self, v):
		current_approx = v
		output_sizes = []
		for l in range(self.n_levels):
			current_approx, current_detail = self.forward_wavelet(current_approx)
			assert current_approx.shape[2] == current_detail[0].shape[2]
			output_sizes.append(current_approx.shape[2])
		return current_approx.size(2), output_sizes

	def tril_init(self):
		print("Initilizing lower triangular kernel")
		with torch.no_grad():
			for dth_dim in range(self.dim_d):
				for kth_dim in range(self.dim_k):
					for k in range(self.K_dense):
						self.dk_pair_dense_weight[k][dth_dim][kth_dim].copy_(torch.tril(self.dk_pair_dense_weight[k][dth_dim][kth_dim]))