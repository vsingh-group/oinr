'''
Model file to fit O-INR
and to compute it's derivative
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class OINR_2D(nn.Module):
	def __init__(self, inchannel=40, outchannel=3):
		super(OINR_2D, self).__init__()

		self.inchannel = inchannel
		self.outchannel = outchannel

		self.approx_conv1 = nn.Conv2d(inchannel, 64, 3, 1, padding='same', bias=False)
		self.approx_conv2 = nn.Conv2d(64, 128, 3, 1, padding='same', bias=False)
		self.approx_conv3 = nn.Conv2d(128, 128, 3, 1, padding='same', bias=False)
		self.approx_conv4 = nn.Conv2d(128, 64, 3, 1, padding='same', bias=False)
		self.approx_conv5 = nn.Conv2d(64, outchannel, 3, 1, padding='same', bias=False)

	def forward(self, x, verbose=False):
		x = self.approx_conv1(x)
		x = torch.sin(x)
		x = self.approx_conv2(x)
		x = torch.sin(x)
		x = self.approx_conv3(x)
		x = torch.sin(x)
		x = self.approx_conv4(x)
		x = torch.sin(x)
		x = self.approx_conv5(x)
		signal = x
		return signal


class Derivative2D(nn.Module):
	def __init__(self, inchannel, outchannel, trained_INR):
		super(Derivative2D, self).__init__()

		self.inchannel = inchannel
		self.outchannel = outchannel

		self.approx_conv1 = nn.Conv2d(inchannel, 64, 3, 1, padding='same', bias=False)
		self.approx_conv2 = nn.Conv2d(64, 128, 3, 1, padding='same', bias=False)
		self.approx_conv3 = nn.Conv2d(128, 128, 3, 1, padding='same', bias=False)
		self.approx_conv4 = nn.Conv2d(128, 64, 3, 1, padding='same', bias=False)
		self.approx_conv5 = nn.Conv2d(64, outchannel, 3, 1, padding='same', bias=False)

		# Match the weights
		self.approx_conv1.weight.data = trained_INR.approx_conv1.weight.data.clone()
		self.approx_conv2.weight.data = trained_INR.approx_conv2.weight.data.clone()
		self.approx_conv3.weight.data = trained_INR.approx_conv3.weight.data.clone()
		self.approx_conv4.weight.data = trained_INR.approx_conv4.weight.data.clone()
		self.approx_conv5.weight.data = trained_INR.approx_conv5.weight.data.clone()

	def forward(self, x, der_x):

		der_prev2_ip = self.approx_conv1(der_x)
		prev2_ip = self.approx_conv1(x)
		prev2 = torch.mul(torch.cos(prev2_ip), der_prev2_ip)
		der_prev3_ip = self.approx_conv2(prev2)
		prev3_ip = self.approx_conv2(torch.sin(self.approx_conv1(x)))
		prev3 = torch.mul(torch.cos(prev3_ip), der_prev3_ip)
		der_prev4_ip = self.approx_conv3(prev3)
		prev4_ip = self.approx_conv3(torch.sin(self.approx_conv2(torch.sin(self.approx_conv1(x)))))
		prev4 = torch.mul(torch.cos(prev4_ip), der_prev4_ip)
		der_prev5_ip = self.approx_conv4(prev4)
		prev5_ip = self.approx_conv4(torch.sin(self.approx_conv3(torch.sin(self.approx_conv2(torch.sin(self.approx_conv1(x)))))))
		prev5 = torch.mul(torch.cos(prev5_ip), der_prev5_ip)
		signal = self.approx_conv5(prev5)
		return signal

	def signal(self, x):
		x = self.approx_conv1(x)
		x = torch.sin(x)
		x = self.approx_conv2(x)
		x = torch.sin(x)
		x = self.approx_conv3(x)
		x = torch.sin(x)
		x = self.approx_conv4(x)
		x = torch.sin(x)
		signal = self.approx_conv5(x)
		return signal