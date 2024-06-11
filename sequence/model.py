'''
Model file which describes O-INR to be used with sequence data
'''

import torch 
import numpy as np 

import torch.nn as nn
import torch.nn.functional as F

class SequenceNet(nn.Module):
	def __init__(self, inchannel=40, outchannel=3):
		super(SequenceNet, self).__init__()

		self.inchannel = inchannel
		self.outchannel = outchannel
		print("model inchannel:", inchannel)
		print("model outchannel:", outchannel)

		self.approx_conv1 = nn.Conv2d(inchannel, 64, 3, 1, padding='same', bias=True)
		self.approx_conv2 = nn.Conv2d(64, 128, 3, 1, padding='same', bias=True)
		self.approx_conv3 = nn.Conv2d(128, 256, 3, 1, padding='same', bias=True)
		self.approx_conv4 = nn.Conv2d(256, 256, 3, 1, padding='same', bias=True)
		self.approx_conv5 = nn.Conv2d(256, 128, 3, 1, padding='same', bias=True)
		self.approx_conv6 = nn.Conv2d(128, 64, 3, 1, padding='same', bias=True)
		self.approx_conv7 = nn.Conv2d(64, outchannel, 3, 1, padding='same', bias=True)


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
		x = torch.sin(x)
		x = self.approx_conv6(x)
		x = torch.sin(x)
		x = self.approx_conv7(x)
		signal = x
		return signal