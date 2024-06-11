'''
File describing O-INR model for 3D volume
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class OINR_3D(nn.Module):
	def __init__(self, inchannel, outchannel):
		super(OINR_3D, self).__init__()

		self.inchannel = inchannel
		self.outchannel = outchannel
		print("model inchannel:", inchannel)
		print("model outchannel:", outchannel)

		self.approx_conv1 = nn.Conv3d(inchannel, 64, 3, 1, padding='same')
		self.approx_conv2 = nn.Conv3d(64, 128, 3, 1, padding='same')
		self.approx_conv3 = nn.Conv3d(128, 128, 3, 1, padding='same')
		self.approx_conv4 = nn.Conv3d(128, 64, 3, 1, padding='same')
		self.approx_conv5 = nn.Conv3d(64, outchannel, 3, 1, padding='same')

	def forward(self, x, verbose=False, autocast=False):
		device = "cuda" if x.is_cuda else "cpu"
		with torch.autocast(device, enabled=autocast):
			return self._forward(x)

	def _forward(self, x):
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