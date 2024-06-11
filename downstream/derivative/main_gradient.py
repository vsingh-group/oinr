'''
Main file to run gradient compuation using pre-trained O-INR
'''

import torch 
import numpy as np 
from torchvision.utils import save_image
from torch import nn
from torch.autograd import Variable
import torchvision.transforms as transforms
from model import OINR_2D, Derivative2D
import cv2
import argparse
import os

import sys
sys.path.append('../../')
from util import get_cos_sine_mgrid
import matplotlib.pyplot as plt
plt.gray()

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pdb
from PIL import Image

def gradient_img_v2(img, nchannel=3):
	x = img
	a = -1/3*np.array([[1, 0, -1],[2,0,-2],[1,0,-1]]) # sobel
	conv1=nn.Conv2d(nchannel, nchannel, kernel_size=3, groups=nchannel, stride=1, padding='same', bias=False)
	conv1.weight=nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0).repeat(nchannel,1,1,1))
	G_x=conv1(Variable(x)).data.view(nchannel,x.shape[2],x.shape[3])

	b = -1/3*np.array([[1, 2, 1],[0,0,0],[-1,-2,-1]]) # sobel
	conv2=nn.Conv2d(nchannel, nchannel, kernel_size=3, groups=nchannel, stride=1, padding='same', bias=False)
	conv2.weight=nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0).repeat(nchannel,1,1,1))
	G_y=conv2(Variable(x)).data.view(nchannel,x.shape[2],x.shape[3])

	G=torch.sqrt(torch.pow(G_x,2)+ torch.pow(G_y,2))
	return G, G_x, G_y

def mat_save(image, fname):
	# Convert the tensor to a NumPy array
	if len(image.shape) == 4:
		image = image.squeeze(0).detach().cpu().numpy()
	else:
		image = image.detach().cpu().numpy()

	# Transpose the array to match the expected shape of an image in matplotlib
	image = np.transpose(image, (1, 2, 0))
	cv2.imwrite(fname, cv2.cvtColor(image * 255, cv2.COLOR_RGB2BGR) ) # 255 for imwrite

def parse_args():
	'''
	Parse input arguments
	'''
	parser = argparse.ArgumentParser()

	parser.add_argument(	"--ckpt_path",
					 		type=str, 
					 		default='derivative_2D',
							help="path to the pre-trained O-INR checkpoint. the final results are stored in the directory containing the check point")
	parser.add_argument("--img_path", type=str, default=None, help="path to the input image. This is not truly required")
	# The arguments below need to match the ones used while training the O-INR, please verify in case of any error.
	parser.add_argument('--noise_channel', default=False, action='store_true', help='Whether to use noise channel as input')
	parser.add_argument('--num_sine', type=int, default=10, help='number of sine non lienarity for the each x and y dimension')
	parser.add_argument('--outchannel', type=int, default=3, help='number of channels in the image')
	parser.add_argument('--img_h', type=int, default=512, help='height of the image O-INR was trained on. Please check in case of error')
	parser.add_argument('--img_w', type=int, default=512, help='width of the image O-INR was trained on. Please check in case of error')

	args = parser.parse_args()
	return args

def main():
	print("Derivative computation")

	args = parse_args()

	results_path = os.path.dirname(os.path.abspath(args.ckpt_path)) + '/gradient'
	if not os.path.exists(results_path):
		os.makedirs(results_path)

	model = OINR_2D(inchannel=4*args.num_sine, outchannel=args.outchannel)
	model = torch.load(args.ckpt_path,map_location=torch.device('cpu')).to('cpu')
	
	dermodel = Derivative2D(inchannel=4*args.num_sine, outchannel=args.outchannel, trained_INR=model)

	img_size = [args.img_h, args.img_w]
	grid_image = get_cos_sine_mgrid(img_size, args.num_sine, noise_channel=args.noise_channel).unsqueeze(0)
	output = model(grid_image.float())
	mat_save(output, os.path.join(results_path, 'OINR_image.png'))

	image = Image.open(args.img_path)
	assert image.size[0] == img_size[0]
	assert image.size[1] == img_size[1]
	true_image = transforms.ToTensor()(np.array(image)).unsqueeze(0)    # shape required: [bs, channel, height, width]
	mat_save(true_image, os.path.join(results_path, 'true_image.png'))
	print("True image:", true_image.shape)

	true_h_grad, true_h_grad_x, true_h_grad_y = gradient_img_v2(true_image, args.outchannel)
	mat_save(true_h_grad,   os.path.join(results_path, 'true_grad.png'))
	mat_save(true_h_grad_x, os.path.join(results_path, 'true_grad_x.png'))
	mat_save(true_h_grad_y, os.path.join(results_path, 'true_grad_y.png'))

	output = dermodel.signal(grid_image.float())
	mat_save(output, os.path.join(results_path, 'model_ouptut_image.png'))

	_, grad_grid_x, grad_grid_y = gradient_img_v2(grid_image, nchannel=4*args.num_sine)

	deroutput_x = dermodel(grid_image.float(), grad_grid_x.unsqueeze(0).float())
	mat_save(deroutput_x, os.path.join(results_path, 'grad_x.png'))

	deroutput_y = dermodel(grid_image.float(), grad_grid_y.unsqueeze(0).float())
	mat_save(deroutput_y, os.path.join(results_path, 'grad_y.png'))

	der_h = torch.sqrt(torch.pow(deroutput_x, 2)+ torch.pow(deroutput_y,2))
	mat_save(der_h, os.path.join(results_path, 'grad.png'))
	
	# List of image file paths
	image_paths = [
		'true_image.png', 'true_grad_x.png', 'true_grad_y.png', 'true_grad.png',
		'OINR_image.png', 'grad_x.png', 'grad_y.png', 'grad.png'
	]

	# List of captions for the images
	captions = [
		'Original Image', 'True Gradient in x', 'True Gradient in y', 'True Gradient',
		'O-INR Image', 'O-INR Gradient in x', 'O-INR Gradient in y', 'O-INR Gradient'
	]

	# Create a figure with 2 rows and 4 columns
	fig, axes = plt.subplots(2, 4, figsize=(15, 8))

	# Flatten the axes array for easy iteration
	axes = axes.flatten()

	# Loop through the images and their captions
	for i, (img_path, caption) in enumerate(zip(image_paths, captions)):
		# Read the image
		full_path = os.path.join(results_path, img_path)
		img = mpimg.imread(full_path)
		# Display the image
		axes[i].imshow(img)
		# Set the caption
		axes[i].set_title(caption)
		# Remove the axes for a cleaner look
		axes[i].axis('off')

	# Adjust layout
	plt.tight_layout()

	# Save the final grid image
	plt.savefig(os.path.join(results_path, 'combine.png'))
	
	print("Done")

if __name__ == '__main__':
	main()