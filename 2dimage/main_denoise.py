'''
Main file to run O-INR on noisy 2d image
'''

import os 
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import numpy as np
import csv
import pickle
from tqdm import tqdm
from PIL import Image
import argparse

import sys
sys.path.append('../')
from util import manual_set_seed, plot_and_save, get_cos_sine_mgrid

from datautils import INR_Single_Image_Dataset
from torch.utils.data import DataLoader
from model import OINR_2D

# Code from https://github.com/vishwa91/wire to add noise to image
def noise_image(x, noise_snr=40, tau=100):
	''' Realistic sensor measurement with readout and photon noise

		Inputs:
			noise_snr: Readout noise in electron count
			tau: Integration time. Poisson noise is created for x*tau.
				(Default is 100)

		Outputs:
			x_meas: x with added noise
	'''
	x_meas = np.copy(x)

	noise = np.random.randn(x_meas.size).reshape(x_meas.shape)*noise_snr

	# First add photon noise, provided it is not infinity
	if tau != float('Inf'):
		x_meas = x_meas*tau

		x_meas[x > 0] = np.random.poisson(x_meas[x > 0])
		x_meas[x <= 0] = -np.random.poisson(-x_meas[x <= 0])

		x_meas = (x_meas + noise)/tau

	else:
		x_meas = x_meas + noise

	x_noise  = x_meas

	return x_noise

def psnr(x, xhat):
	''' Compute Peak Signal to Noise Ratio in dB

		Inputs:
			x: Ground truth signal
			xhat: Reconstructed signal

		Outputs:
			snrval: PSNR in dB
	'''
	err = x - xhat
	denom = np.mean(pow(err, 2))
	snrval = 10*np.log10(np.max(x)/denom)
	return snrval


def normalize(x, fullnormalize=False):
	'''
		Normalize input to lie between 0, 1.

		Inputs:
			x: Input signal
			fullnormalize: If True, normalize such that minimum is 0 and
				maximum is 1. Else, normalize such that maximum is 1 alone.

		Outputs:
			xnormalized: Normalized x.
	'''
	if x.sum() == 0:
		return x
	xmax = x.max()

	if fullnormalize:
		xmin = x.min()
	else:
		xmin = 0

	xnormalized = (x - xmin)/(xmax - xmin)
	return xnormalized




def parse_args():
	'''
	Parse input arguments
	'''
	parser = argparse.ArgumentParser()

	parser.add_argument("--img_path", type=str, required=True, default=None, help="path to the input image")
	parser.add_argument("--seed", type=int, default=0, help="Setting seed for the entire experiment")
	parser.add_argument("--exp", type=str, default='denoise_2D', help="Adjusted in code: Experiment folder name")
	parser.add_argument('--noise_channel', default=False, action='store_true', help='Whether to use noise channel as input')
	parser.add_argument('--num_sine', type=int, default=10, help='number of sine non lienarity for the each x and y dimension')

	# training arguments
	parser.add_argument('--train_bs', default=1, type=int, help='Batchsize for train loader')
	parser.add_argument('--epoch', default=1000, type=int, help='Number of epochs to train')
	parser.add_argument('--lr', default=0.001, type=float, help="Learning rate for the O-INR model")
	parser.add_argument('--model_pred_save_freq', default=100, type=int, help='Saving frequency of model prediction')
	parser.add_argument('--scheduler_en', action='store_true', help='Enable Reduce LR on Plateau scheduler')

	args = parser.parse_args()
	return args

def main(args, device):
	# set seed randomly to check variance
	if args.seed == -1:
		args.seed = np.random.randint(0,100000)

	# Setting seeds for reproducibility
	manual_set_seed(args.seed)
	print(args)

	arg_dict = vars(args)

	# setup the log dirs
	exp_name = args.exp
	print(args)
	if os.path.exists('./result/'+exp_name):
		nth_exp = len(os.listdir('./result/'+exp_name+'/Results'))+1
	else:
		nth_exp = 0
	args.exp = './result/'+exp_name+'/Results/'+str(nth_exp)
	arg_dict['Result_location'] = './Results/'+str(nth_exp)

	if not os.path.exists(args.exp):
		print("Creating experiment directory: ", args.exp)
		os.makedirs(args.exp)

	# list of column names 
	field_names = arg_dict.keys()
	with open('./result/'+exp_name+'/experiment.csv', 'a') as csv_file:
		dict_object = csv.DictWriter(csv_file, fieldnames=field_names) 
		dict_object.writerow(arg_dict)

	argument_file = args.exp+'/arguments.pkl'
	with open(argument_file, 'wb') as f:
		pickle.dump(arg_dict, f)

	#create storage for expt res
	dirname = args.exp+'/model_prediction/'
	if not os.path.exists(dirname):
		os.makedirs(dirname)

	image = Image.open(args.img_path)
	x, y = image.size
	print(image.format)
	print(image.size)
	print(image.mode)
	torch_image = transforms.ToTensor()(np.array(image)).unsqueeze(0)    # shape required: [bs, channel, height, width]
	print("Torch image:", torch_image.shape)

	im = np.array(image).astype(np.float32)
	h, w, c = im.shape
	tau = 3e1					# Photon noise (max. mean lambda)
	noise_snr = 2				# Readout noise (dB)
	# Create a noisy image
	normalized_image = normalize(im, True)
	im_noisy = noise_image(normalized_image, noise_snr, tau)
	torch_im_noise = transforms.ToTensor()(im_noisy).unsqueeze(0).to(torch.float)
	print("Noise Torch image:", torch_im_noise.shape)
	print('Input PSNR: %.2f dB'%psnr(im, im_noisy))

	grid_image = get_cos_sine_mgrid([x,y], args.num_sine, noise_channel=args.noise_channel, dim=2).unsqueeze(0)

	print("Torch image:", grid_image.shape)
	inchannel = grid_image.shape[1]
	outchannel = torch_im_noise.shape[1]

	dataset = INR_Single_Image_Dataset(grid_image, torch_im_noise)
	dataloader = DataLoader(dataset, batch_size=args.train_bs)

	model = OINR_2D(inchannel=inchannel, outchannel=outchannel).to(device)

	pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print("Total number of trainable parameters: ", pytorch_total_params)

	loss_fn = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)

	all_losses = {}
	all_losses['train_total_loss'] = []

	best_mse = torch.tensor(float('inf'))
	best_img = None

	print("Start training")
	for epoch in range(args.epoch):
		epoch_total_loss = 0
		n_batches = 0
		for missing_image, noisy_image in tqdm(dataloader, leave=False):
			missing_image = missing_image.to(device)
			noisy_image = noisy_image.to(device)
			x_pred = model(missing_image, verbose=False)
			loss = loss_fn(x_pred, noisy_image)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			epoch_total_loss += loss.item()
			n_batches += 1

			fname = dirname + 'ep_'+str(epoch)+'_bat_'+str(n_batches)+'_combine.png'
			save_image(torch.cat((torch_image.to(device), noisy_image, x_pred),0), fname)

		all_losses['train_total_loss'].append(epoch_total_loss/n_batches)

		# Learning rate scheduler, on training loss for now
		train_loss = epoch_total_loss/n_batches
		if args.scheduler_en:
			scheduler.step(train_loss)

		plot_and_save(args.exp, all_losses)

		if epoch%args.model_pred_save_freq==0:
			torch.save(model, args.exp+'/model_'+str(epoch)+'.pt')

		# Compute current PSNR

		with torch.no_grad():
			mse_epoch = loss_fn(x_pred, torch_image.to(device))
			psnrval = -10*torch.log10(mse_epoch)
		print("Epoch: {}; Train: Total Loss:{}, PSNR:{}, MSE:{}".format(epoch, epoch_total_loss/n_batches, psnrval, mse_epoch))
		
		if (mse_epoch < best_mse) or (epoch == 0):
			best_mse = mse_epoch
			best_img = x_pred.reshape(h, w, c).detach().cpu().numpy()

	torch.save(model, args.exp+'/final_model.pt')
	print('Best PSNR: %.2f dB'%psnr(im, best_img))

if __name__ == '__main__':
	print("Solving denoising image using O-INR")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print("DEvice=",device)
	args = parse_args()
	main(args, device)