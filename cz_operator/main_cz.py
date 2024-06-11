'''
Main file to run O-INR instantiated
as Calderon-Zygmund operator
'''

import os 
import sys 
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import numpy as np
import math 
import pdb
import csv
import pickle
import random
from tqdm import tqdm
from PIL import Image
import argparse

sys.path.append('../')
from util import manual_set_seed, plot_and_save, get_cos_sine_mgrid

from datautils import INR_Single_Image_Dataset
from torch.utils.data import DataLoader
from model import CDE_BCR
	
def parse_args():
	'''
	Parse input arguments
	'''
	parser = argparse.ArgumentParser(description="Arguments for O-INR")

	parser.add_argument("--img_path", type=str, required=True, default=None, help="path to the input image")
	parser.add_argument("--seed", type=int, default=0, help="Setting seed for the entire experiment")
	parser.add_argument("--exp", type=str, default='learn2D_CZ', help="Adjusted in code: Experiment foler name")

	parser.add_argument('--noise_channel', default=True, action='store_false', help='Whether to use noise channel as input')
	parser.add_argument('--num_sine', type=int, default=10, help='number of sine non lienarity for the each x and y dimension')
	parser.add_argument('--image_size', type=int, default=None, help='Size of square image')
	parser.add_argument('--erase_patch', default=False, action='store_true', help='Whether to earse patch in input image to wavelet model')

	# Setting model arguments
	parser.add_argument('--seq_length', type=int, default=None, help='Total seqeunce length in time series')
	parser.add_argument('--wave', default='db2', type=str, help='Type of pywavelet')
	parser.add_argument('--dim_D_out', default=None, type=int, help="Dimension of predicte variable")
	parser.add_argument('--dim_d', default=20, type=int, help="Latent dimension of evolution")
	parser.add_argument('--dim_k', default=20, type=int, help="Dimension of h_theta (first)")
	parser.add_argument('--num_classes', default=1, type=int, help="Output dimensionality of model")
	parser.add_argument('--nonlinearity', default='relu', type=str, help='Non lienarity to use')
	parser.add_argument('--n_levels', default=10, type=int, help="Number of levels of wavelet decomposition")
	parser.add_argument('--K_dense', default=4, type=int, help="Number of dense layers")
	parser.add_argument('--K_LC', default=4, type=int, help="Number of LC layers per level")
	parser.add_argument('--nb', default=3, type=int, help="Diagonal banded length, dertimine the kernel size")
	parser.add_argument('--num_sparse_LC', default=6, type=int, help="Number of sparse LC unit")
	parser.add_argument('--interpol', default='linear', type=str, help='Interpolation type to use')
	parser.add_argument('--use_cheap_sparse_LC', default=True, action='store_false', help='Whether to use sparse Locally connected layer')
	parser.add_argument('--scheduler_en', action='store_true', help='Enable for scheduler')
	parser.add_argument('--debug', action='store_true', help='enable debugging. this will save the images for each update')

	# training arguments
	parser.add_argument('--train_bs', default=1, type=int, help='Batchsize for train loader')
	parser.add_argument('--valid_bs', default=1, type=int, help='Batchsize for valid loader')
	parser.add_argument('--test_bs', default=1, type=int, help='Batchsize for test loader')
	parser.add_argument('--epoch', default=2000, type=int, help='Number of epochs to train')
	parser.add_argument('--lr', default=0.001, type=float, help="Learning rate for the O-INR model")

	parser.add_argument('--model_pred_save_freq', default=200, type=int, help='Saving frequency of model prediction')

		
	args = parser.parse_args()
	return args

def main():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print("Device=", device)
	args = parse_args()
	# set seed randomly to check variance
	if args.seed == -1:
		args.seed = np.random.randint(0,100000)

	arg_dict = vars(args)

	# empty torch cache
	torch.cuda.empty_cache()

	# Setting seeds for reproducibility
	manual_set_seed(args.seed)
	print(args)

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
	if image.size[0] % 2 != 0:
		image = transforms.Pad((1,1,0,0), padding_mode='edge')(image)
	if args.image_size is None:
		args.image_size = image.size[0]
		args.seq_length = image.size[0] * image.size[1]
		args.dim_D_out = len(image.getbands())

	print(image.format)
	print(image.size)
	print(image.mode)
	torch_image = transforms.ToTensor()(np.array(image)).unsqueeze(0)    # shape required: [bs, channel, height, width]
	print("Torch image:", torch_image.shape)

	grid_image = get_cos_sine_mgrid([args.image_size, args.image_size], args.num_sine, noise_channel=args.noise_channel, dim=2).unsqueeze(0)

	print("Torch image:", grid_image.shape)
	inchannel = grid_image.shape[1]

	dataset = INR_Single_Image_Dataset(grid_image, torch_image)
	dataloader = DataLoader(dataset, batch_size=args.train_bs)

	model = CDE_BCR(wave=args.wave,
					D=inchannel, D_out=args.dim_D_out, d=args.dim_d, k=args.dim_k, original_length=args.seq_length, 
					num_classes=args.num_classes, nonlinearity=args.nonlinearity, n_levels=args.n_levels, 
					K_dense=args.K_dense, K_LC=args.K_LC, nb=args.nb, 
					num_sparse_LC=args.num_sparse_LC, use_cheap_sparse_LC=args.use_cheap_sparse_LC, interpol=args.interpol, conv_bias=True, 
					predict=False, masked_modelling=False).to(device)

	pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print("Total number of trainable parameters: ", pytorch_total_params)

	loss_fn = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.epoch, eta_min=5e-4, verbose=False)

	all_losses = {}
	all_losses['train_total_loss'] = []
	all_losses['valid_total_loss'] = []
	all_losses['test_total_loss'] = []

	print("Start training")
	for epoch in range(args.epoch):
		epoch_total_loss = 0
		n_batches = 0
		for missing_image, true_image in tqdm(dataloader, leave=False):
			missing_image = missing_image.to(device)
			true_image = true_image.to(device)

			missing_image = missing_image.contiguous().view(1, inchannel, -1).transpose(1,2)
			x_pred = model(missing_image).transpose(1,2).view(1, args.dim_D_out, args.image_size, args.image_size)
			loss = loss_fn(x_pred, true_image)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			epoch_total_loss += loss.item()
			n_batches += 1

			fname = dirname + 'ep_'+str(epoch)+'_bat_'+str(n_batches)+'_combine.png'
			# save_image(torch.cat((missing_image, true_image, x_pred),0), fname)
			if args.debug:
				save_image(torch.cat((true_image, x_pred),0), fname)

		print("Epoch: {}; Train: Total Loss:{}".format(epoch, epoch_total_loss/n_batches))
		all_losses['train_total_loss'].append(epoch_total_loss/n_batches)

		# Learning rate scheduler, on training loss for now
		train_loss = epoch_total_loss/n_batches
		if args.scheduler_en:
			scheduler.step()

		if args.debug:
			plot_and_save(args.exp, all_losses)

		if epoch%args.model_pred_save_freq==0  and epoch != 0:
			torch.save(model, args.exp+'/model_'+str(epoch)+'.pt')

	# final mse and psnr
	model.eval()
	with torch.no_grad():
		for missing_image, true_image in tqdm(dataloader, leave=False):
			missing_image = missing_image.to(device)
			true_image = true_image.to(device)

			missing_image = missing_image.contiguous().view(1, inchannel, -1).transpose(1,2)
			final_pred = model(missing_image).transpose(1,2).view(1, args.dim_D_out, args.image_size, args.image_size)
			loss = loss_fn(final_pred, true_image)

			# save image
			save_image(true_image, os.path.join(args.exp, f'true_image.png'))
			save_image(final_pred, os.path.join(args.exp, f'pred_image.png'))

			# compute psnr
			psnr = -10*torch.log10(loss.cpu())

			# write psnr
			mdict = {'img_file': args.img_path,
						'wave': args.wave,
						'nonlin': args.nonlinearity,
						'n_levels': args.n_levels,
						'dim_d': args.dim_d,
					 'mse': loss.item(), 
					 'psnr': psnr.item()}
			with open('/nobackup/harsha/pseudo_result/'+exp_name+'/pseudo_results.csv', 'a') as csv_file:
				dict_object = csv.DictWriter(csv_file, fieldnames=mdict.keys()) 
				dict_object.writerow(mdict)

	plot_and_save(args.exp, all_losses)
	# torch.save(model, args.exp+'/final_model.pt')

if __name__ == '__main__':
	main()