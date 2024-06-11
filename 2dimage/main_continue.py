'''
Main file to run O-INR on 2d image
implemented using continuous convolution.
This can handle training using
multiple resolutions of the same image
'''

import os 
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.utils.data import DataLoader

import numpy as np
import csv
import pickle
import random
from tqdm import tqdm
from PIL import Image
import argparse

import sys
sys.path.append('../')
from util import manual_set_seed, plot_and_save, create_multiple_data_v3

from datautils import INR_Multi_Image_Dataset_v3
import continous_model
from continous_model import OINR_cont_2D

def parse_args():
	'''
	Parse input arguments
	'''
	parser = argparse.ArgumentParser(description="Arguments for O-INR")

	parser.add_argument('--img_path', required=True, type=str, help='path to the image')
	parser.add_argument("--seed", type=int, default=0, help="Setting seed for the entire experiment")
	parser.add_argument("--exp", type=str, default='OINR_conti_2D', help="Adjusted in code: Experiment foler name")
	parser.add_argument("--fft", action="store_true", help="whether to use the convolution theorem to accelerate")
	parser.add_argument('--noise_channel', default=True, action='store_false', help='Whether to use noise channel as input')
	parser.add_argument('--num_sine', type=int, default=10, help='number of sine non lienarity for the each x and y dimension')
	parser.add_argument('--min_image_size', type=int, default=256, help='Size of square image')
	parser.add_argument('--max_image_size', type=int, default=360, help='Size of square image')

	# training arguments
	parser.add_argument('--train_bs', default=1, type=int, help='Batchsize for train loader')
	parser.add_argument('--valid_bs', default=1, type=int, help='Batchsize for valid loader')
	parser.add_argument('--test_bs', default=1, type=int, help='Batchsize for test loader')
	parser.add_argument('--epoch', default=400, type=int, help='Number of epochs to train')
	parser.add_argument('--lr', default=0.0005, type=float, help="Learning rate for the O-INR model")

	parser.add_argument('--model_pred_save_freq', default=20, type=int, help='Saving frequency of model prediction')
			
	args = parser.parse_args()
	return args

def main():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print("DEvice=",device)
	args = parse_args()

	# set conv type
	continous_model.conv_cfg['use_fft'] = args.fft
	if args.fft:
		print("Using FFT convolution")
	else:
		print("Using vanilla convolution")

	# set seed randomly to check variance
	if args.seed == -1:
		args.seed = np.random.randint(0,100000)

	arg_dict = vars(args)

	# Setting seeds for reproducibility
	manual_set_seed(args.seed)
	print(args)

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

	grid_list, image_list = create_multiple_data_v3(args.img_path, min_size=args.min_image_size, max_size=args.max_image_size,
													noise_channel=args.noise_channel, num_sine=args.num_sine)
	temp = list(zip(grid_list, image_list))
	random.shuffle(temp)
	grid_list, image_list = zip(*temp)
	grid_list, image_list = list(grid_list), list(image_list)

	inchannel = grid_list[0].shape[0]
	print("Inchannel:",inchannel)
	total_data = len(image_list)

	train_dataset = INR_Multi_Image_Dataset_v3(grid_list[:int(0.8*total_data)], image_list[:int(0.8*total_data)])
	train_loader = DataLoader(train_dataset, batch_size=args.train_bs)
	valid_dataset = INR_Multi_Image_Dataset_v3(grid_list[int(0.8*total_data):int(0.9*total_data)], image_list[int(0.8*total_data):int(0.9*total_data)])
	valid_loader = DataLoader(valid_dataset, batch_size=args.valid_bs)
	test_dataset = INR_Multi_Image_Dataset_v3(grid_list[int(0.9*total_data):], image_list[int(0.9*total_data):])
	test_loader = DataLoader(test_dataset, batch_size=args.test_bs)
	print("Train set size:",len(train_dataset))
	print("Valid set size:",len(valid_dataset))
	print("Test set size:",len(test_dataset))

	model = OINR_cont_2D(inchannel=inchannel, outchannel=3).to(device)

	pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print("Total number of trainable parameters: ", pytorch_total_params)

	loss_fn = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.75, patience=35, verbose=True)

	all_losses = {}
	all_losses['train_total_loss'] = []
	all_losses['valid_total_loss'] = []
	all_losses['test_total_loss'] = []

	print("Start training")
	for epoch in range(args.epoch):
		epoch_total_loss = 0
		n_batches = 0
		for missing_image, true_image in tqdm(train_loader, leave=False):
			missing_image = missing_image.to(device)
			true_image = true_image.to(device)
			x_pred = model(missing_image)
			loss = loss_fn(x_pred, true_image)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			epoch_total_loss += loss.item()
			n_batches += 1

			if epoch%args.model_pred_save_freq==0:
				# save only first one for train
				fname = dirname + 'ep_'+str(epoch)+'_bat_'+str(n_batches)+'_combine.png'
				save_image(torch.cat((true_image, x_pred),0), fname)
				# save_image(torch.cat((missing_image, true_image, x_pred),0), fname)

		print("Epoch: {}; Train: Total Loss:{}".format(epoch, epoch_total_loss/n_batches))
		all_losses['train_total_loss'].append(epoch_total_loss/n_batches)

		epoch_total_loss = 0
		n_batches = 0
		with torch.no_grad():
			model.eval()			
			for missing_image, true_image in tqdm(valid_loader, leave=False):
				missing_image = missing_image.to(device)
				true_image = true_image.to(device)
				x_pred = model(missing_image)
				loss = loss_fn(x_pred, true_image)

				epoch_total_loss += loss.item()
				n_batches += 1

				if epoch%args.model_pred_save_freq==0:
					# save only first one for train
					fname = dirname + 'valid_ep_'+str(epoch)+'_bat_'+str(n_batches)+'_combine.png'
					save_image(torch.cat((true_image, x_pred),0), fname)
					# save_image(torch.cat((missing_image, true_image, x_pred),0), fname)

		print("\t Valid: Total Loss:{}".format(epoch_total_loss/n_batches))
		all_losses['valid_total_loss'].append(epoch_total_loss/n_batches)
		scheduler.step(epoch_total_loss/n_batches)

		epoch_total_loss = 0
		n_batches = 0
		with torch.no_grad():
			model.eval()
			for missing_image, true_image in tqdm(test_loader, leave=False):
				missing_image = missing_image.to(device)
				true_image = true_image.to(device)
				x_pred = model(missing_image)
				loss = loss_fn(x_pred, true_image)

				epoch_total_loss += loss.item()
				n_batches += 1

				if epoch%args.model_pred_save_freq==0 or epoch==args.epoch-1:
					fname = dirname + 'test_ep_'+str(epoch)+'_bat_'+str(n_batches)+'_combine.png'
					save_image(torch.cat((true_image, x_pred),0), fname)
					# save_image(torch.cat((missing_image, true_image, x_pred),0), fname)

		print("\t  Test: Total Loss:{}".format(epoch_total_loss/n_batches))
		all_losses['test_total_loss'].append(epoch_total_loss/n_batches)

		plot_and_save(args.exp, all_losses)

		if epoch%args.model_pred_save_freq==0:
			# torch.save(model.state_dict(), args.exp+'/model_'+str(epoch)+'.pt')
			torch.save(model, args.exp+'/model_'+str(epoch)+'.pt')

	# torch.save(model.state_dict(), args.exp+'/final_model.pt')
	torch.save(model, args.exp+'/final_model.pt')

if __name__ == '__main__':
	main()