'''
Main file to run O-INR on sequential data
'''

import torch 
import numpy as	np 
import os
import argparse
from torchvision.utils import save_image

import csv

import pickle
from tqdm import tqdm
import matplotlib.pyplot as	plt

import torch.nn	as nn
from datasets import manual_set_seed, plot_and_save, Sequence_Dataset_v2, Sequence_Dataset_h
from model import SequenceNet
from torch.utils.data import DataLoader
import logging

def	parse_args():
	'''
	Parse input	arguments
	'''
	parser = argparse.ArgumentParser(description="Arguments	for	O-INR")

	parser.add_argument("--seed", type=int,	default=0, help="Setting seed for the entire experiment")
	parser.add_argument("--exp", type=str, default='adni', help="Adjusted in code: Experiment foler	name")
	parser.add_argument('--basedir', type=str, default='./result/')
	parser.add_argument('--dataset', type=str, default='adni', help='Video dataset name')
	parser.add_argument('--video_path',	type=str, default=None,	help='path to the video	file')

	parser.add_argument('--frame_gap', type=int, default=None, help='Number	of frames to skip while	sample using Sequence_Dataset_v4')
	parser.add_argument('--noise_channel', default=True, action='store_false', help='Whether to	use	noise channel as input')
	parser.add_argument('--num_sine', type=int,	default=10,	help='number of	sine non lienarity for the each	x and y	dimension')
	parser.add_argument('--desired_w', type=int, default=None, help='desired width of sequence')
	parser.add_argument('--desired_h', type=int, default=None, help='desired height	of sequence')
	parser.add_argument('--desired_len', type=int, default=None, help='desired length of sequence')

	# training arguments
	parser.add_argument('--train_bs', default=8, type=int, help='Batchsize for train loader')
	parser.add_argument('--epoch', default=1000, type=int,	help='Number of	epochs to train')
	parser.add_argument('--lr',	default=0.001, type=float, help="Learning rate for the O-INR")
	parser.add_argument('--scheduler_en', action='store_true', help='Enable	for	scheduler')

	parser.add_argument('--model_pred_save_freq', default=100,	type=int, help='Saving frequency of	model prediction')
	parser.add_argument('--data_save_freq',	default=100, type=int,	help='Saving frequency of model	output')
	parser.add_argument('--check_loss_every', type=int,	default=None, help='check loss at a	multiple of	this number	of epochs')
	parser.add_argument('--sequence_dataset', type=str,	default='Sequence_Dataset_h', help='dataset	to be used')

	args = parser.parse_args()
	return args


def	main(args, device):
	logger = logging.getLogger(__name__)

	# Setting seeds	for	reproducibility
	manual_set_seed(args.seed)
	print(args)

	arg_dict = vars(args)

	exp_name = args.exp

	if os.path.exists(args.basedir+exp_name):
		nth_exp	= len(os.listdir(args.basedir+exp_name+'/Results'))+1
	else:
		nth_exp	= 0
	args.exp = args.basedir+exp_name+'/Results/'+str(nth_exp)
	arg_dict['Result_location']	= './Results/'+str(nth_exp)

	if not os.path.exists(args.exp):
		print("Creating	experiment directory: ", args.exp)
		os.makedirs(args.exp)

	# logging file setup
	logging.basicConfig(filename=os.path.join(args.exp,	'logfile.log'),
						format='%(asctime)s	- %(levelname)s	- %(name)s - %(message)s',
						datefmt='%m/%d/%Y %H:%M:%S',
						level=logging.INFO)
	logger.info(f'currently	running	file: {args.video_path}')

	#	list of	column names 
	field_names	= arg_dict.items()
	with open(args.basedir+exp_name+'/experiment.csv', 'a')	as csv_file:
		dict_object	= csv.DictWriter(csv_file, fieldnames=field_names, extrasaction='ignore') 
		dict_object.writerow(arg_dict)

	argument_file =	args.exp+'/arguments.pkl'
	with open(argument_file, 'wb') as f:
		pickle.dump(arg_dict, f)

	#create	storage	for	expt res
	dirname	= args.exp+'/model_prediction/'
	if not os.path.exists(dirname):
		os.makedirs(dirname)

	# log args
	for	_k,_v in vars(args).items():
		logger.info(f'{_k},{_v}')

	video_path = args.video_path
	if args.sequence_dataset ==	'Sequence_Dataset_v2':
		dataset	= Sequence_Dataset_v2(video_path, args.num_sine, args.desired_h, args.desired_w, args.desired_len)
	elif args.sequence_dataset == 'Sequence_Dataset_h':
		dataset	= Sequence_Dataset_h(video_path, args.num_sine, args.desired_h, args.desired_w, args.desired_len, frame_gap=args.frame_gap,	datasetName=args.dataset)

	print("Length:",len(dataset))
	logger.info(f"Length:{len(dataset)}")
	inchannel =	dataset[0][0].shape[0]
	print("Inchannel:",inchannel)
	logger.info(f"Inchannel:{inchannel}")
	dataloader = DataLoader(dataset, batch_size=args.train_bs)

	outchannel = dataset[0][1].shape[0]
	if 'feta' in args.dataset or 'adni'	in args.dataset:
		outchannel = 1
	
	model =	SequenceNet(inchannel=inchannel, outchannel=outchannel).to(device)
	print(model)
	logger.info(model)

	pytorch_total_params = sum(p.numel() for p in model.parameters() if	p.requires_grad)
	print("Total number	of trainable parameters: ",	pytorch_total_params)
	logger.info(f"Total	number of trainable	parameters:	{pytorch_total_params}")

	loss_fn	= nn.MSELoss()
	optimizer =	torch.optim.Adam(model.parameters(), lr=args.lr)
	scheduler =	torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max	= args.epoch, eta_min=5e-4,	verbose=False)

	all_losses = {}
	all_losses['train_total_loss'] = []
	all_losses['valid_total_loss'] = []
	all_losses['test_total_loss'] =	[]

	print("Start training")
	logger.info("Start training")

	best_train_loss	= 100
	best_train_epoch = 0
	for	epoch in range(args.epoch):
		epoch_total_loss = 0
		n_batches =	0
		for	missing_image, true_image in dataloader:
			missing_image =	missing_image.to(device)
			true_image = true_image.to(device)
			x_pred = model(missing_image, verbose=False)
			loss = loss_fn(x_pred, true_image)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			epoch_total_loss +=	loss.item()
			n_batches += 1

			if epoch%args.data_save_freq==0:
				fname =	dirname	+ 'ep_'+str(epoch)+'_bat_'+str(n_batches)+'_combine.png'
				save_image(torch.cat((true_image, x_pred),0), fname)

		if args.scheduler_en:
			scheduler.step()

		if args.check_loss_every is	not	None:
			if epoch % args.check_loss_every ==	0:
				curr_epoch_loss	= epoch_total_loss / n_batches
				if curr_epoch_loss < best_train_loss:
					best_train_loss	= curr_epoch_loss 
					best_train_epoch = epoch
					torch.save(model, args.exp+'/model_best.pt')

		print("Epoch: {}; Train: Total Loss:{}".format(epoch, epoch_total_loss/n_batches))
		logger.info("Epoch:	{};	Train: Total Loss:{}".format(epoch,	epoch_total_loss/n_batches))
		all_losses['train_total_loss'].append(epoch_total_loss/n_batches)

		plot_and_save(args.exp,	all_losses)

		if epoch%args.model_pred_save_freq==0:
			torch.save(model, args.exp+'/model_'+str(epoch)+'.pt')

	torch.save(model, args.exp+'/final_model.pt')
	logger.info(f'best train loss  = {best_train_loss} at epoch	= {best_train_epoch}')
	result_log = {'best_train_loss'	: best_train_loss, 'best_train_epoch': best_train_epoch}
	with open(os.path.join(args.exp, 'final_results.pkl'), 'wb') as	fie:
		pickle.dump(result_log,	fie)

if __name__	== '__main__':
	import sys
	sys.path.append('..')
	print("Frame sequence")
	logger = logging.getLogger(__name__)

	device = torch.device("cuda" if	torch.cuda.is_available() else "cpu")
	print("DEvice=",device)
	args = parse_args()
	# set seed randomly	to check variance
	if args.seed ==	-1:
		args.seed =	np.random.randint(0,100000)

	main(args, device)