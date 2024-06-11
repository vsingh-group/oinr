'''
Main file to run O-INR on 3d volumetric data
'''

import os
from tqdm import tqdm

import numpy as np
from scipy import io
from scipy import ndimage

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import re
import logging
import argparse

from model import OINR_3D

import sys
sys.path.append('../')
from util import get_cos_sine_mgrid_3d, manual_set_seed
from datautils import INR_Single_Image_Dataset

from utils_3D import get_IoU, march_and_save

def parse_args():
	'''
	Parse input arguments
	'''
	parser = argparse.ArgumentParser(description="Arguments for O-INR")

	parser.add_argument('--img_path', required=True, type=str, help="path to 3D volume")
	parser.add_argument("--seed", type=int, default=0, help="Setting seed for the entire experiment")
	parser.add_argument("--exp", type=str, default='INR_3D', help="Adjusted in code: Experiment foler name")
	parser.add_argument('--full_volume', default=False, action='store_true', help='Full volume, can be memory intensive')
	parser.add_argument('--basedir', type=str, default='./results')

	parser.add_argument('--noise_channel', action='store_true', help='Whether to use noise channel as input')
	parser.add_argument('--num_sine', type=int, default=8, help='number of sine non lienarity for the each x and y dimension')
	parser.add_argument("--slice_width", type=int, default=80, help="width of each slice for the dataset")

	# training arguments
	parser.add_argument('--epoch', default=1000, type=int, help='Number of epochs to train')
	parser.add_argument('--lr', default=0.005, type=float, help="Learning rate for the O-INR model")
	parser.add_argument('--scheduler_en', action='store_true', help='Enable for scheduler')

	args = parser.parse_args()
	return args


def main(args):

	manual_set_seed(args.seed)
	
	scale = 1.0                 # Run at lower scales to testing
	mcubes_thres = 0.5          # Threshold for marching cubes
		
	exp_name = args.exp
	print(args)
	
	if os.path.exists(args.basedir+exp_name):
		nth_exp = len(os.listdir(args.basedir+exp_name))+1
	else:
		nth_exp = 0
	img_name = re.split(r'/|\.',args.img_path)[-2] + '_'
	args.exp = args.basedir + exp_name + "/" + img_name + '_e' + str(args.epoch) + '_' + str(nth_exp) + '/'
	if not os.path.exists(args.exp):
		print("Creating experiment directory: ", args.exp)
		os.makedirs(args.exp)

	logging.basicConfig(filename=args.exp + 'logfile.log', format='%(asctime)s %(message)s', level=logging.INFO) 
	logging.info('-----------------------------EXP start -----------------------------')

	# Load image and scale
	im = io.loadmat(args.img_path)['hypercube'].astype(np.float32)
	im = ndimage.zoom(im/im.max(), [scale, scale, scale], order=0) # im is binary: {0, 255}

	# If the volume is an occupancy, clip to tightest bounding box
	hidx, widx, tidx = np.where(im > 0.99)
	im = im[hidx.min():hidx.max(),
			widx.min():widx.max(),
			tidx.min():tidx.max()]
		
	imten = torch.tensor(im)
	_x,_y,_z = imten.shape
	
	# pad the image to make it even
	if _x %2 != 0:
		imten = torch.cat((imten, imten[-1,:,:].unsqueeze(0)),dim = 0)
	if _y %2 != 0:
		imten = torch.cat((imten, imten[:,-1,:].unsqueeze(1)),dim = 1)
	if _z %2 != 0:
		imten = torch.cat((imten, imten[:,:,-1].unsqueeze(2)),dim = 2)

	print(imten.shape)
	H,W,T = imten.shape
	grid_image = get_cos_sine_mgrid_3d([H,W,T], args.num_sine, False, 3) 
	inchannel = grid_image.shape[0]
	print("Inchannel:",inchannel)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = OINR_3D(inchannel=inchannel, outchannel=1).to(device)

	pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print("Total number of trainable parameters: ", pytorch_total_params)

	loss_fn = torch.nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
	
	if not args.full_volume:
		grid_image = grid_image.unsqueeze(0)
		imten = imten.unsqueeze(0)
		grid_image = grid_image[:, :, 100:180, 180:260, :]
		imten = imten[:, 100:180, 180:260, :]
		print("Final grid shape:",grid_image.shape)
		print("Final 3d shape:",imten.shape)	
		dt = INR_Single_Image_Dataset(grid_image, imten)
		dl = DataLoader(dt, batch_size=1, shuffle=False)
	else:
		gimage_list = []
		imten_list = []
		i = 0
		while i < W:
			temp_grid = grid_image[:, :, i:min(W,i+args.slice_width), :]
			temp_img = imten[:, i:min(W,i+args.slice_width), :]
			i+= args.slice_width
			print(temp_grid.shape)
			print(temp_img.shape)
			gimage_list.append(temp_grid)
			imten_list.append(temp_img)
		dt = INR_Single_Image_Dataset(gimage_list, imten_list)
		dl = DataLoader(dt, batch_size=1, shuffle=False)


	losses = []
	losses_per_epoch = []
	for epoch in range(args.epoch):
		curr_epoch_loss = []
		epoch_total_loss = 0
		n_batches = 0
		for inp, outp in tqdm(dl, leave=False):
			inp, outp = inp.to(device), outp.to(device)
			preds = model(inp)
			loss = loss_fn(preds.squeeze(1), outp)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			losses.append(loss.item())
			curr_epoch_loss.append(loss.item())

			epoch_total_loss += loss.item()
			n_batches += 1

		print("Epoch:",epoch, epoch_total_loss/n_batches)
		
		losses_per_epoch.append(np.mean(curr_epoch_loss))

	plt.plot(losses)
	plt.savefig(args.exp + 'loss_vs_batch.png')
	plt.close()
	# plt.show()
	plt.plot(losses_per_epoch)
	plt.savefig(args.exp + 'loss_vs_epoch.png')
	plt.close()

	# inference
	outs = []
	dl_final = DataLoader(dt, batch_size=1, shuffle=False)
	with torch.no_grad():
		for inp, outp in dl_final:
			inp, outp = inp.to(device), outp.to(device)
			preds = model(inp)
			outs.append(preds.squeeze(1))

	final_out = torch.cat(outs, dim=2)
	print(final_out.shape)
	print(imten.shape)

	iou_error = get_IoU(final_out, imten, mcubes_thres)
	print(iou_error)

	march_and_save(final_out.squeeze().detach().cpu().numpy(), mcubes_thres, args.exp + '/pred.dae', True)

	logging.info(args.__dict__)
	logging.info(f"iou_error: {iou_error.item()}")
	pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	logging.info(f'num model params = {pytorch_total_params}')

if __name__ == '__main__':
	args = parse_args()
	main(args)
