'''
Main file to vectorize the
trained OINR of a given dataset.
It requires the data has been already partitined into train and test set
'''

import torch
import os
import argparse
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle as pkl

# Dataset class to process the trained OINR
# and vectorize them
class CustomDataset(Dataset):
	def __init__(self, data_dir, means_var=None):
		self.data_dir = data_dir
		if means_var is not None:
			self.mean = means_var['mean']
			self.std = means_var['std']
		else: 
			self.mean = None
			self.var = None

		self.data_files = self._get_data_files()

	def _get_data_files(self):
		# Build a list of file paths with supported extensions 
		data_files = []
		for cls_dir in os.listdir(self.data_dir):
			cls_path = os.path.join(self.data_dir, cls_dir)
			for _o in os.listdir(cls_path):
				data_path = os.path.join(cls_path, _o)
				data_files.append((data_path, int(cls_dir)))

		return data_files

	def __getitem__(self, index):
		file_path, lbl = self.data_files[index]

		oinr = torch.load(file_path)              # loading state dict
		vectorized_wts = []
		for v in oinr.values(): 
			vectorized_wts.append(v.ravel())      # append flattend parameter

		data = torch.cat(vectorized_wts)
		if self.mean is not None:
			data = (data - self.mean) / self.std

		return data,lbl

	def __len__(self):
		return len(self.data_files)
	

def parse_args():
	'''
	Parse input arguments
	'''
	parser = argparse.ArgumentParser()

	parser.add_argument("--train_path", type=str, required=True, default=None, help="path to the directory containing trained OINR for training")
	parser.add_argument("--test_path", type=str, required=True, default=None, help="path to the directory containing trained OINR for testing")
	parser.add_argument("--exp_name", type=str, required=True, default=None, help="dataset or experiment name")
	
	args = parser.parse_args()
	return args


if __name__ == '__main__':

	args = parse_args()
	train_dir = args.train_path 
	trainset = CustomDataset(data_dir=train_dir)
	trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4) 
	test_dir = args.test_path 
	testset = CustomDataset(data_dir=test_dir)
	testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=4) 

	train_data = []
	train_labels = []
	for d,l in trainloader:
		train_data.append(d)
		train_labels.append(l)

	train_data_cat = torch.cat(train_data, dim=0)
	train_labels_cat = torch.cat(train_labels)
	print(train_data_cat.shape)
	print(train_labels_cat.shape)

	test_data = []
	test_labels = []
	for d,l in testloader:
		test_data.append(d)
		test_labels.append(l)

	test_data_cat = torch.cat(test_data, dim=0)
	test_labels_cat = torch.cat(test_labels)
	print(test_data_cat.shape)
	print(test_labels_cat.shape)

	mdict = {}
	mdict['train_data_npy'] = train_data_cat.numpy()
	mdict['train_labels_npy'] = train_labels_cat.numpy()
	mdict['test_data_npy'] = test_data_cat.numpy()
	mdict['test_labels_npy'] = test_labels_cat.numpy()

	dirname = './results/'
	if not os.path.exists(dirname):
		os.makedirs(dirname)

	fname = dirname + args.exp_name + '.pkl'
	with open(fname, 'wb') as h:
		pkl.dump(mdict, h)