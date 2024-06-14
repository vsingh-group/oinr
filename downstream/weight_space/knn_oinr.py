'''
Main file to perform 
nearest neighbour classification
on the trained and vectorized OINR
'''

import torch  
import numpy as np
import pickle as pkl
from sklearn.neighbors import KNeighborsClassifier
import argparse

def parse_args():
	'''
	Parse input arguments
	'''
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_file", type=str, required=True, default=None, help="path to the vectorized train and test data for classification")
	parser.add_argument("--nclass", type=int, default=10, help="number of class in the data")
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	args = parse_args()
	fname = args.data_file
	with open(fname, 'rb') as h:
		mdict = pkl.load(h)

	clf = KNeighborsClassifier(n_neighbors=args.nclass)
	clf.fit(mdict['train_data_npy'][:10000], mdict['train_labels_npy'][:10000])      # fitting with 10k is sufficient, full data fitting takes more time

	preds = clf.predict(mdict['test_data_npy'])
	true_lbls = mdict['test_labels_npy']

	Accuracy = (preds == true_lbls).sum()/ len(true_lbls)
	print("Test Accuracy:", Accuracy)