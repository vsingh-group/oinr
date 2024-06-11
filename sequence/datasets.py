'''
File describing dataset class for sequential data to be used with O-INR
'''

import torch 
import numpy as np 
import csv
import glob
import math
import os
import random
import pickle
import pdb
import matplotlib.colors as colors
import numpy as np
import scipy.io.wavfile as wavfile
import skvideo
import scipy.ndimage
import scipy.special
import skimage
import skimage.filters
import skvideo.io
from PIL import Image
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize

def get_cos_sine_mgrid(frame_num, sidelen, num_sine, noise_channel=False, dim=2):
    '''
    Here the ouput is positional_ecncoding/frame_num
    '''
    tensors = (torch.linspace(-1, 1, steps=sidelen[0]), torch.linspace(-1, 1, steps=sidelen[1]))
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    
    x_grid = mgrid[:, :, 1]
    y_grid = mgrid[:, :, 0]
    
    all_x = []
    all_y = []
    for i in range(num_sine):
        # add sine value
        x_value = torch.sin((2**i)*math.pi*x_grid)
        y_value = torch.sin((2**i)*math.pi*y_grid)
        all_x.append(x_value)
        all_y.append(y_value)
        # add cosine value
        x_value = torch.cos((2**i)*math.pi*x_grid)
        y_value = torch.cos((2**i)*math.pi*y_grid)
        all_x.append(x_value)
        all_y.append(y_value)       
        
    all_x = torch.stack(all_x)
    all_y = torch.stack(all_y)
    mgrid = torch.cat((all_x, all_y), dim=0)    
    
    if noise_channel:
        noise = torch.randn((sidelen, sidelen)).unsqueeze(0)
        mgrid = torch.cat((mgrid, noise), dim=0)
    else:
        mgrid = mgrid
    mgrid = mgrid/frame_num
    return mgrid


class Sequence_Dataset(Dataset):
    '''
    Here the ouput is positional_ecncoding/frame_num
    '''
    def __init__(self, video_path, start_N, num_sine, desired_h, desired_w, desired_len):
        self.video_path = video_path
        self.num_sine = num_sine
        self.start_N = start_N

        if 'npy' in video_path:
            self.video = np.load(video_path)
        elif 'mp4' in video_path:
            self.video = skvideo.io.vread(video_path).astype(np.single) / 255.
        self.dimension = self.video.shape[:-1]
        self.channels = self.video.shape[-1]

        self.video = torch.from_numpy(self.video).permute(0,3,1,2)
        print("Video shape:",self.video.shape)
        self.nframe = self.video.shape[0]

        h = self.video.shape[2]
        w = self.video.shape[3]
        print("Original dimension:", h, w)

        self.desired_h = desired_h
        self.desired_w = desired_w
        self.desired_len = desired_len
        assert desired_len < self.nframe
        print("Desired dimension:", desired_h, desired_w)
        print("Desired length:", desired_len)

        self.transform = Resize((desired_h,desired_w))

        self.grid_seq = []
        for i in range(start_N, start_N+desired_len):
            grid = get_cos_sine_mgrid(i, (desired_h, desired_w), num_sine, noise_channel=False, dim=2)
            self.grid_seq.append(grid.unsqueeze(0))

        self.grid = torch.cat(self.grid_seq)
        self.image = self.video[:desired_len, :, :, :]

    def __getitem__(self, idx):
        image = self.transform(self.image[idx])
        return self.grid[idx], image

    def __len__(self):
        return len(self.image)



def get_cos_sine_mgrid_v2(frame_num, sidelen, num_sine, noise_channel=False, dim=2):
    '''
    Here the ouput is positional_ecncoding + frame_num
    '''
    tensors = (torch.linspace(-1, 1, steps=sidelen[0]), torch.linspace(-1, 1, steps=sidelen[1]))
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    
    x_grid = mgrid[:, :, 1]
    y_grid = mgrid[:, :, 0]
    
    all_x = []
    all_y = []
    for i in range(num_sine):
        # add sine value
        x_value = torch.sin((2**i)*math.pi*x_grid)
        y_value = torch.sin((2**i)*math.pi*y_grid)
        all_x.append(x_value)
        all_y.append(y_value)
        # add cosine value
        x_value = torch.cos((2**i)*math.pi*x_grid)
        y_value = torch.cos((2**i)*math.pi*y_grid)
        all_x.append(x_value)
        all_y.append(y_value)       
        
    all_x = torch.stack(all_x)
    all_y = torch.stack(all_y)
    mgrid = torch.cat((all_x, all_y), dim=0)    
    
    if noise_channel:
        noise = torch.randn((sidelen, sidelen)).unsqueeze(0)
        mgrid = torch.cat((mgrid, noise), dim=0)
    else:
        mgrid = mgrid
    mgrid = mgrid + frame_num
    return mgrid


class Sequence_Dataset_v2(Dataset):
    '''
    Here we choose the first n frame
    '''
    def __init__(self, video_path, num_sine, desired_h, desired_w, desired_len):
        print("Sequence2")
        self.video_path = video_path
        self.num_sine = num_sine

        if 'npy' in video_path:
            self.video = np.load(video_path)
        elif 'mp4' in video_path:
            self.video = skvideo.io.vread(video_path).astype(np.single) / 255.
        elif 'mat' in video_path:
            from scipy import io
            self.video = io.loadmat(video_path)['hypercube'][...,None]
        
        self.dimension = self.video.shape[:-1]
        self.channels = self.video.shape[-1]

        self.video = torch.from_numpy(self.video).permute(0,3,1,2)
        print("Video shape:",self.video.shape)
        self.nframe = self.video.shape[0]

        h = self.video.shape[2]
        w = self.video.shape[3]
        print("Original dimension:", h, w)

        self.desired_h = desired_h
        self.desired_w = desired_w
        self.desired_len = desired_len
        assert desired_len < self.nframe
        print("Desired dimension:", desired_h, desired_w)
        print("Desired length:", desired_len)

        self.transform = Resize((desired_h,desired_w))

        self.grid_seq = []
        for i in np.linspace(0,0.1,desired_len):
            grid = get_cos_sine_mgrid_v2(i, (desired_h, desired_w), num_sine, noise_channel=False, dim=2)
            self.grid_seq.append(grid.unsqueeze(0))

        self.grid2 = torch.cat(self.grid_seq)
        self.image2 = self.video[:desired_len, :, :, :]

        # self.grid = self.grid2[::2,...]
        # self.image = self.image2[::2,:,:]
        self.grid = self.grid2
        self.image = self.image2

    def __getitem__(self, idx):
        image = self.transform(self.image[idx])
        return self.grid[idx], image

    def __len__(self):
        return len(self.image)

class Sequence_Dataset_v4(Dataset):
    '''
    Here we choose n frames equally spaced, where the default space is 10 
    '''
    def __init__(self, video_path, num_sine, desired_h, desired_w, desired_len, frame_gap=10):
        print("Sequence4")
        self.video_path = video_path
        self.num_sine = num_sine

        if 'npy' in video_path:
            self.video = np.load(video_path)
        elif 'mp4' in video_path:
            self.video = skvideo.io.vread(video_path).astype(np.single) / 255.
        self.dimension = self.video.shape[:-1]
        self.channels = self.video.shape[-1]

        self.video = torch.from_numpy(self.video).permute(0,3,1,2)
        print("Video shape:",self.video.shape)
        self.nframe = self.video.shape[0]

        h = self.video.shape[2]
        w = self.video.shape[3]
        print("Original dimension:", h, w)

        self.desired_h = desired_h
        self.desired_w = desired_w
        self.desired_len = desired_len
        assert desired_len < self.nframe
        print("Desired dimension:", desired_h, desired_w)
        print("Desired length:", desired_len)

        self.transform = Resize((desired_h,desired_w))

        self.grid_seq = []
        for i in np.linspace(0,0.1,desired_len):
            grid = get_cos_sine_mgrid_v2(i, (desired_h, desired_w), num_sine, noise_channel=False, dim=2)
            self.grid_seq.append(grid.unsqueeze(0))

        self.grid = torch.cat(self.grid_seq)

        self.image_seqe = []
        for i in range(self.nframe):
            if i%frame_gap == 0:
                print(i)
                self.image_seqe.append(self.video[i, :, :, :].unsqueeze(0))
            if len(self.image_seqe) == desired_len:
                break

        self.image = torch.cat(self.image_seqe)

    def __getitem__(self, idx):
        image = self.transform(self.image[idx])
        return self.grid[idx], image

    def __len__(self):
        return len(self.image)


class Sequence_Dataset_v3(Dataset):
    '''
    Here we choose the first n frame
    The difference with Sequence_Dataset_v2 is that we
    are varying the partinioninig of time differently
    '''
    def __init__(self, video_path, num_sine, desired_h, desired_w, desired_len):
        print("Sequence3")
        self.video_path = video_path
        self.num_sine = num_sine

        if 'npy' in video_path:
            self.video = np.load(video_path)
        elif 'mp4' in video_path:
            self.video = skvideo.io.vread(video_path).astype(np.single) / 255.
        self.dimension = self.video.shape[:-1]
        self.channels = self.video.shape[-1]

        self.video = torch.from_numpy(self.video).permute(0,3,1,2)
        print("Video shape:",self.video.shape)
        self.nframe = self.video.shape[0]

        h = self.video.shape[2]
        w = self.video.shape[3]
        print("Original dimension:", h, w)

        self.desired_h = desired_h
        self.desired_w = desired_w
        self.desired_len = desired_len
        assert desired_len < self.nframe
        print("Desired dimension:", desired_h, desired_w)
        print("Desired length:", desired_len)

        self.transform = Resize((desired_h,desired_w))

        self.grid_seq = []
        for i in np.linspace(0,0.4,4*desired_len):
            grid = get_cos_sine_mgrid_v2(i, (desired_h, desired_w), num_sine, noise_channel=False, dim=2)
            self.grid_seq.append(grid.unsqueeze(0))

        self.grid = torch.cat(self.grid_seq)
        self.image = self.video[:desired_len, :, :, :]

    def __getitem__(self, idx):
        image = self.transform(self.image[idx])
        return self.grid[idx], image

    def __len__(self):
        return len(self.image)

def manual_set_seed(seed):
    print("Setting all seeds to: ", seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_and_save(exp, all_losses):
    loss_file = exp+'/all_losses.pkl'
    with open(loss_file, 'wb') as f:
        pickle.dump(all_losses, f)

    plt.plot(all_losses['train_total_loss'], label='Training')
    plt.plot(all_losses['test_total_loss'], label='Test')   
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Total loss")
    plt.savefig(exp+'/TotalLoss.png')
    plt.cla()
    plt.clf()
    plt.close()

    plt.semilogy(all_losses['train_total_loss'], label='Training')
    plt.semilogy(all_losses['test_total_loss'], label='Test')
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Total loss log")
    plt.savefig(exp+'/TotalLossLog.png')
    plt.cla()
    plt.clf()
    plt.close()

class Sequence_Dataset_h(Dataset):
        '''
        Here we choose the first n frame
        '''
        def __init__(self, video_path, num_sine, desired_h, desired_w, desired_len, frame_gap=None, datasetName=None):
                print("Sequence_h ")
                self.video_path = video_path
                self.num_sine = num_sine

				# to make skvideo work
                np.float = np.float64
                np.int = np.int_
                if 'npy' in video_path:
                        self.video = np.load(video_path).astype(np.float32)
                        if datasetName == 'adni':
                                self.video = np.flip(np.moveaxis(self.video, [0,1,2], [2,0,1]), axis = 1)
                        if self.video.max() > 1.0:
                                self.video = self.video / 255.
                        if len(self.video.shape) == 3:
                                self.video = self.video[...,None]
                elif 'mp4' in video_path or 'gif' in video_path or 'avi' in video_path:
                        self.video = skvideo.io.vread(video_path).astype(np.single) / 255.
                elif 'mat' in video_path:
                        from scipy import io
                        self.video = io.loadmat(video_path)['hypercube'][...,None]

                self.dimension = self.video.shape[:-1]
                self.channels = self.video.shape[-1]

                self.video = torch.from_numpy(self.video).permute(0,3,1,2)
                print("Video shape:",self.video.shape)
                self.nframe = self.video.shape[0]

                h = self.video.shape[2]
                w = self.video.shape[3]
                print("Original dimension:", h, w)

                if desired_h is not None:
                        self.desired_h = desired_h
                        self.desired_w = desired_w
                        self.desired_len = desired_len
                        self.transform = Resize((desired_h,desired_w))

                        assert desired_len < self.nframe
                        print("Desired dimension:", desired_h, desired_w)
                        print("Desired length:", desired_len)
                else:
                        self.desired_h = h
                        self.desired_w = w
                        self.desired_len = self.nframe
                        self.transform = None

                        print("Desired dimension:", self.desired_h, self.desired_w)

                # our working sine cos positional encoding
                self.grid_seq = []
                for i in np.linspace(0,0.1,self.desired_len):
                        grid = get_cos_sine_mgrid_v2(i, (self.desired_h, self.desired_w), num_sine, noise_channel=False, dim=2)
                        self.grid_seq.append(grid.unsqueeze(0))

                self.grid2 = torch.cat(self.grid_seq)
                self.image2 = self.video[:self.desired_len, :, :, :]

                if frame_gap is None:
                        self.grid = self.grid2
                        self.image = self.image2
                else:
                        self.grid = self.grid2[::frame_gap,...]
                        self.image = self.image2[::frame_gap,:,:]

        def __getitem__(self, idx):
                if self.transform is not None:
                        image = self.transform(self.image[idx])
                else:
                        image = self.image[idx]
                return self.grid[idx], image

        def __len__(self):
                return len(self.image)

