import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import os, shutil, json
import argparse

import torchvision.models as models
import torch.nn.functional as F

from tools.Trainer import ModelNetTrainer
from tools.ImgDataset import MultiviewImgDataset, SingleImgDataset
from models.MVCNN import MVCNN, SVCNN

import torchvision
from tqdm import tqdm

from pdb import set_trace

parser = argparse.ArgumentParser()
parser.add_argument('-name', '--name', type=str, help='Name of the experiment', default='mvcnn')
parser.add_argument('-bs', '--batchSize', type=int, help='Batch size for the second stage', default=8) # there will be *12 images in each batch for mvcnn
parser.add_argument('-num_models', type=int, help='number of models per class', default=1000)
parser.add_argument('-lr', type=float, help='learning rate', default=5e-5)
parser.add_argument('-weight_decay', type=float, help='weight decay', default=0.0)
parser.add_argument('-no_pretraining', dest='no_pretraining', action='store_true')
parser.add_argument('-cnn_name', '--cnn_name', type=str, help='cnn model name', default='vgg11')
parser.add_argument('-num_views', type=int, help='number of views', default=12)
parser.add_argument('-train_path', type=str, default='modelnet40_images_new_12x/*/train')
parser.add_argument('-val_path', type=str, default='modelnet40_images_new_12x/*/test')
parser.set_defaults(train=False)

parser.add_argument('-ct', '--constraint', type=str, default=None)

parser.add_argument('-prefix', type=str, default='./')
#parser.add_argument('-prefix', type=str, default='/vulcan/scratch/yxren/mvcnn/')

if __name__ == '__main__':
	args = parser.parse_args()
	
	selected_dir = './selected/'
	
	n_models_train = args.num_models * args.num_views
	
	train_dataset = MultiviewImgDataset(args.train_path, scale_aug=False, rot_aug=False, num_models=n_models_train, num_views=args.num_views)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=False, num_workers=0) # shuffle needs to be false! it's done within the trainer
	
	val_dataset = MultiviewImgDataset(args.val_path, scale_aug=False, rot_aug=False, num_views=args.num_views)
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchSize, shuffle=False, num_workers=0)
	print('num_train_files: '+str(len(train_dataset.filepaths)))
	print('num_val_files: '+str(len(val_dataset.filepaths)))


	# STAGE 1
	cnet = SVCNN(args.name, nclasses=40, pretraining=False, cnn_name=args.cnn_name)
	
	cnet.cuda()
	
	log_dir = args.prefix + 'ckpt/svcnn/model-00023.pth'
	model = torch.load(log_dir)
	cnet.load_state_dict(model)
#	set_trace()
	
	all_loss = np.zeros(args.num_views)
	all_conf = np.zeros(args.num_views)
	
	for i, data in enumerate(tqdm(train_loader)):
		x = data[1]
		N, V, C, H, W = x.size()
		x = x.view(-1, C, H, W).cuda()
#		set_trace()
		
		y, _ = cnet(x)
		target = torch.repeat_interleave(data[0], V).cuda()
		
		loss_fn = nn.CrossEntropyLoss(reduction='none')
		loss = loss_fn(y, target).view(N, V)
		
		pred = F.softmax(y, dim=1)
		conf = pred[range(N * V), target].view(N, V)
		
		all_loss += loss.sum(dim=0).cpu().detach().numpy()
		all_conf += conf.sum(dim=0).cpu().detach().numpy()
#		set_trace()
	set_trace()
	
	
	
	# STAGE 2
	cnet_2 = MVCNN(args.name, cnet, nclasses=40, cnn_name=args.cnn_name, num_views=args.num_views, constraint=args.constraint)
	del cnet
	
	cnet_2.cuda()
	
#	log_dir = args.prefix + 'runs/24/' + args.name + '_stage_2/mvcnn/model-00005.pth'
#	log_dir = args.prefix + 'ckpt/mvcnn/model-00024.pth'
	log_dir = args.prefix + 'model-00025.pth'
	model = torch.load(log_dir)
	cnet_2.load_state_dict(model)
#	set_trace()
	
	all_ww = np.zeros(args.num_views)
	top1_idx = np.zeros(args.num_views)
	
	for i, data in enumerate(tqdm(train_loader)):
		x = data[1]
		N, V, C, H, W = x.size()
		x = x.view(-1, C, H, W).cuda()
#		set_trace()
		
		_, ww = cnet_2(x)
		if args.constraint == 'temperature':
			ww = F.softmax(ww / 0.9, dim=1)
#		set_trace()
		
		_, idx = torch.max(ww, dim=1)
#		set_trace()
		
		'''
		x = x.view(N, V, C, H, W)
		for j in range(N):
#			set_trace()
			torchvision.utils.save_image(x[j, idx[j], :, :, :], selected_dir + str(N * i + j).zfill(4) + '.png')
		'''
		
		all_ww += ww.sum(dim=0).cpu().detach().numpy()
#		top1_idx[idx] += 1
	
	set_trace()


