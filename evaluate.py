import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import os,shutil,json
import argparse

import torchvision.models as models
import torch.nn.functional as F

from tools.Trainer import ModelNetTrainer
from tools.ImgDataset import MultiviewImgDataset, SingleImgDataset
from models.MVCNN import MVCNN, SVCNN

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

parser.add_argument('-prefix', type=str, default='./')
#parser.add_argument('-prefix', type=str, default='/vulcan/scratch/yxren/mvcnn/')

if __name__ == '__main__':
	args = parser.parse_args()
	
	if args.cnn_name == 'alexnet':
		net_1 = models.alexnet().features
#		self.net_2 = models.alexnet().classifier
	elif args.cnn_name == 'vgg11':
		net_1 = models.vgg11().features
#		self.net_2 = models.vgg11().classifier
	
	
	if args.cnn_name == 'alexnet':
		inout = [256 * 6 * 6, 512, 32, 1]
	elif args.cnn_name == 'vgg11':
		inout = [512 * 7 * 7, 2048, 64, 1]
	inout = np.array(inout) * args.num_views
	
	main_net = nn.Sequential(
		nn.Linear(inout[0], inout[1]),
		nn.ReLU(),
		nn.Linear(inout[1], inout[2]),
		nn.ReLU(),
		nn.Linear(inout[2], inout[3]),
		nn.Softmax(dim=1))
	'''
	self.main = nn.Sequential(
		nn.Linear(inout[0], inout[3]),
		nn.Softmax(dim=1))
	'''
	
	if torch.cuda.is_available():
		net_1.cuda()
		main_net.cuda()

	
	n_models_train = args.num_models * args.num_views
	
	train_dataset = MultiviewImgDataset(args.train_path, scale_aug=False, rot_aug=False, num_models=n_models_train, num_views=args.num_views)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=False, num_workers=0) # shuffle needs to be false! it's done within the trainer

	val_dataset = MultiviewImgDataset(args.val_path, scale_aug=False, rot_aug=False, num_views=args.num_views)
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchSize, shuffle=False, num_workers=0)
	print('num_train_files: '+str(len(train_dataset.filepaths)))
	print('num_val_files: '+str(len(val_dataset.filepaths)))
	
	
	log_dir = args.prefix + args.name + '_stage_2/mvcnn/model-00000.pth'
	model = torch.load(log_dir)
	net_1.load_state_dict(model, strict=False)
	net_1.eval()
	main_net.load_state_dict(model, strict=False)
	main_net.eval()
	
#	set_trace()
	
	for i, data in enumerate(train_loader):
		N, V, C, H, W = data[1].size()
		x = data[1].view(-1, C, H, W).cuda()
		
		y = net_1(x)
		ww = main_net(y.view(N, -1))
		
		set_trace()


