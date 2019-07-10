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
import matplotlib.pyplot as plt

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
parser.add_argument('-w_m', type=float, default=1e-1)
parser.add_argument('-T', type=float, default=1)

parser.add_argument('-pt', '--preType', type=str, default=None)

parser.add_argument('-stm', '--svcnn_training_mode', type=str, default=None)
parser.add_argument('-f', '--freeze', type=bool, default=False)

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
	
	total_num = len(train_dataset.filepaths)
	
	
	# STAGE 1
	cnet = SVCNN(args.name, nclasses=40, pretraining=False, cnn_name=args.cnn_name)
	'''
	cnet.cuda()
	
	log_dir = args.prefix + 'ckpt/svcnn/model-00025.pth'
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
	
	np.savez_compressed('stage1', all_loss=all_loss, all_conf=all_conf)
	print('Npz file saved')
	'''
	
	'''
	# STAGE 2
	cnet_2 = MVCNN(args.name, cnet, nclasses=40, cnn_name=args.cnn_name, num_views=args.num_views, constraint=args.constraint, w_m=args.w_m, T=args.T, preType=args.preType)
	del cnet
	
	cnet_2.cuda()
	
#	log_dir = args.prefix + 'runs/24/' + args.name + '_stage_2/mvcnn/model-00005.pth'
#	log_dir = args.prefix + 'ckpt/mvcnn/model-00024.pth'
	log_dir = args.prefix + 'model-00023.pth'
	model = torch.load(log_dir)
	cnet_2.load_state_dict(model)
#	set_trace()
	
	all_ww = np.zeros(args.num_views)
	top3_idx = np.zeros(args.num_views)
	principal_idx = np.zeros(args.num_views)
	rank3_idx = np.zeros([3, args.num_views])
	
	for i, data in enumerate(tqdm(train_loader)):
		x = data[1]
		N, V, C, H, W = x.size()
		x = x.view(-1, C, H, W).cuda()
#		set_trace()
		
		_, ww = cnet_2(x)
		if args.constraint == 'temperature':
			ww = F.softmax(ww / 0.9, dim=1)
#		set_trace()
		
#		_, idx = torch.max(ww, dim=1)
		_, top3 = torch.topk(ww, 3, dim=1)
		unique, counts = torch.unique(top3, return_counts=True)
		dd = dict(zip(unique.cpu().detach().numpy(), counts.cpu().detach().numpy()))
#		set_trace()
		
		
#		x = x.view(N, V, C, H, W)
#		for j in range(N):
#			set_trace()
#			torchvision.utils.save_image(x[j, idx[j], :, :, :], selected_dir + str(N * i + j).zfill(4) + '.png')
		
		
		all_ww += ww.sum(dim=0).cpu().detach().numpy()
		for key in dd:
			top3_idx[key] += dd[key]
		principal_idx += (ww > 0.15).sum(dim=0).cpu().detach().numpy()
		
		top3 = top3.transpose(0, 1)
		for i in range(top3.size()[0]):
			unique, counts = torch.unique(top3[i], return_counts=True)
			dd = dict(zip(unique.cpu().detach().numpy(), counts.cpu().detach().numpy()))
			for key in dd:
				rank3_idx[i, key] += dd[key]
#			set_trace()
	
	np.savez_compressed('stage2', all_ww=all_ww, top3_idx=top3_idx, principal_idx=principal_idx, rank3_idx=rank3_idx)
	print('Npz file saved')
	
#	set_trace()
	'''
	
	name_list = [str(i) for i in range(1, args.num_views+1)]
	
	stage1 = np.load('stage1.npz')
	
	all_loss = stage1['all_loss']
	all_conf = stage1['all_conf']
	
#	set_trace()
	
	X = np.arange(1, args.num_views+1, dtype=float) * 1
	width = 0.3
	
	plt.figure()
	plt.bar(X, all_loss/total_num, width=width, label='avg_loss', tick_label=name_list, color='r')
	plt.legend()
	
#	plt.show()
	plt.savefig('all_loss.pdf')
	
	
	X += width
	
	plt.figure()
	plt.bar(X, all_conf/total_num, width=width, label='avg_conf', tick_label=name_list, color='b')
	plt.legend()
	
#	plt.show()
	plt.savefig('all_conf.pdf')
	
#	set_trace()
	
	
	
	stage2 = np.load('stage2.npz')
	
	all_ww = stage2['all_ww']
	top3_idx = stage2['top3_idx']
	principal_idx = stage2['principal_idx']
	rank3_idx = stage2['rank3_idx']
	
#	set_trace()
	
	X = np.arange(1, args.num_views+1, dtype=float) * 3
	width = 0.3
	
	plt.figure(figsize=(6.4*2, 4.8))
	plt.bar(X, all_ww, width=width, label='all_ww', tick_label=name_list, color='r')
	X += width
	plt.bar(X, top3_idx, width=width, label='top3_idx', tick_label=name_list, color='b')
	X += width
	plt.bar(X, principal_idx, width=width, label='principal_idx', tick_label=name_list, color='g')
	for i in range(rank3_idx.shape[0]):
		X += width
		plt.bar(X, rank3_idx[i], width=width, label='rank3_idx: '+str(i+1), tick_label=name_list, color='y')
	plt.xticks(X-width*2, name_list)
	plt.yscale('log')
	plt.legend()
	
#	plt.show()
	plt.savefig('all.pdf')
	
	set_trace()


