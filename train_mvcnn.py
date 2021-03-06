import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import os, shutil, json
import argparse

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

parser.add_argument('-ct', '--constraint', type=str, default=None)
parser.add_argument('-w_m', type=float, default=1e-1)
parser.add_argument('-T', type=float, default=1)

parser.add_argument('-pt', '--preType', type=str, default=None)

parser.add_argument('-stm', '--svcnn_training_mode', type=str, default=None)
parser.add_argument('-f', '--freeze', type=bool, default=False)

parser.add_argument('-prefix', type=str, default='./')
#parser.add_argument('-prefix', type=str, default='/vulcan/scratch/yxren/mvcnn/')

def create_folder(log_dir):
	# make summary folder
	if not os.path.exists(log_dir):
		os.mkdir(log_dir)
	else:
		print('WARNING: summary folder already exists!! It will be overwritten!!')
		shutil.rmtree(log_dir)
		os.mkdir(log_dir)

if __name__ == '__main__':
	args = parser.parse_args()
	
	if args.svcnn_training_mode == None and args.freeze:
		print('Invalid args!')
		exit()
	
	
	ckpt_dir = args.prefix + 'runs'
	if not os.path.exists(ckpt_dir):
		os.makedirs(ckpt_dir)
#	set_trace()
	runs = sorted(map(int, next(os.walk(ckpt_dir))[1]))
	run_nr = 0 if len(runs) == 0 else runs[-1] + 1
	run_folder = str(run_nr).zfill(2)
	run_dir = os.path.join(ckpt_dir, run_folder)
	if not os.path.exists(run_dir):
		os.makedirs(run_dir)
	
	
	pretraining = not args.no_pretraining
	log_dir = run_dir + '/' + args.name
	create_folder(log_dir)
	config_f = open(os.path.join(log_dir, 'config.json'), 'w')
	json.dump(vars(args), config_f)
	config_f.close()
	
	n_models_train = args.num_models * args.num_views
	
	# STAGE 1
	cnet = SVCNN(args.name, nclasses=40, pretraining=pretraining, cnn_name=args.cnn_name)

	if args.svcnn_training_mode == 'train':
		log_dir = run_dir + '/' + args.name + '_stage_1'
		create_folder(log_dir)
		
		optimizer = optim.Adam(cnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)

		train_dataset = SingleImgDataset(args.train_path, scale_aug=False, rot_aug=False, num_models=n_models_train, num_views=args.num_views)
		train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)

		val_dataset = SingleImgDataset(args.val_path, scale_aug=False, rot_aug=False, test_mode=True)
		val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
		print('num_train_files: '+str(len(train_dataset.filepaths)))
		print('num_val_files: '+str(len(val_dataset.filepaths)))
		trainer = ModelNetTrainer(cnet, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), 'svcnn', log_dir, num_views=1)
		trainer.train(30)
	elif args.svcnn_training_mode == 'load':
		cnet.cuda()
		log_dir = args.prefix + 'ckpt/svcnn/model-00025.pth'
#		log_dir = ckpt_dir + '/20/mvcnn_stage_1/mvcnn/model-00025.pth'
		model = torch.load(log_dir)
		cnet.load_state_dict(model)
		print('SVCNN trained model loaded!')


	# STAGE 2
	log_dir = run_dir + '/' + args.name + '_stage_2'
	create_folder(log_dir)
	cnet_2 = MVCNN(args.name, cnet, nclasses=40, cnn_name=args.cnn_name, num_views=args.num_views, constraint=args.constraint, w_m=args.w_m, T=args.T, preType=args.preType)
	del cnet

	new_params = list(cnet_2.main_net.parameters()) + list(cnet_2.main_net.parameters()) if args.preType != None else cnet_2.main_net.parameters()
	params = new_params if args.freeze else cnet_2.parameters()
	optimizer = optim.Adam(params, lr=args.lr/8*args.batchSize, weight_decay=args.weight_decay, betas=(0.9, 0.999))
	
	train_dataset = MultiviewImgDataset(args.train_path, scale_aug=False, rot_aug=False, num_models=n_models_train, num_views=args.num_views)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=False, num_workers=0) # shuffle needs to be false! it's done within the trainer

	val_dataset = MultiviewImgDataset(args.val_path, scale_aug=False, rot_aug=False, num_views=args.num_views)
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchSize, shuffle=False, num_workers=0)
	print('num_train_files: '+str(len(train_dataset.filepaths)))
	print('num_val_files: '+str(len(val_dataset.filepaths)))
	trainer = ModelNetTrainer(cnet_2, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), 'mvcnn', log_dir, num_views=args.num_views)
	trainer.train(30)


