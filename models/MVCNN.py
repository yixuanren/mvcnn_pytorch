import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from .Model import Model

from pdb import set_trace

mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cuda()
std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cuda()

def flip(x, dim):
	xsize = x.size()
	dim = x.dim() + dim if dim < 0 else dim
	x = x.view(-1, *xsize[dim:])
	x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
					  -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
	return x.view(xsize)


class SVCNN(Model):

	def __init__(self, name, nclasses=40, pretraining=True, cnn_name='vgg11'):
		super(SVCNN, self).__init__(name)

		self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
						 'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
						 'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
						 'person','piano','plant','radio','range_hood','sink','sofa','stairs',
						 'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']

		self.name = name
		self.nclasses = nclasses
		self.pretraining = pretraining
		self.cnn_name = cnn_name
		self.use_resnet = cnn_name.startswith('resnet')
		
		self.constraint = ''
		
		self.mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cuda()
		self.std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cuda()

		if self.use_resnet:
			if self.cnn_name == 'resnet18':
				self.net = models.resnet18(pretrained=self.pretraining)
				self.net.fc = nn.Linear(512,40)
			elif self.cnn_name == 'resnet34':
				self.net = models.resnet34(pretrained=self.pretraining)
				self.net.fc = nn.Linear(512,40)
			elif self.cnn_name == 'resnet50':
				self.net = models.resnet50(pretrained=self.pretraining)
				self.net.fc = nn.Linear(2048,40)
		else:
			if self.cnn_name == 'alexnet':
				self.net_1 = models.alexnet(pretrained=self.pretraining).features
				self.net_2 = models.alexnet(pretrained=self.pretraining).classifier
			elif self.cnn_name == 'vgg11':
				self.net_1 = models.vgg11(pretrained=self.pretraining).features
				self.net_2 = models.vgg11(pretrained=self.pretraining).classifier
			elif self.cnn_name == 'vgg16':
				self.net_1 = models.vgg16(pretrained=self.pretraining).features
				self.net_2 = models.vgg16(pretrained=self.pretraining).classifier
#			set_trace()
			self.net_2._modules['6'] = nn.Linear(self.net_2._modules['6'].in_features, self.nclasses)

	def forward(self, x):
		if self.use_resnet:
			return self.net(x)
		else:
			y = self.net_1(x)
#			set_trace()
			return self.net_2(y.view(y.shape[0], -1)), torch.zeros(1, 1).cuda()


class MVCNN(Model):

	def __init__(self, name, model, nclasses=40, cnn_name='vgg11', num_views=12, constraint=None):
		super(MVCNN, self).__init__(name)

		self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
						 'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
						 'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
						 'person','piano','plant','radio','range_hood','sink','sofa','stairs',
						 'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']

		self.name = name
		self.nclasses = nclasses
		self.num_views = num_views
		
		self.constraint = constraint
		
		self.mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cuda()
		self.std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cuda()

		self.use_resnet = cnn_name.startswith('resnet')

		if self.use_resnet:
			self.net_1 = nn.Sequential(*list(model.net.children())[:-1])
			self.net_2 = model.net.fc
		else:
			self.net_1 = model.net_1
			self.net_2 = model.net_2
#		set_trace()
		
		if cnn_name == 'alexnet':
			inout = [self.net_2._modules['1'].in_features, 512, 32, 1]
		elif cnn_name == 'vgg11':
			inout = [self.net_2._modules['0'].in_features, 2048, 64, 1]
		inout = np.array(inout) * self.num_views
		
		self.main_net = nn.Sequential(
				nn.Linear(inout[0], inout[1]),
				nn.ReLU(inplace=True),
				nn.Linear(inout[1], inout[2]),
				nn.ReLU(inplace=True),
				nn.Linear(inout[2], inout[3]))
		'''
		self.main_net = nn.Sequential(
			nn.Linear(inout[0], inout[3]))
		'''
		if self.constraint == None or self.constraint == 'maxmax' or self.constraint == 'argmax':
			self.main_net.add_module(str(len(self.main_net)), nn.Softmax(dim=1))
#		set_trace()

	def forward(self, x):
		y = self.net_1(x) # (96, 256, 6, 6)
		N1, C, H, W = y.size()
		N2 = int(N1 / self.num_views)
		y = y.view((N2, self.num_views, C, H, W)) # (8, 12, 256, 6, 6)
#		y = y.permute(1, 0, 2, 3, 4).contiguous()
#		set_trace()
		
		ww = self.main_net(y.view(N2, -1)) # (8, 12)
		
		if self.constraint == 'temperature':
			ww = F.softmax(ww / 0.5, dim=1) # Temperature
#		set_trace()
		
		if self.constraint == 'argmax':
			# NOT DIFFERENTIABLE!
			m, i = torch.max(ww, dim=1)
			ww = torch.zeros_like(ww)
			ww[range(N2), i] = 1
#		set_trace()
		
		wwy = torch.bmm(ww.unsqueeze(1), y.view(N2, self.num_views, -1)) # (8, 1, 9216) = (8, 1, 12) * (8, 12, 9216)
		
		return self.net_2(wwy.view(N2, -1)), ww
		
		'''
		y = y.view((int(x.shape[0]/self.num_views), self.num_views, y.shape[-3], y.shape[-2], y.shape[-1])) # (8, 12, 512 ,7, 7)
		
		return self.net_2(torch.max(y, 1)[0].view(y.shape[0], -1))
		'''

