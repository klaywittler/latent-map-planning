import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import pdb

class siameseCVAE(nn.Module):
	def __init__(self,batch=4):
		super().__init__()
		d = 0.4
		self.z_size = 64
		self.small = 256
		self.hidden = 1024
		ch_sz = 1
		c1 = 64
		c2 = 16
		last_conv = 4
		self.tensor = (batch,last_conv,150,200)
		flat = np.prod(self.tensor)
		flat2 = flat*2

		# channel_in, c_out, kernel_size, stride, padding
		def convbn(ci,co,ksz,s=1,pz=0):		#ReLu nonlinearity
			return nn.Sequential(
				nn.Conv2d(ci,co,ksz,stride=s,padding=pz),
				nn.ReLU(),
				nn.BatchNorm2d(co))
		def convout(ci,co,ksz,s=1,pz=0):	#Sigmoid nonlinearity
			return nn.Sequential(
				nn.Conv2d(ci,co,ksz,stride=s,padding=pz),
				nn.Sigmoid(),
				nn.BatchNorm2d(co))
		def mlp(in_size,hidden):
			return nn.Sequential(
				nn.Dropout(d),
				nn.Linear(in_size,hidden),
				nn.ReLU())

		#Encoder NN
		self.encx = nn.Sequential(
				nn.Dropout(d),
				convbn(ch_sz,c1,3,1,1),
				convbn(c1,c2,3,1,1),
				convbn(c2,last_conv,3,1,1))
		self.ency = nn.Sequential(
				nn.Dropout(d),
				convbn(ch_sz,c1,3,1,1),
				convbn(c1,c2,3,1,1),
				convbn(c2,last_conv,3,1,1))
		self.m1 = nn.Sequential(
				nn.Dropout(d),
				mlp(flat2,self.hidden),
				mlp(self.hidden, self.small))
		self.zmean = nn.Linear(self.small,self.z_size)
		self.zlogvar = nn.Linear(self.small,self.z_size)

		#Decoder NN
		self.expand_z = nn.Linear(self.z_size,self.small)
		self.mx = nn.Sequential(
				nn.Dropout(d),
				mlp(self.small,self.hidden),
				mlp(self.hidden,flat))
		self.my = nn.Sequential(
				nn.Dropout(d),
				mlp(self.small,self.hidden),
				mlp(self.hidden,flat))
		self.decx = nn.Sequential(
				nn.Dropout(d),
				convbn(last_conv,c2,3,1,1),
				convbn(c2,c1,3,1,1),
				convout(c1,ch_sz,3,1,1))
		self.decy = nn.Sequential(
				nn.Dropout(d),
				convbn(last_conv,c2,3,1,1),
				convbn(c2,c1,3,1,1),
				convout(c1,ch_sz,3,1,1))

	def encoder(self, x, y):
		# Flatten enc output
		h_x = self.encx(x).view(-1)
		h_y = self.ency(y).view(-1)
		# Concatenate flat convs
		h_layer = torch.cat((h_x,h_y))
		h = self.m1(h_layer)
		return h

	def bottleneck(self, x):
		z_mean = self.zmean(x)
		z_logvar = self.zlogvar(x)
		#reparam to get z latent sample
		std = torch.exp(0.5*z_logvar)
		eps = torch.randn_like(std)
		z = z_mean + eps*std
		return z, z_mean, z_logvar

	def decoder(self, z):
		#check the nonlinearities of this layer
		h = self.expand_z(z)
		#exand z to each decoder head
		h_x = self.mx(h)
		h_y = self.my(h)
		#make sure to reshape data correctly and decode
		x = self.decx(h_x.view(self.tensor))
		y = self.decy(h_x.view(self.tensor))
		return x, y

	def forward(self, x, y):
		h = self.encoder(x, y)
		z, z_mean, z_logvar = self.bottleneck(h)
		x_hat, y_hat = self.decoder(z)
		return x_hat, y_hat, z, z_mean, z_logvar

	def encode_get_z(self, x, y):
		h = self.encoder(x, y)
		z, z_mean, z_logvar = self.bottleneck(h)
		return z, z_mean, z_logvar