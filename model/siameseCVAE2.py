import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import pdb

#input is a tuple of images t and t+1 with lidar and control input
class siameseCVAE(nn.Module):
	def __init__(self):
		super().__init__()
		d = 0.4
		self.z_size = 64
		self.hidden = 1024
		self.small = 256
		ch_sz = 3
		last_conv = 4
		self.tensor = (1,last_conv,300,400)
		flat = np.prod(self.tensor)*2
		self.flat = flat

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
				convbn(ch_sz,64,3,1,1),
				convbn(64,16,1,1),
				convbn(16,last_conv,1,1))
		self.ency = nn.Sequential(
				nn.Dropout(d),
				convbn(ch_sz,64,3,1,1),
				convbn(64,16,1,1),
				convbn(16,last_conv,1,1))
		self.m1 = nn.Sequential(
				nn.Dropout(d),
				mlp(flat,self.hidden),
				mlp(self.hidden, self.small))
		self.zmean = nn.Linear(self.small,self.z_size)
		self.zstdev = nn.Linear(self.small,self.z_size)

		#Decoder NN
		self.expand_z = nn.Linear(self.z_size,self.small)
		self.m2 = nn.Sequential(
				nn.Dropout(d),
				mlp(self.small,self.hidden),
				mlp(self.hidden,flat))
		self.decx = nn.Sequential(
				nn.Dropout(d),
				convbn(last_conv,16,1,1),
				convbn(16,64,1,1),
				convout(64,ch_sz,1,1))
		self.decy = nn.Sequential(
				nn.Dropout(d),
				convbn(last_conv,16,1,1),
				convbn(16,64,1,1),
				convout(64,ch_sz,1,1))

	def encoder(self, x, y, ctrl):
		h_x = torch.flatten(self.encx(x))
		h_y = torch.flatten(self.ency(y))	
		# Concatenate flat convs
		h_layer = torch.cat((h_x,h_y))
		h = self.m1(h_layer)
		return h

	def bottleneck(self, x):
		z_mean = self.zmean(x)
		z_stdev = self.zstdev(x)
		#reparam to get z latent sample
		std = torch.exp(0.5*z_stdev)
		eps = torch.randn_like(std)
		z = z_mean + eps*std
		return z, z_mean, z_stdev

	def decoder(self, z):
		#check the nonlinearities of this layer
		h = self.expand_z(z)
		h_layer = self.m2(h)
		#Split in 2
		h_x = h_layer[:int(self.flat/2)]
		h_y = h_layer[int(self.flat/2):]
		#make sure to reshape data correctly and decode
		x = self.decx(torch.reshape(h_x,(self.tensor)))
		y = self.decy(torch.reshape(h_x,(self.tensor)))
		return x, y

	def forward(self, x, y, ctrl):
		h = self.encoder(x, y, ctrl)
		z, z_mean, z_stdev = self.bottleneck(h)
		x_hat, y_hat = self.decoder(z)
		return x_hat, y_hat, z, z_mean, z_stdev