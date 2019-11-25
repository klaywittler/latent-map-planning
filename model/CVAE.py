import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import pdb

#input is a tuple of images t and t+1 with lidar and control input
class CVAE(nn.Module):
	def __init__(self):
		super().__init__()
		d = 0.4
		self.z_size = 64
		self.hidden = 256
		ch_sz = 3
		last_conv = 16
		self.tensor = (2,last_conv,300,400)
		flat = np.prod(self.tensor)

		# channel_in, c_out, kernel_size, stride, padding
		def convbn(ci,co,ksz,s=1,pz=0):
			return nn.Sequential(
				nn.Conv2d(ci,co,ksz,stride=s,padding=pz),
				nn.ReLU(),
				nn.BatchNorm2d(co))
		def mlp(in_size,hidden):
			return nn.Sequential(
				nn.Dropout(d),
				nn.Linear(in_size,hidden),
				nn.ReLU())

		#Encoder NN
		self.enc = nn.Sequential(
				nn.Dropout(d),
				convbn(ch_sz,64,3,1,1),
				convbn(64,32,1,1),
				convbn(32,last_conv,1,1))
		self.m1 = mlp(flat,self.hidden)
		self.zmean = nn.Linear(self.hidden,self.z_size)
		self.zstdev = nn.Linear(self.hidden,self.z_size)

		#Decoder NN
		self.expand_z = nn.Linear(self.z_size,self.hidden)
		self.m2 = mlp(self.hidden,flat)
		self.dec = nn.Sequential(
				nn.Dropout(d),
				convbn(last_conv,32,1,1),
				convbn(32,64,1,1),
				convbn(64,ch_sz,1,1))

	def encoder(self, x):
		h_layer = torch.flatten(self.enc(x))	
		# Get shapes for decoder
		# shapes1 = self.enc(x).shape
		# shapes2 = len(h_layer)
		# pdb.set_trace()
		# add control input in the following layer
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
		h1 = self.m2(h)
		#make sure to reshape data correctly
		x = torch.reshape(h1,(self.tensor))
		out = self.dec(x)
		return out

	def forward(self, x):
		h = self.encoder(x)
		z, z_mean, z_stdev = self.bottleneck(h)
		out = self.decoder(z)
		return out, z, z_mean, z_stdev