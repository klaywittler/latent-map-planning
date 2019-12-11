import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import pdb

#input is a tuple of images t and t+1 with lidar and control input
class CVAE(nn.Module):
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

		# channel_in, c_out, kernel_size, stride, padding
		def convbn(ci,co,ksz,s=1,pz=0):		#ReLU nonlinearity
			return nn.Sequential(
				nn.Conv2d(ci,co,ksz,stride=s,padding=pz),
				nn.ReLU(),
				nn.BatchNorm2d(co))
		def decconvbn(ci,co,ksz,s=1,pz=0):		#ReLU nonlinearity
			return nn.Sequential(
				nn.ConvTranspose2d(ci,co,ksz,stride=s,padding=pz),
				nn.ReLU(),
				nn.BatchNorm2d(co))
		def convlast(ci,co,ksz,s=1,pz=0):	#Sigmoid nonlinearity
			return nn.Sequential(
				nn.ConvTranspose2d(ci,co,ksz,stride=s,padding=pz),
				nn.Sigmoid())
		def mlp(in_size,hidden):
			return nn.Sequential(
				nn.Dropout(d),
				nn.Linear(in_size,hidden),
				nn.ReLU())

		#Encoder NN
		self.enc = nn.Sequential(
				nn.Dropout(d),
				convbn(ch_sz,c1,3,1,1),
				convbn(c1,c2,3,1,1),
				convbn(c2,last_conv,3,1,1))
		# self.m1 = mlp(flat,self.hidden)
		self.m1 = nn.Sequential(
				nn.Dropout(d),
				mlp(flat,self.hidden),
				mlp(self.hidden, self.small))
		self.zmean = nn.Linear(self.small,self.z_size)
		self.zstdev = nn.Linear(self.small,self.z_size)

		#Decoder NN
		self.expand_z = nn.Linear(self.z_size,self.small)
		# self.m2 = mlp(self.hidden,flat)
		self.m2 = nn.Sequential(
				nn.Dropout(d),
				mlp(self.small,self.hidden),
				mlp(self.hidden,flat))
		self.dec = nn.Sequential(
				nn.Dropout(d),
				decconvbn(last_conv,c2,3,1,1),
				decconvbn(c2,c1,3,1,1),
				convlast(c1,ch_sz,3,1,1))

	def encoder(self, x):
		# h_layer = torch.flatten(self.enc(x))	
		h_layer = self.enc(x).view(-1)
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
		# x = torch.reshape(h1,(self.tensor))
		x = h1.view(self.tensor)
		out = self.dec(x)
		return out

	def forward(self, x):
		h = self.encoder(x)
		z, z_mean, z_stdev = self.bottleneck(h)
		out = self.decoder(z)
		return out, z, z_mean, z_stdev