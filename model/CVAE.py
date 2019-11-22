import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import pdb

#input is a tuple of images t and t+1 with lidar and control input
class CVAE(nn.Module):
	def __init__(self):
		super(CVAE, self).__init__()
		self.d = 0.4
		self.z_size = 64
		self.hidden = int(256)
		#hidden must be > last layer conv enc layer or first dec layer
		self.im_size = 4
		self.shapes = {}

	# channel_in, c_out, kernel_size, stride, padding
	def convbn(self,ci,co,ksz,s=1,pz=1):
		return nn.Sequential(
			nn.Conv2d(ci,co,ksz,stride=s,padding=pz),
			nn.ReLU(),
			nn.BatchNorm2d(co))

	def mlp(self,in_size,hidden):
		#Fully connected layer
		return nn.Sequential(
			nn.Dropout(self.d),
			nn.Linear(in_size,hidden),
			nn.ReLU())

	def encoder(self, x):
		enc = nn.Sequential(
			nn.Dropout(self.d),
			self.convbn(self.im_size,64,3,1,1),
			self.convbn(64,32,1,1),
			self.convbn(32,16,1,1))
		
		h_layer = torch.flatten(enc(x))	
		# Get shapes for decoder
		self.shapes['tensor'] = enc(x).shape
		self.shapes['len'] = len(h_layer)
		# add control input in the following layer
		m = self.mlp(self.shapes['len'],self.hidden)
		h = m(h_layer)
		return h

	def bottleneck(self, x):
		bneck = nn.Linear(self.hidden,self.z_size)
		z_mean = bneck(x)
		z_stdev = bneck(x)
		#reparam to get z latent sample
		std = torch.exp(0.5*z_stdev)
		eps = torch.randn_like(std)
		z = z_mean + eps*std
		return z, z_mean, z_stdev

	def decoder(self, z):
		m = nn.Linear(self.z_size,self.hidden)
		m1 = self.mlp(self.hidden,self.shapes['len'])
		h = m(z)
		h1 = m1(h)
		x = torch.reshape(h1,(self.shapes['tensor']))
		#make sure to reshape data correctly
		dec = nn.Sequential(
			nn.Dropout(self.d),
			self.convbn(16,32,1,1),
			self.convbn(32,64,1,1),
			self.convbn(64,self.im_size,1,1)
			)
		out = dec(x)
		#maybe put a nonlinearity on the output dec
		return out

	def forward(self, x):
		h = self.encoder(x)
		z, z_mean, z_stdev = self.bottleneck(h)
		out = self.decoder(z)
		pdb.set_trace()
		return out, z, z_mean, z_stdev