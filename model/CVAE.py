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
		self.hidden = 256
		#hidden must be > last layer conv enc layer or first dec layer
		self.im_size = 4

	# channel_in, c_out, kernel_size, stride, padding
	def convbn(self,ci,co,ksz,s=1,pz=1):
		return nn.Sequential(
			nn.Conv2d(ci,co,ksz,stride=s,padding=pz),
			nn.ReLU(True),
			nn.BatchNorm2d(co))

	def mlp(self,x,hidden):
		#Fully connected layer
		return nn.Sequential(
			nn.Dropout(self.d),
			nn.Linear(x,hidden),
			nn.ReLU(True))

	def encoder(self, x):
		enc = nn.Sequential(
			nn.Dropout(self.d),
			self.convbn(self.im_size,64,3,1,1),
			self.convbn(64,32,1,1),
			self.convbn(32,16,1,1))

		h = enc(x[0])
		# h = torch.flatten(enc(x))
		pdb.set_trace()
		# self.shapes = enc.shape() 
		# need to get last layer size
		#add control input in the following layer
		# h_layer = mlp(torch.flatten(enc),self.hidden)
		# return h_layer
		return h

	def bottleneck(self, x):
		z_mean = nn.Linear(x,self.z_size)
		z_stdev = nn.Linear(x,self.z_size)
		#reparam to get z latent sample
		std = torch.exp(0.5*logvar)
		eps = torch.randn_like(std)
		z = z_mean + eps*std
		return z, z_mean, z_stdev

	def decoder(self, z):
		h_layer = mlp(z,self.hidden)
		x = torch.reshape(h_layer,(self.shapes))
		#make sure to reshape data correctly
		dec = nn.Sequential(
			nn.Dropout(d),
			convbn(16,32,1,1),
			convbn(32,64,1,1),
			convbn(64,self.im_size,1,1)
			)
		#maybe put a nonlinearity on the output dec
		return dec

	def forward(self, x):
		h = self.encoder(x)
		pdb.set_trace()
		z, z_mean, z_stdev = self.bottleneck(h)
		out = self.decoder(z)
		return out, z, z_mean, z_stdev