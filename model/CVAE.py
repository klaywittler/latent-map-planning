import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F


#input is a tuple of images t and t+1 with lidar and control input
class CVAE(nn.Module):
	def __init__(self, im_size):
		super(CVAE, self).__init__()
		d = 0.4
		z_size = 64
		hidden = 256
		#hidden must be > last layer conv enc layer or first dec layer
		self.im_size = im_size

		# channel_in, c_out, kernel_size, stride, padding
        def convbn(ci,co,ksz,s=1,pz=1):
            return nn.Sequential(
                nn.Conv2d(ci,co,ksz,stride=s,padding=pz),
                nn.ReLU(True),
                nn.BatchNorm2d(co))

        def mlp(x,hidden):
        	#Fully connected layer
        	return nn.Sequential(
        		nn.Dropout(d)
        		nn.Linear(x,hidden),
        		nn.ReLU(True))

	def encoder(self, x):
		enc = nn.Sequential(
			nn.Dropout(d),
			convbn(self.im_size,64,3,1,1),
			convbn(64,32,1,1),
			convbn(32,16,1,1))
			self.shapes = enc.shape() 
		# need to get last layer size
		#add control input in the following layer
		h_layer = mlp(torch.flatten(enc),hidden,z_size)
		return h_layer

	def bottleneck(self, x):
		z_mean = nn.Linear(x,z_size)
		z_stdev = nn.Linear(x,z_size)
		#reparam to get z latent sample
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = z_mean + eps*std
		return z, z_mean, z_stdev

	def decoder(self, z):
		h_layer = mlp(z,hidden)
		x = torch.reshape(h_layer,(self.shapes))
		#make sure to reshape data correctly
		dec = nn.Sequential(
			nn.Dropout(d)
			convbn(16,32,1,1)
			convbn(32,64,1,1)
			convbn(64,self.im_size,1,1)
			)
		#maybe put a nonlinearity on the output dec
		return dec

	def forward(self, x):
		h = encoder(x)
		z, z_mean, z_stdev = bottleneck(h)
		out = decoder(z)
		return out, z, z_mean, z_stdev