import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F


#input is a tuple of images t and t+1 with lidar and control input
class CVAE(nn.Module):
	def __init__(self):
		super(VAE, self).__init__()


	def encoder(self, x):
		pass

	def bottleneck(self, mu, logvar):
		pass

	def decoder(self, z):
		pass

	def forward(self, x):
		pass