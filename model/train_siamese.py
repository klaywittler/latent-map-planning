import pdb, argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as thv
from torchvision import transforms

from torch.utils.data import Dataset, DataLoader
from CarlaDataset import CarlaDataset
from CVAE import CVAE
from siameseCVAE import siameseCVAE


def main():
    model = siameseCVAE()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999),
                            eps=1e-08, weight_decay=0)
    epochs = 10
    dl = DataLoader(CarlaDataset('../data/'))
    for epoch in range(epochs):
        for i, X in enumerate(dl):
            img = X[0]
            img_ = X[1]
            ctrl_inputs = X[2]

            img = (img/255).float()
            img_ = (img_/255).float()
            img1 = img[:,0,:,:,:]
            img2 = img[:,1,:,:,:]
            img1_ = img_[:,0,:,:,:]
            img2_ = img_[:,1,:,:,:]

            xhat, yhat, z, z_mean, z_stdev = model.forward(img1,img2,ctrl_inputs)
            lossx = criterion(xhat,img1_)
            lossy = criterion(yhat,img2_)
            optimizer.zero_grad()
            lossx.backward()
            lossy.backward()

            if (i+1) % 10 == 0:
              print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, i+1, total_step, loss.item()))


            pdb.set_trace()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch_size', type=int, default=32)
    # parser.add_argument('--lr', type=int, default=0.005)

    args = parser.parse_args()

    main()