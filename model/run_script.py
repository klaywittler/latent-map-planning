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


def main():
    model = CVAE()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999),
                            eps=1e-08, weight_decay=0)
    epochs = 10
    dl = DataLoader(CarlaDataset('../data/'))
    for epoch in range(epochs):
        for i, X in enumerate(dl):
            img1 = X[0]
            img2 = X[1]
            ctrl_inputs = X[2]

            im_1 = img1[0].view(2,4,300,-1)
            im_1 = (im_1/255).float()
            im_2 = img2[0].view(2,4,300,-1)
            im_2 = (im_2/255).float()

            out, z, z_mean, z_stdev = model.forward(im_1)
            loss = criterion(out,im_2)
            optimizer.zero_grad()
            loss.backward()

            if (i+1) % 10 == 0:
              print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, i+1, total_step, loss.item()))


            pdb.set_trace()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch_size', type=int, default=32)
    # parser.add_argument('--lr', type=int, default=0.005)

    args = parser.parse_args()

    main()