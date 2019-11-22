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


def main(args):
    dl = DataLoader(CarlaDataset('../data/'))
    for i, X in enumerate(dl):
        img1 = X[0]
        img2 = X[1]
        ctrl_inputs = X[2]

        im_t = img1[0].view(2,4,300,-1)
        im_t = (im_t/255).float()
        out, z, z_mean, z_stdev = CVAE().forward(im_t)

        pdb.set_trace()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train_samples', type=int, default=30000)
    parser.add_argument('--val_samples', type=int, default=5000)
    parser.add_argument('--built_in', type=bool, default=False)
    args = parser.parse_args()

    main(args)