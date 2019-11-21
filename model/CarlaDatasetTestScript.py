from torch.utils.data import Dataset, DataLoader
from CarlaDataset import CarlaDataset

dl = DataLoader(CarlaDataset('../data/'))
for i, X in enumerate(dl):
    img1 = X[0]
    img2 = X[1]
    ctrl_inputs = X[2]
    print(">>>>>>Time t: {}".format(img1))
    print(">>>>>>Time t+1: {}".format(img2))
    print(">>>>>>Control Inputs: {}".format(ctrl_inputs))
