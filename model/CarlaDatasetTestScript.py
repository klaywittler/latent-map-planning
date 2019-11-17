from CarlaDataset import CarlaDataset
from torch.utils.data import Dataset, DataLoader

carla_ds = CarlaDataset('../data/')

dl = DataLoader(CarlaDataset('../data/'))
for ctrl_inputs, r in enumerate(dl):
    print(r)
