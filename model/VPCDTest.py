import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils # We should use this eventually.
from VelocityPredictionCarlaDataset import VelocityPredictionCarlaDataSet
import matplotlib.pyplot as plt

transform = transforms.Compose([
        transforms.Resize((150,200)),
        transforms.ToTensor()])
batch = 1
vpcd = VelocityPredictionCarlaDataSet(
    '../data/synced_single_camera',
    delta=100,
    load_as_grayscale=True,
    transform=transform)
dl = DataLoader(vpcd, batch_size=batch)

for i, X in enumerate(dl):
    if i > 10:
        break
    limg = X[0]
    print(limg.shape)
    rimg = X[1]
    print(rimg.shape)
    ctrl_inputs = X[2]
    if i == 0:
        fig = plt.figure()
        limg = limg.squeeze().numpy()
        rimg = rimg.squeeze().numpy()
        ax1 = fig.add_subplot(2,2,1)
        ax1.imshow(limg, cmap='gray')
        ax2 = fig.add_subplot(2,2,2)
        ax2.imshow(rimg, cmap='gray')
        plt.show()