from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms, utils
from CarlaDataset import CarlaDataset
import matplotlib.pyplot as plt

transform = transforms.Compose([
        transforms.Resize((150,200)),
        transforms.ToTensor()])
batch = 1
dl = DataLoader(CarlaDataset('../data/double_camera', load_as_grayscale=True, transform=transform), batch_size=batch)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')     


for i, X in enumerate(dl):
    limg = X[0][:,1,:,:,:]
    rimg = X[0][:,0,:,:,:]
    limg.to(device)
    rimg.to(device)
    
    if i == 25:
        fig = plt.figure()
        limg = limg.squeeze().numpy()
        rimg = rimg.squeeze().numpy()
        ax1 = fig.add_subplot(2,2,1)
        ax1.imshow(limg, cmap='gray')
        ax2 = fig.add_subplot(2,2,2)
        ax2.imshow(rimg, cmap='gray')
        plt.show()

    if i == 50:
        break
