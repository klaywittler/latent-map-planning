import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as thv
from torchvision import transforms

from torch.utils.data import Dataset, DataLoader
from VelocityPredictionCarlaDataSet import *
from CVAE import CVAE
from siameseCVAE import siameseCVAE
from velcoityNN import *

base = ''

def trainVAE(net, optimizer, criterion, epochs, dataloader, exp_name):
    model = net.to(device)
    total_step = len(dataloader)
    overall_step = 0
    losses, kl_loss, mseX_loss, mseY_loss = [], [], [], []
    for epoch in range(epochs):
        model.train()
        total = 0 
        running_loss, kl_running, mseX_running, mseY_running = 0.0, 0.0, 0.0, 0.0
        for i, X in enumerate(dataloader):
            t0 = X[0].float().to(device)
            tk = X[1].float().to(device)

            xhat, yhat, z, z_mean, z_logvar = model.forward(t0,tk)
            loss, MSE_X, MSE_Y, KLD = criterion(xhat,t0, yhat, tk, z_mean, z_logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            kl_running += KLD.item()
            mseX_running += MSE_X.item()
            mseY_running += MSE_Y.item()
            total += X[2].size(0)

            if (i+1) % 10 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, i+1, total_step, loss.item()))
            
            if i == 25:
                break
        
        if (epoch+1) % 10 == 0:
            chpt_path = base+'checkpoints/'+exp_name+'.pt'
            torch.save(model.state_dict(), chpt_path)

        losses.append(running_loss/total)     
        kl_loss.append(kl_running/total)
        mseX_loss.append(mseX_running/total)
        mseY_loss.append(mseY_running/total)  
    
    ells = {'elbo':losses,
            'kl':kl_loss,
            'mseX':mseX_loss,
            'mseY':mseY_loss}

    with open(base+'logs/'+exp_name+'_losses.pickle', 'wb') as f:
        pickle.dump(ells, f)

    return ells

def testVAE(net, criterion, dataloader):
    t0_list, tk_list = [], []
    recon_t0_list, recon_tk_list = [], []
    for i, X in enumerate(dataloader):
        model.eval()
        t0 = X[0].float().to(device)
        tk = X[1].float().to(device)
        u = X[2].float().to(device)

        #Forward Pass
        xhat, yhat, z, z_mean, z_stdev = model.forward(t0,tk)

        t0_ = t0.cpu().squeeze().numpy()
        tk_ = tk.cpu().squeeze().numpy()
        xhat_ = xhat.cpu().detach().squeeze().numpy()
        yhat_ = yhat.cpu().detach().squeeze().numpy()

        t0_list.append(t0_)
        tk_list.append(tk_)
        recon_t0_list.append(xhat_)
        recon_tk_list.append(yhat_)
        if i == 40:
            break

    t0_list = np.asarray(t0_list)
    tk_list = np.asarray(tk_list)
    recon_t0_list = np.asarray(recon_t0_list)
    recon_tk_list = np.asarray(recon_tk_list)

    result = {'t0':t0_list, 'tk':tk_list, 'recon_t0':recon_t0_list, 'recon_tk':recon_tk_list}

    return result


def input2classification(control, lb, ub, num_class):
    interval_list = torch.linspace(lb,ub,num_class).to(device)
    val = abs(interval_list-control.unsqueeze(1))
    idx = torch.argmin(val, axis=1)
    return idx

def trainVel(netVAE, netVel, optimizer, criterion, epochs, dataloader, exp_name):
    modelVAE = netVAE.to(device)
    modelVel = netVel.to(device)
    total_step = len(dataloader)
    overall_step = 0
    accuracy, loss = [], []
    for epoch in range(epochs):
        modelVel.train()
        modelVAE.eval()
        correct = 0
        total = 0
        total_loss = 0
        for i, X in enumerate(dataloader):
            t0 = X[0].float().to(device)
            tk = X[1].float().to(device)
            u = X[2].float().to(device)

            label_vel = input2classification(u[:,1],0,50,6)
            label_steer = input2classification(u[:,0],-1,1,11)

            #Forward Pass
            xhat, yhat, z, z_mean, z_stdev = modelVAE.forward(t0,tk)
            vel, steer = modelVel.forward(z)

            loss = criterion(vel, label_vel) + criterion(steer, label_steer) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item()

            acc = 100*(predicted == labels).sum().item()/labels.size(0)
            overall_step += 1

            if (i+1) % 10 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {}'.format(epoch+1, epochs, overall_step, total_step*epochs, loss.item(), acc))

            if i == 40:
                break
            
        if (epoch+1) % 10 == 0:
            chpt_path = base+'checkpoints/'+exp_name+'.pt'
            torch.save(model.state_dict(), chpt_path)

        accuracy.append(correct/total)
        loss.append(total_loss/total)

    ells = {'accuracy':accuracy,'loss':loss}

    return ells

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((150,200)),
        transforms.ToTensor()])

    batch = 1
    path = base + "project_data/synced_single_camera/"

    print(os.listdir(base))

    dl = DataLoader(VelocityPredictionCarlaDataSet(path, load_as_grayscale=True, transform=transform), batch_size=batch)



    #Run from here
    exp_name = 'siamese_single_test'
    model = siameseCVAE(batch=batch)

    epochs = 1
    criterion = ELBO_loss
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-3)

    ells = trainVAE(model, optimizer, criterion, epochs, dl, exp_name)



        #Run from here
    modelVAE = siameseCVAE(batch=batch)
    checkpoint = torch.load(base+'checkpoints/siamese_chpt.pt')
    model.load_state_dict(checkpoint) 


    exp_name = 'velocity_single_test'
    modelVel = velocityNN()

    epochs = 1
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-3)

    ells = trainVel(modelVAE, modelVel, optimizer, criterion, epochs, dl, exp_name)