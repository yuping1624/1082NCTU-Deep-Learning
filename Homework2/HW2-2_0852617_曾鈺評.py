# -*- coding: utf-8 -*-
"""0852617_dl_hw2-2_3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16LIfH8HaDIYgBqOP46q8NGFLhs0eRXGE
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# gauth = GoogleAuth()
# gauth.LoadCredentialsFile("mycreds.txt")
# if gauth.credentials is None:
#   auth.authenticate_user()   
#   gauth.credentials = GoogleCredentials.get_application_default()
    
# elif gauth.access_token_expired:
#   gauth.Refresh()
    
# else:
#   gauth.Authorize()
# gauth.SaveCredentialsFile("mycreds.txt")
# drive = GoogleDrive(gauth)

path = os.getcwd()
if path == "/content":
  from pydrive.auth import GoogleAuth
  from pydrive.drive import GoogleDrive
  from google.colab import auth
  from google.colab import drive
  from tqdm.notebook import tqdm
  from oauth2client.client import GoogleCredentials

  gauth = GoogleAuth()
  gauth.LoadCredentialsFile("mycreds.txt")
  if gauth.credentials is None:
    auth.authenticate_user()   
    gauth.credentials = GoogleCredentials.get_application_default()
    
  elif gauth.access_token_expired:
    gauth.Refresh()
    
  else:
    gauth.Authorize()
  gauth.SaveCredentialsFile("mycreds.txt")
  drive.mount('/content/drive', force_remount=True)
  drive1 = GoogleDrive(gauth)

  savepath = "/content/drive/My Drive/Colab Notebooks/"

  link = 'https://drive.google.com/open?id=1InSMrD5S_si_x75QU2xhPR6Bn7yu8CAh'
  fluff, id = link.split('=')
  print (id)
  downloaded = drive1.CreateFile({'id':id}) 
  downloaded.GetContentFile('hw2_2.npy') 

else:
  savepath = ""
print("savepath: {}".format(savepath))

#drive

# from google.colab import drive
# print(gauth)



data_anime = np.load('{}hw2_2.npy'.format(savepath)).transpose(0,3,1,2)/255.

data_anime.shape

data_anime = (data_anime).astype(np.float32)
#print(data_anime[0])

latent_dim = 32
class AE(nn.Module):
  def __init__(self, latent_dim):
    super(AE, self).__init__()

    self.encoder = nn.Sequential(
        nn.Conv2d(3, 64, 3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, stride=2, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )
    self.mean = nn.Sequential(
        nn.Linear(128*8*8, 256),
        nn.ReLU(),
        nn.Linear(256, 100),
        nn.ReLU(),
        nn.Linear(100, latent_dim),
        nn.ReLU()
    )
    self.variance = nn.Sequential(
        nn.Linear(128*8*8, 256),
        nn.LeakyReLU(),
        nn.Linear(256, 100),
        nn.LeakyReLU(),
        nn.Linear(100, latent_dim),
        nn.LeakyReLU()
    )
    self.decoder = nn.Sequential(
        nn.Linear(latent_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 128*4),
        nn.ReLU(),
        nn.Linear(128*4, 256*4),
        nn.ReLU(),
        nn.Linear(256*4, 64*64*3),
        nn.Sigmoid()
    )
  def forward(self, x):
    x1 = self.encoder(x).reshape(-1, 128*8*8)
    mean = self.mean(x1)
    logvar = self.variance(x1)
    x = self.decoder(mean + logvar.exp().sqrt()*torch.randn_like(logvar))
    return x1, mean, logvar, x



autoencoder = AE(latent_dim=latent_dim).cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001, weight_decay=0.00001)
autoencoder.train()
batch_size = 64
n_epoch = 6
image_dataloader = DataLoader(data_anime, batch_size = batch_size, shuffle=True)
total_loss = []
for epoch in tqdm(range(n_epoch)):
  for data in image_dataloader:
    img = data
    img = img.cuda()

    output1, mean, logvar, output = autoencoder(img)
    loss = criterion(output, img.view(-1,64*64*3))*64*64*3 + 0.5*((logvar.exp() - (1 + logvar) + mean**2).sum(dim = 1).mean())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #assert not torch.isnan(loss.data).any()
  if (epoch+1) % 10 == 2:
    #print(epoch+1)
    torch.save(autoencoder.state_dict(), '{}checkpoint_{}.pth'.format(savepath, epoch+1))
  if torch.isnan(loss.data).any():
    break

  total_loss.append(loss)
  print('epoch [{}/{}], loss:{:.5f}'.format(epoch+1, n_epoch, loss.data))









torch.save(autoencoder.state_dict(), '{}checkpoint_all.pth'.format(savepath))







total_loss_np = [i.data.cpu() for i in total_loss]
fig = plt.figure(figsize=(20,10)) 
ax = plt.axes()
x = np.linspace(0, len(total_loss_np), len(total_loss_np))
ax.plot(x, total_loss_np)
#plt.xticks([])
#plt.yticks([])
plt.savefig('{}loss_image'.format(savepath))
plt.show()

exp_anime = (data_anime[0:64,:,:,:]).astype(np.float32)


fig = plt.figure(figsize=(20,20))
for i in range(len(exp_anime)):
  fig.add_subplot(8, 8, i+1)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(exp_anime[i,:].transpose(1,2,0)[:,:,::-1])
plt.savefig('{}origin_images'.format(savepath))
plt.show()



exp_dataloader = DataLoader(exp_anime, batch_size = 64, shuffle=True)
for epoch in tqdm(range(1)):
  for data in exp_dataloader:
    img = data
    img = img.cuda()

    exp_output1, exp_mean, exp_var, exp_output = autoencoder(img)
    #loss = criterion(output, img.view(-1,64*64*3))*64*64*3 + (var.sqrt().exp() - (1 + var.sqrt()) + mean**2).sum(dim = 1).mean()
    #optimizer.zero_grad()
    #loss.backward()
    #optimizer.step()
    #if (epoch+1) % 10 == 0:
    #  pass
    #  #torch.save(autoencoder.state_dict(), './checkpoints/checkpoint_{}.pth'.format(epoch+1))
  #total_loss.append(loss)


exp_output.view(-1, 3, 64, 64).permute(0, 2, 3, 1)[0].shape
len(exp_output)

fig = plt.figure(figsize=(20,20))
for i in range(len(exp_output)):
  fig.add_subplot(8, 8, i+1)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(exp_output.view(-1, 3, 64, 64).permute(0, 2, 3, 1)[i].data.cpu().numpy()[:,:,::-1])
plt.savefig('{}exp_images'.format(savepath))
plt.show()

z = torch.randn(8, latent_dim).cuda()
image_seq = []
for i in range(8):
  image = autoencoder.decoder(z[i]).view(3,64,64).permute(1,2,0).data.cpu().numpy()
  image_seq.append(image)
fig = plt.figure(figsize=(20,10))
for i in range(8):
  fig.add_subplot(1, 8, i+1)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(image_seq[i][:,:,::-1])
plt.savefig('{}prior_images'.format(savepath))
plt.show()


plt.imshow(autoencoder(torch.from_numpy(data_anime).reshape(*data_anime.shape).cuda()[[1537]])[-1].view(3,64,64).permute(1,2,0).data.cpu().numpy()[:,:,::-1])

plt.imshow(autoencoder(torch.from_numpy(data_anime).reshape(*data_anime.shape).cuda()[[1528]])[-1].view(3,64,64).permute(1,2,0).data.cpu().numpy()[:,:,::-1])

mean_1 = autoencoder(torch.from_numpy(data_anime).reshape(*data_anime.shape).cuda()[[1537]])[1]
logvar_1 = autoencoder(torch.from_numpy(data_anime).reshape(*data_anime.shape).cuda()[[1537]])[2]

mean_2 = autoencoder(torch.from_numpy(data_anime).reshape(*data_anime.shape).cuda()[[1528]])[1]
logvar_2 = autoencoder(torch.from_numpy(data_anime).reshape(*data_anime.shape).cuda()[[1528]])[2]


mean_seq = np.linspace(mean_1.data.cpu(), mean_2.data.cpu(), 9)
logvar_seq = np.linspace(logvar_1.data.cpu(), logvar_2.data.cpu(), 9)


image_seq = []
for i in range(len(mean_seq)):
  mean = torch.from_numpy(mean_seq[i]).cuda()
  logvar = torch.from_numpy(logvar_seq[i]).cuda()
  z = mean + logvar.exp().sqrt()*torch.randn_like(logvar)
  image = autoencoder.decoder(z).view(3,64,64).permute(1,2,0).data.cpu().numpy()
  image_seq.append(image)


fig = plt.figure(figsize=(20,10))
for i in range(len(image_seq)):
  fig.add_subplot(1, 10, i+1)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(image_seq[i][:,:,::-1])
plt.savefig('{}syn_images'.format(savepath))
plt.show()







#autoencoder.load_state_dict(torch.load('{}checkpoint_all.pth'.format(savepath)))