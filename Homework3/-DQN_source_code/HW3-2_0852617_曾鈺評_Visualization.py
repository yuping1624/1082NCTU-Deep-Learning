#!/usr/bin/env python
# coding: utf-8

# In[93]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate
from datetime import datetime


# In[80]:


df_1 = pd.read_csv('episode_reward_learning.csv')
df_2 = pd.read_csv('episode_reward_evaluation.csv')
x_1 = df_1['Episode']
y_1 = df_1['Reward']
rolling_mean_1 = y_1.rolling(window=5).mean()
x_2 = df_2['Episode']
y_2 = df_2['Reward']
rolling_mean_2 = y_2.rolling(window=5).mean()
df_2


# In[89]:


fig = plt.figure(figsize=(20, 20))
fig.add_subplot(2, 1, 1)
plt.plot(x_1, y_1, color='#66B3FF')
plt.plot(x_1, rolling_mean_1, color='blue')
plt.title('Training DQN Policy Reward')
plt.xlim(0,810)
plt.xlabel('Episode')
plt.ylabel('Reward')
fig.add_subplot(2, 1, 2)
plt.plot(x_2, y_2, color='orange')
plt.plot(x_2, rolling_mean_2, color='red')
plt.title('Evaluated DQN Policy Reward')
plt.xlim(0,810)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.savefig('DQN_reward_good')
plt.show()


# In[82]:


a = open(r"D:\1082NCTU\Deep Learning\Homework3\DL_HW3\DQN_source_code\log_0.txt")
text = a.readlines()


# In[91]:


y3 = [float(tex.split(", reward:")[1][:3]) for tex in text if tex.find(", reward:") != -1]
y4 = [float(tex.split(", evaluate reward:")[1][:3]) for tex in text if tex.find(", evaluate reward:") != -1]
rolling_mean_3 = pd.DataFrame(y3).rolling(window=5).mean()
rolling_mean_4 = pd.DataFrame(y4).rolling(window=5).mean()
fig = plt.figure(figsize=(20,20))
fig.add_subplot(2,1,1)
plt.title('Training DQN Policy Reward')
plt.plot(y3, color='#66B3FF')
plt.plot(rolling_mean_3, color='blue')
plt.xlabel('Episode')
plt.ylabel('Reward')
fig.add_subplot(2,1,2)
plt.title('Evaluated DQN Policy Reward')
plt.plot(y4, color='orange')
plt.plot(rolling_mean_4, color='red')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.savefig('DQN_reward_bad')
plt.show()


# In[ ]:


fig = plt.figure(figsize=(20,20))
for i in range(len(2)):
  fig.add_subplot(8, 8, i+1)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(exp_output.view(-1, 3, 64, 64).permute(0, 2, 3, 1)[i].data.cpu().numpy()[:,:,::-1])
plt.savefig('{}exp_images'.format(savepath))
plt.show()

