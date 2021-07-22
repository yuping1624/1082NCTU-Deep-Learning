#!/usr/bin/env python
# coding: utf-8

# In[ ]:


##Write down your visualization code here


# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.core.display import display, HTML


# In[3]:


## Animation for your generation
##input : image_list (size = (the number of sample times, how many samples created each time, image )   )
path = r'D://1082NCTU//Deep Learning//Homework3//image//'


# ### Part 1 (Adam, normalized data)

# In[5]:


img_list = [path + 'samples3_epoch{}.png'.format(i+1) for i in range(5)]
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(plt.imread(file_name), animated=True)] for file_name in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=700, repeat_delay=500, blit=True)

HTML(ani.to_jshtml())


# ### Part 2 (RMSprop, normalized data)

# In[6]:


img_list = [path + 'samples2_epoch{}.png'.format(i+1) for i in range(5)]
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(plt.imread(file_name), animated=True)] for file_name in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=700, repeat_delay=500, blit=True)

HTML(ani.to_jshtml())


# ### Part 3 (RMSprop, none normalized data)

# In[7]:


img_list = [path + 'samples1_epoch{}.png'.format(i+1) for i in range(5)]
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(plt.imread(file_name), animated=True)] for file_name in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=700, repeat_delay=500, blit=True)

HTML(ani.to_jshtml())


# In[ ]:




