#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
import time


# In[4]:


# path = os.getcwd()
# print(path)
filepath = r"C:\Users\Yuping\Pictures\img_align_celeba" 
print(filepath)


# In[5]:


data = []
start = time.time()
for i in range(202599):
    #zero = 
    file = filepath + r'\%s%s.jpg'%('0'*(6-len(str(i+1))), i+1)
    img = cv2.imread(file)
    resizeimg = cv2.resize(img, (64, 64))  #INTER_CUBIC  LANCZOS4
    img = cv2.cvtColor(resizeimg, cv2.COLOR_BGR2RGB)
    #data
    data.append(img)  
end = time.time()
print((end-start)*202599)


# In[11]:


fig = plt.figure(figsize=(14,14))
for i in range(64):
    fig.add_subplot(8,8, i+1)
    plt.imshow(data[i])
    plt.xticks([])
    plt.yticks([])
plt.savefig('original_images')
plt.show()


# In[36]:


data_1 = np.array(data).transpose(0,3,1,2)


# In[16]:


np.save('hw3_1_size64_2', data_1)

