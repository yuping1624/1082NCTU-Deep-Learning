#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install opencv-python')
import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


path = os.getcwd()
#print(path)
filepath = path+r"\Pictures\anime-faces\data"
print(filepath)

test = cv2.imread(filepath+r'\1.png') 


# In[3]:


#test


# In[4]:


data = []
for i in range(21551):
    data.append(cv2.imread(filepath + r'\%s.png'%(i+1)))


# In[8]:


plt.imshow(data[0][:,:,::-1])


# In[ ]:


data = np.array(data)


# In[ ]:


data.shape


# In[ ]:


np.save('hw2_2',data)


# In[ ]:




