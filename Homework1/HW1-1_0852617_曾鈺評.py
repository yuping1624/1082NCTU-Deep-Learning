# -*- coding: utf-8 -*-
"""kernel101a6c8446.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1s1hXiFdvMQ8cUqAbJOTumAXeruqkwe_D
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn.metrics
import time
import sys
import warnings
warnings.filterwarnings("ignore")

seed = 7
np.random.seed(seed)



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train = np.load("../input/dnn-hw1/train.npz")
test = np.load("../input/dnn-hw1/test.npz")
image_train, label_train= train['image'], train['label']
image_test, label_test= test['image'], test['label']

dim  = image_train.shape[1]*image_train.shape[2]
label_train = label_train.reshape(-1)
label_test = label_test.reshape(-1)
image_train = image_train.reshape(image_train.shape[0], dim)
image_test = image_test.reshape(image_test.shape[0], dim)
train_x = np.asmatrix(image_train).astype('float64')
test_x = image_test.astype('float64')
train_y = label_train.reshape(-1).astype('int8')
test_y = label_test.reshape(-1).astype('int8')

print("Shape Train Images: (%d,%d)" % train_x.shape)
print("Shape Labels: (%d)" % train_y.shape)



def normalization(x, mu, sigma):
    
    x_norm = np.zeros_like(x)

    for n in range(len(x)):
        for j in range(len(x[n])):
            if(sigma[j]!=0):
                x_norm[n,j] = (x[n,j] - mu[j]) / sigma[j]
            else:
                x_norm[n,j] = 0
                    
    return x_norm

mu = np.mean(train_x, axis=0)
sigma = np.max(train_x, axis=0)-np.min(train_x, axis=0)

train_x = (train_x - mu ) / sigma

mu_1 = np.mean(test_x, axis=0)
sigma_1 = np.max(test_x, axis=0)-np.min(test_x, axis=0)

x_test = (test_x - mu ) / sigma

print("Test Min: %.2f" % np.min(x_test))
print("Test Max: %.2f" % np.max(x_test))
print("Train Min: %.2f" % np.min(train_x))
print("Train Max: %.2f" % np.max(train_x))

train_y = pd.get_dummies(train_y).as_matrix()

y_test = pd.get_dummies(test_y).as_matrix()

def ReLu(x, derivative=False):
    if(derivative==False):
        return np.multiply(x,(x > 0))
    else:
        return np.multiply(1,(x > 0))

def ReLu1(x, derivative=False):
    if(derivative==False):
        return x*(x > 0)
    else:
        return 1*(x > 0)

def Softmax(x):
    x -= np.max(x)
    sm = (np.exp(x).T / np.sum(np.exp(x),axis=1)).T
    return sm

def CreateWeights(ninputs=784, nh1 = 500, nh2 = 300, nclass=10):

    
    #layer1
    w1 = np.random.normal(0, ninputs**-0.5, [ninputs,nh1])
    b1 = np.random.normal(0, ninputs**-0.5, [1,nh1])
    
    #Layer2
    w2 = np.random.normal(0, nh1**-0.5, [nh1,nh2])
    b2 = np.random.normal(0, nh1**-0.5, [1,nh2])

    #Layer3
    w3 = np.random.normal(0, nh2**-0.5, [nh2,nclass])
    b3 = np.random.normal(0, nh2**-0.5, [1,nclass])
    
    return [w1,w2,w3,b1,b2,b3]

def Dropout(x, dropout_percent):
    mask = np.random.binomial([np.ones_like(x)],(1-dropout_percent))[0]  / (1-dropout_percent)
    return np.multiply(x, mask)

def predict(weights_in, x, dropout_percent=0):
    
    w1,w2,w3,b1,b2,b3  = weights_in
    
    #1-Hidden Layer
    first = ReLu(x@w1+b1)
    first = Dropout(first, dropout_percent)

    #2-Hidden Layer
    second_ori = first@w2+b2
    second = ReLu(first@w2+b2)
    second = Dropout(second, dropout_percent)
    
    #Output Layer
    third = second@w3+b3
    
    return [first, second, second_ori, Softmax(third)]

def accuracy(output, y):
    hit = 0
    output = np.argmax(output, axis=1)
    y = np.argmax(y, axis=1)
    for y in zip(output, y):
        if(y[0]==y[1]):
            hit += 1

    p = (hit*100)/output.shape[0]
    return p

def log2(x):
    if(x!=0):
        return np.log(x)
    else:
        return -np.inf
    
def log(y):
    return [[log2(nx) for nx in x]for x in y]

def cost(Y_predict, Y_right, weights, nabla):
    w1,w2,w3,b1,b2,b3  = weights
    weights_sum_square = np.mean(w1**2) + np.mean(w2**2) + np.mean(w3**2)
    Loss = -np.mean(Y_right*log(Y_predict) + (1-Y_right)*log(1-Y_predict)) + nabla/2 *  weights_sum_square
    return Loss

porcent_valid = 0
VALID_SIZE = round(train_x.shape[0]*porcent_valid)

index_data = np.arange(train_x.shape[0])
np.random.shuffle(index_data)

x_train_1 = train_x[index_data[VALID_SIZE:]]
x_valid_1 = train_x[index_data[:VALID_SIZE]]


y_train_1 = train_y[index_data[VALID_SIZE:]]
y_valid_1 = train_y[index_data[:VALID_SIZE]]


#print(train_x.shape[0])

def SGD(weights, x, t, outputs, eta, nabla, cache=None):
    
    w1,w2,w3,b1,b2,b3  = weights
    
    gamma  = 0
    if(cache==None):
            vw1 = np.zeros_like(w1)
            vw2 = np.zeros_like(w2)
            vw3 = np.zeros_like(w3)
            vb1 = np.zeros_like(b1)
            vb2 = np.zeros_like(b2)
            vb3 = np.zeros_like(b3)
    else:
        vw1,vw2,vw3,vb1,vb2,vb3 = cache
    
    first, second, third, y = outputs
   
    w3_delta = (t-y)
   
    w2_error = w3_delta@w3.T
    
    w2_delta = w2_error * ReLu(second,derivative=True)

    w1_error = w2_delta@w2.T
    w1_delta = w1_error * ReLu(first,derivative=True)
    
    eta = -eta/x.shape[0]
 
    vw3 = gamma*vw3 + eta * (second.T@w3_delta + nabla*w3)
    vb3 = gamma*vb3 + eta * w3_delta.sum(axis=0)

    vw2 = gamma*vw2 + eta * (first.T@w2_delta + nabla*w2)
    vb2 = gamma*vb2 + eta * w2_delta.sum(axis=0)

    vw1 = gamma*vw1 + eta * (x.T@w1_delta + nabla*w1)
    vb1 = gamma*vb1 + eta * w1_delta.sum(axis=0)
    
    
    w3 -= vw3
    b3 -= vb3

    w2 -= vw2
    b2 -= vb2

    w1 -= vw1
    b1 -= vb1
    
    weights = [w1,w2,w3,b1,b2,b3]
    cache = [vw1,vw2,vw3,vb1,vb2,vb3]
    
    return weights, cache

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

def elastic_transform(image, alpha, sigma, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    return map_coordinates(image, indices, order=1).reshape(shape)

def run(weights, x_train, y_train, x_valid, y_valid, epochs = 10, nbatchs=25, alpha = 1e-3, decay = 0, l2 = 0.001, dropout_percent = 0):
    
    pross = x_train.shape[0]*0.05
    
    history = [[],[],[],[],[]]
    
    index = np.arange(x_train.shape[0])
    cache = None
    print("Train data: %d" % (x_train.shape[0]))
    print("Test data: %d \n" % (x_valid.shape[0]))
    mtime = 0
    
    r_weights = []
    max_accuracy_valid = 0
    
    for j in range(epochs):
        np.random.shuffle(index)
        t = 0
        iterations = round(x_train.shape[0]/nbatchs)
        prog = ""
        sacurr = 0
        sloss = 0
        sys.stdout.write("\nEpochs: %2d \ %2d \n"% (j+1,epochs))
        stime = 0
        timeIT = time.time()
        for i in range(iterations):
            timeI = time.time()
            f = i*nbatchs
            l = f+nbatchs
            
            if(l>(x_train.shape[0]-1)):
                l = x_train.shape[0]
                
            x = np.array([elastic_transform(xx.reshape(28,28),15,3).reshape(784) for xx in x_train[index[f:l]]])
            y = y_train[index[f:l]]

            outputs = predict(weights, x, dropout_percent)
            
            loss = cost(outputs[-1], y, weights, l2)
            
            
            accuracy_t = accuracy(outputs[-1], y)
            
            sacurr += accuracy_t
            sloss += loss
            
            accuracy_train = sacurr/(i+1)
            loss_train = sloss/(i+1)
            
            weights, cache = SGD(weights, x, y, outputs, alpha, l2, cache)
            
            t+= x.shape[0]
            
            qtd = round(t/pross)
            prog = "["
            for p in range(20):
                if(p<qtd-1):
                    prog += "="
                elif(p==qtd-1):
                    prog += ">"
                else:
                    prog += "."
            prog += "]"

            
            stime += time.time()-timeI
            mtime = stime/(i+1)
            mTimeT = mtime * (iterations-i-1)
            
            sys.stdout.write("\r%5d/%5d %s ETA: %3d s - loss: %.4f  acc: %.4f" % (t, x_train.shape[0], prog, mTimeT, loss_train, accuracy_train))
            
            history[0].append([loss_train, accuracy_train])
        
        mtime = time.time()-timeIT
        alpha = alpha - (alpha*decay)
        
        outputs = predict(weights, x_valid)
        
        loss_valid = cost(outputs[-1], y_valid, weights, l2)
        accuracy_valid = accuracy(outputs[-1], y_valid)
        
        sys.stdout.write("\r%5d/%5d %s ETA: %3d s - loss: %.4f  acc: %.4f - lossValid: %.4f  accValid: %.4f " % ( t, x_train.shape[0], prog, mtime, loss_train, accuracy_train, loss_valid, accuracy_valid))
        history[1].append([loss_valid, accuracy_valid])
            
        if (accuracy_valid >= max_accuracy_valid):
            w1,w2,w3,b1,b2,b3 = weights
            r_weights = [w1.copy(),w2.copy(),w3.copy(),b1.copy(),b2.copy(),b3.copy()]
            max_accuracy_valid = accuracy_valid
        if (j % 10 == 9):
            history[2].append(outputs[-2])
            history[3].append(y_valid)
        if (j == epochs-1):
            history[4].append([y_valid, outputs[-1]])

        
    return r_weights, history

'''
weights_1 = CreateWeights(nh1=500, nh2=120)

alpha = 5e-2
epochs = 300
nbatchs = 100
h_weights_1, history_1 = run(weights_1, x_train_1, y_train_1, np.array(x_test), y_test, epochs = epochs, nbatchs=nbatchs, 
              alpha = alpha, decay = 0.0, l2 = 0, dropout_percent = 0.0)
'''

'''
weights_2 = CreateWeights(nh1=500, nh2=60)

alpha = 5e-2
epochs = 300
nbatchs = 100
h_weights_2, history_2 = run(weights_2, x_train_1, y_train_1, np.array(x_test), y_test, epochs = epochs, nbatchs=nbatchs, 
              alpha = alpha, decay = 0.00, l2 = 0, dropout_percent = 0.0)
'''

weights_3 = CreateWeights(nh1=500, nh2=2)

alpha = 5e-2
epochs = 300   # 300
nbatchs = 100
h_weights_3, history_3 = run(weights_3, x_train_1, y_train_1, np.array(x_test), y_test, epochs = epochs, nbatchs=nbatchs, 
              alpha = alpha, decay = 0.00, l2 = 0, dropout_percent = 0.0)

'''
weights_4 = CreateWeights(nh1=2, nh2=2)

alpha = 5e-2
epochs = 300
nbatchs = 100
h_weights_4, history_4 = run(weights_4, x_train_1, y_train_1, np.array(x_test), y_test, epochs = epochs, nbatchs=nbatchs, 
              alpha = alpha, decay = 0.00, l2 = 0, dropout_percent = 0.0)
'''

'''
weights_5 = CreateWeights(nh1=500, nh2=2)

alpha = 1e-2
epochs = 300
nbatchs = 100
h_weights_5, history_5 = run(weights_5, x_train_1, y_train_1, np.array(x_test), y_test, epochs = epochs, nbatchs=nbatchs, 
              alpha = alpha, decay = 0.00, l2 = 0, dropout_percent = 0.0)
'''

'''
weights_6 = CreateWeights(nh1=500, nh2=2)

alpha = 5e-3
epochs = 300
nbatchs = 100
h_weights_6, history_6 = run(weights_6, x_train_1, y_train_1, np.array(x_test), y_test, epochs = epochs, nbatchs=nbatchs, 
              alpha = alpha, decay = 0.00, l2 = 0, dropout_percent = 0.0)
              '''

'''
weights_7 = CreateWeights(nh1=500, nh2=2)

alpha = 1e-3
epochs = 300
nbatchs = 100
h_weights_7, history_7 = run(weights_7, x_train_1, y_train_1, np.array(x_test), y_test, epochs = epochs, nbatchs=nbatchs, 
              alpha = alpha, decay = 0.00, l2 = 0, dropout_percent = 0.0)
              '''

'''
weights_11 = CreateWeights(nh1=500, nh2=2)
weights_zero = []
for l in range(len(weights_11)):
    w = np.zeros_like(weights_11[l])
    weights_zero.append(w)

#print(weights_zero[0].shape)
alpha = 5e-2
epochs = 300
nbatchs = 100
h_weights_11, history_11 = run(weights_zero, x_train_1, y_train_1, np.array(x_test), y_test, epochs = epochs, nbatchs=nbatchs, 
              alpha = alpha, decay = 0.00, l2 = 0, dropout_percent = 0.0)
'''

def plot_loss_error(history):
    train_history = np.array(history[0])
    
    epo_lim = np.arange(epochs)
    lim = (np.arange(epochs)+1)*int(x_train_1.shape[0]/nbatchs)
    
    tr_loss = train_history[lim-1,:1]
    tr_err = (100 - train_history[lim-1,1:2]) / 100

    test_history = np.array(history[1])
    te_acc = test_history[:,1:2]
    #print(te_err)
    te_err = (100 - te_acc) / 100
    #print(te_err)
    plt.figure(figsize=(12,4))

    plt.subplot(1, 3, 1)    
    plt.plot(epo_lim,tr_err)
    plt.title("Train error rate")
    plt.xlabel("epochs")
    plt.ylabel("error rate")

    plt.subplot(1, 3, 2)
    plt.plot(epo_lim,te_err)
    plt.title("Test error rate")
    plt.xlabel("epochs")
    plt.ylabel("error rate")

    plt.subplot(1, 3, 3)
    plt.plot(epo_lim,tr_loss)
    plt.title("Training loss")
    plt.xlabel("epochs")
    plt.ylabel("Average Cross Entropy")
    
    
    plt.show()

#plot_1 = plot_loss_error(history_1)

#plot_2 = plot_loss_error(history_2)

plot_3 = plot_loss_error(history_3)

fig = plt.figure(figsize=(20, 20))
n_dot=1000
second = np.array(history_3[2])[:,:n_dot,:]
target = np.array(history_3[3])[:,:n_dot,:]
target = np.argmax(target[0,:,:], axis=1)
color = ['r', 'g', 'b', 'k', 'c', 'm', 'y', 'orange', 'gold', 'lightcoral']
plt.title("")
for j in range(9):
    plt.subplot(3,3,j+1)
    epoch_no = (j+1)*10
    plt.title("2D feature %s epochs" %epoch_no)
    for i in range(10):
        #target = np.argmax(target[0,:,:], axis=1)
        index = np.where(target == i)
        #print(index)
        plt.scatter(second[j,index,0], second[j,index,1], s=20, color=color[i], alpha=0.8, label='%s' %i)  #[epoch, row, x1 or x2]
    plt.legend(loc='upper left')

plt.savefig("latent_feature.png")
plt.show()

y_true = np.argmax(np.array(history_3[4])[0,0,:,:], axis=1)
y_pred = np.argmax(np.array(history_3[4])[0,1,:,:], axis=1)
confu_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
pd.DataFrame(confu_matrix)

#plot_4 = plot_loss_error(history_4)

#plot_5 = plot_loss_error(history_5)

#plot_6 = plot_loss_error(history_6)

#plot_7 = plot_loss_error(history_7)

#plot_11 = plot_loss_error(history_11)

h_weights_3[0]

h_weights_3[1]

h_weights_3[2]

h_weights_3[3]

h_weights_3[4]

h_weights_3[5]

#h_weights_11[0]

#h_weights_11[1]

#h_weights_11[2]

#h_weights_11[3]

#h_weights_11[4]

#h_weights_11[5]