# -*- coding: utf-8 -*-

import sklearn.datasets
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import random

#use_cuda = torch.cuda.is_available()
use_cuda = False
device = torch.device("cuda:0" if use_cuda else "cpu")

np.random.seed(0)
X, y = sklearn.datasets.make_moons(200,noise=0.2)


import matplotlib.pyplot as plt

plt.scatter(X[:,0],X[:,1],s=40,c=y,cmap=plt.cm.binary)

import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset, TensorDataset

#our class must extend nn.Module
class ClsNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ClsNet,self).__init__()
        #Our network consists of 3 layers. 1 input, 1 hidden and 1 output layer
        self.hidden_size = hidden_size

        #This applies Linear transformation to input data. 
        self.fc1 = nn.Linear(input_size+hidden_size, 24)
        
        #This applies linear transformation to produce output data
        self.fc2 = nn.Linear(24, 6)

        self.fc3 = nn.Linear(6, 1)

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
    
    def forward(self, x, hidden):
        hidden = hidden.repeat((x.size()[0], 1))
        combined = torch.cat((x, hidden), 1)
        # Output hidden layer
        hidden = self.i2h(combined)
        #Output of the first layer
        x = self.fc1(combined)
        #Activation function is Relu. Feel free to experiment with this
        x = torch.relu(x)
        #This produces output
        x = self.fc2(x)
        x = torch.relu(x)
        output = self.fc3(x)
        hidden = torch.mean(hidden, dim=0)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size).to(device)

    #todo: finish it         
    def predict(self, x, hidden):
        #Apply softmax to output
        output, hidden = self.forward(x, hidden)
        pred = torch.sigmoid(output).detach() > 0.5
        return pred, hidden
        
def save_model(model, file):
    torch.save(model.state_dict(), file)

def load_model(model, file, default_device='cpu'):
    device = torch.device(default_device)
    model.load_state_dict(torch.load(file, map_location=device))
    model.eval()
    return model


def adjust_learning_rate(optimizer, epoch, init_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.4 ** (epoch // 1000))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(model, feats, ys, optimizer):
    global device
    hidden = model.init_hidden()
    criterion = nn.BCEWithLogitsLoss()
    for i in range(len(feats)):
        tx = feats[i]
        ty = ys[i]
        if tx.size()[0] == 0:
            return
        output, hidden = model.forward(tx, hidden)
        loss = criterion(output, ty)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

def predict(model, feats, ys, acc, acc_all):
    global device
    hidden = model.init_hidden()
    for i in range(len(feats)):
        tx = feats[i]
        ty = ys[i]
        if tx.size()[0] == 0:
            return
        _y, hidden = model.predict(tx, hidden)
        acc.append(accuracy_score(ty, _y))
        acc_all.append(_y.size()[0])
        
class RNN_Ranker():
    def __init__(self, timestamp, load=True):
        self.model = ClsNet(12, 3)
        load_model(self.model, timestamp+'.pth')
        self.factor = 1.0
        self.hidden = self.model.init_hidden()
    
    def set_factor(self, factor):
        self.factor = factor
        self.hidden = self.model.init_hidden()

    def rank_features(self, features):
        _features = np.copy(features)
        for f in _features:
            f[1] *= self.factor
            f[4] *= self.factor
            f[5] *= self.factor
        # return np.array([0, 1, 2, 3, 4])

        test_x = []
        for i in range(len(_features)):
            for j in range(len(_features)):
                if i == j:
                    continue
                test_x.append(np.concatenate(
                (_features[i], _features[j]), axis=0))
        
        test_x = np.array(test_x)
        print(test_x.shape)
        test_x = torch.from_numpy(test_x).type(torch.FloatTensor).to(device)
        y, self.hidden = self.model.predict(test_x, self.hidden)
        y = y.detach().cpu().numpy().reshape(len(_features), len(_features)-1)
        y = np.sum(y, axis=1)
        # print(y)
        return np.argsort(y)[::-1]
