# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 15:09:05 2021

@author: obouldjedr
"""
import torch
import torch.nn as nn
from functools import partial
from sklearn.metrics import classification_report
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from sklearn.metrics import top_k_accuracy_score
import argparse
import datetime
import numpy as np
import sklearn.metrics as metrics
from torchvision import datasets, transforms, models
from torchmetrics import ConfusionMatrix
import time
import torch
import torch.backends.cudnn as cudnn
import json
import gc
import os
from pathlib import Path
from torch import nn, optim
from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma
import seaborn as sns
from torchmetrics.functional import cohen_kappa
from torchmetrics.functional import average_precision
from torchmetrics.functional import precision
#from datasets import build_dataset
#from engine import train_one_epoch, evaluate
#from samplers import RASampler
#import models
#import utils
import torchvision
from sklearn.metrics import confusion_matrix
import pickle
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import timm
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()


#### F1,kappa,more digits.
#### store the accuracies in a vector 
####sqme for other metrics







#########################################################






#######################################################


#train_confmat = ConfusionMatrix(num_classes=11)

#val_confmat = ConfusionMatrix(num_classes=11)

#test_confmat = ConfusionMatrix(num_classes=11)

#Train_predicted=[]


#target_names = ['Boire', 'Dormir', 'Enfiler_une_veste','Faire_menage','Faire_pates','Faire_the',
#                'Faire_vaisselle','Laver_dents','Laver_mains','Lire_livre','Manger','Marcher','Mettre_des_chaussures',
#                'Prendre_medicaments','Utiliser_ordinateur']


list_act=['BrushingTeeth','DoingHousework','Drinking','Eating','GettingDressedUndressed','GoingToilet','PuttingAwayCleanLaundry',
          #'PuttingAwayDishes',
          'ReadingBook',
          'Resting','Sleeping','TakingShower','UsingCompSPhone']
           #,'WashingDishes']


target_names = ['BrushingTeeth','DoingHousework','Drinking','Eating','GettingDressedUndressed','GoingToilet','PuttingAwayCleanLaundry',
                #'PuttingAwayDishes',
                'ReadingBook',
                'Resting','Sleeping','TakingShower','UsingCompSPhone']
                #'Faire_vaisselle','Laver_dents','Laver_mains','Lire_livre','Manger','Marcher','Mettre_des_chaussures',
                #'Prendre_medicaments','Utiliser_ordinateur']

################################################################################
################################################################################
################################################################################



###############################


#train_report_file = open("train_report_file3_lvoo5.txt", "a")
#test_report_file = open("test_report_file3_lvoo5.txt", "a")
#val_report_file = open("val_report_file3_lvoo5.txt", "a")

#######################################################################################################

##### read the data set


#path = 'C:\Users\obouldjedr\Desktop\UWB_dataset\Butter_Windo_LVOO_Datasets_Oussama\LVOO_0\swin_large\'

timestep_window=['10sec']#,'10sec']#,'15sec']


window_size_list=[10]#,10]#,15]


#MODEL_PATH = 'C:/Users/obouldjedr/Desktop/latest_dataset_liara/UWB/classification1/str1/15sec_overlapping0.9/model_run1.pth'

#root = os.path.join(dataset_dir, 'train')
#dataset_train = datasets.ImageFolder(root)#, transform=transform)


#dataset = torchvision.datasets.ImageFolder(root=dataset_dir+"train")#, transform = train_tfms)
#train_loader = torch.utils.data.DataLoader(dataset, batch_size = 32, shuffle=True)#, num_workers = 2)

BATCH_SIZE=64

#classes=['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14']

classes=['BrushingTeeth','DoingHousework','Drinking','Eating','GettingDressedUndressed','GoingToilet','PuttingAwayCleanLaundry',
          #'PuttingAwayDishes',
          'ReadingBook',
          'Resting','Sleeping','TakingShower','UsingCompSPhone']
          #,'WashingDishes']

def train_plot_confusion_matrix(cfm,epoch):
    df_cfm = pd.DataFrame(cfm, index = classes, columns = classes)
    plt.figure(figsize = (50,40))
    #cfm_plot = sn.heatmap(df_cfm, annot=True)
    cfm_plot = sns.heatmap(df_cfm, cmap='coolwarm', annot=True, fmt=".1f")

    
    
    cfm_plot.figure.savefig(str(dataset_dir_train_mat)+"train_cfm_epoch"+str(epoch)+'.png')
    plt.close()

    return()



def val_plot_confusion_matrix(cfm,epoch):
    df_cfm = pd.DataFrame(cfm, index = classes, columns = classes)
    plt.figure(figsize = (50,40))
    #cfm_plot = sn.heatmap(df_cfm, annot=True)
    cfm_plot = sns.heatmap(df_cfm, cmap='coolwarm', annot=True, fmt=".1f")
    
    
    cfm_plot.figure.savefig(str(dataset_dir_val_mat)+"val_cfm_epoch"+str(epoch)+'.png')
    plt.close()

    return()



def test_plot_confusion_matrix(cfm):
    df_cfm = pd.DataFrame(cfm, index = classes, columns = classes)
    plt.figure(figsize = (50,20))
    #cfm_plot = sn.heatmap(df_cfm, annot=True)
    cfm_plot = sns.heatmap(df_cfm, cmap='coolwarm', annot=True, fmt=".1f")
    cfm_plot.figure.savefig(str(dataset_dir_test_mat)+'test_cfm.png')
    plt.close()

    return()

###########################################


#transform_norm = transforms.Compose([
    #transforms.ToTensor(),
#    transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                             std=[0.229, 0.224, 0.225])
#])

###try with another mean std for this data set.

#######################################################################################################


class UWBnet10(nn.Module):
  def __init__(self):
    super(UWBnet10, self).__init__()
    
    
    self.batchnorm0=nn.BatchNorm2d(3)
    

    self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, 
                           kernel_size = 7, stride = 2, padding = 0)
    
    
    self.batchnorm1=nn.BatchNorm2d(32)
    
    self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, 
                           kernel_size = 5, stride = 2, padding = 0)
    
    self.batchnorm2=nn.BatchNorm2d(64)
    
    
    self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 64,    ###it was 120
                           kernel_size = 5, stride = 2, padding = 0)
    
    
    self.batchnorm3=nn.BatchNorm2d(64)
    
    self.conv4 = nn.Conv2d(in_channels = 64, out_channels = 128, 
                           kernel_size = 3, stride = 2, padding = 0)
    
   
    
    
    self.conv5 = nn.Conv2d(in_channels = 128, out_channels = 128, 
                           kernel_size = 3, stride = 2, padding = 0)
    
    
    self.batchnorm5=nn.BatchNorm2d(128)
    
    
    self.conv6 = nn.Conv2d(in_channels = 128, out_channels = 128, 
                           kernel_size = 3, stride = 2, padding = 0)
    
    
     
    self.batchnorm6=nn.BatchNorm2d(128)
    
    
    
    
    self.linear1 = nn.Linear(128*6*1, 200)   ### updat this input  /// it was 128
    
    #self.batchnorm1=nn.BatchNorm2d(200)
    
    self.linear2 = nn.Linear(200, 100)
    
    #self.batchnorm1=nn.BatchNorm2d(100)
    
    self.linear3=  nn.Linear(100,12)
    
    
    #self.softmax=nn.Softmax(dim=1)
    
    
    self.relu = nn.ReLU()
    

  def forward(self, x):
    #print('x input is ',x[1])
    
    #
    x=self.batchnorm0(x)
    #print('x input is ',x[1])
    #input()
    
    x = self.conv1(x)
    x = self.relu(x)
    x=self.batchnorm1(x)
    #x = self.avgpool(x)
    x = self.conv2(x)
    x = self.relu(x)
    x=self.batchnorm2(x)
    #x = self.avgpool(x)
    x = self.conv3(x)
    x = self.relu(x)
    x=self.batchnorm3(x)
    
    
    x = self.conv4(x)
    x = self.relu(x)
   
    
    x = self.conv5(x)
    x = self.relu(x)
    x=self.batchnorm5(x)

    x = self.conv6(x)
    x = self.relu(x)
    x=self.batchnorm6(x)
    
    
    
    #print('before reshape and after')
    #print(x.size())
    #x = x.reshape(x.shape[0], -1)
    
    #print('x  before linear is ',x.size())
    #x = torch.flatten(x, 1)
    
    x = x.reshape(x.size(0), -1)
    
    #print('x  before linear is ',x.size())
    #input()
    
    x = self.linear1(x)
    x = self.relu(x)
    
    #print('passed')
    #x = self.tanh(x)
    x = self.linear2(x)
    x = self.relu(x)
    
    
    
    x=self.linear3(x)
    
    #print('before softmax',x.size())
    #print('x is ',x)
    #a=x[0].tolist()
    #a=sum(a)
    #print('a is ',a)
    #input()
    #x=self.softmax(x)
    #print('before softmax',x.size())
    #print('x is ',x)
    #b=x[0].tolist()
    #b=sum(b)
    #print('b is ',b)
    #input()
    return x       
    
########################################################################################################


class UWBnet15(nn.Module):
  def __init__(self):
    super(UWBnet15, self).__init__()
    
    
    self.batchnorm0=nn.BatchNorm2d(3)

    self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, 
                           kernel_size = 7, stride = 2, padding = 0)
    
    
    self.batchnorm1=nn.BatchNorm2d(32)
    
    self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, 
                           kernel_size = 5, stride = 2, padding = 0)
    
    self.batchnorm2=nn.BatchNorm2d(64)
    
    
    self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 64,    
                           kernel_size = 5, stride = 2, padding = 0)
    
    
    self.batchnorm3=nn.BatchNorm2d(64)
    
    self.conv4 = nn.Conv2d(in_channels = 64, out_channels = 128, 
                           kernel_size = 3, stride = 2, padding = 0)
    
  
    
    
    self.conv5 = nn.Conv2d(in_channels = 128, out_channels = 128, 
                           kernel_size = 3, stride = 2, padding = 0)
    
    
    self.batchnorm5=nn.BatchNorm2d(128)
    
    
    self.conv6 = nn.Conv2d(in_channels = 128, out_channels = 128, 
                           kernel_size = 3, stride = 2, padding = 0)
    
    
     
    self.batchnorm6=nn.BatchNorm2d(128)
    
    
    
    
    self.linear1 = nn.Linear(128*10*1, 200)   ### updat this input  /// it was 128
    
    #self.batchnorm1=nn.BatchNorm2d(200)
    
    self.linear2 = nn.Linear(200, 100)
    
    #self.batchnorm1=nn.BatchNorm2d(100)
    
    self.linear3=  nn.Linear(100,12)
    
    
    
    self.relu = nn.ReLU()
    

  def forward(self, x):
    #print('x input is ',x.size())
    
    #print('x input is ',x.size())
    #input()
    
    
    
    x=self.batchnorm0(x)
    
    #print('x input is ',x.size())
    #input()
    
    x = self.conv1(x)
    x = self.relu(x)
    x=self.batchnorm1(x)
    #x = self.avgpool(x)
    x = self.conv2(x)
    x = self.relu(x)
    x=self.batchnorm2(x)
    #x = self.avgpool(x)
    x = self.conv3(x)
    x = self.relu(x)
    x=self.batchnorm3(x)
    
    
    x = self.conv4(x)
    x = self.relu(x)
    
    
    x = self.conv5(x)
    x = self.relu(x)
    x=self.batchnorm5(x)

    x = self.conv6(x)
    x = self.relu(x)
    x=self.batchnorm6(x)
    
    
    
    #print('before reshape and after')
    #print(x.size())
    #x = x.reshape(x.shape[0], -1)
    
    #print('x  before linear is ',x.size())
    #x = torch.flatten(x, 1)
    
    x = x.reshape(x.size(0), -1)
    
    #print('x  before linear is ',x.size())
    #input()
    
    x = self.linear1(x)
    x = self.relu(x)
    
    #print('passed')
    #x = self.tanh(x)
    x = self.linear2(x)
    x = self.relu(x)
    
    x=self.linear3(x)
    
    return x       
    
      
    
   
    
   #################################################################################################
    
class UWBnet(nn.Module):
  def __init__(self):
    super(UWBnet, self).__init__()
    
    self.batchnorm0=nn.BatchNorm2d(3)
    

    self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, 
                           kernel_size = 7, stride = 2, padding = 0)
    
    
    self.batchnorm1=nn.BatchNorm2d(32)
    
    self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, 
                           kernel_size = 5, stride = 2, padding = 0)
    
    self.batchnorm2=nn.BatchNorm2d(64)
    
    
    self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 64,    
                           kernel_size = 5, stride = 2, padding = 0)
    
    
    self.batchnorm3=nn.BatchNorm2d(64)
    
    self.conv4 = nn.Conv2d(in_channels = 64, out_channels = 128, 
                           kernel_size = 3, stride = 2, padding = 0)
    
  
    
    
    self.conv5 = nn.Conv2d(in_channels = 128, out_channels = 128, 
                           kernel_size = 3, stride = 2, padding = 0)
    
    
    self.batchnorm5=nn.BatchNorm2d(128)
    
    
    self.conv6 = nn.Conv2d(in_channels = 128, out_channels = 128, 
                           kernel_size = 3, stride = 2, padding = 0)
    
    
     
    self.batchnorm6=nn.BatchNorm2d(128)
    
    
    
    
    self.linear1 = nn.Linear(128*2*1, 200)   ### updat this input  /// it was 128
    
    #self.batchnorm1=nn.BatchNorm2d(200)
    
    self.linear2 = nn.Linear(200, 100)
    
    #self.batchnorm1=nn.BatchNorm2d(100)
    
    self.linear3=  nn.Linear(100,12)
    
    
    
    self.relu = nn.ReLU()
    

  def forward(self, x):
    #print('x input is ',x.size())
    
    #print('x input is ',x.size())
    #input()
    
    x=self.batchnorm0(x)
    
    #print('x input is ',x.size())
    #input()
    
    
    x = self.conv1(x)
    x = self.relu(x)
    x=self.batchnorm1(x)
    #x = self.avgpool(x)
    x = self.conv2(x)
    x = self.relu(x)
    x=self.batchnorm2(x)
    #x = self.avgpool(x)
    x = self.conv3(x)
    x = self.relu(x)
    x=self.batchnorm3(x)
    
    
    x = self.conv4(x)
    x = self.relu(x)
    
    
    x = self.conv5(x)
    x = self.relu(x)
    x=self.batchnorm5(x)

    x = self.conv6(x)
    x = self.relu(x)
    x=self.batchnorm6(x)
    
    
    
    #print('before reshape and after')
    #print(x.size())
    #input()
    #x = x.reshape(x.shape[0], -1)
    
    #print('x  before reshape  is 5sec ',x.size())
    #x = torch.flatten(x, 1)
    #x=x.view(x.shape[0],-1)
    
    x = x.reshape(x.size(0), -1)
    
    #print('x  after reshape is 5sec ',x.size())
    #input()
    
    x = self.linear1(x)
    x = self.relu(x)
    
    #print('passed')
    #x = self.tanh(x)
    x = self.linear2(x)
    x = self.relu(x)
    
    x=self.linear3(x)
    
    return (x)    





###########################################################################################################
class CustomDataset_train(Dataset):
    def __init__(self):
        
        #super(EchoDataset).__init__()
        self.data_path = dataset_dir_train
        file_list = glob.glob(self.data_path + "*")
        #print('file list is ',file_list)
        #input()
        self.data = []
        for class_path in file_list:
            class_name = class_path.split("\\")[-1]
            #print('classe name is ',class_name)
            #input()
            for data_path in glob.glob(class_path + "/*.pickle"):
                #print('class parth is ',class_path)
                #input()
                self.data.append([data_path, class_name])
                #print('self data is ',self.data)
                #input()
        #print('the data is ',self.data)
        #print()
        #print('self.data shape is ',self.data[0])
        #print('self.data shape is ',len(self.data[0]))
        #input()
        self.class_map = {'BrushingTeeth': 0,
                          'DoingHousework':1,
                          'Drinking':2,
                          'Eating':3,
                          'GettingDressedUndressed':4,
                          'GoingToilet':5,
                          'PuttingAwayCleanLaundry':6,
                          'ReadingBook':7,
                          'Resting':8,
                          'Sleeping':9,
                          'TakingShower':10,
                          'UsingCompSPhone':11,
                          ###update the rest of activites once data available
                          #'Marcher_':11,
                          #'Mettre_des_chaussures_':12,
                          #'Prendre_medicaments_':13,
                          #'Utiliser_ordinateur_':14
                          
                          
                          }
        
        #print('the data is ',len(self.data))
        
        #self.transform = transform_norm
        
        #print('len is ',len(self.data))
        #print('self data ',type(self.data))
        #print('self.data',len(self.data[0]))
        #print('self data ',type(self.data[0]))
        #print('self data ',self.data)
        #input()
        ##################################
        
        #data = self.to_tensor(data)
        #self.transformations = transforms.Compose([
        #                        transforms.ToTensor()]
        
        #self.to_tensor = transforms.ToTensor()
        #self.img_dim = (416, 416)
    def __len__(self):
        #print('the data is ',len(self.data))
        return len(self.data)
    def __getitem__(self, idx):
        #print('get item')
        #print()
        data_path, class_name = self.data[idx]
        #data = cv2.imread(img_path)
        #img = cv2.resize(img, self.img_dim)
        data = pickle.load(file=open(data_path, "rb"))
        #print('data before is ',data)
        #print('data before is ',type(data))
        #print('data before is ',data.shape)
        #input()
        #print()
        
        
        #data=data[data!= float('inf')]
        
        #print('data after is ',data.shape)
        
        class_id = self.class_map[class_name]
        
        
        fused_tensor_data=torch.from_numpy(data)
        
        #fused_tensor_data=self.transform(fused_tensor_data)
        
        
        #fused_tensor_data=torch.squeeze(fused_tensor_data)
        #print('after transform',fused_tensor_data.size())
        #print(fused_tensor_data)
        #input()
        
        #input()
        
        return (fused_tensor_data, class_id)
    
###########################################################################################################


class CustomDataset_val(Dataset):
    def __init__(self):
        self.data_path = dataset_dir_val
        file_list = glob.glob(self.data_path + "*")
        #print('file list is ',file_list)
        #input()
        self.data = []
        for class_path in file_list:
            #print('self.data len is ',len(self.data)) 
            class_name = class_path.split("\\")[-1]
            #print('classe name is ',class_name)
            #print('class path ',class_path)
            #input()
            for data_path in glob.glob(class_path + "/*.pickle"):
                self.data.append([data_path, class_name])
                #print('self data is ',len(self.data))
                #print('data path is ',data_path)
                
            #print('self.data len is ',len(self.data))    
            #input()
        self.class_map = {'BrushingTeeth': 0,
                          'DoingHousework':1,
                          'Drinking':2,
                          'Eating':3,
                          'GettingDressedUndressed':4,
                          'GoingToilet':5,
                          'PuttingAwayCleanLaundry':6,
                          'ReadingBook':7,
                          'Resting':8,
                          'Sleeping':9,
                          'TakingShower':10,
                          'UsingCompSPhone':11,
                          ###update the rest of activites once data available
                          #'Marcher_':11,
                          #'Mettre_des_chaussures_':12,
                          #'Prendre_medicaments_':13,
                          #'Utiliser_ordinateur_':14
                          
                          
                          }        
                
                
                
                
        #print('the data is ',len(self.data))
        #print()
        #input()
        
        #self.transform = transform_norm
        ##################################
        
        #data = self.to_tensor(data)
        #self.transformations = transforms.Compose([
        #                        transforms.ToTensor()]
        
        #self.to_tensor = transforms.ToTensor()
        #self.img_dim = (416, 416)
    def __len__(self):
        #print('data leng is ',len(self.data))
        return len(self.data)
    def __getitem__(self, idx):
        data_path, class_name = self.data[idx]
        #data = cv2.imread(img_path)
        #img = cv2.resize(img, self.img_dim)
        data = pickle.load(file=open(data_path, "rb"))
        #print('data before is ',data)
        #print()
        #print('data before is ',data.shape)
        #input()
        #print()
        
        
        #data=data[data!= float('inf')]
        
        #print('data after is ',data.shape)
        #input()
        
        
        
        class_id = self.class_map[class_name]
        fused_tensor_data=torch.from_numpy(data)
        #print(fused_tensor_data)
        #input()
        return (fused_tensor_data, class_id)

###########################################################################################


class CustomDataset_test(Dataset):
    def __init__(self):
        self.data_path = dataset_dir_test
        file_list = glob.glob(self.data_path + "*")
        #print('file list is ',file_list)
        #input()
        self.data = []
        for class_path in file_list:
            class_name = class_path.split("\\")[-1]
            #print('class path ',class_path)
            #print('classe name is ',class_name)
            #input()
            for data_path in glob.glob(class_path + "/*.pickle"):
                self.data.append([data_path, class_name])
                #print('data path is ',data_path)
                #print('self data is ',self.data)
                #input()
        #print('the data is ',self.data)
        #print()
        #input()
        self.class_map = {'BrushingTeeth': 0,
                          'DoingHousework':1,
                          'Drinking':2,
                          'Eating':3,
                          'GettingDressedUndressed':4,
                          'GoingToilet':5,
                          'PuttingAwayCleanLaundry':6,
                          'ReadingBook':7,
                          'Resting':8,
                          'Sleeping':9,
                          'TakingShower':10,
                          'UsingCompSPhone':11,
                          ###update the rest of activites once data available
                          #'Marcher_':11,
                          #'Mettre_des_chaussures_':12,
                          #'Prendre_medicaments_':13,
                          #'Utiliser_ordinateur_':14
                          
                          
                          }        
                
        
        ##################################
        #print('the data is ',len(self.data))
        
    def __len__(self):
        #print('the data is ',len(self.data))
        return len(self.data)
    def __getitem__(self, idx):
        data_path, class_name = self.data[idx]
        #data = cv2.imread(img_path)
        #img = cv2.resize(img, self.img_dim)
        data = pickle.load(file=open(data_path, "rb"))
        
        #print('data before is ',data.shape)
        #input()
        #print()
        
        
        #data=data[data!= float('inf')]
        
        #print('data after is ',data.shape)
        
        #input()
        
        #print('data before is ',data)
        class_id = self.class_map[class_name]
        
        
        fused_tensor_data=torch.from_numpy(data)
        #print(fused_tensor_data)
        #input()
        return (fused_tensor_data, class_id)



###########################################################################################################


    


#####################################################################################################################
#####################################################################################################################
#######################################################################################################################
####################################################################################################################

def get_data():
    
    #train_transforms = transforms.Compose([transforms.ToTensor()])

    #val_transforms = transforms.Compose([transforms.ToTensor()])
    
    #test_transforms = transforms.Compose([transforms.ToTensor()])
    
    
    #print('transformers defined')
    
    train_dataset=CustomDataset_train()
    
    
    #train_dataset= torchvision.datasets.DatasetFolder(root=dataset_dir_train, loader=pickle_loader, extensions='.pickle')
    
    print('passed training')

    valid_dataset=CustomDataset_val()
    
    #valid_dataset= torchvision.datasets.DatasetFolder(root=dataset_dir_val, loader=pickle_loader, extensions='.pickle')

    test_dataset=CustomDataset_test()
    
    #test_dataset= torchvision.datasets.DatasetFolder(root=dataset_dir_test, loader=pickle_loader, extensions='.pickle')
    
    
    
    print('data set created')
    
    
    print('train_dataset is ',len(train_dataset))
    print('val_dataset is ',len(valid_dataset))
    print('test_dataset is ',len(test_dataset))
    #input()
    
    return(train_dataset,valid_dataset,test_dataset)
 

def get_data_loader(train_dataset,valid_dataset,test_dataset):
    
   

    
    
    
    
    

     

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = BATCH_SIZE,shuffle=True, num_workers=4,pin_memory = True)

    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = BATCH_SIZE,shuffle=True,num_workers=4,pin_memory = True)#,shuffle = True)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = BATCH_SIZE,shuffle=True,num_workers=4,pin_memory = True)#,shuffle = True)
    
    
    return(train_loader,valid_loader,test_loader)   

#############################################################################################################

def train(model, optimizer, train_loader, epoch,writer):
        model.train()
        
        train_loss=0
        train_accuracy=0
        correct=0
       
        
        train_predicted=[]
        train_target=[]
        
        print('inside train epoch ',epoch)
        top2=0
        
        
        
        top3=0
        top4=0
        top5=0
        #compt=0
        #for batch_idx, (data, target) in enumerate(train_loader):
        for data, target in train_loader:    
         
            if use_cuda:    
                data, target = data.cuda(), target.cuda()
                #print('data',data.size())
                data=data.permute(0, 3, 1,2)
             
                data=data.float()
                #print('data is ',data)
                #input()
                
            #data, target = Variable(data), Variable(target, requires_grad=False)
            optimizer.zero_grad()
            
           
            output = model(data)
        
            
            _, predicted = torch.max(output.data, 1)
         
            #_, predicted = torch.max(output.data, 1)
            
            target = target.long() 
            
            
            
            
            
            
            
            target_top_numpy=target.cpu().detach().numpy()
            
            predicted_top_numpy=output.cpu().detach().numpy()
            
            
            
            
            
            
          
            
          
            
       
            top2=top2+top_k_accuracy_score(target_top_numpy, predicted_top_numpy, k=2,normalize=False,labels=range(12))
            
            
           
           
            top3=top3+top_k_accuracy_score(target_top_numpy, predicted_top_numpy, k=3,normalize=False,labels=range(12))
            top4=top4+top_k_accuracy_score(target_top_numpy, predicted_top_numpy, k=4,normalize=False,labels=range(12))
            top5=top5+top_k_accuracy_score(target_top_numpy, predicted_top_numpy, k=5,normalize=False,labels=range(12))
            loss = criterion(output,target)  
           
            loss.backward()
            optimizer.step()
            
            #pred = output.data.max(1, keepdim=True)[1]
            
        
            #train_loss+=loss.data 
            
            train_loss+=loss.item()*data.size(0)
            
            
            optimizer.zero_grad()
            
            
         
            predicted_cpu=predicted.cpu()
            predicted_list=predicted.tolist()
            
            target_cpu=target.cpu()
            target_list=target.tolist()
            
            #train_predicted.append(predicted_list)
            train_predicted=train_predicted+predicted_list
            
            #train_target.append(target_list)
            train_target=train_target+target_list
            
            
            correct +=(target==predicted).sum().item()
            
       
        
        
        #################################################################################
        
        train_top2_epoch=100*top2/len(train_loader.dataset)
        print('top 2 accuracy is ',train_top2_epoch)
        
        train_top3_epoch=100*top3/len(train_loader.dataset)
        print('top 3 accuracy is ',train_top3_epoch)
        
        train_top4_epoch=100*top4/len(train_loader.dataset)
        print('top 4 accuracy is ',train_top4_epoch)
        
        train_top5_epoch=100*top5/len(train_loader.dataset)
        print('top 5 accuracy is ',train_top5_epoch)
        
        
        
        train_accuracy=100. * correct / len(train_loader.dataset)
        print('accuracy is ',train_accuracy)
        #input()
        #train_loss /= len(train_loader)
        
        train_loss = train_loss/len(train_loader.sampler)    
        
        
        #train_loss    
            
            #if batch_idx % args.log_interval == 0:
        print('Train Epoch: {} '.format(epoch))
        #print('IMU is ',IMU)
            #               100. * batch_idx / len(train_loader), loss.data))
        print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
               train_loss, correct, len(train_loader.dataset),
               100. * correct / len(train_loader.dataset)))
        
        acc=metrics.accuracy_score(train_target, train_predicted)*100
        balanced_acc=metrics.balanced_accuracy_score(train_target,train_predicted)*100
        
        precision=metrics.precision_score(train_target,train_predicted,average='weighted')*100
        recall=metrics.recall_score(train_target,train_predicted,average='weighted')*100
        f1_score=metrics.f1_score(train_target,train_predicted,average='weighted')*100
        
        train_kappa=metrics.cohen_kappa_score(train_target,train_predicted)*100
        
        train_predicted_torch=torch.LongTensor(train_predicted)
        train_target_torch=torch.LongTensor(train_target)
        
        #torch.LongTensor
        
        train_report=classification_report(train_target, train_predicted, target_names=target_names) 
       
        cnf_matrix=confusion_matrix(train_target, train_predicted)
        
        train_plot_confusion_matrix(cnf_matrix,epoch)
        
        #print('train report',train_report)
        #input()
      
        with open(train_report_file, "a") as train_text_file:
            train_text_file.write('****************************************************')
            train_text_file.write('\n')
            train_text_file.write('epoch ')
            train_text_file.write(str(epoch))
            train_text_file.write('\n')
            train_text_file.write('train report ')
            train_text_file.write('\n')
            train_text_file.write(train_report)
            train_text_file.write('\n')
            train_text_file.write('train loss ')
            train_text_file.write(str(train_loss))
            train_text_file.write('\n')
            train_text_file.write('train accuracy ')
            train_text_file.write(str(train_accuracy))
            
            
            train_text_file.write('\n')
            train_text_file.write(' train top 2 accuracy is ')
            train_text_file.write(str(train_top2_epoch))
            train_text_file.write('\n')
            train_text_file.write(' train top 3 accuracy is ')
            train_text_file.write(str(train_top3_epoch))
            train_text_file.write('\n')
            train_text_file.write(' train top 4 accuracy is ')
            train_text_file.write(str(train_top4_epoch))
            train_text_file.write('\n')
            train_text_file.write(' train top 5 accuracy is ')
            train_text_file.write(str(train_top5_epoch))
            
            
            
            
            
            #print()
            #train_text_file.write('\n')
            #train_text_file.write('accuracy ')
            #train_text_file.write(str(acc))
            train_text_file.write('\n')
            train_text_file.write('balanced accuracy ')
            train_text_file.write(str(balanced_acc))
            train_text_file.write('\n')
            train_text_file.write('recall ')
            train_text_file.write(str(recall))
            train_text_file.write('\n')
            train_text_file.write('f1 score ')
            train_text_file.write(str(f1_score))
            train_text_file.write('\n')
            train_text_file.write('train kappa ')
            train_text_file.write(str(train_kappa))
            train_text_file.write('\n')
            train_text_file.write('train precision ')
            train_text_file.write(str(precision))
            train_text_file.write('\n')
            #train_text_file.write('\n')    
            #train_text_file.write('train conf mat ')
            #train_text_file.write('\n')
            #train_text_file.write(str(cnf_matrix))
            train_text_file.write('\n')
            
            
            ###########################################################
                
                
          
            
            
            
            
            #print('loss is '.format())
        train_text_file.close()
   
        
        
        
        
            
        return (train_accuracy,train_loss,balanced_acc,train_kappa,f1_score,precision,recall,train_top2_epoch,train_top3_epoch,train_top4_epoch,train_top5_epoch)
       ######################################################################################################
                ##################################################################################

def val(model, optimizer, val_loader, epoch,writer):
        model.eval()
        with torch.no_grad():
            val_loss = 0
            correct = 0
            val_accuracy=0
            val_top2=0
            val_top3=0
            val_top4=0
            val_top5=0
            val_target=[]
            val_predicted=[]
            for data, target in val_loader:
              
                if use_cuda:    
                    data, target = data.cuda(), target.cuda()
                  
                    data=data.permute(0, 3, 1,2)
                    
                    data=data.float()
                    #print('target',target)
                    #print(target.size())
                    #input()
               
                output = model(data)
                
                _, predicted = torch.max(output.data, 1)
                
                target = target.long() 
                
                target_top_numpy=target.cpu().detach().numpy()
            
                predicted_top_numpy=output.cpu().detach().numpy()
                
                
                val_top2=val_top2+top_k_accuracy_score(target_top_numpy, predicted_top_numpy, k=2,normalize=False,labels=range(12))
            
            #print('top2 is ',top2)
            #input()
                val_top3=val_top3+top_k_accuracy_score(target_top_numpy, predicted_top_numpy, k=3,normalize=False,labels=range(12))
                val_top4=val_top4+top_k_accuracy_score(target_top_numpy, predicted_top_numpy, k=4,normalize=False,labels=range(12))
                val_top5=val_top5+top_k_accuracy_score(target_top_numpy, predicted_top_numpy, k=5,normalize=False,labels=range(12))
                
                
                
                
                
                
                loss=criterion(output,target)
                
                val_loss += loss.item()*data.size(0)
                            #loss.item()*data.size(0)
                #pred = probs.data.max(1, keepdim=True)[1]  # get the index of the max probability
                #correct += pred.eq(target.data.view_as(pred)).cpu().sum()
                target_cpu=target.cpu()
                predicted_cpu=predicted.cpu()
                predicted_list=predicted_cpu.tolist()
                target_list=target_cpu.tolist()
                val_predicted=val_predicted+predicted_list#.append(predicted)
                val_target=val_target+target_list#.append(target)
                correct+=(predicted == target).sum().item()
                
            val_accuracy=100. * correct / len(val_loader.dataset)
            
            
            
            val_top2_epoch=100*val_top2/len(val_loader.dataset)
            print('top 2 accuracy is ',val_top2_epoch)
        
            val_top3_epoch=100*val_top3/len(val_loader.dataset)
            print('top 3 accuracy is ',val_top3_epoch)
        
            val_top4_epoch=100*val_top4/len(val_loader.dataset)
            print('top 4 accuracy is ',val_top4_epoch)
        
            val_top5_epoch=100*val_top5/len(val_loader.dataset)
            print('top 5 accuracy is ',val_top5_epoch)
            
            
            
            
            
            
            
            
            val_loss = val_loss/len(val_loader.sampler)
            
            
            #val_loss /= len(val_loader.dataset)
            print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            val_loss, correct, len(val_loader.dataset),
            100. * correct / len(val_loader.dataset)))
            
            
            acc=metrics.accuracy_score(val_target, val_predicted)*100
            balanced_acc=metrics.balanced_accuracy_score(val_target,val_predicted)*100
            recall=metrics.recall_score(val_target,val_predicted,average='weighted')*100
            precision=metrics.precision_score(val_target,val_predicted,average='weighted')*100
            f1_score=metrics.f1_score(val_target,val_predicted,average='weighted')*100
            val_kappa=metrics.cohen_kappa_score(val_target,val_predicted)*100
            
        
            
            
            
         
            
            val_report=classification_report(val_target, val_predicted, target_names=target_names) 
            
            
            val_cnf_matrix=confusion_matrix(val_target, val_predicted)
            #print('confusion matrix is ',val_cnf_matrix)
            
            val_plot_confusion_matrix(val_cnf_matrix,epoch)
            
            #input()
            #val_cnf_matrix=val_confmat(torch.LongTensor(val_target),torch.LongTensor(val_predicted))
            
       
            with open(val_report_file, "a") as val_text_file:
                val_text_file.write('****************************************************')
                val_text_file.write('\n')
                val_text_file.write('epoch ')
                val_text_file.write(str(epoch))
                val_text_file.write('\n')
                val_text_file.write(' valid report is ')
                val_text_file.write('\n')
                val_text_file.write(val_report)
                val_text_file.write('\n')
                val_text_file.write(' valid loss is ')
                val_text_file.write(str(val_loss))
                val_text_file.write('\n')
                val_text_file.write('accuracy ')
                val_text_file.write(str(val_accuracy))
                val_text_file.write('\n')
                val_text_file.write('\n')
                #val_text_file.write('accuracy ')
                #val_text_file.write(str(acc))
                
                
                
                val_text_file.write('\n')
                val_text_file.write(' val top 2 accuracy is ')
                val_text_file.write(str(val_top2_epoch))
                val_text_file.write('\n')
                val_text_file.write(' val top 3 accuracy is ')
                val_text_file.write(str(val_top3_epoch))
                val_text_file.write('\n')
                val_text_file.write(' val top 4 accuracy is ')
                val_text_file.write(str(val_top4_epoch))
                val_text_file.write('\n')
                val_text_file.write(' val top 5 accuracy is ')
                val_text_file.write(str(val_top5_epoch))
            
            
            
            
                #print()
                
                val_text_file.write('\n')
                val_text_file.write('balanced accuracy ')
                val_text_file.write(str(balanced_acc))
                val_text_file.write('\n')
                val_text_file.write('recall ')
                val_text_file.write(str(recall))
                val_text_file.write('\n')
                val_text_file.write('f1 score ')
                val_text_file.write(str(f1_score))
                val_text_file.write('\n')
                val_text_file.write('cohen kappa ')
                val_text_file.write(str(val_kappa))
                val_text_file.write('\n')
                val_text_file.write('val precision ')
                val_text_file.write(str(precision))
                val_text_file.write('\n')
             
                
                val_text_file.write('\n')
                
                
                
                
                ###########################################################
                
                
              
                
                
                #print('loss is '.format())
            val_text_file.close()
            
            
         
          
            return (val_accuracy,val_loss,balanced_acc,val_kappa,f1_score,precision,recall,val_top2_epoch,val_top3_epoch,val_top4_epoch,val_top5_epoch)         
                
                

                ############################################################################################
                ###########################################################################################
                #############################################################################################
                

def test(model, optimizer, test_loader):
        model.eval()
        
        with torch.no_grad():
            
            test_loss = 0
            correct = 0
            
            test_accuracy=0
            test_top2=0
            test_top3=0
            test_top4=0
            test_top5=0
            
            test_target=[]
            test_predicted=[]
            
            
            for data, target in test_loader:
                
                if use_cuda:    
                    data, target = data.cuda(), target.cuda()
                    #print('data',data)
                    #print('data size ',data.size())
                    #print()
                    data=data.permute(0, 3, 1,2)
                    
                    data=data.float()
                 
                    
               
                
                output= model(data)
                
                _, predicted = torch.max(output.data, 1)
                
                target = target.long() 
                
                
                target_top_numpy=target.cpu().detach().numpy()
            
                predicted_top_numpy=output.cpu().detach().numpy()
                
                
                
                test_top2=test_top2+top_k_accuracy_score(target_top_numpy, predicted_top_numpy, k=2,normalize=False,labels=range(12))
            
            #print('top2 is ',top2)
            #input()
                test_top3=test_top3+top_k_accuracy_score(target_top_numpy, predicted_top_numpy, k=3,normalize=False,labels=range(12))
                test_top4=test_top4+top_k_accuracy_score(target_top_numpy, predicted_top_numpy, k=4,normalize=False,labels=range(12))
                test_top5=test_top5+top_k_accuracy_score(target_top_numpy, predicted_top_numpy, k=5,normalize=False,labels=range(12))
                
                
                
                
                
                
                
                
                loss=criterion(output,target)
                
                
                
                
                
                
                
                #test_loss += loss.data
                
                
                test_loss += loss.item()*data.size(0)
                
                
                
                
              
                predicted_cpu=predicted.cpu()
                target_cpu=target.cpu()
                
                predicted_list=predicted_cpu.tolist()
                target_list=target_cpu.tolist()
                
                test_predicted=test_predicted+predicted_list
                
                test_target=test_target+target_list
            
                
                correct+=(predicted == target).sum().item()
              
            test_accuracy=100. * correct / len(test_loader.dataset)
            
            test_loss = test_loss/len(test_loader.sampler)
            
            #test_loss /= len(test_loader)
            
            
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
            
            test_top2=100*test_top2/len(test_loader.dataset)
            print('top 2 accuracy is ',test_top2)
        
            test_top3=100*test_top3/len(test_loader.dataset)
            print('top 3 accuracy is ',test_top3)
        
            test_top4=100*test_top4/len(test_loader.dataset)
            print('top 4 accuracy is ',test_top4)
        
            test_top5=100*test_top5/len(test_loader.dataset)
            print('top 5 accuracy is ',test_top5)
            
            
            
            
            
            
           
            acc=metrics.accuracy_score(test_target, test_predicted)*100
            balanced_acc=metrics.balanced_accuracy_score(test_target,test_predicted)*100
            
            #### add accuracy here with weighted 
            #### add the precision here with weighted 
            
            
            precision=metrics.precision_score(test_target,test_predicted,average='weighted')*100
            recall=metrics.recall_score(test_target,test_predicted,average='weighted')*100
            f1_score=metrics.f1_score(test_target,test_predicted,average='weighted')*100
            test_kappa=metrics.cohen_kappa_score(test_target,test_predicted)*100
            
            testreport=classification_report(test_target, test_predicted, target_names=target_names)  
            
            
            test_cnf_matrix=confusion_matrix(test_target, test_predicted)
            test_plot_confusion_matrix(test_cnf_matrix)
            
            #input()
            #test_cnf_matrix=test_confmat(torch.LongTensor(test_target),torch.LongTensor(test_predicted))
            
            with open(test_report_file, "a") as test_text_file:
                test_text_file.write('****************************************************')
                test_text_file.write('\n')
              
                
                test_text_file.write('\n')
                test_text_file.write('\n')
                test_text_file.write('accuracy ')
                test_text_file.write(str(acc))
                test_text_file.write('\n')
                test_text_file.write('\n')
                
                test_text_file.write('test_report ')
                test_text_file.write('\n')
                test_text_file.write(testreport)
            
               
                test_text_file.write('\n')
                test_text_file.write(' test top 2 accuracy is ')
                test_text_file.write(str(test_top2))
                test_text_file.write('\n')
                test_text_file.write(' test top 3 accuracy is ')
                test_text_file.write(str(test_top3))
                test_text_file.write('\n')
                test_text_file.write(' test top 4 accuracy is ')
                test_text_file.write(str(test_top4))
                test_text_file.write('\n')
                test_text_file.write(' test top 5 accuracy is ')
                test_text_file.write(str(test_top5))
                
                test_text_file.write('\n')
                test_text_file.write('loss is ')
                test_text_file.write(str(test_loss))
                test_text_file.write('\n')
                
                test_text_file.write('\n')
                test_text_file.write('balanced accuracy ')
                test_text_file.write(str(balanced_acc))
                test_text_file.write('\n')
                test_text_file.write('recall ')
                test_text_file.write(str(recall))
                test_text_file.write('\n')
                test_text_file.write('f1 score ')
                test_text_file.write(str(f1_score))
                test_text_file.write('\n')
                test_text_file.write('cohen kappa ')
                test_text_file.write(str(test_kappa))
                test_text_file.write('\n')
                test_text_file.write('test precision ')
                test_text_file.write(str(precision))
                test_text_file.write('\n')
              
                ############################################################
                test_text_file.write('the best validation epoch was ')
                test_text_file.write(str(best_epoch))
                test_text_file.write('\n')
                
                
                test_text_file.write('\n')
                ###########################################################
                
                
               
            test_text_file.close()
            
            
           
       
        return (test_accuracy,test_loss,balanced_acc,test_kappa,f1_score,precision,recall,test_top2,test_top3,test_top4,test_top5)




if __name__ == '__main__':
 for window_split in timestep_window:
  run=11  #change this to t1
  while run<=11: ### change this  to 
    Train_accuracy_array=[]

    Train_accuracy_balanced=[]

    Train_kappa=[]

    Train_f1=[]

    Train_precision=[]

    Train_recall=[]


    Train_loss=[]

    Train_top2=[]

    Train_top3=[]

    Train_top4=[]

    Train_top5=[]


    ###########################################################



    Val_accuracy_array=[]


    Val_accuracy_balanced=[]


    Val_kappa=[]

    Val_f1=[]

    Val_precision=[]

    Val_recall=[]


    Val_loss=[]


    Val_top2=[]

    Val_top3=[]

    Val_top4=[]
    
    Val_top5=[]



##########################################################
    best_accuracy=0

    best_epoch=0         
   
    epoches_id=[]
   
    
    Train_loss_array=[]

    Val_loss_array=[]

    Test_loss_array=[]
    
    
    dataset_dir_train ="C:/Users/obouldjedr/Desktop/latest_dataset_liara/train_dataset_filtred_"+str(window_split)+"_overlapping0.9/"


    dataset_dir_val ="C:/Users/obouldjedr/Desktop/latest_dataset_liara/val_dataset_filtred_"+str(window_split)+"_overlapping0.9/"


    dataset_dir_test ="C:/Users/obouldjedr/Desktop/latest_dataset_liara/test_dataset_filtred_"+str(window_split)+"_overlapping0.9/"


    ###########################################

    dataset_dir_train_mat ="C:/Users/obouldjedr/Desktop/latest_dataset_liara/UWB/classification1/str1/"+str(window_split)+"_filtred/train/run"+str(run)+"/"


    dataset_dir_val_mat ="C:/Users/obouldjedr/Desktop/latest_dataset_liara/UWB/classification1/str1/"+str(window_split)+"_filtred/val/run"+str(run)+"/"


    dataset_dir_test_mat ="C:/Users/obouldjedr/Desktop/latest_dataset_liara/UWB/classification1/str1/"+str(window_split)+"_filtred/test/run"+str(run)+"/"


    #performence_file='C:/Users/obouldjedr/Desktop/latest_dataset_liara/UWB/classification1/str1/5sec_filtred/'+"performances_UWB_classification1_str1_5s_filtred_run1.txt"


    train_report_file=dataset_dir_train_mat+'train_report_UWB_classification1_str1_'+str(window_split)+'_filtred_run'+str(run)+'.txt'

    val_report_file=dataset_dir_val_mat+'val_report_UWB_classification1_str1_'+str(window_split)+'_filtred_run'+str(run)+'.txt'

    test_report_file=dataset_dir_test_mat+'test_report_UWB_classification1_str1_'+str(window_split)+'_filtred_run'+str(run)+'.txt' 
    
    window_size_index= timestep_window.index(window_split)
    print('window size index',window_size_index)
    window_size=window_size_list[window_size_index]
    print()
    print('window size',window_size)
    #input()
   
    '''
    print('dataset_dir_train',dataset_dir_train)
    print('dataset_dir_val',dataset_dir_val)
    print('dataset_dir_test',dataset_dir_test)
    print()
    print('dataset_dir_train',dataset_dir_train_mat)
    print('dataset_dir_test',dataset_dir_test_mat)
    print('dataset_dir_val',dataset_dir_val_mat)
    print()
    print(train_report_file)
    print()
    print(val_report_file)
    print()
    print(test_report_file)
    input()
    '''
    
   
    
   
    
   
    
   
    


    train_dataset,valid_dataset,test_dataset=get_data()

    train_loader,val_loader,test_loader=get_data_loader(train_dataset,valid_dataset,test_dataset)

    print()
    print()
    print()

    print('train_loader',train_loader)
    print()
    print('val_loader',val_loader)
    print()
    print('test_loader',test_loader)


    #input()
    #input()
    #input()
####
       
    if window_size==5:
        model=UWBnet()
    if window_size==10:
        model=UWBnet10()
    if window_size==15:
        model=UWBnet15()
        
    print('model',model) 
    #input()
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    optimizer = optim.Adam(model.parameters(),lr=0.001, betas=(0.9, 0.999), eps=1e-08)
    print('optimiser',optimizer)
    #input()
    #scheduler = ExponentialLR(optimizer, gamma=0.9)
    
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
    
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, threshold = 0.9)
    
    print('cretirion is ',criterion)
    #input()
    
    device = 'cuda'
    model.to(device)
    if use_cuda:
        print('hello cuda')
        model=model.cuda()
        
    
    #####################################################################################
    
    
#######################################################################################
    loaders_transfer = {'train':train_loader,'valid':val_loader,'test':test_loader}

    writer = SummaryWriter()


    #print('summury is ',summary,(3,224,224))
    #if window_size==5:
    #    print(summary(model,(3,250,164)))
    #if window_size==10:
    #    print(summary(model,(500,164,3)))
    #if window_size==15:
    #    print(summary(model,(3,750,184)))
    #    print(summary(model,(3,750,164)))
    
    
    #input('before training start')
    
    
    #model_transfer = train(n_epochs, loaders_transfer, model_transfer, optimizer_transfer, criterion_transfer, use_cuda,'model_transfer.pt')
    
    #train(n_epochs, loaders_transfer, model2, optimizer_transfer, criterion_transfer, use_cuda,'model_transfer_caps.pt')
    
 

    for epoch in range(1, 50):
        
        train_accuracy,train_loss,train_balanced_accuracy,train_kappa,train_f1,train_precision,train_recall,train_top2,train_top3,train_top4,train_top5=train(model, optimizer, train_loader, epoch,writer)
        #writer.add_scalar("accuracy/train", train_accuracy, epoch)
        #writer.add_scalar("loss/train", train_loss, epoch)
        #writer.add_scalar("kappa/train", train_kappa, epoch)
        #writer.add_scalar("f1/train", train_f1, epoch)
        #writer.add_scalar("precision/train", train_precision, epoch)
        #writer.add_scalar("recall/train", train_recall, epoch)
        #writer.add_scalar("balanced accuracy/train", train_balanced_accuracy, epoch)
        
        
        
        #writer.add_scalar("top2/train", train_top2, epoch)
        #writer.add_scalar("top3/train", train_top3, epoch)
        #writer.add_scalar("top4/train", train_top4, epoch)
        #writer.add_scalar("top5/train", train_top5, epoch)
        
        
        
        
        Train_accuracy_array.append(train_accuracy)
        Train_loss_array.append(train_loss)
        Train_accuracy_balanced.append(train_balanced_accuracy)
        Train_kappa.append(train_kappa)
        Train_f1.append(train_f1)
        Train_precision.append(Train_precision)
        Train_recall.append(Train_recall)
        Train_top2.append(train_top2)
        Train_top3.append(train_top3)
        Train_top4.append(train_top4)
        Train_top5.append(train_top5)
        
        
        
        
        
        
        val_accuracy,val_loss,val_balanced_accuracy,val_kappa,val_f1,val_precision,val_recall,val_top2,val_top3,val_top4,val_top5=val(model, optimizer, val_loader, epoch,writer)
        #writer.add_scalar("accuracy/val", val_accuracy, epoch)
        #writer.add_scalar("loss/val", val_loss, epoch)
        #writer.add_scalar("kappa/val", val_kappa, epoch)
        #writer.add_scalar("f1/val", val_f1, epoch)
        #writer.add_scalar("precision/val", val_precision, epoch)
        #writer.add_scalar("recall/val", val_recall, epoch)
        #writer.add_scalar("balanced accuracy/val", val_balanced_accuracy, epoch)
        
        #writer.add_scalar("top2/val", val_top2, epoch)
        #writer.add_scalar("top3/val", val_top3, epoch)
        #writer.add_scalar("top4/val", val_top4, epoch)
        #writer.add_scalar("top5/val", val_top5, epoch)
        
        #print('optimiser step',optimizer.)
        #for param_group in optimizer.param_groups:
        #    print(param_group['lr'])
        
        
        #scheduler.step(val_accuracy)
        
        #for param_group in optimizer.param_groups:
        #    print(param_group['lr'])        
        
        Val_accuracy_array.append(val_accuracy)
        Val_loss_array.append(val_loss)
        Val_accuracy_balanced.append(val_balanced_accuracy)
        Val_kappa.append(val_kappa)
        Val_f1.append(val_f1)
        Val_precision.append(Val_precision)
        Val_recall.append(Val_recall)
        Val_top2.append(val_top2)
        Val_top3.append(val_top3)
        Val_top4.append(val_top4)
        Val_top5.append(val_top5)
        
        epoches_id.append(epoch)
    
        ######################## validation of the accuracy##################################
        if val_accuracy> best_accuracy:
           
           print('found out a better state ')
           print('old accuracy is ',best_accuracy)
           print('new accuracy is ',val_accuracy)
           print('old best epoch is ',best_epoch)
           print('new best epoch is ',epoch)
           best_accuracy=val_accuracy 
           best_epoch=epoch
           
           
           
           
           torch.save(model.state_dict(), 'C:/Users/obouldjedr/Desktop/latest_dataset_liara/UWB/classification1/str1/'+str(window_split)+'_filtred/test/run'+str(run)+'/model_'+str(window_split)+'_filtred_str1_clf1_run'+str(run)+'.pth')
            #### save the model curent state we have an improvement
            
            
        
    writer.close() 
    
    ##### convert to np array instead of lists  and plot them
    
    
    Train_accuracy_array=np.array(Train_accuracy_array)
    #Train_accuracy_array=np.array(Train_accuracy_array)
    
    epoches_id=np.array(epoches_id)
    Train_accuracy_balanced=np.array(Train_accuracy_balanced)
    Train_loss_array=np.array(Train_loss_array)
    
    
    Train_top2_array=np.array(Train_top2)
    Train_top3_array=np.array(Train_top3)
    Train_top4_array=np.array(Train_top4)
    Train_top5_array=np.array(Train_top5)
    
    Train_kappa_array=np.array(Train_kappa)
    Train_f1_array=np.array(Train_f1)
    
    Val_accuracy_array=np.array(Val_accuracy_array)
    #epoches_id=np.array(epoches_id)
    Val_accuracy_balanced=np.array(Val_accuracy_balanced)
    Val_loss_array=np.array(Val_loss_array)
    
    Val_top2_array=np.array(Val_top2)
    Val_top3_array=np.array(Val_top3)
    Val_top4_array=np.array(Val_top4)
    Val_top5_array=np.array(Val_top5)
    
    Val_kappa_array=np.array(Val_kappa)
    Val_f1_array=np.array(Val_f1)
    

    plt.plot( epoches_id,Train_accuracy_array)
    plt.title("Train_accuracy")
    plt.xlabel("epochs")
    plt.ylabel("Train_accuracy");
    plt.savefig(dataset_dir_train_mat+'train_accuracy.png', format="png")
    plt.close()
    #plt.show()
    
    #train_accuracy_fig.savefig('train_accuracy_fig_lvoo4_round5.png')   # save the figure to file
    #plt.close(train_accuracy_fig)
    
    
    #train_balanced_accuracy_fig=plt.plot(Train_accuracy_balanced,epoches_id)
    #train_balanced_accuracy_fig.savefig('train_balanced_accuracy_fig_lvoo4_round5.png')   # save the figure to file
    #plt.close(train_balanced_accuracy_fig)
    plt.plot(epoches_id,Train_loss_array)
    plt.title("Train_loss")
    plt.xlabel("epoch")
    plt.ylabel("Train loss");
    plt.savefig(dataset_dir_train_mat+'train_loss.png', format="png")
    plt.close()
    
    
    
    plt.plot( epoches_id,Train_top2_array)
    plt.title("Train_accuracy top2")
    plt.xlabel("epochs")
    plt.ylabel("Train_accuracy top2");
    plt.savefig(dataset_dir_train_mat+'train_top2.png', format="png")
    plt.close()
    
    
    plt.plot( epoches_id,Train_top3_array)
    plt.title("Train_accuracy top 3")
    plt.xlabel("epochs")
    plt.ylabel("Train_accuracy top 3");
    plt.savefig(dataset_dir_train_mat+'train_top3.png', format="png")
    plt.close()
    
    
    
    plt.plot( epoches_id,Train_top4_array)
    plt.title("Train_accuracy top4")
    plt.xlabel("epochs")
    plt.ylabel("Train_accuracy top4");
    plt.savefig(dataset_dir_train_mat+'train_top4.png', format="png")
    plt.close()
    
    
    plt.plot( epoches_id,Train_top5_array)
    plt.title("Train_accuracy top5")
    plt.xlabel("epochs")
    plt.ylabel("Train_accuracy top5");
    plt.savefig(dataset_dir_train_mat+'train_top5.png', format="png")
    plt.close()
    
    
    plt.plot( epoches_id,Train_f1_array)
    plt.title("Train_f1")
    plt.xlabel("epochs")
    plt.ylabel("Train_f1");
    plt.savefig(dataset_dir_train_mat+'train_f1.png', format="png")
    plt.close()
    
    
    plt.plot( epoches_id,Train_kappa_array)
    plt.title("Train_kappa")
    plt.xlabel("epochs")
    plt.ylabel("Train_kappa");
    plt.savefig(dataset_dir_train_mat+'train_kappa.png', format="png")
    plt.close()
    
    
    
    plt.plot(epoches_id,Val_accuracy_array)
    plt.title("Val_accuracy")
    plt.xlabel("epoch")
    plt.ylabel("Val Accuracy");
    plt.savefig(dataset_dir_val_mat+'val_accuracy.png', format="png")
    plt.close()
    
    #val_accuracy_fig.savefig('val_accuracy_fig_lvoo4_round5.png')   # save the figure to file
    #plt.close(val_accuracy_fig)
    
    #val_balanced_accuracy_fig=plt.plot(Val_accuracy_balanced,epoches_id)
    plt.plot(epoches_id,Val_loss_array)
    plt.title("Val_loss")
    plt.xlabel("loss")
    plt.ylabel("Val loss");
    plt.savefig(dataset_dir_val_mat+'val_loss.png', format="png")
    plt.close()
    
    
    
    plt.plot(epoches_id,Val_top2_array)
    plt.title("Val_accuracy top2")
    plt.xlabel("epochs")
    plt.ylabel("Val_accuracy top2");
    plt.savefig(dataset_dir_val_mat+'val_top2.png', format="png")
    plt.close()
    
    
    
    plt.plot( epoches_id,Val_top3_array)
    plt.title("Val_accuracy top 3")
    plt.xlabel("epochs")
    plt.ylabel("Val_accuracy top 3");
    plt.savefig(dataset_dir_val_mat+'val_top3.png', format="png")
    plt.close()
    
    
    
    plt.plot( epoches_id,Val_top4_array)
    plt.title("Val_accuracy top4")
    plt.xlabel("epochs")
    plt.ylabel("Val_accuracy top4");
    plt.savefig(dataset_dir_val_mat+'val_top4.png', format="png")
    plt.close()
    
    
    
    
    plt.plot(epoches_id,Val_top5_array)
    plt.title("Val_accuracy top5")
    plt.xlabel("epochs")
    plt.ylabel("Val_accuracy top5");
    plt.savefig(dataset_dir_val_mat+'val_top5.png', format="png")
    plt.close()
    
    
    plt.plot(epoches_id,Val_f1_array)
    plt.title("Val_f1")
    plt.xlabel("epochs")
    plt.ylabel("Val_f1");
    plt.savefig(dataset_dir_val_mat+'val_f1.png', format="png")
    plt.close()
    
    
    plt.plot(epoches_id,Val_kappa_array)
    plt.title("Val_kappa")
    plt.xlabel("epochs")
    plt.ylabel("Val_kappa");
    plt.savefig(dataset_dir_val_mat+'val_kappa.png', format="png")
    plt.close()
    
################################################################################################
#################################################################################################


####load the besrt state of the torch model and 
    if window_size==5:
        model=UWBnet()
    if window_size==10:
        model=UWBnet10()
    if window_size==15:
        model=UWBnet15()    


    device = 'cuda'
    model.to(device)
    if use_cuda:
        print('hello cuda')
        model=model.cuda()
    
    
    #model.load_state_dict(torch.load('model_5sec_overlap0.9_str1_clf1_run1.pth'﻿)﻿)
    
    
    model.load_state_dict(torch.load('C:/Users/obouldjedr/Desktop/latest_dataset_liara/UWB/classification1/str1/'+str(window_split)+'_filtred/test/run'+str(run)+'/model_'+str(window_split)+'_filtred_str1_clf1_run'+str(run)+'.pth'))
    print('best accuracy is ',best_accuracy)
    print('best epoch is ',best_epoch)
    #input('saving model')
    
    #model = torch.load(MODEL_PATH)
    #model.eval()
    print('model is ',model)
    #input('model Loaded')

    test_accuracy,test_loss,test_balanced_accuracy,test_kappa,test_f1,test_precision,test_recall,test_top2,test_top3,test_top4,test_top5 = test(model, optimizer, test_loader)
   
    run=run+1
        