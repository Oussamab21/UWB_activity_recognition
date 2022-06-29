# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 15:25:01 2022

@author: obouldjedr
"""


from __future__ import print_function, absolute_import
import torch
from torch.utils.data import Dataset, DataLoader
import json
from torch.utils.data import DataLoader, IterableDataset
import random
import torch
import torch.nn as nn
from functools import partial
from sklearn.metrics import classification_report
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import argparse
import datetime
import numpy as np
import sklearn.metrics as metrics
from torchvision import datasets, transforms, models
from torchmetrics import ConfusionMatrix
import time
from math import  pi 
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
from torch.utils.tensorboard import SummaryWriter
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
from sklearn.metrics import top_k_accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
import timm
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib.pyplot as plt

from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
import scipy.fftpack

import seaborn as sns 

import plotly.express as px
sns.set_theme()


use_cuda = torch.cuda.is_available()

#torch.backends.cudnn.benchmark = True



#__all__ = ['accuracy']
'''
def accuracy1k(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



##############################################################################
##############################################################################
################################################################################
####################

def accuracy2k(output, target, topk=(2,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
'''

##########validation 1 over the activity files######################
###### activity 1############### brushing teeth

### path
#################################### metrics definitions
epoches_id=[]

#typecnn='1dcnn'
sampling_rate=50

timestep_seconds=15

window_size=sampling_rate*timestep_seconds

print('window size is ',window_size)
overlapping_ratio=0.9

overlaping_step=overlapping_ratio*window_size




step=['5sec','10sec','15sec']


timestep='15sec'

cut_beg=(10*sampling_rate)-1
cut_end=5*sampling_rate


#overlapping=0.9





################## filter info










update_file_pointer=overlaping_step

print('window is ',window_size)
print('overlaping_step is ',overlaping_step)
print('update file is ',update_file_pointer)
input()
import time






##############################################################################################################


#dataset_dir_train_mat ='C:/Users/obouldjedr/Desktop/latest_dataset_liara/train/UWB/strategie2_10s_1/'

#dataset_dir_val_mat ='C:/Users/obouldjedr/Desktop/latest_dataset_liara/val/UWB/strategie2_10s_1/'


#dataset_dir_test_mat ='C:/Users/obouldjedr/Desktop/latest_dataset_liara/test/UWB/strategie2_10s_1/'



UWB1_tmp=[]


####### since charles data is misssing I will comment the user id till the arrival of the missing data.

sensor_list=['_UWB_','_IMU_']

sensor=sensor_list[0]

data='C:/Users/obouldjedr/Desktop/latest_dataset_liara/data/dataset-4.1'





##### list of act

list_act=['BrushingTeeth','DoingHousework','Drinking','Eating','GettingDressedUndressed','GoingToilet','PuttingAwayCleanLaundry',
          #'PuttingAwayDishes',
          'ReadingBook',
          'Resting','Sleeping','TakingShower','UsingCompSPhone']
          #,'WashingDishes']

#list_act=['DoingHousework','Drinking','Eating','GettingDressedUndressed','GoingToilet','PuttingAwayCleanLaundry',
#          #'PuttingAwayDishes',
#          'ReadingBook',
#          'Resting','Sleeping','TakingShower','UsingCompSPhone']
          #,'WashingDishes']
############## list of persons


list_pers=['1b111579','1e1a8f23','4b184c28','4e36eb16','41b00638','80a2ce12','22544a24','32052f20',
           'a6f74a8b',
           'bf7554bb','f87e2ddf']


#target_names=list_act


str_val=1

###############  locations


################# list of iterations


#########################################################################

dict_BrushingTeeth = {1: {'1': '1_Bathroom', '2': '2_Bathroom'},
                      2: {'1': '1_Bathroom', '2': '2_Bathroom'},
                      3: {'1': '1_Bathroom', '2': '2_Bathroom'},
                      4: {'1': '1_Bathroom', '2': '2_Bathroom'},
                      5: {'1': '1_Bathroom', '2': '2_Bathroom'},
                      6: {'1': '1_Bathroom', '2': '2_Bathroom'},
                      7: {'1': '1_Bathroom', '2': '2_Bathroom'},
                      8: {'1': '1_Bathroom', '2': '2_Bathroom'},
                      9: {'1': '1_Bathroom', '2': '2_Bathroom'},
                      10: {'1': '1_Bathroom', '2': '2_Bathroom'},
                      11: {'1': '1_Bathroom', '2': '2_Bathroom'}}
                    
###########################                    






dict_DoingHousework = {1: {'1': "1_Livingroom",'2': "2_Kitchen"},
                      2:  {'1': "1_Livingroom", '2': "2_Kitchen"},
                      3: {'1': "1_Livingroom",'2': "2_Kitchen"},
                      4: {'1': "1_Livingroom",'2': "2_Diningroom"},
                      5: {'1': "1_Livingroom",'2': "2_Diningroom"},
                      6: {'1': "1_Livingroom",'2': "2_Diningroom"},
                      7: {'1': "1_Livingroom",'2': "2_Diningroom"},
                      8: {'1': "1_Bedroom",'2': "2_Diningroom"},
                      9: {'1': "1_Livingroom", '2': "2_Kitchen"},
                      10: {'1': "1_Livingroom", '2': "2_Kitchen"},
                      11: {'1': "1_Bedroom", '2': "2_Kitchen"}}





###########################








dict_Drinking =      {1: {'1': "1_Kitchen", '2': "2_Livingroom"},
                      2: {'1': "1_Kitchen", '2': "2_Bedroom"},
                      3: {'1': "1_Kitchen", '2': "2_Bedroom"},
                      4: {'1': "1_Livingroom", '2': "2_Kitchen"},
                      5: {'1': "1_Livingroom",'2': "2_Diningroom"},
                      6: {'1': "1_Livingroom",'2': "2_Kitchen"},
                      7: {'1': "1_Kitchen", '2': "2_Bedroom"},
                      8: {'1': "1_Bedroom", '2': "2_Kitchen"},
                      9: {'1': "1_Kitchen", '2': "2_Diningroom"},
                      10: {'1': "1_Bedroom", '2': "2_Kitchen"},
                      11: {'1': "1_Kitchen", '2': "2_Livingroom"}}


########################             
   
    
             
  



dict_Eating =        {1: {'1': "1_Diningroom"},
                      2: {'1': "1_Diningroom"}, 
                      3: {'1': "1_Diningroom"},
                      4: {'1': "1_Diningroom"}, 
                      5: {'1': "1_Diningroom"}, 
                      6: {'1': "1_Diningroom"},
                      7: {'1': "1_Diningroom"},
                      8: {'1': "1_Diningroom"},
                      9: {'1': "1_Diningroom"}, 
                      10: {'1':"1_Diningroom"},
                      11: {'1':"1_Diningroom"}}            

        
             


########################



dict_GettingDressedUndressed={1: {'1': "1_Bedroom",'2': "2_Bedroom"},
                              2: {'1': "1_Bedroom",'2': "2_Bedroom"}, 
                              3: {'1': "1_Bedroom",'2': "2_Bedroom"},
                              4: {'1': "1_Bedroom",'2': "2_Bedroom"}, 
                              5: {'1': "1_Bedroom",'2': "2_Bedroom"}, 
                              6: {'1': "1_Bedroom",'2': "2_Bedroom"},
                              7: {'1': "1_Bedroom",'2': "2_Bedroom"},
                              8: {'1': "1_Bedroom",'2': "2_Bedroom"},
                              9: {'1': "1_Bedroom",'2': "2_Bedroom"}, 
                              10: {'1':"1_Bedroom",'2': "2_Bedroom"},
                              11: {'1':"1_Bedroom",'2': "2_Bedroom"}}      
                             

#######################



dict_GoingToilet={1: {'1': "1_Bathroom",'2': "2_Bathroom"},
                  2: {'1': "1_Bathroom",'2': "2_Bathroom"}, 
                  3: {'1': "1_Bathroom",'2': "2_Bathroom"},
                  4: {'1': "1_Bathroom",'2': "2_Bathroom"}, 
                  5: {'1': "1_Bathroom",'2': "2_Bathroom"}, 
                  6: {'1': "1_Bathroom",'2': "2_Bathroom"},
                  7: {'1': "1_Bathroom",'2': "2_Bathroom"},
                  8: {'1': "1_Bathroom",'2': "2_Bathroom"},
                  9: {'1': "1_Bathroom",'2': "2_Bathroom"}, 
                  10: {'1':"1_Bathroom",'2': "2_Bathroom"},
                  11: {'1':"1_Bathroom",'2': "2_Bathroom"}} 



##########################


             
dict_PuttingAwayCleanLaundry= {1: {'1': "1_Bedroom"},
                                2: {'1': "1_Bedroom"}, 
                                3: {'1': "1_Bedroom"},
                                4: {'1': "1_Bedroom"}, 
                                5: {'1': "1_Bedroom"}, 
                                6: {'1': "1_Bedroom"},
                                7: {'1': "1_Bedroom"},
                                8: {'1': "1_Bedroom"},
                                9: {'1': "1_Bedroom"}, 
                                10: {'1':"1_Bedroom"},
                                11: {'1':"1_Bedroom"}} 


##########################





#########################


#dict_PuttingAwayDishes=        {1: {'1': "1_Kitchen"},
#                                2: {'1': "1_Kitchen"}, 
#                                3: {'1': "1_Kitchen"},
#                                4: {'1': "1_Kitchen"}, 
#                                5: {'1': "1_Kitchen"}, 
#                                6: {'1': "1_Kitchen"},
#                                7: {'1': "1_Kitchen"},
#                                8: {'1': "1_Kitchen"},
#                                9: {'1': "1_Kitchen"}, 
#                                10: {'1':"1_Kitchen"},
#                                11: {'1':"1_Kitchen"}} 
########################################################################################



dict_ReadingBook={1: {'1': "1_Livingroom"},
                  2: {'1': "1_Livingroom"}, 
                  3: {'1': "1_Livingroom"},
                  4: {'1': "1_Livingroom"}, 
                  5: {'1': "1_Livingroom"}, 
                  6: {'1': "1_Bedroom"},
                  7: {'1': "1_Bedroom"},
                  8: {'1': "1_Livingroom"},
                  9: {'1': "1_Bedroom"}, 
                  10:{'1': "1_Livingroom"},
                  11:{'1': "1_Livingroom"}} 







##########################

dict_Resting={    1: {'1': "1_Bedroom",'2': "2_Livingroom"},
                  2: {'1': "1_Livingroom",'2': "2_Bedroom"}, 
                  3: {'1': "1_Bedroom",'2': "2_Livingroom"},
                  4: {'1': "1_Livingroom",'2': "2_Bedroom"}, 
                  5: {'1': "1_Livingroom",'2': "2_Bedroom"}, 
                  6: {'1': "1_Livingroom",'2': "2_Bedroom"},
                  7: {'1': "1_Livingroom",'2': "2_Bedroom"},
                  8: {'1': "1_Bedroom",'2': "2_Livingroom"},
                  9: {'1': "1_Bedroom",'2': "2_Livingroom"}, 
                  10:{'1': "1_Livingroom",'2': "2_Bedroom"},
                  11:{'1': "1_Bedroom",'2': "2_Livingroom"}} 


##########################










########################


dict_Sleeping=                 {1: {'1': "1_Bedroom"},
                                2: {'1': "1_Bedroom"}, 
                                3: {'1': "1_Bedroom"},
                                4: {'1': "1_Bedroom"}, 
                                5: {'1': "1_Bedroom"}, 
                                6: {'1': "1_Bedroom"},
                                7: {'1': "1_Bedroom"},
                                8: {'1': "1_Bedroom"},
                                9: {'1': "1_Bedroom"}, 
                                10: {'1':"1_Bedroom"},
                                11: {'1':"1_Bedroom"}} 


#######################


dict_TakingShower=             {1: {'1': "1_Bathroom"},
                                2: {'1': "1_Bathroom"}, 
                                3: {'1': "1_Bathroom"},
                                4: {'1': "1_Bathroom"}, 
                                5: {'1': "1_Bathroom"}, 
                                6: {'1': "1_Bathroom"},
                                7: {'1': "1_Bathroom"},
                                8: {'1': "1_Bathroom"},
                                9: {'1': "1_Bathroom"}, 
                                10: {'1':"1_Bathroom"},
                                11: {'1':"1_Bathroom"}} 

#########################







dict_UsingCompSPhone         = {1: {'1': "1_Kitchen"},
                                2: {'1': "1_Diningroom"}, 
                                3: {'1': "1_Diningroom"},
                                4: {'1': "1_Livingroom"}, 
                                5: {'1': "1_Kitchen"}, 
                                6: {'1': "1_Kitchen"},
                                7: {'1': "1_Diningroom"},
                                8: {'1': "1_Diningroom"},
                                9: {'1': "1_Kitchen"}, 
                                10: {'1': "1_Diningroom"},
                                11: {'1':"1_Bedroom"}} 



#######################


#dict_WashingDishes=            {1: {'1': "1_Kitchen"},
#                                2: {'1': "1_Kitchen"}, 
#                                3: {'1': "1_Kitchen"},
#                                4: {'1': "1_Kitchen"}, 
#                                5: {'1': "1_Kitchen"}, 
#                                6: {'1': "1_Kitchen"},
#                                7: {'1': "1_Kitchen"},#
#                                8: {'1': "1_Kitchen"},
#                                9: {'1': "1_Kitchen"}, 
#                                10: {'1':"1_Kitchen"},
#                                11: {'1':"1_Kitchen"}} 


order=[dict_BrushingTeeth,dict_DoingHousework,dict_Drinking,dict_Eating,dict_GettingDressedUndressed,dict_GoingToilet,
       dict_PuttingAwayCleanLaundry,
       #dict_PuttingAwayDishes,
       dict_ReadingBook,
       dict_Resting,
       
       dict_Sleeping,
       dict_TakingShower,dict_UsingCompSPhone
       #,dict_WashingDishes
       ] 
       


#order=[dict_DoingHousework,dict_Drinking,dict_Eating,dict_GettingDressedUndressed,dict_GoingToilet,
#       dict_PuttingAwayCleanLaundry,
       #dict_PuttingAwayDishes,
#       dict_ReadingBook,
#       dict_Resting,
       
#       dict_Sleeping,
#       dict_TakingShower,dict_UsingCompSPhone
       #,dict_WashingDishes
#       ] 


#iterations=[2,2,2,1,2,2,1,2,1,2,1,1,1,1]
train_files_len=[]
val_files_len=[]
test_files_len=[]

#iterations_dict={'BrushingTeeth':{'1b111579':"Bathroom",'1b111579':"Bathroom"},'DoingHousework':{'1':}}
######################################

data_path='C:/Users/obouldjedr/Desktop/latest_dataset_liara'
 ################################################################################################################# 

def impz(b, a):
   
    # Define the impulse sequence of length 60
    impulse = np.repeat(0., 60)
    impulse[0] = 1.
    x = np.arange(0, 60)
 
    # Compute the impulse response
    response = signal.lfilter(b, a, impulse)
 
    # Plot filter impulse and step response:
    fig = plt.figure(figsize=(10, 6))
    plt.subplot(211)
    plt.stem(x, response, 'm', use_line_collection=True)
    plt.ylabel('Amplitude', fontsize=15)
    plt.xlabel(r'n (samples)', fontsize=15)
    plt.title(r'Impulse response', fontsize=15)
 
    plt.subplot(212)
    step = np.cumsum(response)
     
    # Compute step response of the system
    plt.stem(x, step, 'g', use_line_collection=True)
    plt.ylabel('Amplitude', fontsize=15)
    plt.xlabel(r'n (samples)', fontsize=15)
    plt.title(r'Step response', fontsize=15)
    plt.subplots_adjust(hspace=0.5)
 
    fig.tight_layout()
    plt.show()


##########################################################################################################
#########################################################################################################
def mfreqz(b, a, Fs):
   
    # Compute frequency response of the filter
    # using signal.freqz function
    wz, hz = signal.freqz(b, a)
    
    # Calculate Magnitude from hz in dB
    Mag = 20*np.log10(abs(hz))
 
    # Calculate phase angle in degree from hz
    Phase = np.unwrap(np.arctan2(np.imag(hz), np.real(hz)))*(180/np.pi)
     
    # Calculate frequency in Hz from wz
    Freq = wz*Fs/(2*np.pi)
     
    # Plot filter magnitude and phase responses using subplot.
    fig = plt.figure(figsize=(10, 6))
 
    # Plot Magnitude response
    sub1 = plt.subplot(2, 1, 1)
    sub1.plot(Freq, Mag, 'r', linewidth=2)
    sub1.axis([1, Fs/2, -100, 5])
    sub1.set_title('Magnitude Response', fontsize=20)
    sub1.set_xlabel('Frequency [Hz]', fontsize=20)
    sub1.set_ylabel('Magnitude [dB]', fontsize=20)
    sub1.grid()
 
    # Plot phase angle
    sub2 = plt.subplot(2, 1, 2)
    sub2.plot(Freq, Phase, 'g', linewidth=2)
    sub2.set_ylabel('Phase (degree)', fontsize=20)
    sub2.set_xlabel(r'Frequency (Hz)', fontsize=20)
    sub2.set_title(r'Phase response', fontsize=20)
    sub2.grid()
 
    plt.subplots_adjust(hspace=0.5)
    fig.tight_layout()
    plt.show()
    input()
##################################################################################
def plot_sig(signal,uwb_nbr,cnt_tmp):
    x=range(0, len(signal))
    x=np.asarray(x)
    #print(x)
    #input()
    x=x*(1/sampling_rate)
    y=signal
    plt.plot(x, y)
    #plt.show()
    plt.title("Time domain UWB data")
    plt.xlabel("time")
    plt.ylabel("signal");
    plt.savefig('C:/Users/obouldjedr/Desktop/time_domain/image'+str(cnt_tmp)+''+str(uwb_nbr)+'.png', format="png")
    plt.close()
    #input()



import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack

def plot_fft(signal,uwb_nbr,cnt_tmp):
    # Number of samplepoints
    N = int(len(signal)/50)
    # sample spacing
    T = 1.0 / 50.0
    print(type(T))
    print()
    print(type(N))
    x = np.linspace(0.0, N*T, N)
    y = signal
    yf = scipy.fftpack.fft(y)
    xf = np.linspace(0.0, 1.0/(2.0*T), N/2)

    plt.subplot(2, 1, 1)
    plt.plot(xf, 2.0/N * np.abs(yf[0:N/2]))
    #plt.subplot(2, 1, 2)
    plt.plot(xf[1:], 2.0/N * np.abs(yf[0:N/2])[1:])
    plt.savefig('C:/Users/obouldjedr/Desktop/time_domain/image'+str(cnt_tmp)+''+str(uwb_nbr)+'.png', format="png")
    plt.close() 













################################################################################
'''
def plot_frequency_responce(sos):
    
    w, h = signal.sosfreqz(sos, worN=15)
    
    plt.subplot(2, 1, 1)

    db = 20*np.log10(np.maximum(np.abs(h), 1e-5))

    plt.plot(w/np.pi, db)

    plt.ylim(-75, 75)

    plt.grid(True)

    plt.yticks([60,40,20,0, -20, -40, -60])

    plt.ylabel('Gain [dB]')

    plt.title('Frequency Response')

    #plt.subplot(2, 1, 2)

    #plt.plot(w/np.pi, np.angle(h))

    #plt.grid(True)

    #plt.yticks([-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi],

    #       [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])

    #plt.ylabel('Phase [rad]')

    #plt.xlabel('Normalized frequency (1.0 = Nyquist)')

    plt.show()
    input()    





def plot_frequency_responce_2():
    
    w, h = signal.sosfreqz(sos, worN=15)
    
    plt.subplot(2, 1, 1)

    db = 20*np.log10(np.maximum(np.abs(h), 1e-5))

    plt.plot(w/np.pi, db)

    plt.ylim(-75, 75)

    plt.grid(True)

    plt.yticks([60,40,20,0, -20, -40, -60])

    plt.ylabel('Gain [dB]')

    plt.title('Frequency Response')

    #plt.subplot(2, 1, 2)

    #plt.plot(w/np.pi, np.angle(h))

    #plt.grid(True)

    #plt.yticks([-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi],

    #       [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])

    #plt.ylabel('Phase [rad]')

    #plt.xlabel('Normalized frequency (1.0 = Nyquist)')

    plt.show()
    input() 
'''
##############################################################################################
'''
def plot_scaterring_matrix(data_plot,UWB_id,activity_name,filt_step):
    data_plot=np.absolute(data_plot)
    NY=data_plot.shape[1]
    NX=data_plot.shape[0]
    sns.heatmap(data_plot,xticklabels=500, yticklabels=25,annot = True)
    #plt.ylim(0,NY)
    #plt.xlim(0,NX)
    #plt.show()
    
    
    #plt.plot()
    plt.title('scaterring matrix activity'+str(activity_name)+'UWB_'+str(UWB_id)+'_'+str(filt_step)+'')
    plt.xlabel("Bins")
    plt.ylabel("Frames in time")
    print()
    print('type of time step',type(timestep))
    print()
    print('activity name is ',activity_name)
    #input()
    print('type of ',type(activity_name))  
    
    dataset_dir_sct_mat='C:/Users/obouldjedr/Desktop/latest_dataset_liara/data/dataset-4.1/scat_mat/'
    print('got this 0')
    print()
    print(dataset_dir_sct_mat+'scaterring_matrix_activity'+str(activity_name)+'UWB_'+str(UWB_id)+'_'+str(filt_step))
    plt.savefig(dataset_dir_sct_mat+'scaterring_matrix_activity'+str(activity_name)+'UWB_'+str(UWB_id)+'_'+str(filt_step)+'.png', format="png")
    print('got this 1')
    plt.close()
    return()
    #input()

'''


def plot_scaterring_matrix(data_plot,UWB_id,activity_name,filt_step):
    fig = px.imshow(data_plot, text_auto=True, aspect="auto")
    fig.show()









#######################################################################################################

def mfreqz(b, a, Fs):
   
    # Compute frequency response of the filter
    # using signal.freqz function
    wz, hz = signal.freqz(b, a)
 
    # Calculate Magnitude from hz in dB
    Mag = 20*np.log10(abs(hz))
 
    # Calculate phase angle in degree from hz
    Phase = np.unwrap(np.arctan2(np.imag(hz), np.real(hz)))*(180/np.pi)
     
    # Calculate frequency in Hz from wz
    Freq = wz*Fs/(2*np.pi)
     
    # Plot filter magnitude and phase responses using subplot.
    fig = plt.figure(figsize=(10, 6))
 
    # Plot Magnitude response
    sub1 = plt.subplot(2, 1, 1)
    sub1.plot(Freq, Mag, 'r', linewidth=2)
    sub1.axis([1, Fs/2, -100, 5])
    sub1.set_title('Magnitude Response', fontsize=20)
    sub1.set_xlabel('Frequency [Hz]', fontsize=20)
    sub1.set_ylabel('Magnitude [dB]', fontsize=20)
    sub1.grid()
 
    # Plot phase angle
    sub2 = plt.subplot(2, 1, 2)
    sub2.plot(Freq, Phase, 'g', linewidth=2)
    sub2.set_ylabel('Phase (degree)', fontsize=20)
    sub2.set_xlabel(r'Frequency (Hz)', fontsize=20)
    sub2.set_title(r'Phase response', fontsize=20)
    sub2.grid()
 
    plt.subplots_adjust(hspace=0.5)
    fig.tight_layout()
    plt.show()
 
# Define impz(b,a) to calculate impulse response
# and step response of a system
# input: b= an array containing numerator coefficients,
# a= an array containing denominator coefficients
def impz(b, a):
   
    # Define the impulse sequence of length 60
    impulse = np.repeat(0., 60)
    impulse[0] = 1.
    x = np.arange(0, 60)
 
    # Compute the impulse response
    response = signal.lfilter(b, a, impulse)
 
    # Plot filter impulse and step response:
    fig = plt.figure(figsize=(10, 6))
    plt.subplot(211)
    plt.stem(x, response, 'm', use_line_collection=True)
    plt.ylabel('Amplitude', fontsize=15)
    plt.xlabel(r'n (samples)', fontsize=15)
    plt.title(r'Impulse response', fontsize=15)
 
    plt.subplot(212)
    step = np.cumsum(response)  # Compute step response of the system
    plt.stem(x, step, 'g', use_line_collection=True)
    plt.ylabel('Amplitude', fontsize=15)
    plt.xlabel(r'n (samples)', fontsize=15)
    plt.title(r'Step response', fontsize=15)
    plt.subplots_adjust(hspace=0.5)
 
    fig.tight_layout()
    plt.show()
 





############################################################################################################  
   
def set_all_files(list_files):
    #def __init__(self, files):
        #super(JsonDataset).__init__()
        #self.files = files
        #self.data = []
        #self.data_len=0
        #print('enter the set files')
        #print('len is ',len(list_files))
        #print()
        #print(list_files)
        #input()
        '''
        if mode==1:
        
        
            x_train=[]
            y_train=[]
        if mode==2:
            x_val=[]
            y_val=[]
            
        if mode==3:
            x_test=[]
            y_test=[]     
       '''
        train_files_dataset=[]
        val_files_dataset=[]
        test_files_dataset=[]
       ##################################################################################
       
        for i in range (len(list_files)):
            #print('activity is ',i)
            data_count=0
            list_files_current_activity=list_files[i]
            train_files_current_activity=[]
            val_files_current_activity=[]
            test_files_current_activity=[]
            if len(list_files_current_activity)==22:
                #train_files=16
                #print('list before order is ',list_files_current_activity)
                #input()
                random.shuffle(list_files_current_activity)
                #print('list after ',list_files_current_activity)
                #input()
                
                train_files_current_activity=list_files_current_activity[0:15]
                val_files_current_activity=list_files_current_activity[15:18]
                test_files_current_activity=list_files_current_activity[18:22]
                #val_files=3
                #test_files=3
                
                
            if len(list_files_current_activity)==11:
                
                
                random.shuffle(list_files_current_activity)
               
                
                
                
                train_files_current_activity=list_files_current_activity[0:8]
                val_files_current_activity=list_files_current_activity[8:9]
                test_files_current_activity=list_files_current_activity[9:11]
                
                
                
            
            train_files_dataset.append(train_files_current_activity)
            val_files_dataset.append(val_files_current_activity)
            test_files_dataset.append(test_files_current_activity)
            
            
        #print('train_files_dataset',len(train_files_dataset[]))
        #print('val_files_dataset',len(val_files_dataset))
        #print('test_files_dataset',len(test_files_dataset))
        #input()
        return(train_files_dataset,val_files_dataset,test_files_dataset)   
            
            
                    






###############################################################################################################
def get_files():
 dataset_all=[]
 #val_dataset_all=[]
 #test_dataset_all=[]    
    

 for i in list_act:
    #input()
    #print('current activity  is ',i)
    #print()
    #print(type(i))
    #input()
    list_act_tmp=str(i)
    name_list_activity=str('list_')+str(i)
    #print('hello list_act_tmp',list_act_tmp)
    #print()
    #print('full name',name_list_activity)
    #input()
    if str_val==1:
        
        #list_pers_train=['1b111579','1e1a8f23','4b184c28','4e36eb16','41b00638','80a2ce12','22544a24','32052f20','a6f74a8b']
        #list_pers_val=['bf7554bb']
        #list_pers_test=['f87e2ddf']
        
        ###########
        ### list of files 
        list_files_activity_current_activity=[]  ##### list that contains all files of the current activity
        #i=[]
        list_files_activity_current_activity=[]
        list_files_activity_current_activity=[]
        
        #name_list_activity_train=[]
        #name_list_activity_val=[]
        #name_list_activity_test=[]
        #list_pers_up_train=
        #list_pers_up_val=
        #list_pers_up_test=
    ##### creation of validation strategie 1
    ### go throught the list of persons
    
        ####train
        #print('train data set files') 
        for j in list_pers:     ##### loop over the pers
            #print('pers id is ',j)
            #print()
            #print('index is ',list_act.index(i))
            #input()
            tmp_index_act=(list_act.index(i))        ### trying to add anything to the activity index to get labes equal       
            #print()
            
            tmp=order[tmp_index_act]
            #print('dict is ',tmp)
            #print()
            tmp_pers_index=list_pers.index(j)
            #print('j is ',j)
            #print()
            #print(tmp_pers_index)
            #print()     ######
            #print(tmp)
            #print()
            #print()
            tmp_dict_act_pers=tmp[tmp_pers_index+1]
        
            #print('dict is ',tmp_dict_act_pers)
            #print('dict is ',tmp_dict_act_pers[1])
            #print('dict is ',tmp_dict_act_pers[2])
            #print('dict is ',tmp_dict_act_pers['1'])
            #print('dict is ',tmp_dict_act_pers['2'])
            #print(len(tmp_dict_act_pers))
        
            if len(tmp_dict_act_pers)==1:
                #print(data+'/'+str(j)+'/'+str(i)+'/'+str(j)+'_UWB_'+str(i)+'_'+str(tmp_dict_act_pers['1'])+'.json')
                tmp_name=str(data+'/'+str(j)+'/'+str(i)+'/'+str(j)+sensor+str(i)+'_'+str(tmp_dict_act_pers['1'])+'.json')
                list_files_activity_current_activity.append(tmp_name)
                
                
            else:
                
                    #print(data+'/'+str(j)+'/'+str(i)+'/'+str(j)+'_UWB_'+str(i)+'_'+str(tmp_dict_act_pers['1'])+'.json')
                    #print(data+'/'+str(j)+'/'+str(i)+'/'+str(j)+'_UWB_'+str(i)+'_'+str(tmp_dict_act_pers['2'])+'.json')
                    tmp_name=str(data+'/'+str(j)+'/'+str(i)+'/'+str(j)+sensor+str(i)+'_'+str(tmp_dict_act_pers['1'])+'.json')
                    tmp_name_2=str(data+'/'+str(j)+'/'+str(i)+'/'+str(j)+sensor+str(i)+'_'+str(tmp_dict_act_pers['2'])+'.json')
                    list_files_activity_current_activity.append(tmp_name)
                    list_files_activity_current_activity.append(tmp_name_2)
        
        #input()
        ###val
        
        #list_train_files_all.append(list_files_activity)
        dict_train= {'list_files': list_files_activity_current_activity, 'activity': i}
        #print('dict is ',dict_train)
        #input()
        
        #print('val data set files') 
        
       
        
        #print()
        dataset_all.append(list_files_activity_current_activity)   ### list of all files train  
        
###
 return(dataset_all)    
        



nbr_instances_activity_train=[] 
nbr_instances_activity_val=[]
nbr_instances_activity_test=[]
#################################################################################################################
##################################################################################################################

def dataset_construction(dataset_files,mode):
      
        cnt_tmp=0         
        
        for i in (dataset_files):      ##### liste of files in every activity 
            #print('activity is ',i)
            data_count=0
            list_files_current_activity=i
            nbr_instances_current_activity=0
            
            print('current activity file are ',i)
            #input()
            
            for j in (list_files_current_activity):
                
                #print('new file in the current activity',list_files_current_activity[j])
                #input()
                
                #with open(list_files_current_activity[j]) as file:
                    #########################################################
                    data_UWB1_before=[]
                    data_UWB2_before=[]
                    data_UWB3_before=[]
                    
                    
                    filtred_UWB1=[]
                    filtred_UWB2=[]
                    filtred_UWB3=[]
                    
                    
                    
                    data_UWB1=[]
                    data_UWB2=[]
                    data_UWB3=[]
                    
                    
                    filt_data_UWB1=[]
                    filt_data_UWB2=[]
                    filt_data_UWB3=[]
                    
                    
                    
                    files_lenght1=[]


                    files_lenght2=[]

                    files_lenght3=[]


                    files_instances1=[]

                    files_instances2=[]

                    files_instances3=[]

                    #data_UWB1=[]
                    #data_UWB2=[]
                    #data_UWB3=[]

                    
                    
                    
                    with open(j) as fp:
                         
                        current_file_len=0
                        for line in fp:
                            current_file_len=current_file_len+1
                            res_line= json.loads(str(line))
                            if (res_line['serial_number'] =='100000029171'):
                                   #print('res_line',res_line)
                                   #print()
                                   #print('res_line',res_line['data'])
                                   #input()
                                   tmp=res_line['data']
                                   #tmp2=tmp[0:164]
                                   tmp2=tmp[0:164]
                                   data_UWB1_before.append(tmp2)
                            if (res_line['serial_number'] =='100000029444'):
                                   #print('this is UWB2')
                                   tmp=res_line['data']
                                   #tmp2=tmp[0:164]
                                   tmp2=tmp[0:164]
                                   data_UWB2_before.append(tmp2)
                            if (res_line['serial_number'] =='100000030722'):
                                   #print('this is UWB3')
                                   tmp=res_line['data']
                                   #tmp2=tmp[0:164]
                                   tmp2=tmp[0:164]
                                   data_UWB3_before.append(tmp2)
                    if mode==1:
                       train_files_len.append(current_file_len) 
                    if mode==2:
                       val_files_len.append(current_file_len) 
                    if mode==3:
                       test_files_len.append(current_file_len) 
                    #print('show time')
                    #print(len(data_UWB1))
                    #print(len(data_UWB2))
                    #print(len(data_UWB3))
                    #input()
                    
                    born_UWB1=len(data_UWB1_before)-cut_end
                    born_UWB2=len(data_UWB2_before)-cut_end
                    born_UWB3=len(data_UWB3_before)-cut_end
                    ###### removing the bad side !) seconds in the begening and 5 at the end
                    
                    #print('show time borns')
                    #print(len(data_UWB1_before[1]))
                    #print(len(data_UWB2_before[1]))
                    #print(len(data_UWB3_before[1]))
                    #input()
                    
                    data_UWB1_before=data_UWB1_before[cut_beg:born_UWB1]
                    data_UWB2_before=data_UWB2_before[cut_beg:born_UWB1]
                    data_UWB3_before=data_UWB3_before[cut_beg:born_UWB1]
                    #print(len(data_UWB1_before))
                    #print(len(data_UWB2_before))
                    #print(len(data_UWB3_before))
                    #input()
                    
                    #### plot scatter matrixes
                    
                    
                    
                    ####################################################################################
                    
                    
                        
                    
                    
                    cnt_tmp=cnt_tmp+1
                    #input()
                    
                    ###########################################################
                    '''
                    file = open(list_files_current_activity[j], "r")
                    k=0
                    taille_file=0
                    print('new file is ',list_files_current_activity[j])
                    #nbr_instances_current_file=0
                    while file.readline():
                        taille_file=taille_file+1
                        
                    print('taille file is ',taille_file)
                    input()
                    ###################################################
                    '''
                    #### extracting only uwb1:
                    instances_UWB1_current_file=[]  
                    instances_UWB2_current_file=[] 
                    instances_UWB3_current_file=[] 
                    ################### add the filter
#                    print('type filter_order ',type(filter_order))
                    #print()
#                    print('type Ap ',type(Ap))
                    #print()
                    
#                    print('type wc ',type(wc))
                    #print()
                    
                    #print('data UWB1',type(data_UWB1))
                    #print()
                    #print(len(data_UWB1))
                    #print()
                    #print(len(data_UWB1[1]))
                  
                    
                    
                    #input()
                    #Fs = 50
                    #wc =[1,5] 
                    #Ap = 0.025
                    #filter_order=2
                    
                    
                    
                    #sos = signal.cheby1(filter_order,Ap, wc, 'bandpass',analog=True,output='sos')
                    #y_sos = 
                    Fs = 50
                    #fp =np.array([1, 5]) 
                    Ap = 0.025
                    filter_order=2
                    
                    filter_type = 'bandpass'
                    
                    #wp = fp/(Fs/2)                 
                    #ws = fp/(Fs/2)  
                    
                    #wc = (2*pi*fp)/Fs
                    
                    #wc=fp/(Fs*0.5)
                    #wc=
                    #wc=[6.291466,32.49196]
                    
                    
                    #input()
                    #wc=  [1,5]
                    
                    #wc = (2*pi*fp)/Fs
                    
                    #print('wc is ',wc)
                    
                    #sos=signal.cheby1(filter_order, Ap,wc , 'bandpass',analog=True,output='sos')
                    
                    sos = signal.cheby1(filter_order, 0.025,[0.12, 5],btype=filter_type, output='sos', fs=Fs, analog=False)
                    
                    #sos=signal.cheby1(filter_order, Ap,wc , 'bandpass',analog=True,output='sos')
                    
                    ########## define numpy equivalent#################
                    data_UWB1_before=np.array(data_UWB1_before)
                    data_UWB2_before=np.array(data_UWB2_before)
                    data_UWB3_before=np.array(data_UWB3_before)
                    
                    
                    #print(data_UWB1_before.shape)
                    #input()
                    data_UWB1_before=data_UWB1_before.transpose()
                    data_UWB2_before=data_UWB2_before.transpose()
                    data_UWB3_before=data_UWB3_before.transpose()
                    
                    
                    
                    #sct_act_index = dataset_files.index(i)
                    #sct_act_index=list_act[sct_act_index]
                    
                    #filt_step='before'
                    
                    #plot_scaterring_matrix(data_UWB1_before,1,sct_act_index,filt_step)
                    #plot_scaterring_matrix(data_UWB2_before,2,sct_act_index,filt_step)
                    #plot_scaterring_matrix(data_UWB3_before,3,sct_act_index,filt_step)
                    #print('plotted part 1')
                    #input()
                    #input()
                    #print(data_UWB1_before.shape)
                    #plot_scaterring_matrix(data_UWB2_before)
                    #plot_scaterring_matrix(data_UWB3_before)
                    
                    #print()
                    #print(data_UWB1_before[1])
                    
                    #input()
                    
                    ########plot scatter matrixws
                    #plt.imshow(data_UWB1_before)
                    #plt.plot(data_UWB1_before)
                    #plt.title("scqtter mqt")
                    #plt.xlabel("times frame")
                    #plt.ylabel("bins");
                    #plt.savefig('sctrUWB1.png', format="png")
                    #plt.close()
                    #plt.show()
                    #plt.plot(data_UWB1_before)
                    #plt.title("scqtter mqt")
                    #plt.xlabel("times frame")
                    #plt.ylabel("bins");
                    #plt.savefig('sctrUWB1.png', format="png")
                    #plt.close()
                    ####################################################################
                    #filt_data_UWB1=signal.filtfilt(z, p, data_UWB1_before)
                    #filt_data_UWB2=signal.filtfilt(z, p, data_UWB2_before)
                    #filt_data_UWB3=signal.filtfilt(z, p, data_UWB3_before)
                    #np.savetxt('test_sample.txt',data_UWB1_before[1])# ,data_UWB1[1])
                    #input()
                    
                    #filt_data_UWB1=signal.sosfilt(sos,data_UWB1_before,axis=1)
                    
                    #filt_data_UWB2=signal.sosfilt(sos,data_UWB2_before,axis=1)
                    #filt_data_UWB3=signal.sosfilt(sos,data_UWB3_before,axis=1)
                    
                    
                    
                    
                
                    #print('all line are here')
                    #print(type(filt_data_UWB1))    
                    #input()
                
                    #filt_data_UWB1=filt_data_UWB1.transpose()
                    #filt_data_UWB2=filt_data_UWB2.transpose()
                    #filt_data_UWB3=filt_data_UWB3.transpose()
                    #print(data_UWB1_before)
                    #print()
                    #np.savetxt('test_sample_after.txt',data_UWB1[1])# ,data_UWB1[1])
                    #input()
                    
                    
                    #data_UWB1=filt_data_UWB1.tolist()
                    #data_UWB2=filt_data_UWB2.tolist()
                    #data_UWB3=filt_data_UWB3.tolist()
                    
                    ######################################################################################
                    
                    data_UWB1=signal.sosfilt(sos,data_UWB1_before,axis=1)
                    
                    data_UWB2=signal.sosfilt(sos,data_UWB2_before,axis=1)
                    data_UWB3=signal.sosfilt(sos,data_UWB3_before,axis=1)
                    
                    #print(data_UWB1_before)
                    #print()
                    #np.savetxt('test_sample_after.txt',data_UWB1[1])# ,data_UWB1[1])
                    #input()
                    
                    #plot_scaterring_matrix(data_UWB1)
                    #plot_scaterring_matrix(data_UWB2)
                    #plot_scaterring_matrix(data_UWB3)
                    #print('show time ')
                    #print(data_UWB1)
                    #input()
                
                    #print('all line are here')
                    #print(type(filt_data_UWB1))    
                    #input()
                
                    data_UWB1=data_UWB1.transpose()
                    data_UWB2=data_UWB2.transpose()
                    data_UWB3=data_UWB3.transpose()
                    
                    
                    
                    data_UWB1=data_UWB1.tolist()
                    data_UWB2=data_UWB2.tolist()
                    data_UWB3=data_UWB3.tolist()
                    
                    
                    
                    
                    
                    
                    
                    #print('back to old shape')
                    
                    
                    
                    #print(len(data_UWB1))
                    #print(data_UWB1)
                    #print('nan time')
                    #print(data_UWB1_before)
                    #print()
                    #np.savetxt('test_sample_after.txt',data_UWB1[1])# ,data_UWB1[1])
                    #input()
                    
                    
                 
                    
                    k=0
                    while k <len(data_UWB1):                 
                        if (len(data_UWB1)-k)>=window_size:  ####if we can get an instance here
                            cnt=0
                            data_current_instance1=[]
                            while cnt<window_size:
                                data_frame=data_UWB1[k]
                                #data_frame=frame['data'] 
                                #data_frame=data_frame[0:164]
                                cnt=cnt+1
                                k=k+1
                                data_current_instance1.append(data_frame)
                            instances_UWB1_current_file.append(data_current_instance1)  
                            k=k-update_file_pointer
                            #### add the overlapping update line
                                
                        else:
                            print('small seg')
                            k=len(data_UWB1)+10
                    ####################################################################
                    k2=0
                    while k2 <len(data_UWB2):    
                        if (len(data_UWB2)-k2)>=window_size:
                            cnt=0
                            data_current_instance2=[]
                            while cnt<window_size:
                                data_frame=data_UWB2[k2]
                                #data_frame=frame['data'] 
                                #data_frame=data_frame[0:164]
                                cnt=cnt+1
                                k2=k2+1
                                data_current_instance2.append(data_frame)
                            instances_UWB2_current_file.append(data_current_instance2) 
                            k2=k2-update_file_pointer
                             #### add the overlapping update line    
                        else:
                            print('small seg')
                            k2=len(data_UWB2)+10
                    
                    
                    ###################################################################
                    k3=0
                    while k3 <len(data_UWB3):    
                        if (len(data_UWB3)-k3)>=window_size:
                            cnt=0
                            data_current_instance3=[]
                            while cnt<window_size:
                                data_frame=data_UWB3[k3]
                                #data_frame=frame['data'] 
                                #data_frame=data_frame[0:164]
                                cnt=cnt+1
                                k3=k3+1
                                data_current_instance3.append(data_frame)
                            instances_UWB3_current_file.append(data_current_instance3)    
                             #### add the overlapping update line 
                            k3=k3-update_file_pointer 
                        else:
                            print('small seg')
                            k3=len(data_UWB3)+10
                    
                    #input()
                    #######  saving timer#######
                    if (len(instances_UWB1_current_file)!=len(instances_UWB2_current_file)):
                        input('pr')
                    if (len(instances_UWB1_current_file)!=len(instances_UWB3_current_file)):
                        input('pr')
                    if (len(instances_UWB2_current_file)!=len(instances_UWB3_current_file)):
                        input('pr')
                    for m in range(len(instances_UWB1_current_file)):
                        UWB1_instance_numpy=np.asarray(instances_UWB1_current_file[m])
                        UWB2_instance_numpy=np.asarray(instances_UWB2_current_file[m])
                        UWB3_instance_numpy=np.asarray(instances_UWB3_current_file[m])
                        print('show time is ')
                        print(UWB1_instance_numpy.shape)
                        print(UWB2_instance_numpy.shape)
                        print(UWB3_instance_numpy.shape)
                        
                        np_fusion=np.stack((UWB1_instance_numpy,UWB2_instance_numpy,UWB3_instance_numpy),axis=2)
                        print(np_fusion.shape)
                        #input()
                        activity_index=dataset_files.index(i)
                        current_activity=list_act[activity_index] 
                        
                        if mode==1:
                                
                                savingpath=data_path+'/'+'train'+'_dataset_filtred'+'_'+timestep+'_overlapping'+str(overlapping_ratio)+'/'+current_activity+'/'+'data_'+str(data_count)+'.pickle'
                            
                        if mode==2:
                            
                                savingpath=data_path+'/'+'val'+'_dataset_filtred'+'_'+timestep+'_overlapping'+str(overlapping_ratio)+'/'+current_activity+'/'+'data_'+str(data_count)+'.pickle'
                            
                        if mode==3:
                            
                                savingpath=data_path+'/'+'test'+'_dataset_filtred'+'_'+timestep+'_overlapping'+str(overlapping_ratio)+'/'+current_activity+'/'+'data_'+str(data_count)+'.pickle'    
                            
                            #with open(savingpath,'wb') as f:
                                
                        f = open(savingpath, 'wb')
                                
                        pickle.dump(np_fusion, f,protocol=pickle.HIGHEST_PROTOCOL)
                            
                                #print('saving path is ',savingpath)
                                #print('saving path is ',type(savingpath))
                        data_count=data_count+1
                            
                        f.close()
                      #'''    
                         

        return()            



















################################################################################################
#################################################################################################

if __name__ == '__main__':
    
    
    
    print('before getting to all list files')
    
    print('time step is ',timestep)
    input()
    #print('train',list_train_files_all)
    #print()
    #print('train',train_dataset_all)
    #print()
    #print('train',len(train_dataset_all))
    #input()
    
    dataset_all_files=get_files()
    update_file_pointer=int(update_file_pointer)
    print('update pointer is ',update_file_pointer)
    print('update pointer is ',type(update_file_pointer))
    input()
    print('list of files ordred by activity is ',dataset_all_files)
    #input()
    #input()
    
    #input()
    #input()
    train_files,val_files,test_files=set_all_files(dataset_all_files)
    
    #val_dataset=set_all_files(val_dataset_all2,mode=2)
    #test_dataset=set_all_files(test_dataset_all2,mode=3)
    
    
    print(train_files)
    print(len(train_files))
    #input()
    print(val_files)
    print(len(val_files))
    #input()
    print(test_files)
    print(len(test_files))
    #input()
    
    #train_dataset=TensorDataset()
    #val_dataset_list=set_all_files(val_dataset_all2,mode=2)
    #test_dataset_list=set_all_files(test_dataset_all2,mode=3)
    
    dataset_construction(train_files,mode=1)
    print('done pickeling the train')
    dataset_construction(val_files,mode=2)
    print('done pickeling the val')
    dataset_construction(test_files,mode=3)
    print('done pickeling the test')
    #train_dataset=JsonDataset(train_dataset_all2)
    #val_dataset=JsonDataset(val_dataset_all2)
    #test_dataset=JsonDataset(test_dataset_all2)
    print('done pickeling ')
    
    #files_len3 = [ x-750 for x in files_lenght3 ]
    #files_len2 = [ x-750 for x in files_lenght2 ]
    #input()
    #    tmp_index_act=list_act.index(i)
    #data_loader_train=DataLoader(train_dataset, batch_size=512,shuffle = True, num_workers=4,pin_memory=True)
    #data_loader_val=DataLoader(val_dataset, batch_size=512,num_workers=4)
    #data_loader_test=DataLoader(test_dataset, batch_size=512,num_workers=4)
    
   