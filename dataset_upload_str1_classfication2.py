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

locations=['Diningroom','Livingroom','Kitchen','Bathroom','Bedroom']

liste_of_location=[['Diningroom','Livingroom','Kitchen'],['Bathroom'],['Bedroom']]

liste_of_radars=['Diningroom','Bathroom','Bedroom']

ref_radars=['100000029171','100000030722','100000029444']



step=['5sec','10sec','15sec']


timestep='15sec'

cut_beg=(10*sampling_rate)-1
cut_end=5*sampling_rate


#overlapping=0.9



update_file_pointer=overlaping_step

print('window is ',window_size)
print('overlaping_step is ',overlaping_step)
print('update file is ',update_file_pointer)
input()
import time




files_lenght1=[]


files_lenght2=[]

files_lenght3=[]


files_instances1=[]

files_instances2=[]

files_instances3=[]


##############################################################################################################


#dataset_dir_train_mat ='C:/Users/obouldjedr/Desktop/latest_dataset_liara/train/UWB/strategie2_10s_1/'

#dataset_dir_val_mat ='C:/Users/obouldjedr/Desktop/latest_dataset_liara/val/UWB/strategie2_10s_1/'


#dataset_dir_test_mat ='C:/Users/obouldjedr/Desktop/latest_dataset_liara/test/UWB/strategie2_10s_1/'





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
       
#iterations=[2,2,2,1,2,2,1,2,1,2,1,1,1,1]
train_files_len=[]
val_files_len=[]
test_files_len=[]

#iterations_dict={'BrushingTeeth':{'1b111579':"Bathroom",'1b111579':"Bathroom"},'DoingHousework':{'1':}}
######################################

data_path='C:/Users/obouldjedr/Desktop/latest_dataset_liara'
 #################################################################################################################   
   
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
                    data_UWB=[]
                    #data_UWB2=[]
                    #data_UWB3=[]
                    
                    #########################################################
                    
                    
                    current_place_name_last_part = j.split("/")[-1]
                    
                    print('current place name is ',current_place_name_last_part)
                    current_place_name=current_place_name_last_part.split("_")[-1] 
                    print('current place name is ',current_place_name)
                    current_place_name=current_place_name.split(".")[0] 
                    print('current place name is ',current_place_name)
                    if current_place_name  not in locations:
                        input('halt Unknown place')
                    
                    
                    if (current_place_name=='Bathroom'):
                        radar_ref_file='100000030722'
                        
                        
                        
                        
                    if (current_place_name=='Bedroom'):
                        radar_ref_file='100000029444'
                        
                        
                        
                    if (current_place_name=='Diningroom'):
                        
                        radar_ref_file='100000029171'
                    if (current_place_name=='Livingroom'):
                        
                        radar_ref_file='100000029171'    
                    
                    if (current_place_name=='Kitchen'):
                        
                        radar_ref_file='100000029171'
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    #########################################################
                    with open(j) as fp:
                     
                        current_file_len=0
                        for line in fp:
                            current_file_len=current_file_len+1
                            res_line= json.loads(str(line))
                            
                            
                            '''
                            if (res_line['serial_number'] =='100000029171'):
                                   #print('this is UWB1')
                                   data_UWB1.append(res_line)
                            if (res_line['serial_number'] =='100000029444'):
                                   #print('this is UWB2')
                                   data_UWB2.append(res_line)
                            if (res_line['serial_number'] =='100000030722'):
                                   #print('this is UWB3')
                                   data_UWB3.append(res_line)
                            '''
                            
                            
                            if (res_line['serial_number'] == radar_ref_file):
                                   #print('this is UWB1')
                                   data_UWB.append(res_line)       
                    #if mode==1:
                    #   train_files_len.append(current_file_len) 
                    #if mode==2:
                    #   val_files_len.append(current_file_len) 
                    #if mode==3:
                    #   test_files_len.append(current_file_len) 
                    #print('show time')
                    #print(len(data_UWB1))
                    #print(len(data_UWB2))
                    #print(len(data_UWB3))
                    #input()
                    
                    born_UWB=len(data_UWB)-cut_end
                    #born_UWB2=len(data_UWB2)-cut_end
                    #born_UWB3=len(data_UWB3)-cut_end
                    ###### removing the bad side !) seconds in the begening and 5 at the end
                    data_UWB=data_UWB[cut_beg:born_UWB]
                    #data_UWB2=data_UWB2[cut_beg:born_UWB1]
                    #data_UWB3=data_UWB3[cut_beg:born_UWB1]
                    #print('show time')
                    #print(len(data_UWB1))
                    #print(len(data_UWB2))
                    #print(len(data_UWB3))
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
                    instances_UWB_current_file=[]  
                    #instances_UWB2_current_file=[] 
                    #instances_UWB3_current_file=[] 
                    
                    
                    k=0
                    while k <len(data_UWB):                 
                        if (len(data_UWB)-k)>=window_size:  ####if we can get an instance here
                            cnt=0
                            data_current_instance=[]
                            while cnt<window_size:
                                frame=data_UWB[k]
                                data_frame=frame['data'] 
                                data_frame=data_frame[0:164]
                                cnt=cnt+1
                                k=k+1
                                data_current_instance.append(data_frame)
                            instances_UWB_current_file.append(data_current_instance)  
                            k=k-update_file_pointer
                            #### add the overlapping update line
                                
                        else:
                            print('small seg')
                            k=len(data_UWB)+10
                    ####################################################################
                    
                    
                    for m in range(len(instances_UWB_current_file)):
                        UWB_instance_numpy=np.asarray(instances_UWB_current_file[m])
                        #UWB2_instance_numpy=np.asarray(instances_UWB2_current_file[m])
                        #UWB3_instance_numpy=np.asarray(instances_UWB3_current_file[m])
                        print('show time is ')
                        print(UWB_instance_numpy.shape)
                        #print(UWB2_instance_numpy.shape)
                        #print(UWB3_instance_numpy.shape)
                        
                        #np_fusion=np.stack((UWB1_instance_numpy,UWB2_instance_numpy,UWB3_instance_numpy),axis=2)
                        #print(np_fusion.shape)
                        #input()
                        activity_index=dataset_files.index(i)
                        current_activity=list_act[activity_index] 
                        
                        if mode==1:
                                
                                savingpath=data_path+'/'+'train'+'_dataset_str1_clf2'+'_'+timestep+'_overlapping'+str(overlapping_ratio)+'/'+current_activity+'/'+'data_'+str(data_count)+'.pickle'
                            
                        if mode==2:
                            
                                savingpath=data_path+'/'+'val'+'_dataset_str1_clf2'+'_'+timestep+'_overlapping'+str(overlapping_ratio)+'/'+current_activity+'/'+'data_'+str(data_count)+'.pickle'
                            
                        if mode==3:
                            
                                savingpath=data_path+'/'+'test'+'_dataset_str1_clf2'+'_'+timestep+'_overlapping'+str(overlapping_ratio)+'/'+current_activity+'/'+'data_'+str(data_count)+'.pickle'    
                            
                            #with open(savingpath,'wb') as f:
                                
                        f = open(savingpath, 'wb')
                                
                        pickle.dump(UWB_instance_numpy, f,protocol=pickle.HIGHEST_PROTOCOL)
                            
                                #print('saving path is ',savingpath)
                                #print('saving path is ',type(savingpath))
                        data_count=data_count+1
                            
                        f.close()
                          
                         

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
    input()
    
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
    input()
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
    
   