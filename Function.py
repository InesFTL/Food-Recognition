#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 08:13:34 2020

@author: Ines
"""

import cv2
from tqdm import tqdm
import numpy as np 
from keras import backend as K
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def Load_Data(DATA_DIR,filename,IMG_SIZE):
    data = []
    """
    Loading our dataset and preprocessing the images 
    by resizing in it

    """
    for imgs in tqdm(filename):
        img = cv2.imread(DATA_DIR+imgs)
        img = cv2.imread(DATA_DIR+imgs,cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        
        data.append(img)
    
    return img,data


def Norm_Type_Data(data):
    """
    Encoding data & Scaling images to the range [0,1]

    """
    X_data = np.array(data,dtype="float64")/255.0 #Scaling images to the range [0,1]
    return X_data
    

def inter_over_union(y_true,y_pred):
    xA = K.maximum(y_true[:,0],y_pred[:,0])
    yA = K.minimum(y_true[:,1],y_pred[:,1])
    xB = K.minimum(y_true[:,2],y_pred[:,2])
    yB = K.minimum(y_true[:,3],y_pred[:,3])
    
    interArea = K.maximum(0.0, xB-xA)*K.maximum(0.0,yB-yA)
    boxAArea = (y_true[:,2]-y_true[:,0])*(y_true[:,3]-y_true[:,1])
    boxBArea = (y_pred[:,2]-y_pred[:,0]) * (y_pred[:,3]-y_pred[:,1])
    
    iou = interArea / (boxAArea+boxBArea - interArea)
    
    return iou


def metric_iou(y_true,y_pred):
    return inter_over_union(y_true,y_pred)
    



def visualize_data(y_true,y_pred,X_test,labels_legend):
    
    xminP = []
    yminP = []
    xmaxP = []
    ymaxP = []

    xminT = []
    yminT = []
    xmaxT = []
    ymaxT = []

    for i in range(y_pred.shape[0]):
        xminP.append((y_pred[i][0])*(128/5))
        yminP.append((y_pred[i][1])*(128/3.75))
        xmaxP.append((y_pred[i][2])*(128/5))
        ymaxP.append((y_pred[i][3])*(128/3.75))
    

    for i in range(y_true.shape[0]):
        xminT.append((y_true[i][0])*(128/5))
        yminT.append((y_true[i][1])*(128/3.75))
        xmaxT.append((y_true[i][2])*(128/5))
        ymaxT.append((y_true[i][3])*(128/3.75)) 
    
    
    figure = plt.figure(figsize=(8, 8))
    for (i, index) in enumerate(np.random.choice(X_test.shape[0], size=4, replace=False)):
    
        ax = figure.add_subplot(2, 2, i + 1, xticks=[], yticks=[])
 
        ax.imshow(np.squeeze(X_test[index]))
        predict_index = np.argmax(y_pred[index][4])
        true_index = np.argmax(y_true[index][4])
        
        ax.set_title("{} ".format(labels_legend[predict_index]) 
                                  ,
                                  color=("green" if predict_index == true_index else "red"))
        rect = patches.Rectangle((xminP[index],yminP[index]),(xmaxP[index]-xminP[index]),(ymaxP[index]-yminP[index]),linewidth=5,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
        rect = patches.Rectangle((xminT[index],yminT[index]),(xmaxT[index]-xminT[index]),(ymaxP[index]-yminP[index]),linewidth=5,edgecolor='g',facecolor='none')
        ax.add_patch(rect)
        
    
    
    plt.show()

    