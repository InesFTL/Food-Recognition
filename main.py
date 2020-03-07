#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 11:26:02 2020

@author: Ines
"""
from Function import Load_Data,Norm_Type_Data,visualize_data,metric_iou
import os
import numpy as np 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.models import load_model
im_path = "images/"
def detect_and_classify_vrac(im_path):
    filename = os.listdir(im_path)
    IMG_SIZE = 128


    data2 = []
    img,data2 = Load_Data(im_path,filename,IMG_SIZE)    
    data3 = np.squeeze(data2) #Remove earlier dimension
    X = Norm_Type_Data(data3)

    #Loading our Y
    Csv_1 = pd.read_csv('BBox_Data.csv')
    Csv_2 = pd.read_csv('Images_Data.csv')
    Classes_Size_Container = list(Csv_2.iloc[:,1])
    xmin = list(Csv_1.iloc[:,0])
    ymin = list(Csv_1.iloc[:,1])
    xmax = list(Csv_1.iloc[:,2])
    ymax = list(Csv_1.iloc[:,3])


    #Labels of our Size

    labels_size = np.array(Classes_Size_Container)
    le = LabelEncoder().fit(labels_size)
    labels_size = np_utils.to_categorical(le.transform(labels_size), 2)

    #Labels of our BBox
    BBox_arr = np.array([xmin,ymin,xmax,ymax])/IMG_SIZE
    BBox_arr_norm = BBox_arr.T

    Y = np.concatenate( [BBox_arr_norm , labels_size ] , axis=1 )
    #Column 0 : xmin Column 1 : ymin Column 2 :xmax Column 3 : ymax
    #Column 4: Big(Label) Column5 : Small(Label)


    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.1)

    
    MODEL_FILE = "model_DC.h5"
    print(" [INFO] Loading pre-trained network")
    model = load_model(MODEL_FILE)

    model.summary()
    
    labels_legend =["big","small" # index 0              # index 1
        ]

    y_pred = model.predict(X_test)
    visualize_data(Y_test,y_pred,X_test,labels_legend)
    






detect_and_classify_vrac(im_path)    
