# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 14:58:54 2021

@author: ASUS
"""

from os import listdir
from matplotlib import image
import numpy as np
import math
import cv2

loaded_images = list()

path='../AtelierCBIR1/DataSet/obj_decoys'
for filename in listdir(path):
    img_data = image.imread(path+'/' + filename)
    loaded_images.append(img_data)

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(6, 6), dpi=200)
import random


id=random.sample(range(270), 6)
for i in range(6):
  ax1 = fig.add_subplot(2, 3, i+1)
  ax1.imshow(loaded_images[id[i]])
  ax1.set_title('image id: %s '% (id[i]))

img_requete = image.imread('../AtelierCBIR1/ImageRequete.jpg')

distances = {}

import scipy.spatial.distance as dist

def calculateDistance(image1, image2):
    grayImage1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    # convertion ndarray vers matrix
    vecteurImage1 = np.matrix(grayImage1)
    #print(type(vecteurImage1),"gray image type", type(grayImage1))
    a = vecteurImage1.flatten()
    grayImage2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    vecteurImage2 = np.matrix(grayImage2)
    b = vecteurImage2.flatten()
    
    return dist.cdist(a, b)

i=0
for image in loaded_images:
    distances[i]=calculateDistance(image, img_requete)
    i+=1
    

distances = sorted(distances.items(), key=lambda t: t[1])


fig = plt.figure(figsize=(6, 6), dpi=200)
ax1 = fig.add_subplot(2, 3, 1)
ax1.imshow(img_requete)
ax1.set_title('Image Requete')
for i in range(5):
  ax1 = fig.add_subplot(2, 3, i+2)
  ax1.imshow(loaded_images[distances[i][0]])
  ax1.set_title('Similaire N° %s' %(i+1))