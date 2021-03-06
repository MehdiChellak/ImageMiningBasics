# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 13:59:11 2021

@author: ASUS
"""

# la CBIR est constituée de 2 étapes; Indexation et Recherche
# On va créer une fonction pour l'indexation d'une image 
# C'est l'équivalent d'extraire une vecteur descripteur en se basant sur les moments statistiques des couleurs
import numpy as np

from matplotlib import image
import numpy as np
import math


import cv2
from os import listdir
import scipy.spatial.distance as dist


loaded_images = list()


    

def color_Moments(img):
    rouge = img[:,:,0]
    vert = img[:,:,1]
    blue = img[:,:,2]
    colorFeatures = np.zeros((2,3))
    i=0
    for rgb in (rouge, vert, blue):
        colorFeatures[0][i] = np.mean(rgb.flatten())
        colorFeatures[1][i] = np.std(rgb.flatten())
        i+=1
    return colorFeatures

def CBIR_Indexation():
    path='../AtelierCBIR1/DataSet/obj_decoys'
    for filename in listdir(path):
        img_data = image.imread(path+'/' + filename)
        loaded_images.append(img_data)
    i = 0
    features = dict()
    for img in range(len(loaded_images)):
        features[i] = color_Moments(loaded_images[img])
        i+=1
    
    
    return features

def CBIR_Recherche(Imreq,ind_Matrix):
    distances = dict()
    vecteurReq = color_Moments(Imreq)    
    for i in ind_Matrix:        
        distances[i] = dist.euclidean(ind_Matrix[i].flatten(), vecteurReq.flatten())
    distances=sorted(distances.items(), key=lambda t: t[1])
    return distances

index_Matrix=CBIR_Indexation()
#print(index_Matrix[0])

img_requete = image.imread('../AtelierCBIR1/ImageRequete.jpg') 
distanes_CBIR=CBIR_Recherche(img_requete,index_Matrix)


import matplotlib.pyplot as plt


fig = plt.figure(figsize=(6, 6), dpi=200)
ax1 = fig.add_subplot(2, 3, 1)
ax1.imshow(img_requete)
ax1.set_title('Image Requete')
for i in range(5):
  ax1 = fig.add_subplot(2, 3, i+2)
  ax1.imshow(loaded_images[distanes_CBIR[i][0]])
  ax1.set_title('Similaire N° %s' %(i+1))
