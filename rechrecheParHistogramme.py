# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 09:36:16 2021

@author: ASUS
"""

import numpy as np

from matplotlib import image
import numpy as np
import math
from matplotlib import image

import cv2 as cv
from os import listdir
import scipy.spatial.distance as dist

loaded_images = list()

"""

You can usually get better information from a HSV colorspace.
 Let me try and give a personal experience example again:
     Try imagining you have an image of a single-color plane with a 
     shadow on it. In RGB colorspace, the shadow part will most likely 
     have very different characteristics than the part without shadows. 
     In HSV colorspace, the hue component of both patches is more likely
     to be similar: the shadow will primarily influence the value, or maybe
     satuation component, while the hue, indicating the primary "color" 
     (without it's brightness and diluted-ness by white/black) should not
      change so much

"""

def hsvHistogramFeatures(image):
    """ img: image à quantifier dans un espace couleur hsv en 8x2x2 cases identiques
    sortie: vecteur 1x32 indiquant les entités extraites de l'histogramme dans l'espace hsv
    L'Histogramme dans l'espace de couleur HSV est obtenu utilisant une
    quantification par niveau: 
    8 pour H(hue), 2 pour S(saturation), et 2 pour V(Value).
    Le vecteur descripteur de taille 1x32 est calculé et normalisé """
    rows,cols,dd = image.shape
    # convertir l'image RGB en HSV.
    image = cv.cvtColor(image, cv.COLOR_RGB2HSV)
    
    h = image[...,0]
    s = image[...,1]
    v = image[...,2]

    # Chaque composante h,s,v sera quantifiée équitablement en 8x2x2
    # le nombre de niveau de quantification est:
    numberOfLevelsForH = 8
    numberOfLevelsForS = 2
    numberOfLevelsForV = 2

    # Trouver le maximum.
    maxValueForH = np.max(h)
    maxValueForS = np.max(s)
    maxValueForV = np.max(v)

    # Initialiser l'histogramme à des zéro de dimension 8x2x2
    hsvColorHisto = np.zeros((8,2,2))

    # Quantification de chaque composante en nombre niveaux étlablis
    quantizedValueForH = (h*numberOfLevelsForH/maxValueForH)
    quantizedValueForS = (s*numberOfLevelsForS/maxValueForS)
    quantizedValueForV = (v*numberOfLevelsForV/maxValueForV)

    # Créer un vecteur d'indexes
    index = np.zeros((rows*cols,3))
    index[:,0] = quantizedValueForH.flatten()
    index[:,1] = quantizedValueForS.flatten()
    index[:,2] = quantizedValueForV.flatten()

    # Remplir l'histogramme pour chaque composante h,s,v
    # (ex. si h=7,s=2,v=1 Alors incrémenter de 1 la matrice d'histogramme à la position 7,2,1)
    for i in range(len(index[:,0])):
        if(index[i,0]==0 or index[i,1]==0 or index[i,2]==0):
            continue
        hsvColorHisto[int(index[i,0]),int(index[i,1]),int(index[i,2])] +=1
    # normaliser l'histogramme à la somme
    hsvColorHisto = hsvColorHisto.flatten()
    hsvColorHisto /= np.sum(hsvColorHisto)
    return hsvColorHisto.reshape(-1)

def hsv_Moments(img):
    imagehsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    h = imagehsv[:,:,0]
    s = imagehsv[:,:,1]
    v = imagehsv[:,:,2]
    hsvFeatures = np.zeros((2,3))
    i=0
    for hsv in (h, s, v):
        hsvFeatures[0][i] = np.mean(hsv.flatten())
        hsvFeatures[1][i] = np.std(hsv.flatten())
        i+=1
    return hsvFeatures.reshape(-1)
    
def getFeatures(img, fsize):
  # fonction pour créer le vecteur descripteur
  # fsize: taille du descripteur à former
  # il faut prendre en considération les deux cas: 
  # 1- Seulement les moyennes statistiques de couleur
  # 2- 1- et l'histogramme
  hsv = hsvHistogramFeatures(img)
  moments = hsv_Moments(img)
  features = np.concatenate((hsv,moments), axis=None)
  return features

def CBIR_Indexation(fsize):
    path='../AtelierCBIR1/DataSet/obj_decoys'
    for filename in listdir(path):
        img_data = image.imread(path+'/' + filename)
        loaded_images.append(img_data)
    features = dict()
    for i in range(len(loaded_images)):
        features[i] = getFeatures(loaded_images[i],fsize)
    return features

def CBIR_Recherche(Imreq, features):
    distances = dict()
    vecteurReq = getFeatures(Imreq,27)    
    for i in range(len(features)):        
        distances[i] = dist.euclidean(features[i], vecteurReq)
        
    distances=sorted(distances.items(), key=lambda t: t[1])
    return distances

features = CBIR_Indexation(39)  
img_requete = image.imread('../AtelierCBIR1/beta.jpg') 
#plt.imshow(img_requete)
distanes_CBIR=CBIR_Recherche(img_requete,features)

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(6, 6), dpi=200)
ax1 = fig.add_subplot(2, 3, 1)
ax1.imshow(img_requete)
ax1.set_title('Image Requete')
for i in range(5):
  ax1 = fig.add_subplot(2, 3, i+2)
  ax1.imshow(loaded_images[distanes_CBIR[i][0]])
  ax1.set_title('Similaire N° %s' %(i+1))
    