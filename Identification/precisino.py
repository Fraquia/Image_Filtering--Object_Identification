## import packages
import numpy as np
from PIL import Image
from numpy import histogram as hist  # call hist, otherwise np.histogram
import matplotlib.pyplot as plt

import histogram_module
import dist_module
import match_module
import rpc_module


def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


with open('model.txt') as fp:
    model_images = fp.readlines()
model_images = [x.strip() for x in model_images] 

with open('query.txt') as fp:
    query_images = fp.readlines()
query_images = [x.strip() for x in query_images] 



d_type_list = ['chi2','l2','intersect']
hist_type_list = ['dxdy','rg','rgb','grayvalue']
bins_values = [10,15,20]


def match_evaluate(match):
    c = 0                        #numero di match corretti 
    for el in range(len(match)): #l'inidice deve essere uguale al suo elemento
        if el==match[el]:
            c +=1      
    return c 


def match_rate(model_images, query_images, dists, hists, bins):

    query_image_number = len(query_images)
    
    rates = {}
    
    for hist in hists:
        for dist in dists:
            for bin_ in bins:
               best_match, D = match_module.find_best_match(model_images, query_images, dist, hist, bin_) #i parametri di questa specifica conf

               correct_matches = match_evaluate(best_match)
               
               rates[(dist, hist, bin_)] = correct_matches/query_image_number #updatig the dictionary
     
    print(rates)          
    return rates

rate = match_rate(model_images,query_images,d_type_list,hist_type_list,bins_values)



