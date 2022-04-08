import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import histogram_module
import dist_module

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

'''
Implement a function find best match, in
match module.py, which takes a list of model images and a list of query images and for each query image returns the 
index of closest model image. The function should take string parameters, which identify
distance function, histogram function and number of histogram bins. 
'''

# model_images - list of file names of model images
# query_images - list of file names of query images
#
# dist_type - string which specifies distance type:  'chi2', 'l2', 'intersect'
# hist_type - string which specifies histogram type:  'grayvalue', 'dxdy', 'rgb', 'rg'
#
# note: use functions 'get_dist_by_name', 'get_hist_by_name' and 'is_grayvalue_hist' to obtain 
#       handles to distance and histogram functions, and to find out whether histogram function 
#       expects grayvalue or color image

def find_best_match(model_images, query_images, dist_type, hist_type, num_bins):

    hist_isgray = histogram_module.is_grayvalue_hist(hist_type)
    
    model_hists = compute_histograms(model_images, hist_type, hist_isgray, num_bins) #lista di liste con gli istogrammi del modello
    query_hists = compute_histograms(query_images, hist_type, hist_isgray, num_bins) #lista di liste con gli istogrammi delle query
    
    D = np.zeros((len(model_images), len(query_images)))
    
    best_match=[]
    
    for i in range(len(model_hists)):
        for j in range(len(query_hists)):

            value=dist_module.get_dist_by_name(model_hists[i],query_hists[j],dist_type)

            D[i,j]=value

    for j in range(len(query_hists)):
        bv=0
        mi=D[0,j]
        for i in range(len(model_hists)):
            if D[i,j]<mi:
                mi = D[i,j]
                bv=i
        best_match.append(bv)
        
    #best_match risulta una lista di liste contente la top k di ogni query ordinata per minor distanza

    return best_match, D


def compute_histograms(image_list, hist_type, hist_isgray, num_bins):
    image_hist = []
    # Compute hisgoram for each image and add it at the bottom of image_hist
    for i in image_list:
        img=np.array(Image.open(i))
        
        if hist_isgray==True:
            img=rgb2gray(img.astype('double'))
        if hist_type == "grayvalue":
            image_hist.append(histogram_module.get_hist_by_name(img.astype('double'), num_bins, hist_type)[0])
        else:
            image_hist.append(histogram_module.get_hist_by_name(img.astype('double'), num_bins, hist_type))
    return image_hist


# For each image file from 'query_images' find and visualize the 5 nearest images from 'model_image'.
#
# Note: use the previously implemented function 'find_best_match'
# Note: use subplot command to show all the images in the same Python figure, one row per query image
import matplotlib.image as mpimg
def show_neighbors(model_images, query_images, dist_type, hist_type, num_bins):
    
    num_nearest = 5  # show the top-5 neighbors
    best_match, D = find_best_match(model_images, query_images, dist_type, hist_type, num_bins)
    
    best_match=[]
    
    #look for the top-5
    for j in range(len(D[0])):
        mi=[]
        ok=[]
        
        for i in range(len(D)):
            if len(mi)==num_nearest:
                if mi[len(mi)-1][0]>D[i,j]:
                    mi.remove(mi[len(mi)-1])
                    mi.append([D[i,j],i,j])
                    
            else:
                mi.append([D[i,j],i,j])
            mi.sort()
        #print(mi)
        for elem in mi:
            ok.append(elem[1])
        best_match.append(ok)


    #plotting part
    f, axarr = plt.subplots(1,num_nearest)
    for j in range(len(best_match)):

        modellini=[np.array(Image.open("./"+query_images[j]))]
        for i in range(len(best_match[j])):
            modellini.append(np.array(Image.open("./"+model_images[best_match[j][i]])))

        for immagini in range(len(modellini)):
            plt.subplot(1,num_nearest+1,immagini+1)
            plt.imshow(modellini[immagini])

        plt.show()
    #print(best_match)
    return best_match

    