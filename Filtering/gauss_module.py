# import packages: numpy, math (you might need pi for gaussian functions)
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import convolve2d as conv2
from scipy.signal import convolve as conv



"""
Gaussian function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian values Gx computed at the indexes x
"""

def gauss(sigma):
    x=range(int(-3*sigma),int(3*sigma+1),1)
    Gx=[(1/(math.sqrt(2*math.pi)*sigma))*math.exp(-(elem**2)/(2*(sigma**2))) for elem in x]
    Gx = np.array(Gx)
    return Gx, x


"""
Implement a 2D Gaussian filter, leveraging the previous gauss.
Implement the filter from scratch or leverage the convolve2D method (scipy.signal)
Leverage the separability of Gaussian filtering
Input: image, sigma (standard deviation)
Output: smoothed image
"""

"""
# with no separation 
def gaussianfilter(img, sigma):
    Gx,x= gauss(sigma)
    Gx=np.array(Gx)
    kernel=np.outer(Gx,np.matrix.transpose(Gx))
    #kernel=np.outer(Gx,Gx.reshape(1, Gx.size))
    smooth_img = conv2(img, kernel/kernel.sum() ,mode="full",boundary="symm")
    return smooth_img
"""
#with separation
def gaussianfilter(img, sigma):
    Gx,x= gauss(sigma)
    Gx = np.array(Gx)
    Gy = np.matrix.transpose(Gx)
    l=[]
    for el in img:
      l.append(conv(el,Gx,mode='same'))
    l = np.matrix.transpose(np.array(l))
    l1=[]
    for elem in l:
      l1.append(conv(elem,Gy,mode='same'))
    l1=np.array(l1)
    return np.matrix.transpose(l1)
"""
Gaussian derivative function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian derivative values Dx computed at the indexes x
"""
def gaussdx(sigma):
    x=range(int(-3*sigma),int(3*sigma+1),1)
    Dx=[(-(1/(math.sqrt(2*math.pi)*sigma**3)))*elem*math.exp(-(elem**2)/(2*(sigma**2))) for elem in x]
    Dx = np.array(Dx)
    return Dx, x

def gaussderiv(img, sigma):
    smo = gaussianfilter(img, sigma)
    kernel_or = gaussdx(sigma)[0]
    img_Dx = conv2(smo, np.reshape(kernel_or, (1,-1)), mode="same")
    #kernel_vert = np.matrix.transpose(kernel_or)
    img_Dy = conv2(smo, np.reshape(kernel_or, (-1,1)), mode="same")
    return img_Dx, img_Dy

