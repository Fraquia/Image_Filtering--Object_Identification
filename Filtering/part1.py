# -*- coding: utf-8 -*-


import numpy as np
from PIL import Image
from scipy.signal import convolve2d as conv2
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import convolve2d as conv2

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

"""
Gaussian function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian values Gx computed at the indexes x
"""
def gauss(sigma):
  x=np.linspace(-sigma,sigma,(6*int(sigma)+1))
  Gx=[(1/math.sqrt(2*math.pi)*(sigma))*math.exp(-(elem**2)/2*(sigma**2)) for elem in x]
  return Gx, x

"""
Implement a 2D Gaussian filter, leveraging the previous gauss.
Implement the filter from scratch or leverage the convolve2D method (scipy.signal)
Leverage the separability of Gaussian filtering
Input: image, sigma (standard deviation)
Output: smoothed image
"""

def gaussianfilter(img, sigma):
    Gx,x= gauss(sigma)
    kernel=np.outer(Gx,Gx)
    smooth_img = conv2(img, kernel/kernel.sum(),mode="full",boundary="symm")
    return smooth_img

"""
Gaussian derivative function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian derivative values Dx computed at the indexes x
"""
def gaussdx(sigma):
    x=np.random.randint(-3*sigma,3*sigma,1000)
    
    Dx=[]
    for elem in x:
        Dx.append((-1/math.sqrt(2*math.pi)*sigma**3)*elem*math.exp(-(elem**2)/2*(sigma**2)))
    return Dx, x



def gaussderiv(img, sigma):

    #...
    
    return imgDx, imgDy

sigma = 4.0
[Gx, x] = gauss(sigma)
plt.figure(1)
plt.plot(x, Gx, '.-')
plt.show()

img = rgb2gray(np.array(Image.open('graf.png')))

smooth_img =gaussianfilter(img, sigma)

plt.figure(2)
ax1 = plt.subplot(1,2,1)
ax2 = plt.subplot(1,2,2)
plt.sca(ax1)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.sca(ax2)
plt.imshow(smooth_img, cmap='gray', vmin=0, vmax=255)
plt.show()
