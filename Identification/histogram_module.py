import numpy as np
import bisect
import math
from numpy import histogram as hist
from scipy.signal import convolve2d as conv2
from scipy.signal import convolve as conv

#Add the Filtering folder, to import the gauss_module.py file, where gaussderiv is defined (needed for dxdy_hist)
import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
filteringpath = os.path.join(parentdir, 'Filtering')
sys.path.insert(0,filteringpath)

def gaussdx(sigma):
    x=range(int(-3*sigma),int(3*sigma+1),1)
    Dx=[(-(1/(math.sqrt(2*math.pi)*sigma**3)))*elem*math.exp(-(elem**2)/(2*(sigma**2))) for elem in x]
    Dx = np.array(Dx)
    return Dx, x

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

def gaussderiv(img, sigma):
    smo = gaussianfilter(img, sigma)
    kernel_or = gaussdx(sigma)[0]
    img_Dx = conv2(smo, np.reshape(kernel_or, (1,-1)), mode="same")
    #kernel_vert = np.matrix.transpose(kernel_or)
    img_Dy = conv2(smo, np.reshape(kernel_or, (-1,1)), mode="same")
    return img_Dx, img_Dy


def gauss(sigma):
    x=range(int(-3*sigma),int(3*sigma+1),1)
    Gx=[(1/(math.sqrt(2*math.pi)*sigma))*math.exp(-(elem**2)/(2*(sigma**2))) for elem in x]
    return Gx, x


#  compute histogram of image intensities, histogram should be normalized so that sum of all values equals 1
#  assume that image intensity varies between 0 and 255
#
#  img_gray - input image in grayscale format
#  num_bins - number of bins in the histogram
def normalized_hist(img_gray, num_bins):
    assert len(img_gray.shape) == 2, 'image dimension mismatch'
    assert img_gray.dtype == 'float', 'incorrect image type'

    div = 255/num_bins
    bins = np.arange(0,256, div)
    hists = [0]*num_bins
    new_img = img_gray.flatten()
    
    for pixel in new_img:
        if pixel == 255: index = num_bins-1
        else:
            index = int(num_bins * (pixel) / (255))
        hists[index] += 1 
    
    tot = sum(hists)
    norm = []
    
    for i in range(len(hists)):
      norm.append(hists[i]/tot)
      
    norm = np.array(norm)
    return norm, bins


#  Compute the *joint* histogram for each color channel in the image
#  The histogram should be normalized so that sum of all values equals 1
#  Assume that values in each channel vary between 0 and 255
#
#  img_color - input color image
#  num_bins - number of bins used to discretize each channel, total number of bins in the histogram should be num_bins^3
#
#  E.g. hists[0,9,5] contains the number of image_color pixels such that:
#       - their R values fall in bin 0
#       - their G values fall in bin 9
#       - their B values fall in bin 5
def rgb_hist(img_color_double, num_bins):
    assert len(img_color_double.shape) == 3, 'image dimension mismatch'
    assert img_color_double.dtype == 'float', 'incorrect image type'

    div = 255/num_bins
    bins = np.arange(0,256, div)
    #Define a 3D histogram  with "num_bins^3" number of entries
    hists = np.zeros((num_bins, num_bins, num_bins))
    
    # Loop for each pixel i in the image 
    for blocco in img_color_double:
      for pixel in blocco:
        # Increment the histogram bin which corresponds to the R,G,B value of the pixel i
        # I initially wrote /256 to consider a interval made like this (..] but it dind't
        # get me the same neighbours as the pdf, so I changed it to [..) dividing by 255
        # and adding the if condition for the last one
        if pixel[0] == 255: r=num_bins-1
        else:
            r = int(num_bins * (pixel[0]) / (256))

        if pixel[1] == 255: g=num_bins-1
        else:
            g = int(num_bins * (pixel[1]) / (256))

        if pixel[2] == 255: b=num_bins-1
        else:
            b = int(num_bins * (pixel[2]) / (256))
        hists[r][g][b] += 1

    #Return the histogram as a 1D vector
    hists = hists.reshape(hists.size)

    #Normalize the histogram such that its integral (sum) is equal 1
    tot = sum(hists)
    norm = []
    for i in range(len(hists)):
      norm.append(hists[i]/tot)
    final = np.array(norm)

    return final


#  Compute the *joint* histogram for the R and G color channels in the image
#  The histogram should be normalized so that sum of all values equals 1
#  Assume that values in each channel vary between 0 and 255
#
#  img_color - input color image
#  num_bins - number of bins used to discretize each channel, total number of bins in the histogram should be num_bins^2
#
#  E.g. hists[0,9] contains the number of image_color pixels such that:
#       - their R values fall in bin 0
#       - their G values fall in bin 9

def rg_hist(img_color_double, num_bins):
    assert len(img_color_double.shape) == 3, 'image dimension mismatch'
    assert img_color_double.dtype == 'float', 'incorrect image type'

    div = 255/num_bins
    bins = np.arange(0,256, div)
    #Define a 2D histogram  with "num_bins^2" number of entries
    hists = np.zeros((num_bins, num_bins))
    
    # Loop for each pixel i in the image 
    
    for blocco in img_color_double:
      
      for pixel in blocco:
        # Increment the histogram bin which corresponds to the R,G,B value of the pixel i
        # I initially wrote /256 to consider a interval made like this (..] but it dind't
        # get me the same neighbours as the pdf, so I changed it to [..) dividing by 255
        # and adding the if condition for the last one
        if pixel[0] == 255: r=num_bins-1
        else:
            r = int(num_bins * (pixel[0]) / (255))
        if pixel[1] == 255: g=num_bins-1
        else:
            g = int(num_bins * (pixel[1]) / (255))
        hists[r][g] += 1
        
    
    #Return the histogram as a 1D vector
    hists = hists.reshape(hists.size)

    #Normalize the histogram such that its integral (sum) is equal 1
    tot = sum(hists)
    norm = []
    for i in range(len(hists)):
      norm.append(hists[i]/tot)
    final = np.array(norm)
    
    return final


#  Compute the *joint* histogram of Gaussian partial derivatives of the image in x and y direction
#  Set sigma to 3.0 and cap the range of derivative values is in the range [-6, 6]
#  The histogram should be normalized so that sum of all values equals 1
#
#  img_gray - input gray value image
#  num_bins - number of bins used to discretize each dimension, total number of bins in the histogram should be num_bins^2
#
#  Note: you may use the function gaussderiv from the Filtering exercise (gauss_module.py)
    

def cap6(nested):

    nested = [[6 if ( x > 6) else x for x in y] for y in nested]
    nested = [[-6 if ( x < -6) else x for x in y] for y in nested]

    return np.array(nested)

def dxdy_hist(img_gray, num_bins):
    assert len(img_gray.shape) == 2, 'image dimension mismatch'
    assert img_gray.dtype == 'float', 'incorrect image type'

    div = 12/num_bins
    bins = np.arange(-6,6, div)

    #Define a 2D histogram  with "num_bins^2" number of entries
    hists = np.zeros((num_bins, num_bins))

    imgx, imgy = gaussderiv(img_gray, 3.0)
 
    n_imgx = cap6(imgx)
    n_imgy = cap6(imgy)

    for i in range(len(n_imgx)):
      bloccox = n_imgx[i]
      bloccoy = n_imgy[i]
      for j in range(len(bloccox)):
        pixelx = bloccox[j]
        pixely = bloccoy[j]
        #print(pixelx, pixely)
     
       # trova nuovo metodo per negativi 
        x_value = bisect.bisect_left(bins,pixelx) -1
        y_value = bisect.bisect_left(bins,pixely) -1
        hists[x_value][y_value] += 1

    #Return the histogram as a 1D vector
    hists = hists.reshape(hists.size)
    
    tot = sum(hists)
    norm = []
    for i in range(len(hists)):
      norm.append(hists[i]/tot)
    final = np.array(norm)

    return final



def is_grayvalue_hist(hist_name):
  if hist_name == 'grayvalue' or hist_name == 'dxdy':
    return True
  elif hist_name == 'rgb' or hist_name == 'rg':
    return False
  else:
    assert False, 'unknown histogram type'


def get_hist_by_name(img, num_bins_gray, hist_name):
  if hist_name == 'grayvalue':
    return normalized_hist(img, num_bins_gray)
  elif hist_name == 'rgb':
    return rgb_hist(img, num_bins_gray)
  elif hist_name == 'rg':
    return rg_hist(img, num_bins_gray)
  elif hist_name == 'dxdy':
    return dxdy_hist(img, num_bins_gray)
  else:
    assert False, 'unknown distance: %s'%hist_name

