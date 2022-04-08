# Image Filtering and Object Identification

This project was developed by:
1. Caterina Alfano (@cat-erina)
2. Angelo Berardi (@AngeloBerardi)
3. Dario Cappelli (Capp18)
4. Emanuele Fratocchi (@Fraquia)

# Description 

## Image Filtering 

We implemented Image Filtering using Numpy on Python. We implemented **1-D** and **2-D** **Gaussian Filter** functions, where the function should take an image as an input and return the result of convolution of this image with a Gaussian kernel.
We also implemented **Gaussina Derivative Filter** for 1-D. Finally, we observed the results when we applied different combination of these functions. 

## Image Representation 

1. We used the module normalized module and implement the function **normalized hist**, which takes gray-value image as input and returns normalized histogram of pixel intensities. When quantizing the image to compute the histogram we considered that pixel intensities range in [0, 255].
We also compare our implementation with built-in Python function numpy.histogram, producing histograms and histograms that are approximately the same, except for the overall scale, which will be different since normalized hist does not normalize its output.

2. We implemented the histogram distance functions within the **dist_module**. 

## Object Identification

In the identification part, we compared images with several distance functions and evaluate their performance in combination with different image representations. The identification part contains **query** and **model images** for the evaluation, which correspond to the same set of objects photographed from different viewpoints. The files **model.txt** and **query.txt** contain lists of image files arranged so that i-th model image depicts the same object as i-th query image. The placeholder scripts will also be used to test our solution.


1. Having implemented different distance functions and image histograms, we can now test how suitable they are for retrieving images in query-by-example scenario, so we implement a function to find best match, in **match_module.py**, which takes a list of model images and a list of query images and for each query image returns the index of closest model image. The function takes string parameters, which identify distance function, histogram function and number of histogram bins. 

2. We implemented a function called **show_neighbors.py** in **match_module.py**  which takes a list of model images and a list of query images and for each query image visualizes several model images which are closest to the query image according to the specified distance metric. 

3. We used the function **find_best_match.py** to compute recognition rate for different combinations of distance and histogram functions. The recognition rate is given by a ratio between number of correct matches and total number of query images. We tried with different functions and numbers of histogram bins to find combination that works best. 

## Performances Evaluation

Sometimes instead of returning the best match for a query image we would like to return all the model images with distance to the query image below a certain threshold. It is, for example, the case when there are multiple images of the query object among the model images. In order to assess the system performance in such scenario we will use two quality measures: **precision** and **recall**. 

We implemented a function **plot_rpc**, defined in **rpc_module.py**, where we computed precision/recall pairs for a range of threshold values and then output the precision/recall curve (RPC), which gives a good summary of system performance at different levels of confidence.  We also plotted RPC curves for different histogram types, distances and number of bins. 

## Report
As part of the projet we also produced a write a report to explain the theory behind this project, and to summarise and comment results of our functions. 
