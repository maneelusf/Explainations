import numpy as np
import random
import scipy 
from sklearn.metrics import confusion_matrix
import seaborn as sns
from scipy.spatial.distance import cdist
from scipy import sparse
from matplotlib.pyplot import imread
import ipywidgets as widgets
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
from sklearn.datasets import make_blobs
import pandas as pd
import time

color_dict = {0:'#f58231',1:'#bfef45',2:'#42d4f4',3:'#469990',4:'#aaffc3',5:'#fffac8',6:'#ffd8b1',7:'#fabed4'}
def read_image(img):
      
    # loading the png image as a 3d matrix 
    img = imread(img) 
  
    # uncomment the below code to view the loaded image
    plt.imshow(img) # plotting the image
    plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False,
    labelleft = False,
    right = False,
    left = False)
    plt.show() 
      
    # scaling it so that the values are small
    img = img / 255 
  
    return img

def kmeans_plot(X,k,max_iter = 3):
    centroids,labels = kmeans(X,k,max_iter = max_iter)
    fig,axes = plt.subplots(figsize = (16,9))
    axes.scatter(X[:,0],X[:,1]);
    axes.set_title('{} clusters & {} iterations'.format(k,max_iter));
    color_labels = [color_dict[x] for x in labels]
    axes.scatter(X[:,0],X[:,1],c = color_labels);
    axes.scatter(np.array(centroids)[:,0],np.array(centroids)[:,1],c = 'black',marker = '+',s = 100,label = 'centroid');
    axes.legend();
    
def centroidselection(X,centroid_list):
    centroid_distance = []
    centroid_dict = {}
    for centroid in centroid_list:
        broadcast = np.broadcast_to(centroid,(X.shape[0],len(centroid)))
        distances = np.linalg.norm(broadcast - X,axis = 1)
        centroid_distance.append(distances)
    centroidpoints = np.argmin(np.array(centroid_distance),axis = 0)
    for centroid_index,centroid in enumerate(centroid_list):
        centroid_dict[tuple(centroid)] = list(X[np.argwhere(centroidpoints==centroid_index).reshape(1,-1)[0]])
    return centroid_dict,centroidpoints
        
def initialize(X, K):
    C = [X[0]]
    for k in range(1, K):
        D2 = np.array([min([np.inner(c-x,c-x) for c in C]) for x in X])
        probs = D2/D2.sum()
        cumprobs = probs.cumsum()
        r = np.random.rand()
        #import pdb;pdb.set_trace()
        for j,p in enumerate(cumprobs):
            if r < p:
                i = j
                break
        C.append(X[i])
    return C
def kmeans(X,k = 3, centroids='kmeans++',max_iter = 30,tolerance = 0.01):
    shape = X.shape
    max_iter = max_iter
    obs_indices = np.arange(0,shape[0])
    if centroids != 'normal':
        shape_dict = np.percentile(obs_indices,np.linspace(0,100,k)).astype(int)
        centroids = [X[x] for x in shape_dict]
    elif centroids == 'kmeans++':
        centroids = initialize(X,k)
    else:
        
        centroids = np.random.choice(np.arange(0,X.shape[0]),size = k,replace = False)
        centroids = [X[x] for x in centroids]
    for iteration in range(0,max_iter):
        centroid_dict = {}
        ## initialize an empty centroid
        for centroid in centroids:
            centroid_dict[tuple(centroid)] = []
    
        centroid_dict,labels = centroidselection(X,centroids)
        ##calculating distances
        new_centroids = [np.mean(centroid_dict[key],axis = 0) for key in centroid_dict.keys()]
        if np.linalg.norm(np.array(new_centroids) - np.array(centroids)) < tolerance:
            break
        else:
            centroids = new_centroids
    return centroids,labels

def likely_confusion_matrix(y,labels):
    label_count = np.unique(labels)
    y_pred = np.zeros(y.shape)
    for label in label_count:
        pred_value = y[np.argwhere(labels == label)].reshape(1,-1)[0]
        values, counts = np.unique(pred_value, return_counts=True)
        pred_value = values[counts.argmax()]
        y_pred[np.argwhere(labels == label).reshape(1,-1)[0]] = pred_value
    likely_confusion_matrix = confusion_matrix(y,y_pred)
    likely_confusion_matrix = pd.DataFrame(likely_confusion_matrix,columns = ['Actual Positive','Actual Negative'],\
             index = ['Predicted Positive','Predicted Negative'])
    
    return likely_confusion_matrix

def generate_circle_sample_data(r, n, sigma):
    """Generate circle data with random Gaussian noise."""
    angles = np.random.uniform(low=0, high=2*np.pi, size=n)

    x_epsilon = np.random.normal(loc=0.0, scale=sigma, size=n)
    y_epsilon = np.random.normal(loc=0.0, scale=sigma, size=n)

    x = r*np.cos(angles) + x_epsilon
    y = r*np.sin(angles) + y_epsilon
    return x, y

def generate_concentric_circles_data(param_list):
    """Generates many circle data with random Gaussian noise."""
    coordinates = [ 
        generate_circle_sample_data(param[0], param[1], param[2])
     for param in param_list
    ]
    return coordinates

def timegraph(params = [int(x) for x in np.linspace(100,10000,100)]):
    time_list = []
    random_state = 170
    for param in params:
        n_samples = param
        X, y = make_blobs(n_samples=param, random_state=random_state)
        start = time.time()
        centroids,labels = kmeans(X,4)
        time_list.append(time.time()-start)
    time_list = pd.Series(time_list,index = params)
    return time_list