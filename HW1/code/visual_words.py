import os
import multiprocessing as mp
from os.path import join, isfile

import numpy as np
from PIL import Image
import scipy.ndimage
import skimage.color
import random
import matplotlib.pyplot as plt
import tempfile
from sklearn.cluster import KMeans

import math

def extract_filter_responses(opts, img):
    '''
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''
    
    filter_scales = opts.filter_scales
    # ----- TODO -----
    
    if len(img.shape) < 3:
       img = np.dstack((img, img, img))
    
    img_lab = skimage.color.rgb2lab(img)
    img_lab = (img_lab + [0, 128, 128]) / [100, 255, 255]
    filter_responses = np.zeros_like(img)

    for scale in filter_scales:
        gaussian_1 = scipy.ndimage.gaussian_filter(img_lab[:,:,0], sigma = scale)
        gaussian_2 = scipy.ndimage.gaussian_filter(img_lab[:,:,1], sigma = scale)
        gaussian_3 = scipy.ndimage.gaussian_filter(img_lab[:,:,2], sigma = scale)
        filtered_gaussian = np.dstack((gaussian_1,gaussian_2,gaussian_3))

        log_1 = scipy.ndimage.gaussian_laplace(img_lab[:,:,0], sigma = scale)
        log_2 = scipy.ndimage.gaussian_laplace(img_lab[:,:,1], sigma = scale)
        log_3 = scipy.ndimage.gaussian_laplace(img_lab[:,:,2], sigma = scale) 
        filtered_log = np.dstack((log_1,log_2,log_3))

        xDer_1 = scipy.ndimage.gaussian_filter(img_lab[:,:,0], order = [1,0], sigma = scale)
        xDer_2 = scipy.ndimage.gaussian_filter(img_lab[:,:,1], order = [1,0], sigma = scale)
        xDer_3 = scipy.ndimage.gaussian_filter(img_lab[:,:,2], order = [1,0], sigma = scale)
        filtered_xDer = np.dstack((xDer_1,xDer_2,xDer_3))

        yDer_1 = scipy.ndimage.gaussian_filter(img_lab[:,:,0], order = [0,1], sigma = scale)
        yDer_2 = scipy.ndimage.gaussian_filter(img_lab[:,:,1], order = [0,1], sigma = scale)
        yDer_3 = scipy.ndimage.gaussian_filter(img_lab[:,:,2], order = [0,1], sigma = scale)
        filtered_yDer = np.dstack((yDer_1,yDer_2,yDer_3))

        filter_responses = np.append(filter_responses, filtered_gaussian, axis = 2)
        filter_responses = np.append(filter_responses, filtered_log,      axis = 2)
        filter_responses = np.append(filter_responses, filtered_xDer,     axis = 2)
        filter_responses = np.append(filter_responses, filtered_yDer,     axis = 2)
    
    return filter_responses[:,:,3:]

def compute_dictionary_one_image(args):
    '''
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    '''

    # ----- TODO -----;
    image_path, opts, temp_file = args
    img = Image.open(image_path)
    img = np.array(img).astype(np.float32)/255
    filtered = extract_filter_responses(opts, img)
    #print(filtered.shape)
    alpha = opts.alpha
    H,W,C = filtered.shape
    random_samples = random.sample(range(H*W),alpha)
    filtered_2D = np.reshape(filtered, (-1,C))
    sampled_responses = filtered_2D[random_samples]
    
    np.save(temp_file, sampled_responses)
    return temp_file

def compute_dictionary(opts, n_worker=1):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel
    
    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    '''

    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K

    train_files = open(os.path.join(opts.data_dir, 'train_files.txt')).read().splitlines()

    # ----- TODO -----;
    


    with tempfile.TemporaryDirectory() as temp_dir:
        args = [[join(data_dir, image_path), opts, os.path.join(temp_dir, f"response_{i}.npy")] for i, image_path in enumerate(train_files)]
        with mp.Pool(n_worker) as p:
            temp_files = p. map(compute_dictionary_one_image, args)
            
        responses = []
        for temp_file in temp_files:
            response = np.load(temp_file)
            responses.append(response)

        responses = np.vstack(responses)
        kmeans = KMeans(n_clusters = K, random_state = 42).fit(responses)
        dictionary = kmeans.cluster_centers_
        print(dictionary.shape)

    np.save(join(out_dir, 'dictionary.npy'), dictionary)

def get_visual_words(opts, img, dictionary):
    '''
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts:     options
    * img:      numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap:  numpy.ndarray of shape (H,W)
    '''
    
    # ----- TODO -----
    if len(img.shape) < 3:
       img = np.dstack((img, img, img))

    filtered = extract_filter_responses(opts, img)
    filtered_2D = np.reshape(filtered, (-1,filtered.shape[2]))
    wordmap_allchannels = np.zeros_like(filtered_2D)
    wordmap_allchannels = scipy.spatial.distance.cdist(filtered_2D,dictionary, metric ='euclidean')
    #print(wordmap_allchannels.shape)
    
    wordmap = np.argmin(wordmap_allchannels, axis = 1)
    wordmap = np.reshape(wordmap,(img.shape[0],img.shape[1]))
    return wordmap
