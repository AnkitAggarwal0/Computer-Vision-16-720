import os, math, multiprocessing
from os.path import join
from copy import copy

import numpy as np
from PIL import Image

import visual_words
from sklearn import metrics
import functools

def get_feature_from_wordmap(opts, wordmap):
    '''
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    K = opts.K
    # ----- TODO -----

    histogram, bins = np.histogram(wordmap, bins = range(K+1), density= False)
    #histogram /= histogram.sum()
    return histogram

def get_feature_from_wordmap_SPM(opts, wordmap):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^L-1)/3)
    '''
        
    K = opts.K
    L = opts.L
    # ----- TODO -----
    
    hist_all = []

    for l in range(L+1):
        num_cells = 2**l
        cell_height = wordmap.shape[0] // num_cells
        cell_width = wordmap.shape[1] // num_cells

        for i in range(num_cells):
            for j in range(num_cells):
                cell = wordmap[i*cell_height:(i+1)*cell_height, j*cell_width:(j+1)*cell_width]
                hist = get_feature_from_wordmap(opts,cell)
                
                if l == 0 or l == 1:
                    hist = hist * 2**(-L)
                else:
                    hist = hist * 2**(l-L-1)

                hist_all.append(hist)
    
    hist_all = np.array(hist_all).flatten()
    hist_all = hist_all / np.sum(hist_all)

    
    return hist_all
    
def get_image_feature(opts, img_path, dictionary):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K*(4^L-1)/3)
    '''

    # ----- TODO -----
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)/255

    img_wordmap = visual_words.get_visual_words(opts,img,dictionary)
    img_histogram = get_feature_from_wordmap_SPM(opts, img_wordmap)

    return img_histogram

def build_recognition_system(opts, n_worker=1):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
    dictionary = np.load(join(out_dir, 'dictionary.npy'))

    # ----- TODO -----
    train_files = [os.path.join(data_dir,name) for name in train_files]
    partial_get_image_feature = functools.partial(get_image_feature,opts,dictionary=dictionary)
    with multiprocessing.Pool(n_worker) as p:
        features = p .map(partial_get_image_feature, train_files)
        
    print(len(features))

    ## example code snippet to save the learned system
    np.savez_compressed(join(out_dir, 'trained_system.npz'),
         features=features,
         labels=train_labels,
         dictionary=dictionary,
         SPM_layer_num=SPM_layer_num,
    )

def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * hist_dist: numpy.ndarray of shape (N)
    '''

    # ----- TODO -----
    similarity = np.sum(np.minimum(word_hist,histograms),axis=1)
    distance_measure = 1-similarity

    return distance_measure    
    
def evaluate_recognition_system(opts, n_worker=1):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, 'trained_system.npz'))
    dictionary = trained_system['dictionary']

    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system['SPM_layer_num']

    test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, 'test_labels.txt'), np.int32)

    # ----- TODO -----
    test_files = [os.path.join(data_dir,name) for name in test_files]

    trained_labels = trained_system['labels']
    trained_features = trained_system['features']
    
    get_image_feature_partial = functools.partial(get_image_feature, opts, dictionary = dictionary)
    distance_to_set_partial = functools.partial(distance_to_set,histograms = trained_features)
    with multiprocessing.Pool(n_worker) as p:
        histogram_testfile = p .map(get_image_feature_partial, test_files)
        distance_measure = p .map(distance_to_set_partial,histogram_testfile)
    
    min_distance_index = np.argmin(distance_measure,axis=1)
    computed_labels = trained_labels[min_distance_index]

    confusion_matrix = metrics.confusion_matrix(test_labels,computed_labels)
    accuracy = metrics.accuracy_score(test_labels,computed_labels)

    return confusion_matrix, accuracy