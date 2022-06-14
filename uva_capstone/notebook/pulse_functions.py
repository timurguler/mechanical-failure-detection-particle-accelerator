# -*- coding: utf-8 -*-
"""
Pulse Class

Created on Thu Nov 11 15:06:43 2021

@author: tgule
"""
import datetime
import pandas as pd
import numpy as np
import os
import pickle
import boto3
import scipy
from scipy import stats
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
import seaborn as sns

from pyts.approximation import SymbolicFourierApproximation
from pyts.datasets import load_gunpoint
from pyts.bag_of_words import BagOfWords
from pyts.bag_of_words import WordExtractor
from pyts.classification import BOSSVS
from pyts.transformation import BOSS

from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, auc
from sklearn.model_selection import RepeatedKFold

from itertools import product

import pywt

import GPy as gp

##########
# SECTION 1 - PREPROCESSING FUNCTIONS
##########

def get_metadata(s3_object):
    '''
    takes an s3 file object for a csv file with the standard pulse file naming convention
    outputs - dictionary with interval (20 or 400 ns), result (run/fault), timestamp
    test update
    '''
    # split filepath with slashes to get filename
    num_folders = len(s3_object.key.split('/')) - 1
    file_name = s3_object.key.split('/')[num_folders]
    
    # use string parsing to get interval, result, time
    interval = file_name.split('_')[1]
    result = file_name.split('_')[2]
    str_time = file_name.split('_')[3].replace('.csv', '')
    #time = datetime.datetime.strptime(str_time, '%m.%d.%Y-%I.%M.%S%p')
    return {'interval' : interval, 'result' : result, 'str_time' : str_time}


def get_normals(all_pulses, runs=True):
    '''
    This function takes a dictionary of pulses and returns the max value of each component, to be used for normalization
    INPUT - 'all_pulses' dictionary of pulse object, 'runs' boolean, whether to get max of only runs or of all pulses
    OUPUT - pandas series with max value for each component
    '''
    components = all_pulses[list(all_pulses)[0]].data.columns # get column list

    component_maxs = pd.DataFrame() # set up df to store max's for each component in each pulse
    
    if runs==True: # reduce to only runs
        all_pulses = {k : v for k, v in all_pulses.items() if v.result == 'Run'}
    
    # create df with max from each individual pulse
    for k, v in all_pulses.items():
        pulse1_maxs = pd.DataFrame(v.pulse1.abs().max()).transpose()

        component_maxs = pd.concat([component_maxs, pulse1_maxs]).reset_index(drop=True)
    
    # calculate max over all pulses
    return component_maxs.max()


##########
# SECTION 2 - DATA TRANSFORMATION FUNCTIONS
##########

def get_all_pulses(pulses : dict, cols = [], normalized=True, reshape=None):
    '''
    GOAL - This function is intended to pull an array of the raw pulse data for a set of pulses 
    (pulses 1 and 2 for runs, pulse 1 for faults) for a subset of components
    INPUTS - pulses : dictionary of pulse objects, cols : list of components to use, normal : whether to compare
    to the normalized pulse
    OUTPUTS - X : feature tensor (3D), y_fault : vector of booleans, y_type : vector of strings, indices : list of index meaning for X'''
    
    # set up result arrays
    y_fault = []
    y_type = []
    pulse_index = []
    
    ex_pulse = pulses[list(pulses.keys())[0]].pulse1
    
    if len(cols) == 0:
        cols = ex_pulse.columns
        
    X = np.empty((0, len(cols), ex_pulse.loc[::reshape].shape[0]))
    
    # loop through pulses
    for k, v in pulses.items():
    
        # use normalized pulse if specified, otherwise raw pulse
        if normalized:
            obs_pulses = [v.pulse1norm, v.pulse2norm]

        else:
            obs_pulses = [v.pulse1, v.pulse2]
       
        # only use first pulse if fault
        if v.result == 'Fault':
            obs_pulses = [obs_pulses[0]]
            y_fault.append(1)
            y_type.append(v.fault_type)
        
        else:
            _ = [y_fault.append(0) for x in obs_pulses]
            _ = [y_type.append('Run') for x in obs_pulses]
        
        # get timestamp
        _ = [pulse_index.append(k + '-pulse' + str(x+1)) for x in range(len(obs_pulses))]
        
        # concatenate all pulses to overall array
        this_X = np.array([pulse[cols].loc[::reshape].T for pulse in obs_pulses])
        
        
        X = np.concatenate((X, this_X), axis=0)
        
    
    y_fault = np.array(y_fault)
    y_type = np.array(y_type)
    indices = (pulse_index, cols, list(pulses.values())[0].pulse1.loc[::reshape].index)
        
        
    return X, y_fault, y_type, indices


def get_median_pulse(pulses, normalized=True, cols=[], reshape=None):
    '''
    Takes the median of all pulses per component of all runs. 
    These are to be used as a baseline to determine if a run is normal or abnormal.
    INPUT: dictionary of pulse objects
    OUTPUT: median of all first pulses per component  of all runs.
    '''
    
    ex_pulse = pulses[list(pulses.keys())[0]].pulse1
    
    if len(cols) == 0:
        cols = ex_pulse.columns
        
    runs = {k : v for k, v in pulses.items() if v.result == 'Run'}
    X_runs, _, _, _ = get_all_pulses(runs, cols=cols, normalized=normalized, reshape=reshape)
    
    medians = np.median(X_runs, axis=0)
    
    return medians

def get_L2_norms(pulses : dict, cols = [], normalized=True, reshape=None, zscore=True):
    '''
    GOAL - This function is intended to calculate L2 norms of the difference between the actual and expected pulse for a set of components
    (pulses 1 and 2 for runs, pulse 1 for faults)
    INPUTS - pulses : dictionary of pulse objects, cols : list of components to use, normalized : whether to compare
    to the normalized pulse, reshape : the downsampling rate, zscore : whether to incorporate the standard deviation
    OUTPUTS - X : feature tensor (3D), y_fault : vector of booleans, y_type : vector of strings, indices : list of index meaning for X
    '''
    
    ex_pulse = pulses[list(pulses.keys())[0]].pulse1
    
    if len(cols) == 0:
        cols = ex_pulse.columns
        
    runs = {k : v for k, v in pulses.items() if v.result == 'Run'}
    
    
    X_runs, _, _, _ = get_all_pulses(runs, cols, reshape=None)
    
    medians = np.median(X_runs, axis=0)
    stds = np.std(X_runs, axis=0)    
    
    X_raw, y_fault, y_type, indices = get_all_pulses(pulses, cols, reshape=None)
    
    if zscore:
        X = np.linalg.norm(((X_raw - medians)/stds), axis=2)
        
    else:
        X = np.linalg.norm((X_raw - medians), axis=2)
        
    return X, y_fault, y_type, indices

def vectorize(word_array, n_bins=2, word_size=3):
    '''
    GOAL - converts an array of words that represent a BOW sentence (nx1) transformation into a one-hot encoded vector (nxvocab_size)
    '''
    alphabet = 'abcdefghijklmnopqrstuvwxyz' # set alphabet
    column_names = [''.join(x) for x in product(alphabet[:n_bins], repeat=word_size)] # list of all combinations (order matters to keep consistent)
    
    vector = pd.get_dummies(word_array) # convert to vector
    
    # add columns where word not used
    cols_to_add = set(column_names).difference(set(vector.columns))
    for col in cols_to_add:
        vector[col] = 0
    
    # reset order
    vector = vector[column_names]
    
    return vector

def get_bow_results(pulses, cols = [], reshape=None, normalized=True, window_size=10, word_size=3, n_bins=2, method='raw'):
    '''
    GOAL - get eiher a) sentence version of waveform for all observations of a set of components b) vectorized version of (a) or c) histogram representation of word counts (BOSS)
    IMPUTS -  dictionary of pulses, name of component to be used, downsampling rate, window size, word size, number of available letters for vocab
    OUTPUTS - X : feature tensor (3 OR 4D), y_fault : vector of booleans, y_type : vector of strings, indices : list of index meaning for X
    '''
    
    # get raw pulse values
    X_raw, y_fault, y_type, indices_raw = get_all_pulses(pulses, cols=cols, normalized=normalized, reshape=reshape)
    
    # configure bag-of-words model
    bow = BagOfWords(window_size=window_size, word_size=word_size,
                     window_step=window_size, numerosity_reduction=False, n_bins=n_bins)
    

    # transform to BOW and reshape
    X = np.array([np.array([np.array(bow.transform([X_raw[x, y, :]])[0].split()) for y in range(X_raw.shape[1])]) for x in range(X_raw.shape[0])])
    indices = (indices_raw[0], indices_raw[1], range(0, X.shape[2]))
    
    # vectorization and BOSS transformation - relies on helper "vectorize function"
    if method == 'vectorized' or method == 'boss':
        vectorized = np.array([np.array([np.array(vectorize(X[x, y, :])) for y in range(X.shape[1])]) for x in range(X.shape[0])])
        indices_vec = (indices[0], indices[1], indices[2], range(0, vectorized.shape[3]))
        
        if method == 'vectorized':
            X = vectorized
            indices = indices_vec
        
        else: # BOSS transformation if necessary
            X = vectorized.sum(axis=2)
            indices = (indices[0], indices[1], X.shape[2])
            
        
    
    #return outputs
    return X, y_fault, y_type, indices

    
def get_fourier_values(pulses : dict, cols = [], sampling_rate = 0.25, normalized=True, reshape=None):
    '''
    GOAL - This function is intended to calculate Fourier transformations for some or all components across a set of pulses.
    INPUTS - pulses : dictionary of pulse objects, cols : list of components to use, normal : whether to use the normalized pulse, 
    sampling_rate : int?, reshape : int, downsampling rate
    OUTPUTS - X : feature tensor (3D), y_fault : vector of booleans, y_type : vector of strings, indices : list of index meaning for X
    '''
    
    # get raw pulse values
    X_raw, y_fault, y_type, indices_raw = get_all_pulses(pulses, cols=cols, normalized=normalized, reshape=reshape)
    
    # configure bag-of-words model
    size = X_raw.shape[2]    

    # transform to fourier
    X = np.array([np.array([(2.0/size * np.abs(fft(X_raw[x,y,:])[0:size//2])) for y in range(X_raw.shape[1])]) for x in range(X_raw.shape[0])])
    indices = (indices_raw[0], indices_raw[1], range(0, X.shape[2]))
            
    
    #return outputs
    return X, y_fault, y_type, indices


def wavelet_transform(signal, waveletname='cgau5', scales=np.arange(1, 128), sampling_rate=0.3):
    
    # perform transformation
    [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, sampling_rate)
    
    # normalize
    power = (abs(coefficients)) ** 2
    power_normalized = (power - np.min(power))/(np.max(power) - np.min(power))
    
    return power_normalized

def plot_wavelet(pulse : pd.DataFrame, component, reshape=None, waveletname='cgau5', scales=np.arange(1, 128), sampling_rate=0.3):
    '''
    GOAL - this function plots the wavelet of one pulse (e.g. pulse1norm on X date)
    INPUTS - pulse : singular pulse (e.g. pulse1 from a certain runtime), component : name of component, reshape : downsampling rate,
    waveletname : type of wavelet tranform, scales : ?, sampling_rate : ?
    OUTPUT - graph of wavelet
    '''
    
    signal = pulse[component].loc[::reshape]
    wave_data = wavelet_transform(signal, waveletname=waveletname, scales=scales, sampling_rate=sampling_rate)
    
    plt.figure(figsize=(15,10))
    plt.imshow(wave_data, cmap='gray')
    plt.show()
    
def get_wavelet_values(pulses : dict, cols = [], normalized=True, reshape=None, waveletname='cgau5', scales=np.arange(1, 128), sampling_rate = 0.3):
    '''
    GOAL - This function is intended to calculate wavelet transformations for some or all components across a set of pulses.
    INPUTS - pulses : dictionary of pulse objects, cols : list of components to use, normal : whether to use the normalized pulse, 
    sampling_rate : int?, reshape : int, downsampling rate
    OUTPUTS - X : feature tensor (3D), y_fault : vector of booleans, y_type : vector of strings, indices : list of index meaning for X
    '''
    
    # get raw pulse values
    X_raw, y_fault, y_type, indices_raw = get_all_pulses(pulses, cols=cols, normalized=normalized, reshape=reshape)
    
    # configure bag-of-words model  

    # transform to fourier
    X = np.array([np.array([wavelet_transform(X_raw[x, y, :], waveletname=waveletname, scales=scales, sampling_rate=sampling_rate) for y in range(X_raw.shape[1])]) for x in range(X_raw.shape[0])])
    indices = (indices_raw[0], indices_raw[1], scales, range(0, X.shape[3]))
            
    
    #return outputs
    return X, y_fault, y_type, indices


##########
# MODELING AND EVALUATION FUNCTIONS
##########

def fit_and_eval_model(training, testing, transform=get_L2_norms, transform_params : dict = None, model_type=LogisticRegression, model_params = {'random_state' : 2022}):
    '''
    GOAL - Fit and evaluate a model using transformed pulse data
    INPUTS - 
        training : dict of pulses
        testing : dict of pulses
        transform : transformation function to use
        transform_params : dict of parameters for transformation function (e.g. reshape, normalized)
        model_type : sk-learn model
        model_params : parameters to use when running model
        
    OUTPUTS -
        model - the actual model
        cms - list of confusion matrices
        crs - list of classification reports
        roc_rates - list of fprs and tprs for train and test at points along roc curve
        roc_aucs - auc for train and test
            
    '''
    # transform to get feature set
    X_train, y_fault_train, y_type_train, indices = transform(training, **transform_params)
    X_test, y_fault_test, y_type_test, indices = transform(testing, **transform_params)
    
    # define and fit model
    model = model_type(**model_params)
    model.fit(X_train, y_fault_train)
    
    # make predictions
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    # confusion matrices
    cm_train = pd.DataFrame(confusion_matrix(y_fault_train, train_preds), columns=['pred_run', 'pred_fault'], index=['run', 'fault'])
    cm_test = pd.DataFrame(confusion_matrix(y_fault_test, test_preds), columns=['pred_run', 'pred_fault'], index=['run', 'fault'])
    cms = [cm_train, cm_test]
    
    # classification reports
    cr_train = pd.DataFrame(classification_report(y_fault_train, train_preds, target_names=['fault','run'], output_dict=True)).transpose()
    cr_test = pd.DataFrame(classification_report(y_fault_test, test_preds, target_names=['fault','run'], output_dict=True)).transpose()
    crs = [cr_train, cr_test]
    
    # roc info
    fpr_train, tpr_train, threshold = metrics.roc_curve(y_fault_train, train_preds)
    fpr_test, tpr_test, threshold = metrics.roc_curve(y_fault_test, test_preds)
    
    roc_rates = [fpr_train, tpr_train, fpr_test, tpr_test]
    
    roc_aucs = (metrics.auc(fpr_train, tpr_train), metrics.auc(fpr_test, tpr_test))
    
    return model, cms, crs, roc_rates, roc_aucs

def plot_roc(roc_rates, title):
    '''
    GOAL - this function plot's a model's ROC curve
    INPUT - roc_rates - a list of length 4 containing (fpr_train, tpr_train, fpr_test, tpr_test) - from fit_and_eval_model function
    OUTPT - graph of ROC curve
    '''
        
    fpr_train, tpr_train, fpr_test, tpr_test = roc_rates
    roc_auc_train = auc(fpr_train, tpr_train)
    roc_auc_test = auc(fpr_test, tpr_test)
    
    plt.title(f'Receiver Operating Characteristic - {title}')
    plt.plot(fpr_train, tpr_train, 'b', label = 'Training AUC = %0.2f' % roc_auc_train)
    plt.plot(fpr_test, tpr_test, 'r', label = 'Test AUC = %0.2f' % roc_auc_test)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'g--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def fit_and_eval_model(training, testing, transform=get_L2_norms, transform_params : dict = None, model_type=LogisticRegression, model_params = {'random_state' : 2022}):
    '''
    GOAL - Fit and evaluate a model using transformed pulse data
    INPUTS - 
        training : dict of pulses
        testing : dict of pulses
        transform : transformation function to use
        transform_params : dict of parameters for transformation function (e.g. reshape, normalized)
        model_type : sk-learn model
        model_params : parameters to use when running model
        
    OUTPUTS -
        model - the actual model
        cms - list of confusion matrices
        crs - list of classification reports
        roc_rates - list of fprs and tprs for train and test at points along roc curve
        roc_aucs - auc for train and test
            
    '''
    # transform to get feature set
    X_train, y_fault_train, y_type_train, indices = transform(training, **transform_params)
    X_test, y_fault_test, y_type_test, indices = transform(testing, **transform_params)
    
    # define and fit model
    cv = RepeatedKFold(n_splits=3, n_repeats=20, random_state=1)
    
    metrics = {'accuracy' : accuracy_score, 'precision' : precision_score, 'recall' : recall_score, 'auc' : roc_auc_score}
    
    all_is = {k : [] for k, v in metrics.items()}
    all_oos = {k : [] for k, v in metrics.items()}
    all_test = {k : [] for k, v in metrics.items()}

    for train_idx, test_idx in cv.split(X_train):
        model = model_type(**model_params)
        model.fit(X_train[train_idx], y_fault_train[train_idx])

        is_preds = model.predict(X_train[train_idx])
        oos_preds = model.predict(X_train[~train_idx])
        test_preds = model.predict(X_test)

        is_true = y_fault_train[train_idx]
        oos_true = y_fault_train[~train_idx]
        test_true = y_fault_test

        _ = [all_is[name].append(function(is_true, is_preds)) for name, function in metrics.items()]
        _ = [all_oos[name].append(function(oos_true, oos_preds)) for name, function in metrics.items()]
        _ = [all_test[name].append(function(test_true, test_preds)) for name, function in metrics.items()]

    
    df_is = pd.DataFrame({k : np.array(v).mean() for k, v in all_is.items()}, index=[0])
    df_is['sample'] = 'in sample'
 

    df_oos = pd.DataFrame({k : np.array(v).mean() for k, v in all_oos.items()}, index=[1])
    df_oos['sample'] = 'out of sample'

    df_test = pd.DataFrame({k : np.array(v).mean() for k, v in all_test.items()}, index=[2])
    df_test['sample'] = 'test'
    
    df_results = pd.concat([df_is, df_oos, df_test])
    
    
    #df_results['transformation'] = transform
    #df_results['transformation_params'] = str(transform_params)
    #df_results['model'] = model_type
    #df_results['model_params'] = str(model_params)
    
    return model, df_results

def fit_and_eval_model_3D(training, testing, transform=get_L2_norms, transform_params : dict = None, model_type=LogisticRegression, model_params = {'random_state' : 2022}):
    '''
    GOAL - Fit and evaluate a model using transformed pulse data
    INPUTS - 
        training : dict of pulses
        testing : dict of pulses
        transform : transformation function to use
        transform_params : dict of parameters for transformation function (e.g. reshape, normalized)
        model_type : sk-learn model
        model_params : parameters to use when running model
        
    OUTPUTS -
        model - the actual model
        cms - list of confusion matrices
        crs - list of classification reports
        roc_rates - list of fprs and tprs for train and test at points along roc curve
        roc_aucs - auc for train and test
            
    '''
    # transform to get feature set
    X_train, y_fault_train, y_type_train, indices = transform(training, **transform_params)
    X_test, y_fault_test, y_type_test, indices = transform(testing, **transform_params)
    
    new_shape_train = (X_train.shape[0], X_train.shape[2])
    new_shape_test = (X_test.shape[0], X_test.shape[2])
    
    # define and fit model
    cv = RepeatedKFold(n_splits=3, n_repeats=20, random_state=1)
    
    metrics = {'accuracy' : accuracy_score, 'precision' : precision_score, 'recall' : recall_score, 'auc' : roc_auc_score}
    
    all_results = pd.DataFrame()
    
    for col in indices[1]:
        col_idx = indices[1].index(col)
        X_train_col = X_train[:, col_idx, :].reshape(new_shape_train)
        X_test_col = X_test[:, col_idx, :].reshape(new_shape_test)
        
    
        all_is = {k : [] for k, v in metrics.items()}
        all_oos = {k : [] for k, v in metrics.items()}
        all_test = {k : [] for k, v in metrics.items()}

        for train_idx, test_idx in cv.split(X_train):
            model = model_type(**model_params)
            model.fit(X_train_col[train_idx], y_fault_train[train_idx])

            is_preds = model.predict(X_train_col[train_idx])
            oos_preds = model.predict(X_train_col[~train_idx])
            test_preds = model.predict(X_test_col)

            is_true = y_fault_train[train_idx]
            oos_true = y_fault_train[~train_idx]
            test_true = y_fault_test

            _ = [all_is[name].append(function(is_true, is_preds)) for name, function in metrics.items()]
            _ = [all_oos[name].append(function(oos_true, oos_preds)) for name, function in metrics.items()]
            _ = [all_test[name].append(function(test_true, test_preds)) for name, function in metrics.items()]


        df_is = pd.DataFrame({k : np.array(v).mean() for k, v in all_is.items()}, index=[0])
        df_is['sample'] = 'in sample'


        df_oos = pd.DataFrame({k : np.array(v).mean() for k, v in all_oos.items()}, index=[1])
        df_oos['sample'] = 'out of sample'

        df_test = pd.DataFrame({k : np.array(v).mean() for k, v in all_test.items()}, index=[2])
        df_test['sample'] = 'test'

        df_results = pd.concat([df_is, df_oos, df_test])
        df_results['component'] = col
        
        all_results = pd.concat([all_results, df_results]).reset_index(drop=True)
    
    #df_results['transformation'] = transform
    #df_results['transformation_params'] = str(transform_params)
    #df_results['model'] = model_type
    #df_results['model_params'] = str(model_params)
    
    return all_results
    
    #return model, df_results
    
def run_gpc(X, y_fault, y_type, indices, cv, kernel):
    '''
    GOAL - Fit a GPC model in GPy with cross-validation and make predictions
    INPUTS - 
        X : np array of input data (output of transformation functions)
        y_fault : np array of binary fault determination (output of transformation functions)
        y_type : np array of specific fault types (output of transformation functions)
        indices : metadata array (output of transformation functions)
        cv : sk-learn repeated k foldcross validator object
        kernel : valid GPy kernel or composition of GPy kernels
        
    OUTPUTS -
        results - df of log odds distribution for fault likelihoof for each component of each observation in each fold, along with metadata
            
    '''
    
    # set up output df
    results = pd.DataFrame()
    
    # tracker for each fold
    run = 0
    
    # log odds space
    x_logit = np.linspace(-4, 4, 200)

    # loop through repeated k folds
    for train_idx, test_idx in cv.split(X):

        # Test-train split
        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y_fault[train_idx].reshape(y_fault[train_idx].shape[0], 1)
        y_test = y_fault[test_idx].reshape(y_fault[test_idx].shape[0], 1)

        label_train = np.array(indices[0])[train_idx]
        label_test = np.array(indices[0])[test_idx]

        # identifying seen and unseen fault types
        is_types = np.unique(y_type[train_idx])
        oos_types = np.unique(y_type[test_idx])
        unseen_types = list(set(oos_types).difference(set(is_types)))

        # loop through components, define and fit model for each with trianing split, predict on both training and test splits 
        for component in indices[1]:
            # get values for specific component
            comp_idx = indices[1].index(component)
            X_train_comp = X_train[:, comp_idx, :].squeeze()
            X_test_comp = X_test[:, comp_idx, :].squeeze()
            
            # set up dict to store results
            results_train, results_test = {}, {}

            # fit model
            model_gp = gp.models.GPClassification(X_train_comp, y_train, kernel)
            model_gp.optimize()

            # make predictions and get distribution for component
            preds_train =  model_gp.predict(X_train_comp)
            gp_vars_train = model_gp.predict(X_train_comp, include_likelihood=False)

            params_train = np.concatenate((scipy.special.logit(preds_train[0]), gp_vars_train[1]), axis=1)
            posteriors_train = np.array([stats.norm(loc=params_train[ix][0], scale=params_train[ix][1]).pdf(x_logit) for ix in range(params_train.shape[0])])

            results_train['timestamp'] = label_train
            results_train['distribution'] = posteriors_train

            preds_test =  model_gp.predict(X_test_comp)
            gp_vars_test = model_gp.predict(X_test_comp, include_likelihood=False)

            params_test = np.concatenate((scipy.special.logit(preds_test[0]), gp_vars_test[1]), axis=1)
            posteriors_test = np.array([stats.norm(loc=params_test[ix][0], scale=params_test[ix][1]).pdf(x_logit) for ix in range(params_test.shape[0])])

            # store data and metadata
            posteriors_df_train = pd.DataFrame(posteriors_train, columns=x_logit)
            metadata_df_train = pd.DataFrame(label_train, columns=['timestamp'])
            metadata_df_train['run'] = run
            metadata_df_train['component'] = component
            metadata_df_train['type'] = y_type[train_idx]
            metadata_df_train['actual'] = y_train.squeeze()
            metadata_df_train['seen'] = True
            metadata_df_train['group'] = 'train'
            df_train = pd.concat((metadata_df_train, posteriors_df_train), axis=1).set_index(['timestamp', 'run', 'component', 'type', 'actual', 'seen', 'group'])

            posteriors_df_test = pd.DataFrame(posteriors_test, columns=x_logit)
            metadata_df_test = pd.DataFrame(label_test, columns=['timestamp'])
            metadata_df_test['run'] = run
            metadata_df_test['component'] = component
            metadata_df_test['type'] = y_type[test_idx]
            metadata_df_test['actual'] = y_test.squeeze()
            metadata_df_test['seen'] =  ~metadata_df_test.type.isin(unseen_types)
            metadata_df_test['group'] = 'test'
            df_test = pd.concat((metadata_df_test, posteriors_df_test), axis=1).set_index(['timestamp', 'run', 'component', 'type', 'actual', 'seen', 'group'])

            # concatenate
            results = pd.concat([results, df_train, df_test])

        run = run+1
        
    return results

def run_gpc_1D(X, y_fault, y_type, indices, cv, kernel):
    '''
    GOAL - Fit a GPC model in GPy with cross-validation and make predictions
    INPUTS - 
        X : 2 dimensional np array of input data (output of pf.get_L2_norms transformation functions)
        y_fault : np array of binary fault determination (output of transformation functions)
        y_type : np array of specific fault types (output of transformation functions)
        indices : metadata array (output of transformation functions)
        cv : sk-learn repeated k foldcross validator object
        kernel : valid GPy kernel or composition of GPy kernels
        
    OUTPUTS -
        results - df of log odds distribution for fault likelihoof for each component of each observation in each fold, along with metadata
            
    '''
    
    # set up output df
    results = pd.DataFrame()
    
    # tracker for each fold
    run = 0
    
    # log odds space
    x_logit = np.linspace(-4, 4, 200)

    # loop through repeated k folds
    for train_idx, test_idx in cv.split(X):

        # Test-train split
        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y_fault[train_idx].reshape(y_fault[train_idx].shape[0], 1)
        y_test = y_fault[test_idx].reshape(y_fault[test_idx].shape[0], 1)

        label_train = np.array(indices[0])[train_idx]
        label_test = np.array(indices[0])[test_idx]

        # identifying seen and unseen fault types
        is_types = np.unique(y_type[train_idx])
        oos_types = np.unique(y_type[test_idx])
        unseen_types = list(set(oos_types).difference(set(is_types)))
        
        # fit model
        model_gp = gp.models.GPClassification(X_train, y_train, kernel)
        model_gp.optimize()
        
        # set up dict to store results
        results_train, results_test = {}, {}
        
        # make predictions and get distribution for observations
        preds_train =  model_gp.predict(X_train)
        gp_vars_train = model_gp.predict(X_train, include_likelihood=False)

        params_train = np.concatenate((scipy.special.logit(preds_train[0]), gp_vars_train[1]), axis=1)
        posteriors_train = np.array([stats.norm(loc=params_train[ix][0], scale=params_train[ix][1]).pdf(x_logit) for ix in range(params_train.shape[0])])

        results_train['timestamp'] = label_train
        results_train['distribution'] = posteriors_train

        preds_test =  model_gp.predict(X_test)
        gp_vars_test = model_gp.predict(X_test, include_likelihood=False)

        params_test = np.concatenate((scipy.special.logit(preds_test[0]), gp_vars_test[1]), axis=1)
        posteriors_test = np.array([stats.norm(loc=params_test[ix][0], scale=params_test[ix][1]).pdf(x_logit) for ix in range(params_test.shape[0])])

        # store data and metadata
        posteriors_df_train = pd.DataFrame(posteriors_train, columns=x_logit)
        metadata_df_train = pd.DataFrame(label_train, columns=['timestamp'])
        metadata_df_train['run'] = run
        metadata_df_train['type'] = y_type[train_idx]
        metadata_df_train['actual'] = y_train.squeeze()
        metadata_df_train['seen'] = True
        metadata_df_train['group'] = 'train'
        df_train = pd.concat((metadata_df_train, posteriors_df_train), axis=1).set_index(['timestamp', 'run', 'type', 'actual', 'seen', 'group'])

        posteriors_df_test = pd.DataFrame(posteriors_test, columns=x_logit)
        metadata_df_test = pd.DataFrame(label_test, columns=['timestamp'])
        metadata_df_test['run'] = run
        metadata_df_test['type'] = y_type[test_idx]
        metadata_df_test['actual'] = y_test.squeeze()
        metadata_df_test['seen'] =  ~metadata_df_test.type.isin(unseen_types)
        metadata_df_test['group'] = 'test'
        df_test = pd.concat((metadata_df_test, posteriors_df_test), axis=1).set_index(['timestamp', 'run', 'type', 'actual', 'seen', 'group'])

        # concatenate
        results = pd.concat([results, df_train, df_test])


        run = run+1
        
    return results

def agg_by_components(gpc_results : pd.DataFrame(), col_names):
    '''
    GOAL - convert df of component-wise prob distribution to aggregate observation-wise prob distribution
    INPUTS:
        gpc_results - df of results from gpc prediction (output of run_gpc function)
        col_names - the x values for the distribution (typically in the probability space)
    OUTPUTS:
        component_means : df of observation-wise probability distribution
    
    '''
    component_means = gpc_results.groupby(['timestamp', 'run', 'type', 'actual', 'seen', 'group']).mean()
    component_means.columns = col_names
    
    return component_means

def eval_gpc(agg_df : pd.DataFrame(), col_names, x_vals, threshold):
    '''
    GOAL - calculate summary statistics to evalute model
    INPUTS:
        agg_df - df of observation-wise probability distribution (output of agg_by_component function)
        col_names - the column names for agg_df (typically in the probability space)
        x_vals - the x values for the distribution in the log odds space
        threshold - the cutoff for classification
    OUTPUTS:
        results : a dictionary with various evaluation metrics    
    '''
    
    # get metadata
    meta = agg_df.drop(columns=col_names).reset_index()
    
    # normalize probability distribution
    agg_preds = scipy.special.expit(np.divide(np.array(agg_df), np.array(agg_df.sum(axis=1)).reshape(agg_df.shape[0], 1))@x_vals)
    
    # perform calculations
    agg_results = meta.copy()
    agg_results['prob'] = agg_preds
    
    agg_results['pred0'] = agg_results.prob < threshold
    agg_results['pred1'] = agg_results.prob >= threshold
    agg_results['mse'] = (agg_results.actual - agg_results.prob)**2
    agg_results['correct'] = agg_results.pred1 == agg_results.actual
    
    agg_results_train = agg_results.query("group=='train'")
    agg_results_test = agg_results.query("group=='test'")
    agg_results_unseen = agg_results.query("seen==False")
    
    train_confusion = agg_results_train.groupby(['actual'])['pred0', 'pred1'].sum()
    test_confusion = agg_results_test.groupby(['actual'])['pred0', 'pred1'].sum()
    
    accuracies = {}
    accuracies['train'] = {'accuracy' : agg_results_train.correct.mean(),
                           'fpr' : 1 - agg_results_train.query("actual==0").correct.mean(),
                           'tpr' : agg_results_train.query("actual==1").correct.mean()
                          }
    
    accuracies['test'] = {'accuracy' : agg_results_test.correct.mean(),
                           'fpr' : 1 - agg_results_test.query("actual==0").correct.mean(),
                           'tpr' : agg_results_test.query("actual==1").correct.mean()
                          }
    
    accuracies['unseen'] = {'tpr' : agg_results_unseen.correct.mean()}
    
    train_breakdown = agg_results_train.groupby(['type'])['pred0', 'pred1'].sum()
    test_breakdown = agg_results_test.groupby(['type'])['pred0', 'pred1'].sum()
    
    # roc info
    auc_train = roc_auc_score(agg_results.query("group=='train'").actual, agg_results.query("group=='train'").prob)
    auc_test = roc_auc_score(agg_results.query("group=='test'").actual, agg_results.query("group=='test'").prob)
    
    fpr_train, tpr_train, threshold_train = roc_curve(agg_results.query("group=='train'").actual, agg_results.query("group=='train'").prob)
    fpr_test, tpr_test, threshold_test = roc_curve(agg_results.query("group=='test'").actual, agg_results.query("group=='test'").prob)

    roc_rates = [fpr_train, tpr_train, fpr_test, tpr_test]
    
    # combine and return
    results = {'train_confusion' : train_confusion,
               'test_confusion' : test_confusion,
               'accuracies' : accuracies,
               'train_breakdown' : train_breakdown,
               'test_breakdown' : test_breakdown,
               'auc_train' : auc_train,
               'auc_test' : auc_test,
               'roc_rates' : roc_rates
              }
    
    return results

def run_and_eval_gpc(X, y_fault, y_type, indices, cv, kernel, col_names, x_vals, threshold):
    '''
    GOAL - run GPC process and calculate evaluation metrics
    INPUTS - see input descriptions for run_gpc, agg_by_components, eval_gpc
    OUTPUT - dictionaty containing output of run_gpc, agg_by_components, eval_gpc for given params
    '''
    results_gpc = run_gpc(X, y_fault, y_type, indices, cv, kernel)
    component_agg = agg_by_components(results_gpc, col_names)
    results = eval_gpc(component_agg, col_names, x_vals, threshold)
    
    return {'raw' : results_gpc, 'aggregated' : component_agg, 'summary' : results}

def run_and_eval_gpc_1D(X, y_fault, y_type, indices, cv, kernel, col_names, x_vals, threshold):
    '''
    GOAL - run GPC process and calculate evaluation metrics
    INPUTS - see input descriptions for run_gpc, agg_by_components, eval_gpc
    OUTPUT - dictionaty containing output of run_gpc, agg_by_components, eval_gpc for given params
    '''
    results_gpc = run_gpc_1D(X, y_fault, y_type, indices, cv, kernel)
    results = eval_gpc(results_gpc, col_names, x_vals, threshold)
    
    return {'raw' : results_gpc, 'summary' : results}
    
def plot_agg_posterior(agg_df : pd.DataFrame(), timestamp : str, run : int, x_vals):
    '''
    GOAL - This function plots the probability distribution for one observation
    INPUTS:
        agg_df - df of observation-wise probability distribution (output of agg_by_component function)
        timestamp - timestamp of observation
        run - which fold of cross validation
        x_vals - x values of pdf (probability space)
    '''
    with_meta = agg_df.reset_index()
    obs = np.array(agg_df)[(with_meta.timestamp == timestamp) & (with_meta.run == run)].squeeze()
    #plt.plot(agg_results.columns, obs)
    plt.plot(x_vals, obs)
    plt.title(f'Distribution for {timestamp} - Run # {run}')
    plt.show()
    #return obs
    
def run_and_save_1D(X, y_fault, y_type, indices, cv, kernel, col_names, x_vals, threshold, aws_client, temp_folder, aws_folder, filename):
    
    #get results
    results_temp = run_and_eval_gpc_1D(X, y_fault, y_type, indices, cv, kernel, col_names, x_vals, threshold)
    
    # raw results
    raw_filename = filename + '_raw.csv'
    temp_file_raw = os.path.join(temp_folder, raw_filename)
    aws_file_raw = aws_folder + '/raw/' + raw_filename
    results_temp['raw'].to_csv(temp_file_raw)
    aws_client.upload_file(temp_file_raw, 'sds-capstone-jlab', aws_file_raw)
    os.remove(temp_file_raw)
    
    #summary
    summary_filename = filename + '.pickle'
    temp_file_summary = os.path.join(temp_folder, summary_filename)
    aws_file_summary = aws_folder + '/summary/' + summary_filename

    with open(temp_file_summary, 'wb') as f:
        pickle.dump(results_temp['summary'], f)

    aws_client.upload_file(temp_file_summary, 'sds-capstone-jlab', aws_file_summary)
    os.remove(temp_file_summary)
    
def run_and_save_2D(X, y_fault, y_type, indices, cv, kernel, col_names, x_vals, threshold, aws_client, temp_folder, aws_folder, filename):
    
    #get results
    results_temp = run_and_eval_gpc(X, y_fault, y_type, indices, cv, kernel, col_names, x_vals, threshold)
    
    # raw results
    raw_filename = filename + '_raw.csv'
    temp_file_raw = os.path.join(temp_folder, raw_filename)
    aws_file_raw = aws_folder + '/raw/' + raw_filename
    results_temp['raw'].to_csv(temp_file_raw)
    aws_client.upload_file(temp_file_raw, 'sds-capstone-jlab', aws_file_raw)
    os.remove(temp_file_raw)
    
    #aggregated results
    agg_filename = filename + '_agg.csv'
    temp_file_agg = os.path.join(temp_folder, agg_filename)
    aws_file_agg = aws_folder + '/aggregated/' + agg_filename
    results_temp['aggregated'].to_csv(temp_file_agg)
    aws_client.upload_file(temp_file_agg, 'sds-capstone-jlab', aws_file_agg)
    os.remove(temp_file_agg)
    
    #summary
    summary_filename = filename + '.pickle'
    temp_file_summary = os.path.join(temp_folder, summary_filename)
    aws_file_summary = aws_folder + '/summary/' + summary_filename

    with open(temp_file_summary, 'wb') as f:
        pickle.dump(results_temp['summary'], f)

    aws_client.upload_file(temp_file_summary, 'sds-capstone-jlab', aws_file_summary)
    os.remove(temp_file_summary)