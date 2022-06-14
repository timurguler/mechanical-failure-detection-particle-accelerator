"""
THIS NOTEBOOK RUNS GPC AND COLLECTS RESULTS USING THE L2 NORM FOR EACH COMPONENT AS THE INPUT SPACE AND FAULT LIKELIHOOD AS THE OUTPUT SPACE.

This process relies on the GPy package for Gaussian Process Classification, and saves model results in s3.
"""

##########
# STEP 1 - IMPORT PACKAGES
##########

import pandas as pd
import numpy as np
import boto3
import scipy
from scipy import stats
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import dotenv
import sys
import datetime
import pickle
import io
import warnings
from pulse import Pulse
import pulse_functions as pf
warnings.filterwarnings('ignore')

from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, auc

from sklearn.cluster import AgglomerativeClustering

from sklearn.preprocessing import OneHotEncoder

import GPy as gp

##########
# STEP 2 - LOAD DATA FROM S3
##########

# set working directory to location of .env file
env_path = 'C:\\Users\\Administrator\\Documents\\capstone'
original_path = os.getcwd()
os.chdir(env_path)

# Load .env and save variables

dotenv.load_dotenv()
access_key_id=os.getenv('access_key_id')
access_key_secret=os.getenv('access_key_secret')

# change back working directory to notebook location
os.chdir(original_path)

# Connect to S3 and import metadata

s3 = boto3.client(
    service_name='s3',
    region_name='us-east-1',
    aws_access_key_id=access_key_id,
    aws_secret_access_key=access_key_secret
)

# load and consolidate into 1 dict
pulse1 = s3.get_object(Bucket='sds-capstone-jlab', Key='consolidated-data/pulses1.pickle')['Body'].read()
pulse2 = s3.get_object(Bucket='sds-capstone-jlab', Key='consolidated-data/pulses2.pickle')['Body'].read()
pulse3 = s3.get_object(Bucket='sds-capstone-jlab', Key='consolidated-data/pulses3.pickle')['Body'].read()

data = {**pickle.loads(pulse1), **pickle.loads(pulse2), **pickle.loads(pulse3)}

##########
# STEP 3 - CALCULATE NORM VALUES
##########

# subset of components
useful_cols = ['A+ IGBT I', 'A+* IGBT I', 'B+ IGBT I',
'B+* IGBT I', 'C+ IGBT I', '+ CB I', 'C+* IGBT I',
'- CB I', '+ CB V', '- CB V', 'MOD I', 'DTL MPS 1 I',
'A FLUX', 'DTL1 Kly', 'B FLUX',
'DTL2 Kly', 'C FLUX', 'MOD V', 'DV/DT']

# run function to calc norm values
X_fourier, y_fault, y_type, indices_fourier = pf.get_fourier_values(data, cols=useful_cols, reshape=10)

##########
# STEP 4 - Set up Kernels, PDF eval points, and CV
##########

cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=2022)

x_logit = np.linspace(-4, 4, 200)
x_prob = scipy.special.expit(x_logit)

kernels = {
    'rbf' : gp.kern.RBF(input_dim=2, variance=1, lengthscale=1),
    'exponential' : gp.kern.Exponential(input_dim=2, variance=1, lengthscale=1),
    'matern' : gp.kern.Matern52(input_dim=2, variance=1),
    'rat_quad' : gp.kern.RatQuad(input_dim=2, variance=1)
}

##########
# STEP 5 - RUN MODEL AND SAVE RESULTS
##########

for k, v in kernels.items():
    file_name = 'fourier_' + k + '_5fold'
    pf.run_and_save_2D(X_fourier, y_fault, y_type, indices_fourier, cv, v, 
             x_prob, x_logit, 0.3, s3, '..\\temp', 'model-results/gpc', file_name)

