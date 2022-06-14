# -*- coding: utf-8 -*-
"""
CREATING PULSE OBJECTS AND SAVING TO S3

Created on Thu Nov 11 15:00:28 2021

@author: tgule
"""

# Import Necessary Packages

import boto3
import pandas as pd
import numpy as np
from scipy import stats
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
import random
from pulse import Pulse
from pulse_functions import get_metadata, get_median_pulse, get_normals
warnings.filterwarnings('ignore')

# set working directory to location of .env file
env_path = 'C:\\Users\\tgule\\OneDrive\\Documents\\UVA Data Science Masters\\Capstone'
original_path = os.getcwd()
os.chdir(env_path)

# Load .env and save variables

dotenv.load_dotenv()
access_key_id=os.getenv('access_key_id')
access_key_secret=os.getenv('access_key_secret')

# change back working directory to notebook location
os.chdir(original_path)

# Connect to S3 and import metadata

s3 = boto3.resource(
    service_name='s3',
    region_name='us-east-1',
    aws_access_key_id=access_key_id,
    aws_secret_access_key=access_key_secret
)

# Load data and store in dict of pulse objects

pulses = {}

# column names

column_names = ['A+ IGBT I', 'C Phase DV', 'A+* IGBT I', 'Spare 2', 'B+ IGBT I',
       'Spare 3', 'B+* IGBT I', 'Spare 4', 'C+ IGBT I', '+ CB I', 'C+* IGBT I',
       '- CB I', '+ CB V', '- CB V', 'MOD I', 'DTL MPS 1 I', 'AC L1 I',
       'A FLUX', 'AC L1 V', 'DTL1 Kly', 'AC L2 I', 'B FLUX', 'AC L2 V',
       'DTL2 Kly', 'AC L3 I', 'C FLUX', 'AC L3 V', 'Gate In', 'MOD V', 'DV/DT',
       '+ DC I', '- DC I']

for o in s3.Bucket('sds-capstone-jlab').objects.filter(Prefix='RFQ_UVA_Capstone/RFQ_UVA_Capstone'):
           
    # runs
    if o.key.endswith('.csv') and ('Run' in o.key) and ('4.1.2020' not in o.key):
        results = get_metadata(o) # metadata from filename
        results['data'] = pd.read_csv(o.get()['Body'], skiprows=2).drop(columns = 'Signal Name') # data
        results['data'].columns = column_names # consistent column names
        pulses[results['str_time']] = Pulse(results['data'], results['str_time'], results.get('fault_type'), results['interval'], results['result']) # create pulse object
        
    # faults
    elif o.key.endswith('.csv') and ('Fault' in o.key):
        results = get_metadata(o) # metadata from filename
        
        # parse fault type from first row
        first_row = pd.read_csv(o.get()['Body'], nrows=1) # get first row 
        fault_type = first_row.columns[3].split(' ', maxsplit=1)[1].rstrip() # fault type
        results['fault_type'] = fault_type
        
        # data
        results['data'] = pd.read_csv(o.get()['Body'], skiprows=2).drop(columns = 'Signal Name')
        results['data'].columns = column_names # consistent column names
        
        pulses[results['str_time']] = Pulse(results['data'], results['str_time'], results['fault_type'], results['interval'], results['result']) # create pulse object
        

# set pulses

_ = [v.set_pulses() for k, v in pulses.items()]

# set normal pulses

comp_maxs = get_normals(pulses) # get max values for normalization

_ = [v.normalize(comp_maxs) for k, v in pulses.items()]

# split pulses into two files for ease of saving
pulse1 = dict(list(pulses.items())[len(pulese)//2:])
pulse2 = dict(list(train.items())[:len(train)//2])

# write to pickle files
with open('..\\data-cleaned\\train1.pickle', 'wb') as f:
    pickle.dump(train1, f)
    
with open('..\\data-cleaned\\train2.pickle', 'wb') as f:
    pickle.dump(train2, f)cd 
    
with open('..\\data-cleaned\\test.pickle', 'wb') as f:
    pickle.dump(test, f)

# transfer to AWS
s3.meta.client.upload_file('train1.pickle', 'sds-capstone-jlab', 'consolidated-data/train1.pickle')
s3.meta.client.upload_file('train2.pickle', 'sds-capstone-jlab', 'consolidated-data/train2.pickle')
s3.meta.client.upload_file('test.pickle', 'sds-capstone-jlab', 'consolidated-data/test.pickle')