import datetime
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from pyts.approximation import SymbolicFourierApproximation
from pyts.datasets import load_gunpoint
from pyts.bag_of_words import BagOfWords

from pyts.bag_of_words import BagOfWords
from pyts.datasets import load_gunpoint
from pyts.bag_of_words import WordExtractor
from pyts.classification import BOSSVS

import numpy as np
import matplotlib.pyplot as plt
from pyts.datasets import load_gunpoint
from pyts.transformation import BOSS

class Pulse:
    '''
    This class deals with pulse run data and metadata.
    '''
    
    def __init__(self, data, time, fault_type, interval, result):
        self.data = data
        self.time = time
        self.fault_type = fault_type
        self.interval = interval
        self.result = result
        
    def set_pulses(self, pulse_start=829, pulse_length=6100, pulse_gap=41670):
        '''
        Separates the data into 3 pulses by index
        '''
        self.pulse1 = self.data[pulse_start : pulse_start + pulse_length].reset_index(drop=True)
        self.pulse2 = self.data[pulse_start + pulse_gap : pulse_start + pulse_gap + pulse_length].reset_index(drop=True)
        self.pulse3 = self.data[pulse_start + 2*pulse_gap : pulse_start + 2*pulse_gap + pulse_length].reset_index(drop=True)


    def normalize(self, pulse_maxs):
        '''
        INPUTS: self, pulse_maxs (pd series with max value of each component for normalization use - output of get_normals function)
        '''
        self.pulse1norm = self.pulse1/pulse_maxs
        self.pulse2norm = self.pulse2/pulse_maxs
        self.pulse3norm = self.pulse2/pulse_maxs

    def graph_all_pulses(self):
        '''
        Plot all columns - taken from Mackenzye's code
        '''
        columns = self.data.columns

        fig, ax = plt.subplots(8, 4, figsize = (20,20))

        for x in range(len(columns)-1):
            column = x + 1 
            plot_x_index = x % 8
            plot_y_index = x // 8
            title = columns[column]
            SupTitle = 'Plots for: ' + str(self.time)
            ax[plot_x_index, plot_y_index].plot(self.pulse1[columns[column]], label = "Pulse 1")
            ax[plot_x_index, plot_y_index].plot(self.pulse2[columns[column]], label = "Pulse 2")
            ax[plot_x_index, plot_y_index].plot(self.pulse3[columns[column]], label = "Pulse 3")
            ax[plot_x_index, plot_y_index].legend()
            ax[plot_x_index, plot_y_index].set_axis_off()
            ax[plot_x_index, plot_y_index].set_title(title)
            fig.suptitle(SupTitle, fontsize=16)
        
    def graph_component_comparison(self, comparison, cols=None, normed=True):
        '''
        Plot all components with a comparison to the median
        '''

        if normed:
            data = self.pulse1norm
        else:
            data = self.pulse1

        if cols:
            pass
        else:
            cols = data.columns

        pulse1_melted = data[cols].melt().rename(columns={'value' : 'pulse'})
        comparison_melted = comparison[cols].melt().rename(columns={'value' : 'comparison'})

        combined = pulse1_melted.assign(comparison = comparison_melted.comparison)


        g = sns.FacetGrid(combined, col='variable', height=6, col_wrap=3, aspect=1)
        g.map(plt.plot, 'comparison', color='red', label='expected', alpha=0.8)
        g.map(plt.plot, 'pulse', color='blue', label='actual', alpha=0.8)
        g.set_titles('{col_name}', fontsize=48)
        g.add_legend()