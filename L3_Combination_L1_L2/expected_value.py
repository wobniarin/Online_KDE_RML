# Online RML - KDE for flexibility forecast
# Level 3 - combination of previous levels for calculating the expected flexibility value
# Author : Íngrid Munné-Collado
# Date: 25/03/2021

#TODO import libraries
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot as plt
import datetime as dt

def RMSE(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))

def plot_style_LaTex():
    plt.style.use('seaborn-ticks')
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({'font.size': 20})
    # plt.rcParams.update({'font.size': 12})
    plt.rcParams["axes.grid"] = False
    plt.rcParams["legend.loc"] = 'best'
    plt.figure(figsize=[15,10])

# Define paths
flex_value_raw_file = '../data/aggregated_2018_austin_flex_load_hourly.pkl'
prob_level_1_file = '../data/probability_results_20210322_1848.pkl'
exp_value_level_2_file = '../data/L2_Exp_value_20210325_1511.pkl'
fig_path = '../figures/'
folder_path = '../results/'
# read files
with open(flex_value_raw_file, 'rb') as f:
    flex_value_raw_df = pickle.load(f)
    f.close()
with open(prob_level_1_file, 'rb') as f:
    prob_level_1_df = pickle.load(f)
    f.close()
with open(exp_value_level_2_file, 'rb') as f:
    exp_val_level_2_df = pickle.load(f)
    f.close()
# create df
results_df = pd.concat([prob_level_1_df, exp_val_level_2_df, flex_value_raw_df], axis=1)
# fill in missing values where flex is not there with 0.
results_df.fillna(0, inplace=True)

# compute exp_flex_value_L1L2
results_df['exp_val_flex_L1L2'] = results_df['flexibility_0-1'] * results_df['exp_value_L2']
results_df = results_df.reindex(columns=['flexibility_0-1', 'exp_value_L2', 'exp_val_flex_L1L2', 'flex_load_kWh' ])

# compute score
RMSE_score = RMSE(results_df['exp_val_flex_L1L2'], results_df['flex_load_kWh'])