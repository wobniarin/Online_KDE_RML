# Online KDE - Recursive max likelihood for parameters estimation (h_t)
# Functions to be imported into the main script
# 2/2/2021
# Author : Ingrid Munne Collado

# Libraries
import numpy as np
from scipy.stats import uniform, norm
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import mean_squared_error
import seaborn as sns
import datetime
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# defining plot style
def plot_style_LaTex():
    plt.style.use('seaborn-ticks')
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({'font.size': 20})
    # plt.rcParams.update({'font.size': 12})
    plt.rcParams["axes.grid"] = False
    plt.rcParams["legend.loc"] = 'best'
    return

# Initial uniform distribution to start the forecast algorithm based on the dataset values
def initial_distribution(ini, fmax, samples):
    # create uniform distribution associated to that vector
    y = np.linspace(ini, fmax, samples)
    pdf_uni = uniform.pdf(y, loc=ini, scale=fmax)
    return y, pdf_uni

# function to obtain a small dataset - for testing purposes
def smaller_df(df, until_month):
    return df[df.index.month <= until_month]

# function to calculate the histogram based on Fredis-Dicomanis approach and keeping the function in memory
def hist_calculation(df):
    hist, bins = np.histogram(df['flex_load_kWh'].values, bins='fd', density=True)
    width = 0.85 * (bins[1] - bins[0])  # width of the bin
    center = (bins[:-1] + bins[1:]) / 2  # location of the bin
    return hist, bins, width, center

# RMSE score calculation
def RMSE(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

# Once the final distribution ft_y has been calculated, some values are taken (CENTER) to calculate the RMSE score
# against the true distribution (histogram) at the end of the test set. We need the center of the histogram bars
# (points of x to calculate ft_y) and then we have the y_pred vector and we can calculate the RMSE score.
def vector_ypredict_hist(center, ft_y):
    lst = []
    step = round(len(ft_y) / len(center))
    for i in range(0,len(ft_y), step):
       lst.append(ft_y[i])
    return np.array(lst)

# log-likelihood score calculation at time t yi.
def log_score_yi(yi, y, ft_y):
    return - np.log(np.interp(yi, y, ft_y))

#################### Equations for RML parameter estimation
# Function to calculate h_t parameter based on Recursive Maximum Likelihood - EQ 1
def ht_est_RML(ht_est_memory, ddh_St, ddh2_St, it_counter, current_kde):
    if it_counter <= 30: # implementing warm start
        return ht_est_memory
    else:
        index = np.where(current_kde == np.max(current_kde))
        ht_est = np.log(ht_est_memory) - ddh_St[index] / ddh2_St[index]
        # ht_est = ht_est_memory - ddh_St[index] / ddh2_St[index]
        ht_est = float(ht_est)
        ht_value = np.exp(ht_est)
        return ht_value

def ht_est_value(current_kde, ht_est):
    index = np.where(current_kde == np.max(current_kde))
    return index, float(ht_est[index])

# Function to calculate the first derivative of St(h_est) - EQ 2
def d_dh_St(Ut, lambda_value):
    return (lambda_value - 1) * Ut

# Function to calculate the information vector based on ft and d/dh ft - EQ 3
def Ut(d_dh_ft_yi, ft_y):
    u_t = d_dh_ft_yi / ft_y
    return u_t

# Function  to calculate the current distribution based on previous value and kernel - EQ 4
def ft_yi(lambda_value, ft_memory, current_kde):
    ft_y = lambda_value * ft_memory + (1-lambda_value) * current_kde
    return ft_y

# Gaussian kernel function given y, yi and h_t - FOLLOW-UP ON equation 4
def gaussian_kernel(y, yi, h_t):
    # y = np.linspace(0,1,100)
    # yi = np.full(len(y), y[i])
    kernel = 1/(h_t*np.sqrt(2*np.pi)) * np.exp(-(1/2) * ((y-yi)/h_t)**2)
    return kernel

# Function d_dh ft_yi based on recursive formula - EQ 5
def d_dh_ft(d_dh_ft_memory, y, yi, lambda_value, current_kde, ht, index):
    value = lambda_value * d_dh_ft_memory + (1-lambda_value) * current_kde * (1/ht) * ((y-yi)**2/(ht**2)-1)
    return value

# Function  HESSIAN of St - EQ 6
def d2_dh2_St(d2_dh2_St_memory, ut, lambda_value):
    value = lambda_value * d2_dh2_St_memory + (1-lambda_value) * ut**2
    return value


