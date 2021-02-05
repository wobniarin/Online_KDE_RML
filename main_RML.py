# Online KDE - Recursive max likelihood for parameters estimation (h_t)
# Main script for testing functions
# 5/2/2021
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
from functions import *


if __name__ == "__main__":
    print("test starts here")
    plot_style_LaTex()
    # Define paths
    file_path = './data/level_2_input_data.pkl'
    fig_path = './figures/'
    folder_path = './results/'
    # Read input data
    with open(file_path, 'rb') as f:
        series = pickle.load(f)
        f.close()
    df = series.to_frame()  # Convert series into df
    df = smaller_df(df, 1)
    # param values for uniform distribution
    ini = 0
    fmax = 20
    samples = df.shape[0]
    # create initial distribution
    y, uni_pdf = initial_distribution(ini, fmax, samples)
    # Hyper-parameters to tune
    lambda_value = 0.978 # at the moment we are not using CV K-fold to choose the params
    # initialize the function with the uniform distribution
    ft_memory = uni_pdf
    # initial values for memory_vectors
    ht_est_memory = 0.4
    d_dh_ft_memory = 1
    d2_dh2_St_memory = 1
    # counters
    for timeperiod in df.index:
        yi = df.loc[df.index == timeperiod].iloc[0]['flex_load_kWh'] # that is the yi value to calculate the kde
        # TODO - Implementation of the RML in the code!
        # Based on the memory value, I calculate the new kernel based on the new value but the previous ht_est
        # EQ-4
        current_kde = gaussian_kernel(y, yi, ht_est_memory)
        ft_y = ft_yi(lambda_value, ft_memory, current_kde)
        # EQ-5
        d_dh_ft_yi = d_dh_ft(d_dh_ft_memory, y, yi, lambda_value, current_kde, ht_est_memory)
        # EQ-3 information vector
        ut = Ut(d_dh_ft_yi, ft_y)
        # EQ 2
        ddh_St = d_dh_St(ut, lambda_value)
        # EQ 6
        ddh2_St = d2_dh2_St(d2_dh2_St_memory, ut, lambda_value)
        # compute ht_est
        ht_est = ht_est_RML(ht_est_memory, ddh_St, ddh2_St) # EQ-1
        # Take one value from all the possible estimated value
        # TODO - I should check actually which value to take from all the possible values calculated in that vector
        ht_est = ht_est[-1]
        print(ht_est)
        # Once I know the value of ht_est for this current iteration, I calculate the new kernel based on the new value
        # TODO - Check if this is actually correct. I don't know if I should calculate twice this equation.
        current_kde = gaussian_kernel(y, yi, ht_est)
        ft_y = ft_yi(lambda_value, ft_memory, current_kde)
        # keep values in lists for checking plots later
        ht_est_lst.append(ht_est)
        # update memory and run another iteration
        ht_est_memory = ht_est
        d_dh_ft_memory = d_dh_ft_yi
        d2_dh2_St_memory = ddh2_St

    print("test STOP")
