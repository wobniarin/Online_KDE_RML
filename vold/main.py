# Online KDE - Recursive max likelihood for parameters estimation (h_t)
# Main script
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
    h_tilde = 0.6
    lambda_value = 0.978 # at the moment we are not using CV K-fold to choose the params
    # initialize the function with the uniform distribution
    previous_vector = uni_pdf
    sim = 0
    log_score_list = []
    for timeperiod in df.index:
        yi = df.loc[df.index == timeperiod].iloc[0]['flex_load_kWh'] # that is the yi value to calculate the kde
        # Calculate log-score with the realization of the value using the previous curve
        # but not in the first element of the simulation
        if sim == 0:
            pass
        else:
            log_score_list.append(log_score_yi(yi, y, previous_vector)) # calculate log score
        sim += 1
        # TODO - Implementation of the RML in the code!
        h_t_est = ht_est_RML()
        print(h_t_est)
        current_kde = gaussian_kernel(y, yi, h_t_est)
        ft_y = recurrent_formula(lambda_value, previous_vector, current_kde)
        previous_vector = ft_y
    print(f"Plotting last KDE - histogram")
    hist, bins, width, center = hist_calculation(df) # calculate histogram of the entire df
    final_plots_KDE(hist, bins, width, center, y, ft_y) # plot last KDE against histogram of the entire df
    log_score = np.mean(log_score_list) # log-likelihood score of the entire df under study
    print("test STOP")
