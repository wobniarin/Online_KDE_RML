# Online Recursive Maximum Likelihood - HOURLY MODEL
# Author - Íngrid Munné Collado
# Date - 15/03/2021
# Libraries
from Online_RML_KDE_M2_functions import *
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Define paths
file_path = '../data/level_2_input_data.pkl'
fig_path = '../figures/'
folder_path = '../results/'
# Read input data
with open(file_path, 'rb') as f:
    series = pickle.load(f)
    f.close()
df = series.to_frame()  # Convert series into df
# df_train = smaller_df(df, 8) # training split
# df = smaller_df(df,6)
# df_test = df[len(df_train):] # test split
# define lists to store the values of lambda and log-score
lambda_values = []
log_score_values = []
# grid search for lambda
# lambda_grid = np.linspace(0.96, 1, 10)
lambda_grid = [0.96] # optimal value found in training

if __name__ == "__main__":
    print("test starts here")
    for lambda_value in lambda_grid:
        yy, fy, hf, dfy, uf, h_val, log_score_lst = online_KDE(np.array(df['flex_load_kWh']), lambda_value,
                                                               df.index.strftime("%H"))
        a=3
        log_score_values.append([np.mean(i) for i in log_score_lst ])
        lambda_values.append(lambda_value)
    print("test STOP")