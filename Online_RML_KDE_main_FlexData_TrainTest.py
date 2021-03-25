# Online Recursive Maximum Likelihood
# Author - Íngrid Munné Collado
# Date - 11/03/2021
# Libraries
from Online_RML_KDE_functions import *
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import pandas as pd
import datetime as dt


# starting time
start = time.time()
# Define paths
file_path = './data/level_2_input_data.pkl'
fig_path = './figures/'
folder_path = './results/'
# Read input data
with open(file_path, 'rb') as f:
    series = pickle.load(f)
    f.close()
df = series.to_frame()  # Convert series into df
df_train = smaller_df(df, 8) # training split
df_test = df[len(df_train):] # test split
# df = smaller_df(df,1)
# define lists to store the values of lambda and log-score
lambda_values = []
log_score_values = []
# grid search for lambda

# lambda_grid = np.linspace(0.96, 1, 800)
lambda_grid = [0.9973147684605758] # optimal value found in training

if __name__ == "__main__":
    print("test starts here")
    for lambda_value in lambda_grid:
        yy, fy, hf, dfy, uf, h_val, log_score_lst, exp_val_lst = online_KDE(np.array(df['flex_load_kWh']), lambda_value)
        log_score_values.append(np.mean(log_score_lst))
        lambda_values.append(lambda_value)
    # end time
    end = time.time()
    # total time taken
    print(f"Runtime of the program is {end - start}")
    exp_val_L2 = pd.DataFrame(exp_val_lst, index=df.index, columns=['exp_value_L2'])
    L2_path = './data/L2_Exp_value_'+ str(dt.datetime.now().strftime("%Y%m%d_%H%M"))
    exp_val_L2.to_csv(L2_path + '.csv')
    with open(L2_path+'.pkl', 'wb') as f:
        pickle.dump(exp_val_L2, f)
        f.close()
    print("test STOP")