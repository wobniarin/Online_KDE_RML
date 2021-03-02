# Online Recursive Maximum Likelihood
# Author - Íngrid Munné Collado
# Date - 2/03/2021
# Libraries
from Online_RML_KDE_functions import *
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Define paths
file_path = './data/level_2_input_data.pkl'
fig_path = './figures/'
folder_path = './results/'
# Read input data
with open(file_path, 'rb') as f:
    series = pickle.load(f)
    f.close()
df = series.to_frame()  # Convert series into df
# df = smaller_df(df, 6)

if __name__ == "__main__":
    print("test starts here")
    # read input data - Flexibility Data level 1
    yy, fy, hf, dfy, uf, h_val, log_score_lst = online_KDE(np.array(df['flex_load_kWh']), 0.999)
    log_score = np.mean(log_score_lst)
    print("test STOP")