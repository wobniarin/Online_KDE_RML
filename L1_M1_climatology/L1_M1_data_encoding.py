# DATAPORT Austin 2018 - Functions for hierarchical modeling
# Level 1 - Model 1 (Climatology model for probability of Flex YES/NO)
# Script for data encoding YES/NO
# Author : Íngrid Munné-Collado
# Date: 22/03/2021

from L1_M1_climat_funct import *
import pickle

# Define paths
file_path = '../data/aggregated_2018_austin_corrected.pkl'
fig_path = '../figures/'
folder_path = '../results/'
# Read input data
df = load_dataset(file_path, granularity = '1H')
column = ['flex_load_kWh']
threshold = 0.2
encoded_df = encode_threshold(df, column, threshold)
encoded_df.to_csv('../data/L1_M1_encoded_data.csv')
with open('../data/L1_M1_encoded_data.pkl', 'wb') as f:
    pickle.dump(encoded_df, f)
    f.close()