# DATAPORT Austin 2018 - Functions for hierarchical modeling
# Level 1 - Model 1 (Climatology model for probability of Flex YES/NO)
# Author : Íngrid Munné-Collado
# Date: 22/03/2021

from L1_M1_climat_funct import *


# Define paths
file_path = '../data/L1_M1_encoded_data.pkl'
fig_path = '../figures/'
folder_path = '../data/'
# Read input data

with open(file_path, 'rb') as f:
    df = pickle.load(f)
    f.close()
# df_train = smaller_df(df, 8) # training split
# df_test = df[len(df_train):] # test split
# df = smaller_df(df,1)

if __name__ == "__main__":
    print("test starts here")
    prob_df = probability_monthly_hierarchical(df, folder_path) # compute the probability for each month according to
                                                                # climatology model
    monthly_df = monthly_average_prob_df(prob_df) # obtain a smaller df with the avg for each month
    monthly_prob_plot(monthly_df, fig_path)    # plot results by month
    season_prob_plot(monthly_df, fig_path)     # plot results by season
    print("test STOP")

