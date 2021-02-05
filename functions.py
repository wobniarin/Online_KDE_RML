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
    plt.rcParams["legend.loc"] = 'upper right'
    return

# plot function for flexible, inflexible and total share of the dataport dataset
def flex_inflex_total_plot(df):
    plot_style_LaTex()
    plt.figure(figsize=(15, 10))
    plt.rcParams["legend.loc"] = 'upper center'
    plt.plot(df.index, df['total_load_kWh'], label='Total Load', color='darkgrey', linewidth=3.5)

    plt.plot(df.index, df['flex_load_kWh'], label='Flexible Load', color='rebeccapurple', linewidth=2,
             linestyle=(0, (1, 1)))
    plt.plot(df.index, df['inflex_load_kWh'], label='Inflexible Load', color='black', linewidth=2,
             linestyle='dashed')
    # Setting aesthetics of the plot
    plt.xlabel("Time period [day]", labelpad=10)
    plt.ylabel("Energy [kWh]", labelpad=10)
    plt.xlim([dt.date(2018, 8, 27), dt.date(2018, 9, 3)])  # august-sept
    plt.ylim(-3, 62)
    plt.legend(ncol=3)
    plt.savefig('./Results/hierarchical/plots_POWERTECH_2021/total_flex_inflex_FINAL.png', dpi=300)
    plt.show()
    return

# Plot for level 1 of the hierarchy
def probability_plot_season():
    # Read probability csv file results to create probability plot season
    series_2 = pd.read_csv('Results/hierarchical/prob_values.csv', index_col=0)
    series_2.index = pd.to_datetime(series_2.index)
    # Creating plots
    plot_style_LaTex()
    plt.figure(figsize=(15, 10))
    plt.rcParams["legend.loc"] = 'upper center'
    seasons = [1, 4, 7, 10]
    month_list =[]
    for month in seasons:
        month_df = series[series.index.month == month]
        month_list.append(month_df)
    # Setting aesthetics of the plot
    plt.plot(np.arange(24), month_list[0], linewidth=2, color = 'mediumslateblue')
    plt.plot(np.arange(24), month_list[1], linewidth=2, color = 'darkgrey' , linestyle='dashed')
    plt.plot(np.arange(24), month_list[2], linewidth=2, color = 'black', linestyle=(0, (1, 1)))
    plt.plot(np.arange(24), month_list[3], linewidth=2, color = 'mediumaquamarine', linestyle='dashdot')
    plt.xlabel("Time period [h]", labelpad=10)
    plt.ylabel("Probability", labelpad=10)
    plt.ylim(0, 1.1)
    plt.xticks(np.arange(24), ('00', '01', '02', '03', '04', '05', '06', '07', '08', '09',
                               '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
                               '20', '21', '22', '23'))
    plt.legend(['Winter', 'Spring', 'Summer', 'Fall'], ncol=4)
    plt.ylim(0,1.15)
    plt.savefig('./Results/hierarchical/plots_POWERTECH_2021/probability_season_final.png', dpi=250)
    plt.show()
    return

# Initial uniform distribution to start the forecast algorithm based on the dataset values
def initial_distribution(ini, fmax, samples):
    # create uniform distribution associated to that vector
    y = np.linspace(ini, fmax, samples)
    pdf_uni = uniform.pdf(y, loc=ini, scale=fmax)
    return y, pdf_uni

# Calculation of ht based on the first approach (POWERTECH2021). h_tilde is obtained based on GRID SEARCH AND KFOLD CV
def ht_calculation(h_tilde, yi):
    # ht_options = [0.5,0.6,0.75]
    # return random.choice(ht_options)
    return np.sqrt(yi)*h_tilde

# function to obtain a small dataset - for testing purposes
def smaller_df(df, until_month):
    return df[df.index.month <= until_month]

# function to calculate the histogram based on Fredis-Dicomanis approach and keeping the function in memory
def hist_calculation(df):
    hist, bins = np.histogram(df['flex_load_kWh'].values, bins='fd', density=True)
    width = 0.85 * (bins[1] - bins[0])  # width of the bin
    center = (bins[:-1] + bins[1:]) / 2  # location of the bin
    return hist, bins, width, center

# plotting the distribution at each time period desired. It also plots the histogram at that time period t.
def final_plots_KDE(hist, bins, width, center, y, ft_y):
    plt.figure(figsize=(15, 10))
    plt.bar(center, hist, align='center', width=width, color='grey',edgecolor='black',  label= 'histogram')
    # plt.plot(y, uni_pdf, color='cornflowerblue', linewidth=1.5, label='initial distribution')
    plt.plot(y, ft_y, color='gold', linewidth=3, label='resulting pdf')
    # sns.histplot(data=df, x='flex_load_kWh', stat='density', color="grey",
    #              label='histogram_seaborn')  # initial histogram of data
    plt.ylim(0, 0.5)
    plt.ylabel('Density')
    plt.xlabel('Flexibility Value [kWh]')
    # plt.title('Aggregated Flexibility Value' + str(datetime.datetime.now()))
    plt.legend()
    plt.show()
    return

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
def ht_est_RML(ht_est_memory, ddh_St, ddh2_St):
    h_t_est = ht_est_memory - ddh_St/ ddh2_St
    return h_t_est

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
def d_dh_ft(d_dh_ft_memory, y, yi, lambda_value, current_kde, ht):
    value = lambda_value * d_dh_ft_memory + (1-lambda_value) * current_kde * 1/ht * (((y-yi)**2/ht**2)-1)
    return value

# Function     - EQ 6
def d2_dh2_St(d2_dh2_St_memory, ut, lambda_value):
    value = lambda_value * d2_dh2_St_memory + (1-lambda_value) * ut**2
    return value

