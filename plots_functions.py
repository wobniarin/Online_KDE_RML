# Online KDE - Recursive max likelihood for parameters estimation (h_t)
# PLOT Functions to be imported into the main script
# 17/2/2021
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