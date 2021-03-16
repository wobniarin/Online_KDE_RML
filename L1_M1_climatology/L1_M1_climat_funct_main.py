# DATAPORT Austin 2018 - Functions for hierarchical modeling
# Level 1 - Model 1 (Climatology model for probability of Flex YES/NO)
# Author : Íngrid Munné-Collado
# Date: 16/03/2021

import pandas as pd
import matplotlib
# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import seaborn as sns
import datetime as dt
import functions_climat_monthly as clim
import calendar

def encode_threshold(df, column, threshold):
    df_copy = df.copy(deep=True)
    series = df_copy[column]
    series_encoded = series
    series_encoded[series_encoded > threshold] = 1
    series_encoded[series_encoded <= threshold] = 0
    return series_encoded

def hist_encoded(series_encoded, path):
    plt.figure(figsize=(10,7))
    plt.hist(series_encoded, color ='slategray')
    plt.title('Encoded flexible load - Histogram')
    plt.savefig(path + 'encoded_flex_hist.png', dpi=200)
    plt.show()

def remove_0_encoded(df, column, threshold):
    series = df[column].copy(deep=True)
    series.drop(series[series <= threshold].index, inplace=True)
    return series

def hist_0_removed(series, path):
    plt.figure(figsize=(10, 7))
    plt.hist(series, bins=50, color='slategray')
    plt.title('Level 2 flexible load - Histogram')
    # plt.savefig(path + 'removed_0_flex_hist.png', dpi=200)
    plt.show()

# Defining plot style
def plot_style_LaTex():
    plt.style.use('seaborn-ticks')
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({'font.size': 12})
    plt.rcParams["axes.grid"] = False
    plt.rcParams["legend.loc"] = 'upper right'

def plot_timeseries_prob(path):
    plt.figure(figsize=(10, 7))
    ht = [24, 48, 168, 336, 744, 2160, 4380, 6570, 8760]
    p = [0.54167, 27/48, 94/168, 179/336, 410/744, 1144/2160, 2408/4344, 3712/6552, 0.56495]
    plt.plot(ht, p, color='tomato', marker="o")
    # Setting aesthetics of the plot
    plt.title("Probability of Flexibility vs time horizon")
    plt.xlabel("ht time horizon [h]")
    plt.ylabel("Probability")
    plt.xticks(ht, ('1D', '2D', '1W', '2W', '1M', '3M', '6M', '9M', '12M'))
    plt.ylim(0, 1)
    plt.savefig(path + 'probability_ht_p.png', dpi=200)
    plt.show()

def plot_timeseries(df, path):
    """ Function to plot the forecast results for total, flex and inflex"""
    # TODO: Include a function parameter to change the timeperiod to plot?
    plt.figure(figsize=(10, 7))
    plt.plot(df.index, df['total_load_kWh'], label='Total Load')
    plt.plot(df.index, df['flex_load_kWh'], label='Flexible Load')
    plt.plot(df.index, df['inflex_load_kWh'], label='Inflexible Load')
    # Setting aesthetics of the plot
    plt.title("Time Series plot")
    plt.xlabel("Time period [min]")
    plt.ylabel("Energy [kWh]")
    plt.xlim([dt.date(2018, 8, 27), dt.date(2018, 9, 8)])  # august-sept
    plt.legend()
    plt.savefig(path + '_TS_flex_inflex_total.png', dpi=200)
    # plt.close('all')
    plt.show()

def histogram_plot_flex_initial(df,path):
    plt.figure(figsize=(10, 7))
    plt.hist(df['flex_load_kWh'], bins=40, color="slategray")
    # Setting aesthetics of the plot
    plt.title("Flexible load - Histogram")
    plt.savefig(path + 'flex_load_initial_hist.png', dpi=200)
    plt.show()


def probability_monthly_hierarchical(series_encoded, folder_path):
    print("Calculating monthly probabilities")
    forecast_df = pd.DataFrame(columns=['flexibility_0-1'])
    # iterate over test set to see the timeperiods to forecast
    for month in range(1, 13):
        print(f"Analyzing month number {month}")
        month_df = series_encoded[series_encoded.index.month == month]
        for timeperiod in month_df.index:
            # extract hour, minute, second to predict
            print("Extracting hour")
            # check for previous values in train set that match that timeperiod
            # compute mean for all columns under that match. Store in forecast_df
            print(f"Computing forecast for {timeperiod}")
            forecast_df = forecast_df.append(month_df.loc[dt.time(hour=timeperiod.hour, minute=timeperiod.minute,
                                                                  second=timeperiod.second)].mean(axis=0),
                                             ignore_index=True)
            forecast_df.at[forecast_df.index[-1], 'time'] = timeperiod
    # set time as index, same structure as previous df
    forecast_df.set_index('time', inplace=True)
    # defining path to store results
    path = folder_path + 'probability_results_' + str(dt.datetime.now().strftime("%Y%m%d_%H%M"))
    # path = '../test_functions/test'
    # saving results into csv
    forecast_df.to_csv(path+'.csv')
    print("forecast results csv file saved")
    # Saving results into pkl file
    with open(path+'.pkl', 'wb') as f:
        pickle.dump(forecast_df, f)
        f.close()
    # return forecast df
    print("forecast results pkl file saved")
    return forecast_df


def monthly_prob_plot(series, path):
    plt.rcParams["legend.loc"] = 'best'
    plt.figure(figsize=(10, 7))
    for month in range(1,13):
        month_df = series[series.index.month == month]
        plt.plot(np.arange(24), month_df, label=calendar.month_name[month])
    # Setting aesthetics of the plot
    plt.title("Flexibility probability vs time horizon")
    plt.xlabel("ht time  [h]")
    plt.ylabel("Probability")
    plt.ylim(0, 1.1)
    # plt.xticks(np.arange(24), ('00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00',
    #                            '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00',
    #                            '20:00', '21:00', '22:00', '23:00'))
    plt.xticks(np.arange(24), ('00', '01', '02', '03', '04', '05', '06', '07', '08', '09',
                               '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
                               '20', '21', '22', '23'))
    plt.legend(ncol=3)
    plt.savefig(path + 'probability_month_p.png', dpi=200)
    plt.show()

def season_prob_plot(series, path):
    plt.rcParams["legend.loc"] = 'best'
    plt.figure(figsize=(10, 7))
    labels = ['winter', 'spring', 'summer', 'fall']
    seasons = [1,4,7,10]
    for month in seasons:
        month_df = series[series.index.month == month]
        plt.plot(np.arange(24), month_df)
    # Setting aesthetics of the plot
    plt.title("Flexibility probability vs time horizon")
    plt.xlabel("ht time  [h]")
    plt.ylabel("Probability")
    plt.ylim(0, 1.1)
    # plt.xticks(np.arange(24), ('00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00',
    #                            '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00',
    #                            '20:00', '21:00', '22:00', '23:00'))
    plt.xticks(np.arange(24), ('00', '01', '02', '03', '04', '05', '06', '07', '08', '09',
                               '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
                               '20', '21', '22', '23'))
    plt.legend(['winter', 'spring', 'summer', 'fall'],ncol=1)
    plt.savefig(path + 'probability_season_p.png', dpi=200)
    plt.show()


    # TODO: Create a main to check that this code works and obtain the results