# Online KDE - Recursive max likelihood for parameters estimation (h_t) - Using an hourly model
# Functions to be imported into the main script
# 15/3/2021
# Author : Ingrid Munne Collado

# Libraries
import numpy as np
import matplotlib.pyplot as plt
import datetime

def plot_style_LaTex():
    plt.style.use('seaborn-ticks')
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({'font.size': 20})
    # plt.rcParams.update({'font.size': 12})
    plt.rcParams["axes.grid"] = False
    plt.rcParams["legend.loc"] = 'best'
    plt.figure(figsize=[15,10])
    return

def gaussiankernel(y, yi, hy):
    kernel = 1 / (hy * np.sqrt(2 * np.pi)) * np.exp(-(1 / 2) * ((y - yi) / hy) ** 2)
    return kernel

def daterange(start_date, end_date):
    delta = datetime.timedelta(hours=1)
    while start_date < end_date:
        yield start_date
        start_date += delta

def hour_list_func():
    start_date = datetime.datetime(2018, 1, 1, 00, 00)
    end_date = datetime.datetime(2018, 1, 2, 00, 00)
    hours_list = [(single_date.strftime("%H")) for single_date in daterange(start_date, end_date)]
    # hours_list = [(single_date.strftime("%H:%M:%S")) for single_date in daterange(start_date, end_date)]
    return hours_list

# yv: vector of values that I want to calculate the online KDE
# ll: lambda value
def online_KDE(yv, ll, hours_vector):
    hours_lst = hour_list_func() # create list of 24 hours
    # initial value of hh
    hh_dict = {i:-1 for i in hours_lst} # transformed version of the bandwidth htilde
    hy_dict = {i: np.exp(hh_dict.get('00')) for i in hours_lst} # actual value of the bandwidth h_hat
    # creating vector of y where we will calculate the functions
    yy = np.arange(np.min(yv) - 0.01, np.max(yv)+0.01, 0.01) # range over which function are to be calculated and kept
    # initial values for functions and derivatives
    fy_dict = {i:np.repeat(1/(np.max(yv) - np.min(yv)), len(yy)) for i in hours_lst} # start with a uniform distribution
    dfy_dict = {i:np.repeat(0/(np.max(yv) - np.min(yv)), len(yy)) for i in hours_lst} # initial value of the derivative, 0
    hf_dict = {i:np.repeat(1/(np.max(yv) - np.min(yv)), len(yy)) for i in hours_lst}  # initial value of the hessian, uniform distribution
    hv = [[] for i in range(0,24)] # empty list to store values of hh
    hy_hat = [[] for i in range(0,24)]
    tol = 0.05 # tolerance to safe-check that the gradient does not go to 0
    log_score_lst = [[] for i in range(0,24)]
    # iterative calculation
    for i in range(len(yv)):
        print(i)
        # obtain vectors to be updated for that specific hour
        dfy = dfy_dict.get(hours_vector[i])
        fy = fy_dict.get(hours_vector[i])
        hh = hh_dict.get(hours_vector[i])
        hf = hf_dict.get(hours_vector[i])
        # first update parameters
        uf = dfy/fy # calculation of the information vector
        if np.interp(yv[i], yy, fy) < tol: # safety check to avoid problem in the tails grad towards 0
            uf = dfy/ tol
        gf = -(1-ll) * uf # calculation of gradient S
        gfy = np.interp(yv[i], yy, gf) # getting the value of grad S at yi
        hfy = np.interp(yv[i], yy, hf) # getting the value of hessian S hf at yi
        # implementing warm-start
        if i > 50: #implement warm start for each hour. it should be 50*24 = 1200, we start with 50
            # update hh
            hh = hh - gfy/hfy
            hh_dict.update({hours_vector[i]:hh})
            # compute log-likelihood score before updating the fy formula
            log_score_lst[int(hours_vector[i])].append(log_score_yi(yv, yy, fy, i))
            print(hh)
        else:
            pass
        # store value of hh
        hv[int(hours_vector[i])].append(hh)
        hy = np.exp(hh) # calculating actual value of h (h_hat)
        hy_hat[int(hours_vector[i])].append(hy)
        # update functions using recursive formulas
        gaussian = gaussiankernel(yy, yv[i], hy) # remember to remove this line, just for plotting now
        fy = ll * fy + (1 - ll) * gaussiankernel(yy, yv[i], hy) # update density
        dfy = ll * dfy + (1-ll)*((((yy-yv[i])**2)/(hy**2)) -1)*1/hy*gaussiankernel(yy, yv[i], hy) # update derivative
        uf = dfy/fy # update information vector
        hf = ll * hf + (1-ll) * uf**2 # update hessian function
        # update values in dict for next calculation
        fy_dict.update({hours_vector[i]: fy})
        dfy_dict.update({hours_vector[i]: dfy})
        hf_dict.update({hours_vector[i]: hf})
        a = 3
    return yy, fy_dict, hf_dict, dfy_dict, uf, hv, log_score_lst

# log-likelihood score calculation at time t yi.
def log_score_yi(yv, yy, fy, i):
    score = - np.log(np.interp(yv[i], yy, fy))
    return score

def smaller_df(df, until_month):
    return df[df.index.month <= until_month]