# Online KDE - Recursive max likelihood for parameters estimation (h_t)
# Functions to be imported into the main script
# 22/2/2021
# Author : Ingrid Munne Collado

# Libraries
import numpy as np
import matplotlib.pyplot as plt
from numpy import trapz



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

# yv: vector of values that I want to calculate the online KDE
# ll: lambda value
def online_KDE(yv, ll):
    # initial value of hh
    hh = -1 # transformed version of the bandwidth htilde
    hy = np.exp(hh) # actual value of the bandwidth h_hat
    # creating vector of y where we will calculate the functions
    yy = np.arange(np.min(yv) - 0.01, np.max(yv)+0.01, 0.01) # range over which function are to be calculated and kept
    # initial values for functions and derivatives
    fy = np.repeat(1/(np.max(yv) - np.min(yv)), len(yy)) # start with a uniform distribution
    dfy = np.repeat(0/(np.max(yv) - np.min(yv)), len(yy)) # initial value of the derivative, 0
    hf = np.repeat(1/(np.max(yv) - np.min(yv)), len(yy))  # initial value of the hessian, uniform distribution
    hv = [] # empty list to store values of hh
    hy_hat = []
    tol = 0.05 # tolerance to safe-check that the gradient does not go to 0
    log_score_lst = []
    exp_val_lst = []
    # iterative calculation
    for i in range(len(yv)):
        # first update parameters
        fy_old = fy  #remember to remove this line. this is just for plotting now
        uf = dfy/fy # calculation of the information vector
        if np.interp(yv[i], yy, fy) < tol: # safety check to avoid problem in the tails grad towards 0
            uf = dfy / tol
        gf = -(1-ll) * uf # calculation of gradient S
        gfy = np.interp(yv[i], yy, gf) # getting the value of grad S at yi
        hfy = np.interp(yv[i], yy, hf) # getting the value of hessian S hf at yi
        # implementing warm-start
        if i > 50:
            # update hh
            hh = hh - gfy/hfy
            # compute log-likelihood score before updating the fy formula
            log_score_lst.append(log_score_yi(yv, yy, fy, i))
            print(hh)
        else:
            pass
        # store value of hh
        hv.append(hh)
        hy = np.exp(hh) # calculating actual value of h (h_hat)
        hy_hat.append(hy)
        # update functions using recursive formulas
        gaussian = gaussiankernel(yy, yv[i], hy) # remember to remove this line, just for plotting now
        fy = ll * fy + (1 - ll) * gaussiankernel(yy, yv[i], hy) # update density
        dfy = ll * dfy + (1-ll)*((((yy-yv[i])**2)/(hy**2)) -1)*1/hy*gaussiankernel(yy, yv[i], hy) # update derivative
        uf = dfy/fy # update information vector
        hf = ll * hf + (1-ll) * uf**2 # update hessian function
        exp_val_lst.append(expected_value_fy(fy, yy, dx = 0.01))
        a = 3
    return yy, fy, hf, dfy, uf, hv, log_score_lst, exp_val_lst

# log-likelihood score calculation at time t yi.
def log_score_yi(yv, yy, fy, i):
    score = - np.log(np.interp(yv[i], yy, fy))
    return score

def smaller_df(df, until_month):
    return df[df.index.month <= until_month]

def area(fy):
    area = trapz(fy, dx=0.01)
    print("area =", area)

def expected_value_fy(fy, yy, dx):
    area_lst = []
    value_lst = []
    exp_value_lst = []
    for i in range(len(fy)):
        small_fy = fy[i:i+2]
        small_yy = yy[i:i+2]
        area = trapz(small_fy, dx=0.01)
        area_lst.append(area)
        value = np.mean(small_yy)
        value_lst.append(value)
        exp_value_lst.append(area*value)
    a=3
    return np.sum(exp_value_lst)

