# Online KDE - Recursive max likelihood for parameters estimation (h_t)
# Functions to be imported into the main script
# 22/2/2021
# Author : Ingrid Munne Collado

# Libraries
import numpy as np
import matplotlib.pyplot as plt


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
    tol = 0.05
    # iterative calculation
    for i in range(len(yv)):
        # first update parameters
        uf = dfy/fy # calculation of the information vector
        if np.interp(yv[i], yy, fy) < tol:
            # TODO replace values
            # fy = np.where(fy < tol, tol, fy)
            uf = dfy / tol
        gf = -(1-ll) * uf # calculation of gradient S
        gfy = np.interp(yv[i], yy, gf) # getting the value at yi
        hfy = np.interp(yv[i], yy, hf) # getting the value of hf at yi
        # implementing warm-start
        if i > 50:
            # update hh
            hh = hh - gfy/hfy
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
        dfy = ll * dfy + (1-ll)*((((yy-yv[i])**2)/(hy**2)) -1)*gaussiankernel(yy, yv[i], hy) # update derivative
        uf = dfy/fy # update information vector
        hf = ll * hf + (1-ll) * uf**2 # update hessian function
        a = 3
    return yy, fy, hf, dfy, uf, hv

