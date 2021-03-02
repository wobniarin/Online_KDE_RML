from Online_RML_KDE_functions import *
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("test starts here")
    np.random.seed(5465)
    y_normal = np.random.normal(0,1,100000)
    yym = y_normal * np.linspace(0.5, 2, len(y_normal)) - np.linspace(0,2, len(y_normal)) # adding some funny things to make it better to forecast
    yy, fy, hf, dfy, uf, h_val, log_score_lst = online_KDE(yym, 0.999)
    log_score = np.mean(log_score_lst)
    print("test STOP")
