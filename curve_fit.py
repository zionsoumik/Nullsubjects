# -*- coding: utf-8 -*-
from scipy.optimize import curve_fit
import scipy.stats as st
import math
import pandas as pd
from scipy.stats import truncnorm
import matplotlib
matplotlib.use('Agg')
from scipy.stats import truncnorm
from matplotlib import pyplot as plt
import numpy as np
def logifunc(x,x0,k):
    return 1 / (1 + np.exp(-k*(x-x0)))
def func(x, k, c):
    return k*x+c
import csv
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
#df = pd.read_csv('https://gist.githubusercontent.com/shinokada/76070a0927fa1fac01eeaed298757a26/raw/2707a1bd7cba80613a01a2026abeb9f587dbaee5/logisticdata.csv')

#x=df.T.iloc[0]
#y=df.T.iloc[1]

matplotlib.rcParams['font.family'] = 'Times New Roman'

#xdata = np.linspace(0, 4, 50)
#y = func(xdata, 2.5, 1.3, 0.5)
#rng = np.random.default_rng()
#y_noise = 0.2 * rng.normal(size=xdata.size)
#ydata = y + y_noise

#print(type(np.linspace(0, 4, 50)))

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)



def prepare_stats(num):
    y1 = get_truncated_normal(mean=0.4, sd=0.1785, low=0, upp=0.75)
    y2 = get_truncated_normal(mean=0.64, sd=0.43, low=0.25, upp=1)
    y3 = get_truncated_normal(mean=0.9, sd=0.179, low=0.25, upp=1)
    #print(y1.rvs)
    #print("Freedman–Diaconis number of bins:", bins)
    NFS1=y1.rvs(num)
    NFS2=y2.rvs(num)
    NFS3=y3.rvs(num)

    NFS1.sort()
    NFS2.sort()
    NFS3.sort()

    fig, axs = plt.subplots(3, sharex=True, sharey=True)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    axs[0].hist(NFS1, bins=20, label="Data")
    mn, mx = plt.xlim()
    # axs[0].xlim(mn, mx)
    kde_xs = np.linspace(mn, mx, 300)
    kde = st.gaussian_kde(NFS1)
    axs[0].plot(kde_xs, kde.pdf(kde_xs), label="PDF")
    axs[0].set_title("100 e-children in age range 2;6-2;11 with $µ_{IARC}=$0.4 and $σ_{IARC}=$0.178")
    # plt.legend(loc="upper left")
    # plt.xlabel('Data')
    # plt.savefig("NFS_26_3.png")
    axs[1].hist(NFS2, bins=20, label="Data")
    mn, mx = plt.xlim()
    # axs[1].xlim(mn, mx)
    kde_xs = np.linspace(mn, mx, 300)
    kde = st.gaussian_kde(NFS2)
    axs[1].plot(kde_xs, kde.pdf(kde_xs), label="PDF")
    axs[1].set_title("100 e-children in age range 3;0-3;5 with $µ_{IARC}=$0.64 and $σ_{IARC}=$0.43")
    # plt.legend(loc="upper left")
    # plt.xlabel('Data')
    # plt.savefig("NFS_3_36.png")
    axs[2].hist(NFS3, bins=20, label="Data")
    mn, mx = plt.xlim()
    # axs[2].xlim(mn, mx)
    kde_xs = np.linspace(mn, mx, 300)
    kde = st.gaussian_kde(NFS3)
    axs[2].plot(kde_xs, kde.pdf(kde_xs), label="PDF")
    axs[2].set_title("100 e-children in age range 3;6-3;11 with $µ_{IARC}=$0.9 and $σ_{IARC}=$0.179")
    axs[0].legend(loc="upper left")
    axs[1].legend(loc="upper left")
    axs[2].legend(loc="upper left")
    plt.xlabel('IARC')
    plt.savefig("NFS.png")
    plt.show()

    x1 = get_truncated_normal(mean=2.73, sd=0.1, low=2.54, upp=2.96)
    x2 = get_truncated_normal(mean=3.3, sd=0.1, low=3.12, upp=3.48)
    x3 = get_truncated_normal(mean=3.82, sd=0.1, low=3.64, upp=3.98)

    #fig, ax = plt.subplots(3, sharex=True)
    
    age1=x1.rvs(num)
    age2=x2.rvs(num)
    age3=x3.rvs(num)

    age1.sort()
    age2.sort()
    age3.sort()
    fig, axs = plt.subplots(3, sharex=True, sharey=True)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    axs[0].hist(age1, bins=20, label="Data")
    mn, mx = plt.xlim()
    #axs[0].xlim(mn, mx)
    kde_xs = np.linspace(mn, mx, 300)
    kde = st.gaussian_kde(age1)
    axs[0].plot(kde_xs, kde.pdf(kde_xs), label="PDF")
    axs[0].set_title("100 e-children in age range 2;6-3;0 with $µ_{age}=$2.73 and $σ_{age}=$0.1")
    #plt.legend(loc="upper left")
    #plt.xlabel('Data')
    #plt.savefig("age_26_3.png")
    axs[1].hist(age2, bins=20, label="Data", )
    mn, mx = plt.xlim()
    #axs[1].xlim(mn, mx)
    kde_xs = np.linspace(mn, mx, 300)
    kde = st.gaussian_kde(age2)
    axs[1].plot(kde_xs, kde.pdf(kde_xs), label="PDF")
    axs[1].set_title("100 e-children in age range 3;0-3;6 with $μ_{age}=$3.3 and $σ_{age}=$0.1")
    #plt.legend(loc="upper left")
    #plt.xlabel('Age')
    #plt.savefig("age_3_36.png")
    axs[2].hist(age3, bins=20, label="Data")
    mn, mx = plt.xlim()
    #axs[2].xlim(mn, mx)
    kde_xs = np.linspace(mn, mx, 300)
    kde = st.gaussian_kde(age3)
    axs[2].plot(kde_xs, kde.pdf(kde_xs), label="PDF")
    axs[2].set_title("100 e-children in age range 3;6-4;0 with $μ_{age}=$3.82 and $σ_{age}=$0.1")
    axs[0].legend(loc="upper left")
    axs[1].legend(loc="upper left")
    axs[2].legend(loc="upper left")
    plt.xlabel('Age (years)')
    plt.savefig("age.png")
    plt.show()
    x0=[]
    slope=[]
    d_list=[]
    p_list_linear=[]
    p_list_e=[]
    for i in range(0,num):
        d_list.append([NFS1[i],NFS2[i],NFS3[i],age1[i],age2[i],age3[i]])

        x=np.asarray([3566210+((11+11+(age1[i]-2))*365*487*0.5*(age1[i]-2)),5610392+((12+12+(age2[i]-3)*0.5)*365*487*0.5*(age2[i]-3)),5610392+((12+12+(age2[i]-3)*0.5)*365*487*0.5*(age3[i]-3))],dtype=np.float)
        y=np.asarray([NFS1[i],NFS2[i],NFS3[i]],dtype=np.float)
        popt, pcov = curve_fit(func,x,y,p0=[0.000001,5000000])
        popt1, pcov1 = curve_fit(logifunc, x, y, p0=[5000000, 0.000001])
        #print(popt)
        slope.append(popt[0])
        x0.append(popt[1])
        p_list_linear.append([popt[0],popt[1]])
        p_list_e.append([popt1[0], popt1[1]])
        
    with open("linear_param.csv", "w",newline='') as f:
        writer = csv.writer(f)
        writer.writerows(p_list_linear)
    with open("age_NFS.csv", "w",newline='') as f:
        writer = csv.writer(f)
        writer.writerows(d_list)
    with open("e_param.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(p_list_e)
prepare_stats(100)

