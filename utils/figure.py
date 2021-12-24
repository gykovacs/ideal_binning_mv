#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 15:18:32 2020

@author: gykovacs
"""

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(6)

t= np.random.rand(20)*10
w= np.sin(t) + np.random.rand(20)/2

n_bins= 3

figsize= (5.2, 2.5)

t_diff= (np.max(t) - np.min(t))/n_bins
t_bins= [np.min(t) + t_diff*i for i in range(1, n_bins)]
w_diff= (np.max(w) - np.min(w))/n_bins
w_bins= [np.min(w) + w_diff*i for i in range(1, n_bins)]

print(t)
print(sorted(t))
print(w)

plt.figure(figsize=figsize)
plt.vlines(t_bins, np.min(w), np.max(w), linestyles='dashed', label='bin boundaries', colors='black')
plt.scatter(t, w)
plt.xlabel('$\mathbf{x}$')
plt.ylabel('$\mathbf{y}$')
#plt.title('Illustration of PWC nEV')
plt.arrow(t[4] + 0.15, w[4] - 0.15, 1.5 - t[4] - 0.3, 0.8 - w[4] + 0.3, width=0.01, head_width=0.1, head_length=0.1, color='#ff7f0e')
plt.scatter([1.5], [0.8], color="#ff7f0e")
plt.text(8.5, -0.5, 'bin$_3$', horizontalalignment='center', verticalalignment='center')

t_binning= np.digitize(t, t_bins)
w_binning= np.digitize(w, w_bins)

w_means = []
w_stds= []
w_means_adjusted= []
w_stds_adjusted= []
for i in np.unique(t_binning):
    w_means.append(np.mean(w[t_binning == i]))
    w_stds.append(np.std(w[t_binning == i]))

w_adjusted= w.copy()
w_adjusted[4]= 0.8
for i in np.unique(t_binning):
    w_means_adjusted.append(np.mean(w_adjusted[t_binning == i]))
    w_stds_adjusted.append(np.std(w_adjusted[t_binning == i]))
    
def pwc_nev(t, t_binning, w):
    numerator= 0.0
    for i in np.unique(t_binning):
        numerator+= np.sum((np.mean(w[t_binning == i]) - w[t_binning == i])**2)
    return 1.0 - numerator / (len(w)*np.var(w))

def nmi(t, t_binning, w, w_binning):
    p_ij= np.zeros(shape=(len(np.unique(t_binning)), len(np.unique(w_binning))))
    for i in np.unique(t_binning):
        for j in np.unique(w_binning):
            p_ij[i, j]= np.sum((t_binning == i) & (w_binning == j))
    p_i= np.zeros(len(np.unique(t_binning)))
    p_j= np.zeros(len(np.unique(w_binning)))
    for i in np.unique(t_binning):
        p_i[i]= np.sum(t_binning == i)
    for j in np.unique(w_binning):
        p_j[j]= np.sum(w_binning == j)
    
    p_ij= p_ij/np.sum(p_ij)
    p_i= p_i/np.sum(p_i)
    p_j= p_j/np.sum(p_j)
    
    def safe_ln(x, y):
        if x == 0 or y == 0:
            return 0
        return np.log(x/y)
    
    result= 0.0
    for i in np.unique(t_binning):
        for j in np.unique(w_binning):
            result+= p_ij[i,j]*safe_ln(p_ij[i,j], (p_i[i]*p_j[j]))
    
    h_t= np.sum([-p_i[i]*safe_ln(p_i[i], 1) for i in np.unique(t_binning)])
    
    return result/h_t

print('pwc_nev', pwc_nev(t, t_binning, w))
print('nmi', nmi(t, t_binning, w, w_binning))

print(len(w_means), len(t_bins), t_bins)

for i in range(len(w_means)):
    if i == 0:
        bin_min= np.min(t)
    else:
        bin_min= t_bins[i-1]
    if i == len(w_means) - 1:
        bin_max= np.max(t)
    else:
        bin_max= t_bins[i]

    if i == 0:
        plt.hlines(w_means[i], bin_min, bin_max, colors='#1f77b4', label='mean of ${\\bf y}$ in bin')
        plt.vlines((bin_min + bin_max)/2.0, w_means[i] - w_stds[i], w_means[i] + w_stds[i], colors='#1f77b4', linestyles='dashed', label="std. of ${\\bf y}$ in bin")
    else:
        plt.hlines(w_means[i], bin_min, bin_max, colors='#1f77b4')
        plt.vlines((bin_min + bin_max)/2.0, w_means[i] - w_stds[i], w_means[i] + w_stds[i], colors='#1f77b4', linestyles='dashed')

for i in range(len(w_means_adjusted)):
    if i == 0:
        bin_min= np.min(t)
    else:
        bin_min= t_bins[i-1]
    if i == len(w_means_adjusted) - 1:
        bin_max= np.max(t)
    else:
        bin_max= t_bins[i]

    if i == 0:
        plt.hlines(w_means_adjusted[i], bin_min, bin_max, colors='#ff7f0e')
        plt.vlines((bin_min + bin_max)/2.0 + 0.2, w_means_adjusted[i] - w_stds_adjusted[i], w_means_adjusted[i] + w_stds_adjusted[i], colors='#ff7f0e', linestyles='dashed')

plt.legend(fontsize='small')
plt.tight_layout()
plt.savefig('binning_nEV.pdf')

plt.figure(figsize=figsize)
plt.vlines(t_bins, np.min(w), np.max(w), linestyles='dashed', label='bin boundaries', colors='black')
plt.hlines(w_bins, np.min(t), np.max(t), linestyles='dashed', colors='black')
plt.scatter(t, w)
plt.xlabel('$\mathbf{x}$')
plt.ylabel('$\mathbf{y}$')
#plt.title('Illustration of MI')
plt.arrow(t[4] + 0.15, w[4] - 0.15, 1.5 - t[4] - 0.3, 0.8 - w[4] + 0.3, width=0.01, head_width=0.1, head_length=0.1, color='#ff7f0e')
plt.scatter([1.5], [0.8], color="#ff7f0e")
plt.text(8.5, -0.5, 'bin$_{3, 1}$', horizontalalignment='center', verticalalignment='center')
plt.legend(fontsize='small')

plt.tight_layout()
plt.savefig('binning_MI.pdf')

t[4]= 1.5
w[4]= 0.8

print('pwc_nev', pwc_nev(t, t_binning, w))
print('nmi', nmi(t, t_binning, w, w_binning))