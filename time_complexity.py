import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data= pd.read_csv('results_general.csv')

data['a']= data['steps']/data['b_mods']

plt.scatter(data['d_tau'], np.power(data['a'], 1.0/2.0))

data_b= data[data['b_mods'] == 5]

plt.scatter(data_b['d_tau'], data_b['steps'])

grouped= data.groupby('d_tau').agg({'steps': np.mean})

plt.scatter(grouped.index, np.sqrt(grouped))

grouped= data.groupby('b_mods').agg({'steps': np.mean})

plt.scatter(grouped.index, np.sqrt(grouped))

sigmas= data.groupby('sigma').agg({'b': 'count'})
sigmas= sigmas[sigmas['b'] == 5]

data_filtered= data[data['sigma'].isin(sigmas.index)]

grouped= data_filtered.groupby('d_tau').agg({'steps': np.mean})

plt.scatter(grouped.index, np.sqrt(grouped))

grouped= data_filtered.groupby('b_mods').agg({'steps': np.mean})

plt.scatter(grouped.index, np.sqrt(grouped))
