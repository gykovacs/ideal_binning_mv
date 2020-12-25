import os.path
import numpy as np
import pandas as pd

from scipy.stats import ttest_ind, ttest_rel
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from statsmodels.stats.contingency_tables import mcnemar

from config import *

results_general = pd.read_csv(os.path.join(work_dir, 'results_general.csv'))
results_spherical = pd.read_csv(os.path.join(work_dir, 'results_spherical.csv'))

figsize= (5.5, 3.3)

#######################
# general distortions #
#######################

grouped= results_general.groupby(['b']).agg(['mean', 'std'])

grouped= grouped.loc[['2', '5', 'sturges-formula', 'rice-rule', 'square-root']]

print('relative variation for noisy window', abs(grouped[('exact_noise', 'mean')] - grouped[('greedy_noise', 'mean')])/grouped[('exact_noise', 'mean')])
print('relative variation for distorted template (general distortion)', abs(grouped[('exact_distortion', 'mean')] - grouped[('greedy_distorted', 'mean')])/grouped[('greedy_distorted', 'mean')])

fig, ax= plt.subplots(figsize=figsize)
trans0= Affine2D().translate(-0.08, 0.0) + ax.transData
trans1= Affine2D().translate(-0.04, 0.0) + ax.transData
trans2= Affine2D().translate(-0.0, 0.0) + ax.transData
trans3= Affine2D().translate(0.04, 0.0) + ax.transData
trans4= Affine2D().translate(0.08, 0.0) + ax.transData

ax.errorbar(np.arange(len(grouped)), grouped[('exact_noise', 'mean')], grouped[('exact_noise', 'std')], label='$\\overline{\\mathbb{E}_\\xi D(\\mathbf{t},\\xi)}$ by Proposition 1', linestyle='-', linewidth=2.0, transform=trans0)
ax.errorbar(np.arange(len(grouped)), grouped[('greedy_noise', 'mean')], grouped[('greedy_noise', 'std')], label='$\\overline{D(\\mathbf{t}, \\xi)}$ using greedy binning', linewidth=2.0, linestyle=':', transform=trans1)
ax.errorbar(np.arange(len(grouped)), grouped[('exact_distortion', 'mean')], grouped[('exact_distortion', 'std')], label='$\\overline{\\mathbb{E}_\\zeta\\mathbb{E}_\\mathbf{m} D(\\mathbf{t}, S_\\tau \\mathbf{m} + \\zeta)}$ by Proposition 2', linestyle='solid', linewidth=2.0, transform=trans3)
ax.errorbar(np.arange(len(grouped)), grouped[('greedy_distorted', 'mean')], grouped[('greedy_distorted', 'std')], label='$\\overline{D(\\mathbf{t}, S_\\tau \\mathbf{m} + \\zeta)}$ using greedy binning', linewidth=2.0, linestyle=':', transform=trans4)
ax.legend()
ax.set_xlabel('number of bins ($b$)')
ax.set_ylabel('$D(\mathbf{t},\mathbf{w})$')
ax.set_title('Exact values and measurements with general distortions')
ax.set_xticks(np.arange(len(grouped)))
ax.set_xticklabels(['2', '5', 'Sturges-formula', 'Rice-rule', 'Square-root'])
plt.tight_layout()
plt.savefig('fit_general.pdf')

plt.figure(figsize=figsize)
plt.plot(np.arange(len(grouped)), grouped[('eqw_hits', 'mean')], label='EQW binning', linestyle='-', linewidth=2.0)
plt.plot(np.arange(len(grouped)), grouped[('eqf_hits', 'mean')], label='EQF binning', linewidth=2.0, linestyle='-.')
plt.plot(np.arange(len(grouped)), grouped[('kmeans_hits', 'mean')], label='k-means binning', linestyle='--', linewidth=2.0)
plt.plot(np.arange(len(grouped)), grouped[('greedy_hits', 'mean')], label='greedy binning', linewidth=2.0, linestyle=':')
for n in mi_n_neighbors:
    plt.plot(np.arange(len(grouped)), grouped[('mi_' + str(n) + '_hits', 'mean')], label='mi ' + str(n) + ' neighbors', linewidth=2.0, linestyle='-')

plt.legend()
plt.xlabel('number of bins ($b$)')
plt.ylabel('AUC')
plt.title('AUC of recognition for general distortions')
plt.xticks(np.arange(len(grouped)), ['2', '5', 'Sturges-formula', 'Rice-rule', 'Square-root'])
plt.tight_layout()
plt.savefig('auc_general.pdf')

means= np.array([np.mean(results_general['eqw_hits']), np.mean(results_general['eqf_hits']), np.mean(results_general['kmeans_hits']), np.mean(results_general['greedy_hits'])])

p_matrix_general= np.zeros(shape=(len(binning_methods), len(binning_methods)))

for i, b0 in enumerate(binning_methods):
    for j, b1 in enumerate(binning_methods):
        p_matrix_general[i][j]= ttest_rel(results_general[b0 + '_hits'], results_general[b1 + '_hits'])[1]

pd.options.display.float_format = '{:.1e}'.format
p= pd.DataFrame(np.vstack([p_matrix_general, means]), columns= ['EQW', 'EQF', 'k-means', 'greedy'], index=['EQW', 'EQF', 'k-means', 'greedy', 'accuracy']).fillna(1)

print("the matrix of p-values and mean AUCs for general distortions with paired t-test")
print(p.to_latex())


p_matrix_general= np.zeros(shape=(len(binning_methods), len(binning_methods)))

table= np.zeros(shape=(2,2))

for i, b0 in enumerate(binning_methods):
    for j, b1 in enumerate(binning_methods):
        a= results_general[b0 + '_hits'].values
        b= results_general[b1 + '_hits'].values

        for k in [0, 1]:
            for l in [0, 1]:
                table[k,l]= np.sum(np.logical_and(a == k, b == l))

        p_matrix_general[i][j]= mcnemar(table).pvalue

pd.options.display.float_format = '{:.1e}'.format
p= pd.DataFrame(np.vstack([p_matrix_general, means]), columns= ['EQW', 'EQF', 'k-means', 'greedy'], index=['EQW', 'EQF', 'k-means', 'greedy', 'accuracy']).fillna(1)

print("the matrix of p-values and mean AUCs for general distortions with mcnemar test")
print(p.to_latex())

#########################
# spherical distortions #
#########################

grouped= results_spherical.groupby(['b']).agg(['mean', 'std'])

grouped= grouped.loc[['2', '5', 'sturges-formula', 'rice-rule', 'square-root']]

print('relative variation for noisy window', abs(grouped[('exact_noise', 'mean')] - grouped[('kmeans_noise', 'mean')])/grouped[('exact_noise', 'mean')])
print('relative variation for distorted template (spherical distortion)', abs(grouped[('exact_kmeans', 'mean')] - grouped[('kmeans_distorted', 'mean')])/grouped[('exact_kmeans', 'mean')])

fig, ax= plt.subplots(figsize=figsize)
trans0= Affine2D().translate(-0.08, 0.0) + ax.transData
trans1= Affine2D().translate(-0.04, 0.0) + ax.transData
trans2= Affine2D().translate(-0.0, 0.0) + ax.transData
trans3= Affine2D().translate(0.04, 0.0) + ax.transData
trans4= Affine2D().translate(0.08, 0.0) + ax.transData
trans5= Affine2D().translate(0.12, 0.0) + ax.transData

ax.errorbar(np.arange(len(grouped)), grouped[('exact_noise', 'mean')], grouped[('exact_noise', 'std')], label='$\\overline{\\mathbb{E}_\\xi D(\\mathbf{t},\\xi)}$ by Proposition 1', linestyle='-', linewidth=2.0, transform=trans0)
ax.errorbar(np.arange(len(grouped)), grouped[('kmeans_noise', 'mean')], grouped[('kmeans_noise', 'std')], label='$\\overline{D(\\mathbf{t}, \\xi)}$ using k-means clustering', linewidth=2.0, linestyle=':', transform=trans1)
ax.errorbar(np.arange(len(grouped)), grouped[('exact_kmeans', 'mean')], grouped[('exact_kmeans', 'std')], label='$\\overline{\\mathbb{E}_\\zeta\\mathbb{E}_\\mathbf{m} D(\\mathbf{t}, S_\\tau \\mathbf{m} + \\zeta)}$ by Proposition 4', linestyle='solid', linewidth=2.0, transform=trans3)
ax.errorbar(np.arange(len(grouped)), grouped[('kmeans_distorted', 'mean')], grouped[('kmeans_distorted', 'std')], label='$\\overline{D(\\mathbf{t}, S_\\tau \\mathbf{m} + \\zeta)}$ using k-means clustering', linewidth=2.0, linestyle=':', transform=trans4)
ax.legend()
ax.set_xlabel('number of bins ($b$)')
ax.set_ylabel('$D(\mathbf{t},\mathbf{w})$')
ax.set_title('Exact values and measurements with spherical distortions')
ax.set_xticks(np.arange(len(grouped)))
ax.set_xticklabels(['2', '5', 'Sturges-formula', 'Rice-rule', 'Square-root'])
plt.tight_layout()
plt.savefig('fit_spherical.pdf')

plt.figure(figsize=figsize)
plt.plot(np.arange(len(grouped)), grouped[('eqw_hits', 'mean')], label='EQW binning', linestyle='-', linewidth=2.0)
plt.plot(np.arange(len(grouped)), grouped[('eqf_hits', 'mean')], label='EQF binning', linewidth=2.0, linestyle='-.')
plt.plot(np.arange(len(grouped)), grouped[('kmeans_hits', 'mean')], label='k-means binning', linestyle='--', linewidth=2.0)
plt.plot(np.arange(len(grouped)), grouped[('greedy_hits', 'mean')], label='greedy binning', linewidth=2.0, linestyle=':')
for n in mi_n_neighbors:
    plt.plot(np.arange(len(grouped)), grouped[('mi_' + str(n) + '_hits', 'mean')], label='mi ' + str(n) + ' neighbors', linewidth=2.0, linestyle='-')
plt.legend()
plt.xlabel('number of bins ($b$)')
plt.ylabel('AUC')
plt.title('AUC of recognition with spherical distortions')
plt.xticks(np.arange(len(grouped)), ['2', '5', 'Sturges-formula', 'Rice-rule', 'Square-root'])
plt.tight_layout()
plt.savefig('auc_spherical.pdf')

means= np.array([np.mean(results_spherical['eqw_hits']), np.mean(results_spherical['eqf_hits']), np.mean(results_spherical['kmeans_hits']), np.mean(results_spherical['greedy_hits'])])


p_matrix_greedy= np.zeros(shape=(len(binning_methods), len(binning_methods)))

for i, b0 in enumerate(binning_methods):
    for j, b1 in enumerate(binning_methods):
        p_matrix_greedy[i][j]= ttest_rel(results_spherical[b0 + '_hits'], results_spherical[b1 + '_hits'])[1]

pd.options.display.float_format = '{:.1e}'.format
p= pd.DataFrame(np.vstack([p_matrix_greedy, means]), columns= ['EQW', 'EQF', 'k-means', 'greedy'], index=['EQW', 'EQF', 'k-means', 'greedy', 'accuracy']).fillna(1)

print("the matrix of p-values and mean AUCs for spherical distortions")
print(p.to_latex())

for i, b0 in enumerate(binning_methods):
    for j, b1 in enumerate(binning_methods):
        a= results_spherical[b0 + '_hits'].values
        b= results_spherical[b1 + '_hits'].values

        for k in [0, 1]:
            for l in [0, 1]:
                table[k,l]= np.sum(np.logical_and(a == k, b == l))

        p_matrix_greedy[i][j]= mcnemar(table).pvalue

pd.options.display.float_format = '{:.1e}'.format
p= pd.DataFrame(np.vstack([p_matrix_greedy, means]), columns= ['EQW', 'EQF', 'k-means', 'greedy'], index=['EQW', 'EQF', 'k-means', 'greedy', 'accuracy']).fillna(1)

print("the matrix of p-values and mean AUCs for spherical distortions with mcnemar test")
print(p.to_latex())