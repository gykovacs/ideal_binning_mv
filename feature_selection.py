import os
import tqdm
import pandas as pd
import time

from config import *

from binning import *
from data_generation import *
from nev import *

from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor

import mldb.regression as reg

def pwc_nev_regression_ttof(X, 
                       y, 
                       n_bins,
                       binning= 'distortion_aligned',
                       distortions= {#'power': [0.02, 50, 100],
                                     'random_intervals': [100],
                                     #'random_slices': [100],
                                     #'identity': None,
                                     #'difference': None
                                    },
                       random_state=random_seed):
    """
    PWC nev regression target to feature function
    
    Args:
        X (np.array): array of features
        y (np.array): array of target labels
        n_bins (int): the number of bins
        binning (str): the binning method ('distortion_aligned'/'kmeans'/'eqw'/'eqf')
        distortions (dict): the specifications of distortions to be applied to approximate 
                                the covariance matrix
        random_state (int): the random seed to be used
    Returns:
        np.array: the PWC nev scores of the features
    """

    if binning.startswith('distortion_aligned'):
        # if the binning technique specified is distortion aligned,
        # the covariance matrix is approximated by applying the 
        # specified transformations to the target vector
        tau= np.unique(y)
        
        C= np.zeros((len(tau), len(tau)))
        rs= np.random.RandomState(random_state)
        n= 0
        tau= (tau - np.min(tau))/(np.max(tau) - np.min(tau))
        tau= np.log(1 + tau)
        distorted= []
        for k in distortions:
            if k == 'power':
                lower_bound= distortions[k][0]
                upper_bound= distortions[k][1]
                n_sim= distortions[k][2]
                for _ in range(n_sim):
                    p= lower_bound + rs.rand()*(upper_bound - lower_bound)
                    sample= np.power(tau, p)
                    sample= (sample - np.min(sample))/(np.max(sample) - np.min(sample))
                    distorted.append(sample)
                n+= n_sim
            if k == 'random_intervals':
                n_sim= distortions[k][0]
                for _ in range(n_sim):
                    thresholds= np.array(sorted(rs.rand(2)))
                    value= rs.randint(2)
                    mask= np.logical_and(tau >= thresholds[0], tau < thresholds[1])
                    while np.sum(mask) < 2:
                        thresholds= np.array(sorted(rs.rand(2)))
                        mask= np.logical_and(tau >= thresholds[0], tau < thresholds[1])
                    sample= np.where(mask, float(value), 1.0 - value)
                    distorted.append(sample)
                n+= n_sim
            if k == 'random_slices':
                n_sim= distortions[k][0]
                for _ in range(n_sim):
                    threshold= rs.rand()
                    value= rs.randint(2)
                    sample= np.where(tau < threshold, 0.0, 1.0)
                    distorted.append(sample)
                n+= n_sim
            if k == 'identity':
                distorted.append(tau)
                n+= 1

        if not k == 'difference':
            distorted= np.vstack(distorted)
            C= np.dot(distorted.T, distorted)
            C/= float(n)
        else:
            C= np.zeros((len(tau), len(tau)))
            for i in range(len(tau)):
                for j in range(i, len(tau)):
                    C[i,j]= 1.0/(np.abs(tau[i] - tau[j]) + 0.001)
                    C[j,i]= 1.0/(np.abs(tau[j] - tau[i]) + 0.001)
        
        y_binning, steps= distortion_aligned_binning(y, 
                                                        C, 
                                                        n_bins, 
                                                        return_it=True,
                                                        validate=False,
                                                        random_state=random_seed)
        assert len(np.unique(y_binning)) == n_bins
    elif binning.startswith('eqw'):
        y_binning= eqw_binning(y, n_bins)
        it= 0
        while len(np.unique(y_binning)) < n_bins - it:
            y_binning= y_binning= eqw_binning(y, n_bins-it)
            it= it + 1
    elif binning.startswith('eqf'):
        y_binning= eqf_binning(y, n_bins)
    elif binning.startswith('kmeans'):
        y_binning= kmeans_binning(y, n_bins)
        assert len(np.unique(y_binning)) == n_bins
    
    results= []
    for i in range(len(X[0])):
        results.append(pwc_nev(y, X[:,i], y_binning))
        
    return np.array(results)

def pwc_nev_regression_ftot(X, 
                       y, 
                       num_bins,
                       binning= 'distortion_aligned',
                       distortions= {'power': [0.2, 5, 100],
                                     #'random_intervals': [200],
                                     #'random_slices': [100]
                                    },
                       random_state=random_seed):
    """
    PWC nev regression feature to target scores
    
    Args:
        X (np.array): array of features
        y (np.array): array of target labels
        n_bins (int): the number of bins
        binning (str): the binning method ('distortion_aligned'/'kmeans'/'eqw'/'eqf')
        distortions (dict): the specifications of distortions to be applied to approximate 
                                the covariance matrix
        random_state (int): the random seed to be used
    Returns:
        np.array: the PWC nev scores of the features
    """
    scores= []
    for i in range(len(X[0])):
        n_uniques= len(np.unique(X[:,i]))
        if n_uniques > 1:
            num_bins_mod= n_bins(X[:,i], num_bins)
            y_tmp= y[:,np.newaxis]
            X_tmp= X[:,i]

            scores.append(pwc_nev_regression_ttof(X=y_tmp, 
                                                  y=X_tmp, 
                                                  n_bins=num_bins_mod, 
                                                  binning=binning, 
                                                  distortions=distortions,
                                                  random_state=random_state)[0])
        else:
            scores.append(1.0)
    return np.array(scores)

# reading all databases
all_data= reg.get_data_loaders()

results= []

all_n_cases= {}
all_runtimes= []
all_d= {}
all_n= {}
all_rankings= {}
all_r2_scores= {}

# setting up the identifiers of the feature selection methods
# to be used
mi_techniques= ['mi_' + str(n) for n in mi_n_neighbors_feature_selection]
nev_techniques= []
for b in bins:
    for m in binning_methods:
        nev_techniques.append(m + '_' + str(b))

all_techniques= mi_techniques + nev_techniques

# iterating through all databases
for d in all_data:
    data= d()
    X= data['data']
    y= data['target']
    
    if data['name'] in ['residential_building', 'communities', 'ccpp']: #, 'compactiv', 'puma32h', 'laser', 'stock_portfolio_performance', 'diabetes']:
        continue
    
    print("database %s, n: %d, d: %d" % (data['name'], len(X), len(X[0])))
    
    # initializing the variables recording the validation scores
    rankings= []
    r2_scores= []
    
    n_cases= 0
    
    # determining the database folds
    validator= RepeatedKFold(n_repeats=fs_n_repeats, 
                             n_splits=fs_n_splits,
                             random_state=random_seed)
    
    splits= [(train, test) for train, test in validator.split(X)]
    
    # iterating through all database folds
    j= 0
    for train, test in tqdm.tqdm(splits):
        j= j + 1
        X_train, X_test= X[train], X[test]
        y_train, y_test= y[train], y[test]
        
        all_scores= {}
        all_indices= {}
        runtimes= []
        # iterating through all binning techniques
        for m in all_techniques:
            if m.startswith('mi'):
                # determining the mutual information feature scores
                n_neighbors= int(m.split('_')[1])
                start= time.time()
                scores= mutual_info_regression(X_train, 
                                               y_train, 
                                               random_state=random_seed,
                                               n_neighbors=n_neighbors)
                feature_indices= np.array(scores.argsort())
                scores.sort()
                feature_scores= scores
                end= time.time()
            else:
                # determining the PWC nev feature scores
                num_bins= m.split('_')[-1]
                try:
                    num_bins= int(num_bins)
                except:
                    pass
                start= time.time()
                
                scores= pwc_nev_regression_ttof(X_train, 
                                                y_train, 
                                                n_bins=n_bins(y_train, num_bins), 
                                                binning=m, 
                                                random_state=random_seed)
                
                
                feature_indices= np.array(scores.argsort())
                scores.sort()
                feature_scores= scores
                end= time.time()
            
            runtimes.append(end - start)
            
            all_scores[m]= feature_scores
            all_indices[m]= feature_indices
        all_runtimes.append(runtimes)
        
        # iterating through the all subsets of features in according
        # to the ordering implied by the feature scores
        
        #for i in range(1, min(max(2, int(np.sqrt(len(X[0])))), 10)):
        for i in range(1, len(X[0])):
            scores= []
            for m in all_techniques:
                # instantiating a regressor
                regressor= LinearRegression()
                
                # picking the features
                features= np.array(sorted(all_indices[m][:i]))
                
                # fitting the regressor
                regressor.fit(X_train[:,features], y_train)
            
                # computing the r2 score
                r2= r2_score(y_test, regressor.predict(X_test[:,features]))
                
                scores.append(r2)
                
                n_cases+= 1
            # determining which techniques outperformed which other techniques
            unique_scores= np.unique(scores)
            ranks= np.arange(len(unique_scores))[np.argsort(unique_scores)]
            ranks= len(unique_scores) - ranks
            rank_dict= {unique_scores[i]: ranks[i] for i in range(len(unique_scores))}
            
            final_ranking= np.array([rank_dict[s] for s in scores])
            if j == 1 and i == int(len(X[0])/2):
                print(scores)
                print(unique_scores)
                print(final_ranking)
            rankings.append(final_ranking)
            r2_scores.append(np.array(scores))
            
    # recording the results
    all_n_cases[data['name']]= n_cases
    all_d[data['name']]= len(X[0])
    all_n[data['name']]= len(X)
    all_rankings[data['name']]= np.mean(np.array(rankings), axis=0)
    all_r2_scores[data['name']]= np.mean(np.array(r2_scores), axis=0)
    print(pd.DataFrame(all_rankings.values(), index=all_rankings.keys(), columns=all_techniques))
    print(pd.DataFrame(all_rankings.values(), index=all_rankings.keys(), columns=all_techniques).mean())
    
    print(pd.DataFrame(all_r2_scores.values(), index=all_r2_scores.keys(), columns=all_techniques))
    print(pd.DataFrame(all_r2_scores.values(), index=all_r2_scores.keys(), columns=all_techniques).mean())

pd.DataFrame(all_r2_scores.values(), index=all_r2_scores.keys(), columns=all_techniques).to_csv('feature_selection_results.csv')
pd.DataFrame([np.mean(all_runtimes, axis=0)], index=[0], columns=all_techniques).to_csv('feature_selection_runtimes.csv')
