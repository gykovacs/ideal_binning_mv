# working directory
work_dir='.'

# random seed to be used
random_seed=5

# number of trials
n_trials=10000

# lower and upper bounds of the dimensionality
d_lower=100
d_upper=1000

# lower and upper bounds of the white noise standard deviation
sigma_lower=0.1
sigma_upper=2.0

# lower and upper bounds of the spherical distortion standard deviation
sigma_m_lower=0.1
sigma_m_upper=2.0

# number of bins
bins=[2, 5, 'square-root', 'sturges-formula', 'rice-rule']

# binning methods
binning_methods= ['eqw', 'eqf', 'kmeans', 'distortion_aligned']

# mutual information n_neighbors
mi_n_neighbors_simulation= [3, 7, 11, 21, 31, 41, 51]
mi_n_neighbors_feature_selection= [3, 7, 11, 21, 31]

# feature selection 
fs_n_repeats= 20
fs_n_splits= 5