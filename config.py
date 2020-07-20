# working directory
work_dir='.'

# random seed to be used
random_seed=5

# number of trials
n_trials=5000

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
binning_methods= ['eqw', 'eqf', 'kmeans', 'greedy']
