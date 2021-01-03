import numpy as np
from binning import *

def unique_binning(t):
    """
    Carries out the unique binning for a template
    
    Args:
         t (np.array): a template vector
        
    Returns:
        np.array: the unique binning vector
    """
    return np.digitize(t, np.unique(t), right=True)

def generate_t(d, spherical=False):
    """
    Generates a random template
    
    Args:
        d (int): dimensionality of the template
    
    Returns:
        np.array: the template
    """
    # choosing the type of the template
    typ= np.random.randint(3)
    if typ == 0:
        t= np.random.normal(size=d)
    elif typ == 1:
        t= np.random.rand(d)
    elif typ == 2:
        t= np.zeros(d)
        for i in range(d):
            if np.random.randint(2) == 0:
                t[i]= np.random.normal()
            else:
                t[i]= np.random.normal(loc=5.0)
    
    t= (t - np.min(t))/(np.max(t) - np.min(t))
    
    exponent= np.random.rand()*5 + 1
    
    if np.random.randint(2) == 0:
        t= t**exponent
    else:
        t= t**(1.0/exponent)
    
    if spherical:
        return t
    else:
        return np.round(t, decimals=3)

def generate_noisy_window(d, sigma= 1):
    """
    Generates a noisy window
    
    Args:
        d (int): the dimensionality of the window
        sigma (float): the standard deviation of the noise
        
    Returns:
        np.array: the noisy window
    """
    return np.random.normal(scale=sigma, size=d)    

def generate_S_from_binning(t_binning):
    """
    Generates slice matrix from a binning
    
    Args:
        t_binning (np.array): a binning vector
    
    Returns:
        np.array: the slice matrix
    """
    S= np.zeros(shape=(len(t_binning), len(np.unique(t_binning))))
    for i, t_ in enumerate(t_binning):
        S[i][t_]= 1
    return S

def generate_n_tau(t):
    """
    Generates the vector n_tau containing the umbers of unique elements
    
    Args:
        t (np.array): a template vector
    
    Returns:
        np.array: the n_tau vector
    """
    return np.unique(t, return_counts=True)[1]

def generate_tau(t):
    """
    Generates the tau vector of unique elements

    Args:
        t (np.array): a template vector
    
    Returns:
        np.array: the tau vector
    """
    return np.unique(t)

def generate_S_tau(t):
    """
    Generates the S_tau matrix for a template
    
    Args:
        t (np.array): the template vector
    
    Returns:
        np.array: the S_tau matrix
    """
    t_binning = unique_binning(t)
    return generate_S_from_binning(t_binning)

def generate_C(t, spherical=False, sigma_m=1.0, eigen_th= 0.01):
    """
    Generates a covariance matrix for distortion
    
    Args:
        t (np.array): the template
        spherical (bool): whether the distortion is spherical
        sigma_m (float): the standard deviation of a spherical distortion
        eigen_th (float): threshold on eigenvalues
    
    Returns:
        np.array: the covariance matrix
    """
    if not spherical:
        n_tau= generate_n_tau(t)
        tmp= np.random.rand(len(n_tau), len(n_tau))
        matrix= (tmp + tmp.T)/2.0
        eigv, eigw= np.linalg.eigh(matrix)
        eigv[eigv < eigen_th]= eigen_th
        eigv= eigv*(sigma_m*sigma_m)
        matrix= np.dot(np.dot(eigw.T, np.diag(eigv)), eigw)
        return matrix
    else:
        n_tau= generate_n_tau(t)
        matrix= np.eye(len(n_tau))*(sigma_m*sigma_m)
        return matrix

def generate_distorted_t(t, C, distortion_mean, sigma):
    """
    Generates the distorted template
    
    Args:
        t (np.array): the template vector
        C (np.array): the covariance matrix of the distortion
        sigma (float): the standard deviation of the white noise
    
    Returns:
        np.array: the distorted template
    """
    S_tau = generate_S_tau(t)
    
    m= np.random.multivariate_normal(mean= distortion_mean, cov=C)
    noise= generate_noisy_window(len(t), sigma)
    return np.dot(S_tau, m) + noise

def generate_A_from_S(S):
    """
    Generates the projection matrix A from S
    
    Args:
        S (np.array): slice matrix
    
    Returns:
        A (np.array): the projection matrix
    """
    return np.dot(np.dot(S, np.linalg.inv(np.dot(S.T, S))), S.T)

def generate_A_from_binning(t_binning):
    """
    Generates the projection matrix from binning
    
    Args:
        t_binning (np.array): the binning matrix
    
    Returns:
        A (np.array): the projection matrix
    """
    return generate_A_from_S(generate_S_from_binning(t_binning))

def generate_A_from_binning_eff(t_binning):
    S= generate_S_from_binning(t_binning)
    _, counts= np.unique(t_binning, return_counts=True)
    return np.dot(S, np.dot(np.diag(1.0/counts), S.T))
