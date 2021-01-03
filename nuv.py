import numpy as np

from data_generation import *

def pwc_nuv(t, w, t_binning):
    """
    Piecewise constant approximation of the normalized Unexplained Variance measure.
    
    Let this measure be denoted by D(t,w,q). 1 - D(t,w,q)*var(w) is basically
    the r2 score of regressing w to t by a piecewise constant function.
    
    Args:
        t (np.array): template vector
        w (np.array): window vector
        t_binning (np.array): the binning vector
    
    Returns:
        float: the PWC nUV dissimilarity of t and w
    """

    # computing the optimal solution of the least squares approximation \hat\beta
    hat_beta= []
    for i in range(max(t_binning)+1):
        try:
            hat_beta.append(np.mean(w[t_binning == i]))
        except:
            print(i)
            print(t_binning)
            print(np.unique(t_binning))
            raise ValueError("empty slice")
    hat_beta= np.array(hat_beta)
    
    # reconstructing the (S . \hat\beta) vector (the vector approximating w)
    hat_w= hat_beta[t_binning]
    
    # computing the measure
    return np.sum((w - hat_w)**2)/(np.var(w)*len(w))

def exact_nuv_noise(d=10, b=3):
    """
    Computes the exact expectation of the dissimilarity of t and the white noise

    For more details, see Proposition 1 of the paper.
    
    Args:
        d (int): dimensionality of the window
        b (int): number of bins
    
    Returns:
        float: the expectation
    """
    return (d-b)/(d-1)

def exact_nuv_general(C, t, A, sigma, n_bins):
    """
    Computes the exact expectation of the dissimilarity of t and the distorted t

    For more details, see Proposition 2 of the paper.
    
    Args:
        C (np.array): the cross-product matrix
        t (np.array): the template
        A (np.array): the projection matrix
        sigma (float): the standard deviation of the white noise
        n_bins (int): the number of bins
    """
    
    num= 0
    denom= 0
    
    n_tau = generate_n_tau(t)
    S_tau = generate_S_tau(t)
    
    num+= np.dot(n_tau, np.diag(C)) - np.sum(A*np.dot(S_tau, np.dot(C, S_tau.T)))
    
    num+= sigma**2*(len(t) - n_bins)
    
    denom+= np.dot(n_tau, np.diag(C))/len(t)
    denom-= np.dot(n_tau, np.dot(C, n_tau))/len(t)**2
    denom+= sigma**2*(1 - 1/len(t))
    denom*= len(t)
    
    return num/denom

def exact_nuv_spherical(t, A, sigma, sigma_m, n_bins):
    """
    Exact value for the spherical distortion

    For more details, see Proposition 4 of the paper.

    Args:
        t (np.array): the template
        A (np.array): the ideal k-means clustering projection matrix
        sigma (float): the standard deviation of the white nosie
        sigma_m (float): the standard deviation of the diagonal distortion
        n_bins (int): the number of bins
        t_binning (np.array): 
    """
    num= 0
    denom= 0
    
    n_tau= generate_n_tau(t)
    
    num+= np.dot(np.dot(A, t) - t, np.dot(A, t) - t)
    num+= sigma_m**2*(len(t) - n_bins)
    num+= sigma**2*(len(t) - n_bins)
    denom+= len(t)*np.var(t) + sigma**2*(len(t) - 1)
    denom+= sigma_m**2*(len(t) - 1)
    
    return num/denom
