import os
import tqdm
import pandas as pd

from config import *

from binning import *
from data_generation import *
from nuv import *

def simulation(spherical=False,
               d_lower=d_lower,
               d_upper=d_upper,
               sigma_lower=sigma_lower,
               sigma_upper=sigma_upper,
               sigma_m_lower=sigma_m_lower,
               sigma_m_upper=sigma_m_upper,
               bins=bins,
               binning_methods= binning_methods,
               n_trials=n_trials,
               random_seed=random_seed):
    """
    The function implementing the numerical simulations

    Args:
        spherical (bool): whether the distortion is spherical or not
        d_lower (int): lower bound of the dimensionality
        d_upper (int): upper bound of the dimensionality
        sigma_lower (float): lower bound of the standard deviation
        sigma_upper (float): upper bound of the standard deviation
        sigma_m_lower (float): lower bound of the spherical distortion standard deviation
        sigma_m_upper (float): upper bound of the spherical distortion standard deviation
        bins (list): numbers of bins or n_bin selection strategies
        binning_methods (list): the names of the binning methods to be used
        n_trials (int): number of trials
    
    Returns:
        pd.DataFrame: the results of the simulation
    """

    # fixing the random seed
    np.random.seed(random_seed)

    # initialization
    exact_noise, exact_distortion, exact_kmeans= [], [], []
    d_noise, d_distortion, hits= {}, {}, {}
    ds, bs, b_mods, sigmas, sigma_ms= [], [], [], [], []
    
    for binning in binning_methods:
        d_noise[binning]= []
        d_distortion[binning]= []
        hits[binning]= []

    pbar = tqdm.tqdm(total=n_trials)   
    n_tests= 0
    
    # repeating the test case n_trials times
    while n_tests < n_trials:
        # random dimensionality
        d= np.random.randint(d_lower, d_upper)

        # random sigma for white noise
        sigma= sigma_lower + np.random.rand()*(sigma_upper - sigma_lower)

        # random sigma for spherical distribution (used only if spherical = True)
        sigma_m= sigma_m_lower + np.random.rand()*(sigma_m_upper - sigma_m_lower)
        
        # generating a template
        t= generate_t(d, spherical)

        # generating a covariance structure
        C= generate_C(t, spherical, sigma_m)
        # generating a mean vector
        distortion_mean= None
        if spherical:
            distortion_mean= generate_tau(t)
        else:
            distortion_mean= np.random.normal(size=len(C))
        cross_product= C + np.outer(distortion_mean, distortion_mean)
        A= None
        
        for b in bins:
            # for all number of bins specified

            # determining the true number of bins
            b_mod= n_bins(t, b)
            
            binnings= []
            for binning in binning_methods:
                # for all binning methods carry out the binning
                if binning == 'eqw':
                    t_binning = eqw_binning(t, b_mod)
                elif binning == 'eqf':
                    t_binning = eqf_binning(t, b_mod)
                elif binning == 'kmeans':
                    t_binning = kmeans_binning(t, b_mod)
                elif binning == 'greedy':
                    t_binning = greedy_binning(t, cross_product, b_mod)
                    A= generate_A_from_binning(t_binning)
                if len(np.unique(t_binning)) != b_mod:
                    # skipping the case if the template is such that the
                    # binning fails (EQW, because not all bins will contain at
                    # least one element which is a requirement for MTM)
                    break
                binnings.append(t_binning)
            
            if len(binnings) != len(binning_methods):
                # if any of the binnings did not succeed (EQW), continue with
                # the next test case
                continue
            
            # recording the dimensionality, the binning, the true number
            # of bins and the standard deviations of the noises
            ds.append(d)
            bs.append(b)
            b_mods.append(b_mod)
            sigmas.append(sigma)
            sigma_ms.append(sigma_m)
        
            # generating a noisy window
            w_noise= generate_noisy_window(d, sigma)

            # generating a distorted template
            w_distorted= generate_distorted_t(t, C, distortion_mean, sigma)
                
            for i, binning_method in enumerate(binning_methods):
                # for each binning compute the dissimilarity scores
                mtm_noise= pwc_nuv(t, w_noise, binnings[i])
                mtm_distorted= pwc_nuv(t, w_distorted, binnings[i])
                
                # record the dissimilarity scores
                d_noise[binning_method].append(mtm_noise)
                d_distortion[binning_method].append(mtm_distorted)
                
                # record the hit for pattern recognition
                if mtm_noise > mtm_distorted:
                    hits[binning_method].append(1)
                else:
                    hits[binning_method].append(0)
            
            # record the exact values
            exact_noise.append(exact_nuv_noise(d, b_mod))
            exact_distortion.append(exact_nuv_general(cross_product, t, A, sigma, b_mod))

            if spherical:
                exact_kmeans.append(exact_nuv_spherical(t, A, sigma, sigma_m, b_mod))
            else:
                exact_kmeans.append(-1)
        
        n_tests+= 1
        pbar.update(1)
        
    pbar.close()
    
    results= pd.DataFrame({'d': ds,
                           'b': bs,
                           'b_mods': b_mods,
                           'sigma': sigmas,
                           'sigma_m': sigma_ms,
                           'exact_noise': exact_noise,
                           'exact_distortion': exact_distortion,
                           'exact_kmeans': exact_kmeans})
    
    for b in binning_methods:
        results[b + '_noise']= d_noise[b]
        results[b + '_distorted']= d_distortion[b]
        results[b + '_hits']= hits[b]
    
    return results

def main():
    #######################
    # General distortions #
    #######################

    results_general= simulation(spherical=False)
    results_general.to_csv(os.path.join(work_dir, 'results_general.csv'), index=False)

    #########################
    # Spherical distortions #
    #########################

    results_spherical= simulation(spherical=True)
    results_spherical.to_csv(os.path.join(work_dir, 'results_spherical.csv'), index=False)


if __name__ == "__main__":
    main()

