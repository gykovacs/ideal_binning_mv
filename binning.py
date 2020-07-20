import numpy as np
from sklearn.cluster import KMeans

epsilon= 1e-8

def unique_binning(t):
    """
    Carries out the unique binning for a template
    
    Args:
         t (np.array): a template vector
        
    Returns:
        np.array: the unique binning vector
    """
    diff= np.unique(t)
    diff= diff[1:] - diff[:-1]
    diff = np.min(diff)/2
    return np.digitize(t, np.hstack([np.unique(t) + diff]))

def eqw_binning(t, n_bins):
    """
    Carries out equal width binning
    
    Args:
        t (np.array): template to bin
        n_bins (int): number of bins
    
    Returns:
        np.array: the binning vector
    """
    
    t_diff= (np.max(t) - np.min(t))/n_bins
    t_bins= np.hstack([np.array([np.min(t) + t_diff*i for i in range(1, n_bins)]), [np.max(t) + epsilon]])
    t_binning= np.digitize(t, t_bins)
    return t_binning

def eqf_binning(t, n_bins):
    """
    Carries out equal frequency binning
    
    Args:
        t (np.array): template to bin
        n_bins (int): number of bins
    
    Returns:
        np.array: the binning vector
    """
    t_bins= []
    t= sorted(t)
    n_items= int(len(t)/n_bins)

    for i in range(1, n_bins):
        t_bins.append(t[int(i*n_items)])
    t_bins.append(np.max(t) + epsilon)
    t_binning= np.digitize(t, t_bins)
    return t_binning

def kmeans_binning(t, n_bins, n_trials=20):
    """
    Carries out kmeans binning
    
    Args:
        t (np.array): the template
        n_bins (int): the number of bins
        n_trials (int): the number of trials
    
    Returns:
        np.array: the binning vector
    """
    best_clustering = None
    best_score = None
    
    for _ in range(n_trials):
        kmeans= KMeans(n_clusters=n_bins, random_state= np.random.randint(100))
        kmeans.fit(t.reshape(-1, 1))
        score= kmeans.score(t.reshape(-1, 1))
        if best_score is None or score > best_score:
            best_score= score
            best_clustering= kmeans.labels_
    
    clusters= np.unique(best_clustering)
    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            if np.mean(t[best_clustering == clusters[i]]) > np.mean(t[best_clustering == clusters[j]]):
                tmp_clustering= best_clustering.copy()
                tmp_clustering[best_clustering == clusters[j]]= clusters[i]
                tmp_clustering[best_clustering == clusters[i]]= clusters[j]
                best_clustering= tmp_clustering
    
    means= []
    for i in np.unique(best_clustering):
        means.append(np.mean(t[best_clustering == i]))
    
    return best_clustering

def block_sum(i, bins, C, n_tau):
    """
    Computes the sum of a block for greedy binning
    
    Args:
        i (int): bin index
        bins (np.array): bin boundary vector
        C (np.array): covariance matrix
        n_tau (np.array): vector of the number of unique elements
    
    Returns:
        float: the contribution of bin i
    """
    s= 0.0
    for j in range(bins[i], bins[i+1]):
        for k in range(bins[i], bins[i+1]):
            s+= C[j][k]*n_tau[j]*n_tau[k]
    return s

def row_col_sums(i, b_j, bins, C, n_tau):
    """
    Computes the contribution of row and column i for bin_j
    
    Args:
        i (int): index
        b_j (int): bin index
        bins (np.array): bin boundary vector
        C (np.array): covariance matrix
        n_tau (np.array): vector of the number of unique elements
    
    Returns:
        float: the contribution
    """
    s= C[i][i]*n_tau[i]*n_tau[i]
    for j in range(bins[b_j], bins[b_j+1]):
        if i != j:
            s+= (C[i][j] + C[j][i])*n_tau[i]*n_tau[j]
    return s

def changes(i, step, bins, C, n_tau, ns, sums):
    """
    Computes the changes in the objective function and temporary arrays
    
    Args:
        i (int): bin boundary index
        step (int): step being made
        bins (np.array): bin boundary vector
        C (np.array): covariance matrix
        n_tau (np.array): vector of the number of unique elements
        ns (np.array): temporary array of numbers
        sums (np.array): temporary array of sums
    
    Returns:
        float, float, float, float, float: the change in the objective function,
                    in sums[i], sums[i-1], ns[i], ns[i-1]
    """
    offset= int((step-1)/2)
    sum_i= sums[i] + (-1)*step*row_col_sums(bins[i]+offset, i, bins, C, n_tau)
    sum_im1= sums[i-1] + step*row_col_sums(bins[i]+offset, i-1, bins, C, n_tau)
    ns_i = (ns[i] - step*n_tau[bins[i]+offset])
    ns_im1 = (ns[i-1] + step*n_tau[bins[i]+offset])
    
    change= (sum_i)/ns_i - sums[i]/ns[i] + (sum_im1)/ns_im1 - sums[i-1]/ns[i-1]
    
    return change, sum_i, sum_im1, ns_i, ns_im1

def greedy_binning(t, C, n_bins, maxit= 1000):
    """
    Carries out greedy binning
    
    Args:
        t (np.array): the template vector
        C (np.array): the covariance matrix
        n_bins (int): the number of bins
        maxit (int): the maximum number of iterations
    
    Returns:
        np.array: the binning vector
    """
    b= n_bins
    _, n_tau= np.unique(t, return_counts=True)
    d= len(n_tau)
    cum_n_tau= np.hstack([[0], np.cumsum(n_tau)])
    tau= np.unique(t)
    tau= np.hstack([tau, [np.max(tau) + 0.1]])
    
    splits= sorted(np.random.randint(1, d, b-1))
    while len(np.unique(splits)) < b-1:
        splits= sorted(np.random.randint(1, d, b-1))    
    bins= np.array([0] + splits + [d])

    #print('cum', cum_n_tau)
    #print('bins', bins)
    
    sums= np.repeat(0.0, n_bins)

    for i in range(n_bins):
        sums[i]= block_sum(i, bins, C, n_tau)
    
    ns= np.repeat(0.0, n_bins)
    for i in range(n_bins):
        ns[i]= cum_n_tau[bins[i+1]] - cum_n_tau[bins[i]]
    
    objective= 0.0
    
    for i in range(n_bins):
        objective+= sums[i]/ns[i]

    cum_n_tau= np.hstack([[0], np.cumsum(n_tau)])
    
    it= 0
    while True and it < maxit:
        it+= 1
        
        change_obj, change_idx, step_, new_sum_i, new_sum_im1, new_ns_i, new_ns_im1= 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
        for i in range(1, n_bins):
            for step in [-1, 0]:
                if ns[i + step] > n_tau[bins[i] + step]:
                    change, sum_i, sum_im1, ns_i, ns_im1 = changes(i, step*2 + 1, bins, C, n_tau, ns, sums)
                    if change > change_obj:
                        change_obj, change_idx, step_, new_sum_i, new_sum_im1, new_ns_i, new_ns_im1= change, i, step*2 + 1, sum_i, sum_im1, ns_i, ns_im1
        
        if change_obj > 0.0:
            objective= objective + change_obj
            bins[change_idx]+= step_
            sums[change_idx]= new_sum_i
            sums[change_idx-1]= new_sum_im1
            ns[change_idx]= new_ns_i
            ns[change_idx-1]= new_ns_im1
        else:
            break
    
    t_binning= []
    for i in range(len(t)):
        for j in range(len(bins)):
            if t[i] >= tau[bins[j]] and t[i] < tau[bins[j+1]]:
                t_binning.append(j)
    t_binning= np.array(t_binning)
    
    #print('greedy', tau[bins])
    
    return t_binning

def square_root_rule(n):
    return int(np.ceil(np.sqrt(n)))

def sturges_formula(n):
    return int(np.ceil(np.log2(n))) + 1

def rice_rule(n):
    return int(np.ceil(2*n**(1/3)))

def n_bins(t, method):
    """
    Determins the number of bins
    
    Args:
        t (np.array): template
        method (int/str): the binning method
    """
    n= len(np.unique(t))
    
    if isinstance(method, int):
        return min([method, n])
    
    if method == 'square-root':
        return square_root_rule(n)
    if method == 'sturges-formula':
        return sturges_formula(n)
    if method == 'rice-rule':
        return rice_rule(n)
