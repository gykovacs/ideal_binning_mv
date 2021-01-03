import numpy as np
from data_generation import generate_A_from_binning_eff, generate_S_tau
import kmeans1d

epsilon= 1e-3

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
    t_u= sorted(np.unique(t))
    
    n_items= max(max(1, int(np.floor(len(t_u)/n_bins))), int(np.floor(len(t)/n_bins)))

    idx= 0
    for i in range(1, n_bins):
        idx+= n_items
        while idx < len(t) and ((len(t_bins) > 0 and t[idx] <= t_bins[-1]) or (i == 1 and t[idx] == t[0])):
            idx+= 1
        if idx < len(t):
            t_bins.append(t[idx])
        else:
            break
    t_bins.append(np.max(t) + epsilon)
    t_binning= np.digitize(t, t_bins)
    
    """
    try:
        
    except:
        print(t_bins)
        print(t)
        print(n_bins)
        raise ValueError("incraseing")
    if len(np.unique(t_binning)) != n_bins:
        print(len(np.unique(t_binning)))
        print(n_bins)
        print(np.unique(t_binning))
        raise ValueError("not equal")
    """
    return t_binning




def kmeans_binning(t, n_bins):
    """
    Carries out kmeans binning
    
    Args:
        t (np.array): the template
        n_bins (int): the number of bins
    
    Returns:
        np.array: the binning vector
    """
    clusters, centroids= kmeans1d.cluster(t, n_bins)

    return np.array(clusters)

def eqw_bins(t, n_bins):
    t_diff= (np.max(t) - np.min(t))/n_bins
    t_bins= np.hstack([np.array([np.min(t) + t_diff*i for i in range(1, n_bins)]), [np.max(t) + epsilon]])
    return t_bins

def eqw_binning(t, n_bins):
    """
    Carries out equal width binning
    
    Args:
        t (np.array): template to bin
        n_bins (int): number of bins
    
    Returns:
        np.array: the binning vector
    """
    
    t_bins= eqw_bins(t, n_bins)
    t_binning= np.digitize(t, t_bins)

    return t_binning

class DistortionAlignedBinning:
    def __init__(self, 
                 template, 
                 C, 
                 n_bins, 
                 random_state=5,
                 validate=False):
        """
        Constructor of the distortion aligned binning method
        
        Args:
            template (np.array): the template to bin
            C (np.ndarray): the cross-product matrix of the expected distortion
            n_bins (int): the number of bins
            random_state (int): the random state
            validate (bool): True for validation (computing the exact objective function)
        """
        self.template= template
        
        self.C= C
        self.n_bins= n_bins
        self.random_state= random_state
        self.validate= validate
        self.rs= np.random.RandomState(random_state)
        
        self.S_tau= generate_S_tau(template)
        self.SCS= np.dot(np.dot(self.S_tau, C), self.S_tau.T)
        self.tau, self.n_tau= np.unique(self.template, return_counts=True)
        
        # initial binning        
        self.tau_binning= kmeans_binning(self.tau, self.n_bins)
        
        #print(len(np.unique(self.tau_binning)))
        
        # if the initial binning could not identify n_bins, 
        # falling back to a simple linear binning
        if self.n_bins != len(np.unique(self.tau_binning)):
            self.tau_binning= np.hstack([np.arange(self.n_bins-1), 
                                         np.repeat(self.n_bins, 
                                                   len(self.tau) - self.n_bins - 1)])
         
        # determining the counts of the bins and their borders in terms of 
        # indices (lower inclusive)
        self.bin_counts= np.repeat(0, n_bins)
        self.bin_borders= np.repeat(0, n_bins+1)
        last_bin= -1
        for i, t in enumerate(self.tau_binning):
            if t != last_bin:
                self.bin_borders[t]= i
                last_bin= t
            self.bin_counts[t]+= 1
        self.bin_borders[-1]= len(self.tau)
        
        # computing the components of the objective function (numerators and
        # denominators for each bin)
        self.terms_num= np.zeros(n_bins)
        self.terms_denom= np.zeros(n_bins)
        
        for i in range(n_bins):
            self.terms_denom[i]= np.sum([self.n_tau[j] for j in range(self.bin_borders[i], self.bin_borders[i+1])])
            C_sub= self.C[self.bin_borders[i]:self.bin_borders[i+1], self.bin_borders[i]:self.bin_borders[i+1]]
            n_tau_sub= (self.n_tau[self.bin_borders[i]:self.bin_borders[i+1]])
            self.terms_num[i]= np.sum(np.dot(np.dot(n_tau_sub, C_sub), n_tau_sub))
        
        # computing the initial value of the objective function
        self.objective= np.inner(self.terms_num, 1.0/self.terms_denom)
        
    def compute_exact_objective(self, bin_counts= None):
        """
        Compute the exact objective (inefficient) at any iteration for
        the validation of the optimized method.
        
        Args:
            bin_counts (np.array): optional update bin counts
        
        Returns:
            float: the computed exact objective function value
        """
        if bin_counts is None:
            bin_counts= self.bin_counts
        
        tau_binning= []
        for i in range(len(bin_counts)):
            tau_binning.append(np.repeat(i, bin_counts[i]))
        self.t_binning= np.dot(self.S_tau, np.hstack(tau_binning)).astype(int)
        
        A= generate_A_from_binning_eff(self.t_binning)
        exact_objective= np.sum(np.multiply(self.SCS, A))
        return exact_objective
    
    def evaluate(self):
        """
        Evaluates the objective function with the sampled changes,
        the calculations are optimized by updating only that part
        of the objective function which has changed
        """
        new_objective= self.objective
        
        if self.upper:
            # bin_idx i shrinks to upper half, values from bin_idx : bin_idx + splitting_idx move to the bin below
            # bin i moves to bin_idx + splitting_idx
            bin_i_lower= self.bin_borders[self.bin_idx]
            bin_ip1_lower= self.bin_borders[self.bin_idx+1]
            bin_im1_lower= self.bin_borders[self.bin_idx-1]
            new_bin_i_lower= bin_i_lower + self.splitting_idx
            
            n_tau_change= self.n_tau[bin_i_lower:new_bin_i_lower]
            n_tau_sides= self.n_tau[new_bin_i_lower:bin_ip1_lower]
            n_tau_extension= self.n_tau[bin_im1_lower:bin_i_lower]

            C_sides= self.C[bin_i_lower:new_bin_i_lower,new_bin_i_lower:bin_ip1_lower]
            C_square= self.C[bin_i_lower:new_bin_i_lower, bin_i_lower:new_bin_i_lower]
            C_extension= self.C[bin_i_lower:new_bin_i_lower, bin_im1_lower:bin_i_lower]
            
            C_square_change= np.sum(np.dot(np.dot((n_tau_change), C_square), (n_tau_change)))
            
            self.term_num_change_i= -(2*np.sum(np.dot(n_tau_sides,np.dot(C_sides.T,n_tau_change))) + C_square_change)
            self.term_denom_change_i= -np.sum(n_tau_change)
            self.term_denom_change_im1= -self.term_denom_change_i
            self.term_num_change_im1= 2*np.sum(np.dot(n_tau_extension.T, np.dot(n_tau_change, C_extension))) + C_square_change
            
            new_objective= new_objective - self.terms_num[self.bin_idx]/self.terms_denom[self.bin_idx]
            if not self.bin_idx == 0:
                new_objective= new_objective - self.terms_num[self.bin_idx-1]/self.terms_denom[self.bin_idx-1]
            
            new_objective= new_objective + (self.terms_num[self.bin_idx] + self.term_num_change_i)/(self.terms_denom[self.bin_idx] + self.term_denom_change_i)
            if not self.bin_idx == 0:
                new_objective= new_objective + (self.terms_num[self.bin_idx-1] + self.term_num_change_im1)/(self.terms_denom[self.bin_idx-1] + self.term_denom_change_im1)
        else:
            # bin_idx i shrinks to lower half, values from bin_idx + splitting_idx : bin_idx + 1 move to the bin above
            # bin i+1's moves down to bin_idx + splitting_idx
            bin_i_lower= self.bin_borders[self.bin_idx]
            bin_ip1_lower= self.bin_borders[self.bin_idx+1]
            bin_ip2_lower= self.bin_borders[self.bin_idx+2]
            new_bin_ip1_lower= bin_i_lower + self.splitting_idx
            
            n_tau_change= self.n_tau[new_bin_ip1_lower:bin_ip1_lower]
            n_tau_sides= self.n_tau[bin_i_lower:new_bin_ip1_lower]
            n_tau_extension= self.n_tau[bin_ip1_lower:bin_ip2_lower]
            
            C_sides= self.C[new_bin_ip1_lower:bin_ip1_lower, bin_i_lower:new_bin_ip1_lower]
            C_square= self.C[new_bin_ip1_lower:bin_ip1_lower, new_bin_ip1_lower:bin_ip1_lower]
            C_extension= self.C[new_bin_ip1_lower:bin_ip1_lower, bin_ip1_lower:bin_ip2_lower]
            
            C_square_change= np.sum(np.dot(np.dot((n_tau_change), C_square), (n_tau_change)))
            
            self.term_num_change_i= -(2*np.sum(np.dot(n_tau_sides, np.dot(C_sides.T,n_tau_change))) + C_square_change)
            self.term_denom_change_i= -np.sum(n_tau_change)
            self.term_denom_change_ip1= -self.term_denom_change_i
            self.term_num_change_ip1= 2*np.sum(np.dot(n_tau_extension.T,np.dot(n_tau_change, C_extension))) + C_square_change
            
            new_objective= new_objective - self.terms_num[self.bin_idx]/self.terms_denom[self.bin_idx]
            if not self.bin_idx == self.n_bins - 1:
                new_objective= new_objective - self.terms_num[self.bin_idx+1]/self.terms_denom[self.bin_idx+1]
                
            new_objective= new_objective + (self.terms_num[self.bin_idx] + self.term_num_change_i)/(self.terms_denom[self.bin_idx] + self.term_denom_change_i)
            if not self.bin_idx == self.n_bins - 1:
                new_objective= new_objective + (self.terms_num[self.bin_idx+1] + self.term_num_change_ip1)/(self.terms_denom[self.bin_idx+1] + self.term_denom_change_ip1)
        
        if self.validate:
            # updating the bin borders
            bin_borders= self.bin_borders.copy()
            bin_counts= self.bin_counts.copy()
            if self.upper:
                bin_borders[self.bin_idx]+= self.splitting_idx
            else:
                bin_borders[self.bin_idx+1]= bin_borders[self.bin_idx] + self.splitting_idx
            
            # updating the bin counts
            if self.bin_idx == 0:
                pass
            else:
                bin_counts[self.bin_idx-1]= bin_borders[self.bin_idx] - bin_borders[self.bin_idx-1]
            if self.bin_idx == 0:
                bin_counts[self.bin_idx]= bin_borders[self.bin_idx+1]
            else:
                bin_counts[self.bin_idx]= bin_borders[self.bin_idx+1] - bin_borders[self.bin_idx]
            if self.bin_idx == self.n_bins-1:
                pass
            else:
                bin_counts[self.bin_idx+1]= bin_borders[self.bin_idx+2] - bin_borders[self.bin_idx+1]
            
            exact_objective= self.compute_exact_objective(bin_counts)
            print('optimized changed objective: %f, exact changed objective: %f' % (new_objective, 
                                                                                    exact_objective))
        
        return new_objective        
    
    def accept(self):
        """
        Accepting the changes by updating the precomputed components of the
        objective function
        """
        self.objective-= self.terms_num[self.bin_idx]/self.terms_denom[self.bin_idx]
        self.terms_num[self.bin_idx]+= self.term_num_change_i
        self.terms_denom[self.bin_idx]+= self.term_denom_change_i
        self.objective+= self.terms_num[self.bin_idx]/self.terms_denom[self.bin_idx]
        
        if not self.upper and not self.bin_idx == self.n_bins - 1:
            self.objective-= self.terms_num[self.bin_idx+1]/self.terms_denom[self.bin_idx+1]
            self.terms_num[self.bin_idx+1]+= self.term_num_change_ip1
            self.terms_denom[self.bin_idx+1]+= self.term_denom_change_ip1
            self.objective+= self.terms_num[self.bin_idx+1]/self.terms_denom[self.bin_idx+1]
        if self.upper and not self.bin_idx == 0:
            self.objective-= self.terms_num[self.bin_idx-1]/self.terms_denom[self.bin_idx-1]
            self.terms_num[self.bin_idx-1]+= self.term_num_change_im1
            self.terms_denom[self.bin_idx-1]+= self.term_denom_change_im1
            self.objective+= self.terms_num[self.bin_idx-1]/self.terms_denom[self.bin_idx-1]
        
        # updating the bin borders
        if self.upper:
            self.bin_borders[self.bin_idx]+= self.splitting_idx
        else:
            self.bin_borders[self.bin_idx+1]= self.bin_borders[self.bin_idx] + self.splitting_idx
        
        # updating the bin counts
        if self.bin_idx == 0:
            pass
        else:
            self.bin_counts[self.bin_idx-1]= self.bin_borders[self.bin_idx] - self.bin_borders[self.bin_idx-1]
        if self.bin_idx == 0:
            self.bin_counts[self.bin_idx]= self.bin_borders[self.bin_idx+1]
        else:
            self.bin_counts[self.bin_idx]= self.bin_borders[self.bin_idx+1] - self.bin_borders[self.bin_idx]
        if self.bin_idx == self.n_bins-1:
            pass
        else:
            self.bin_counts[self.bin_idx+1]= self.bin_borders[self.bin_idx+2] - self.bin_borders[self.bin_idx+1]
    
    def get_binning(self):
        """
        Assemble the current binning
        
        Returns:
            np.array: the binning vector
        """
        tau_binning= []
        for i in range(len(self.bin_counts)):
            tau_binning.append(np.repeat(i, self.bin_counts[i]))
        tau_binning= np.hstack(tau_binning)

        t_binning= np.dot(self.S_tau, tau_binning).astype(int)

        return t_binning
    
    def sample(self):
        """
        Sample random changes: a bin index, a splitting point within the bin,
        and whether the upper or lower half should be kept.
        """
        # pick a bin with more than 1 elements
        if len(self.tau) > self.n_bins:
            self.bin_idx= self.rs.randint(self.n_bins)
            
            while self.bin_counts[self.bin_idx] < 2:
                self.bin_idx= self.rs.randint(self.n_bins)
            
            if self.bin_counts[self.bin_idx] == 2:
                self.splitting_idx= 1
            else:
                self.splitting_idx= self.rs.randint(1, self.bin_counts[self.bin_idx]-1)
            
            if self.bin_idx == 0:
                self.upper= False
            elif self.bin_idx == self.n_bins-1:
                self.upper= True
            else:
                self.upper= self.rs.choice([True, False])

def distortion_aligned_binning(t, 
                               C, 
                               n_bins, 
                               maxit=1000,
                               annealing='auto',
                               t0= 1e-5,
                               return_it= False,
                               stopping_condition= lambda x: np.std(x[-20:]) == 0,
                               validate=False,
                               random_state=5):
    """
    Distortion aligned binning
    
    Args:
        t (np.array): the template to bin
        C (np.ndarray): the cross product matrix of the distortion
        n_bins (int): the number of bins
        maxit (int): the maximum number of iterations
        annealing (float/'auto'): the annealing of the system
        t0 (float): the absolute zero
        return_it (bool): whether to return the number of iterations
        validate (bool): whether to validate the optimized objective calculation
        random_state (int): the random state of the random search
    
    Returns:
        np.array, int: the binning vector and the number of iterations
    """
    tau= np.unique(t)
    if len(tau) < n_bins:
        n_bins= len(tau)
    
    dab= DistortionAlignedBinning(template=t, 
                                  C=C, 
                                  n_bins=n_bins, 
                                  validate=validate,
                                  random_state=random_state)
    
    if len(dab.tau) == n_bins:
        if return_it:
            return dab.get_binning(), 0
        else:
            return dab.get_binning()
    
    # initial objective function
    rs= np.random.RandomState(random_state)
    
    temperature= 1.0
    obj= -np.inf
    best_obj= -np.inf
    best_binning= dab.get_binning()
    it= 0
    objective_values= []
    
    if maxit == 'auto':
        maxit= int(np.log(t0)/np.log(annealing))
    elif annealing == 'auto':
        annealing= np.exp(np.log(t0)/maxit)
    
    while it < maxit:
        dab.sample()
        new_obj= dab.evaluate()
        
        if new_obj > obj or np.exp((new_obj - obj)/temperature) > rs.rand():
            dab.accept()
            obj= new_obj
            if obj > best_obj:
                best_obj= obj
                best_binning= dab.get_binning()

        objective_values.append(obj)
        temperature= temperature*annealing
        it= it + 1
        
        if (obj > -np.inf) and it > 1 and stopping_condition(objective_values):
            break
    
    if return_it:
        return best_binning, it
    else:
        return best_binning
a= 1
def square_root_rule(n):
    return int(np.ceil(np.sqrt(n)))

def sturges_formula(n):
    return int(np.ceil(np.log2(n))) + 1

def rice_rule(n):
    return min(n, int(np.ceil(2*n**(1/3))))

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

###################################################
# the greedy binning methods below are depracated #
# and replaced by the distortion aligned binning  #
###################################################

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
    
def greedy_binning(t, C, n_bins, maxit= 1000000):
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
    cache= {}
    while True:
        it+= 1
        
        change_obj, change_idx, step_, new_sum_i, new_sum_im1, new_ns_i, new_ns_im1= 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
        for i in range(1, n_bins-1):
            for step in [-1, 0]:
                if ns[i + step] > n_tau[bins[i] + step]:
                    bin_lower= bins[i-1]
                    bin_upper= bins[i+1]
                    key= (bin_lower, bin_upper, step, i, bins[i])
                    if key in cache:
                        change, sum_i, sum_im1, ns_i, ns_im1= cache[key]
                    else:
                        change, sum_i, sum_im1, ns_i, ns_im1 = changes(i, step*2 + 1, bins, C, n_tau, ns, sums)
                        cache[key]= (change, sum_i, sum_im1, ns_i, ns_im1)
                    
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
    
    return t_binning, it, d
