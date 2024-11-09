import copy as cp
import numpy as np
from tqdm import tqdm






def random_cluster_walk(ntrajs, tlength, ncv,
                        centers, centers_var, ctype='random', start=None,
                        residence=1, var=0.3, probs=1,
                        seed=None, less_mem=None, lout=False):
    '''
    generate random trajectories of multi-cluster walk in d-dimensional space
    
    THEORY::
        lets take, c clusters on a d-dimensional surface either in random or with a covraince
        a trajectory starts from a random datapoint t of a random cluster c i.e., x_c(t), then
            
                     | x_c(r), if i < tau _c 
            x(t+i) = | 
                     | x_p(r), if i >= tau _c    || x_i(r) ~ ith univariate d-dimensional distribution
                     
        given i >= tau_c (tau is controlled by residence) a random chance is applied to exit the 
        ith distribution. this random chance is analogous to free energy barrier
                     
                     | x_c,    if not random chance
            x_p    = |
                     | x_q,    if random chance   || random chance is controlled by 'var'
                     
        a new distribution is choosen randomly but governed by their probability of getting selected,
        analogous to downhills of free energy surface
        
                 q = i^p , p is probability of i  || p is controlled by probs
                 
        Note that, for
                 c = 1 (only one cluster)
        the above model becomes random walk, i.e.,
            x(t+i) = r(j), r(j) ~ N (single distribution)
               
                         
    
    INPUTS::
        ntrajs      - [int] no. of timelines to generate
        tlength     - [int,arr] timestep lengths of ntrajs (int, or int 1d-arr of len(ntrajs) )
        ncv         - [int] no of dimensions
        centers     - [int,arr] integer (no of centers), 1d-arr (center positions in uniform ncv space),
                                 2d-arr (center positions, shape[1]==ncv)
                                 (measure of metastable states)
        centers_var - [int,arr] center variance, same as centers, 
                                (+ 1d-arr (len=ncv)) - corr to feature importances
                                WARNING: if len(centers) = ncv: the above option is preferred
        ctype       - [str] E [random, covar] type of centers
                                random - the d dimensions has no correlation
                                covar  - with a particular d dimensional covariance
                                        for covar: 
                                            centers     - [2d-arr] of (n,cv) defining centers
                                            centers_var - [3d-arr] of (n,cv,cv) defining covariance
        start       - [int] d[None] the starting center                       
        residence   - [float,arr] [>0] d[1] no of minimum steps to maintain in a cluster, 
                                (measure of residence time) (WARNING: infinite loop for 0)
        var         - [float,arr] [0-1] d[0.3] probability of jumping to another cluster at next step
        probs       - [arr] [0-1] d[1] probability of choosing this cluster
                                            (measure of free energy barriers/surface)
        seed        - [int] d[None] random seed for reproducibility
        less_mem    - [int] d[None] if less memory (maybe tlength are too large), any positive integer number
        lout        - [bool] d[False] return labels
        
    OUTPUTS::
        data        - (ntrajs, tlength(i), ncv) trajectory data
        labels      - (ntrajs, tlength(i)) labels (indices of centers, if given)
        
    '''
    if type(seed)==int:
        np.random.seed(seed)
        
    if type(tlength)==int:
        tlength = [ tlength for i in range(ntrajs) ]
    elif not len(tlength)==ntrajs:
        raise ValueError('len(tlength) != ntrajs')

        
    maxi=np.max(tlength)
    if type(less_mem)==int:
        maxi=less_mem
    if ctype == 'random':
        if type(centers) == int:
            centers = np.random.randint(0, 2*centers, centers)

        if isinstance(centers_var, (float,int)) or ( isinstance(centers_var, (list,np.ndarray)) and len(centers_var) == ncv ):
            centers_var = [centers_var for i in centers ]

        dcenters = [
            np.random.normal( centers[i], centers_var[i], (maxi, ncv) )
        for i in range(len(centers))]
    
    
    elif ctype == 'covar':
        if not centers.shape[1]==ncv or not any(i.shape == (ncv,ncv) for i in centers_var):
            raise ValueError('problem with centers and/or centers_var')
        dcenters = [
            np.random.multivariate_normal(mean=centers[i], cov=centers_var[i], size=maxi)
        for i in range(len(centers))]
    
    
    else:
        raise ValueError('ctype E [random, covar]')
    
    
        
    if type(residence)==int:
        residence = [residence for i in centers]
    elif isinstance(residence, (list,np.ndarray)):
        if not len(residence)==len(centers): raise ValueError('residence != centers')
            
    if isinstance(var, (float,int)) and 0 <= var <= 1:
        var = [var for i in centers]
    elif isinstance(var, (list,np.ndarray)):
        if not len(var) == len(centers): raise ValueError('var != centers')
            
    if isinstance(probs, (float,int)) and 0 <= probs <= 1:
        probs = np.array([probs for i in centers]).astype(float)
    elif isinstance(probs, (list,np.ndarray)):
        if not len(probs)==len(centers): raise ValueError('probs != centers')
    probs /= np.sum(probs)
        
        
    
    
    data = []
    indices = np.arange(maxi)
    labels = []
    
    if type(start)==int and start < len(centers):
        n=start
    else:
        n = np.random.choice( np.arange(len(centers)), p=probs )
    
    for t in range(ntrajs):
        
        dd = dcenters[n][ np.random.choice( indices, residence[n] ) ]
        ll = np.zeros((residence[n])).astype(int) + n
        
        while len(dd) < tlength[t]:
            if np.random.random() > 1-var[n]:
                n = np.random.choice( np.arange(len(centers)), p=probs )
                
            dd = np.concatenate(( dd, dcenters[n][ np.random.choice(indices, residence[n]) ] ))
            ll = np.concatenate(( ll, np.zeros((residence[n])).astype(int) + n ))
            
        data.append(dd[:tlength[t]])
        labels.append(ll[:tlength[t]])
        
     
    if lout:
        return data, labels
    else:
        return data
                    
    
    

def random_additive_cluster_walk(ntrajs, tlength, ncv,
                        centers, centers_var, ctype='random', start=None,
                        residence=1, var=0.3, probs=1,
                        seed=None, less_mem=None):
    '''
    generate random trajectories of multi-cluster walk in d-dimensional space same as 
    FUNCTION "random_cluster_walk", except 
            x(t+i) = x(t) + x(t+i) [as per above function]
    based on c=1, this biols down to "random_additive_walk",
            x(t+i) = x(t) + r(j), r(j) ~ N
    see functions "random_additive_cluster_walk" and "random_additive_walk" for details.
    
    INPUTS::
        ntrajs      - [int] no. of timelines to generate
        tlength     - [int,arr] timestep lengths of ntrajs (int, or int 1d-arr of len(ntrajs) )
        ncv         - [int] no of dimensions
        centers     - [int,arr] integer (no of centers), 1d-arr (center positions in uniform ncv space),
                                 2d-arr (center positions, shape[1]==ncv)
                                 (measure of metastable states)
        centers_var - [int,arr] center variance, same as centers, 
                                (+ 1d-arr (len=ncv)) - corr to feature importances
                                WARNING: if len(centers) = ncv: the above option is preferred
        ctype       - [str] E [random, covar] type of centers
                                random - the d dimensions has no correlation
                                covar  - with a particular d dimensional covariance
                                        for covar: 
                                            centers     - [2d-arr] of (n,cv) defining centers
                                            centers_var - [3d-arr] of (n,cv,cv) defining covariance
        start       - [int] d[None] the starting center
        residence    - [float,arr] [>0] d[1] no of minimum steps to maintain in a cluster, 
                                (measure of residence time) (WARNING: infinite loop for 0)
        var         - [float,arr] [0-1] d[0.3] probability of jumping to another cluster at next step
        probs       - [arr] [0-1] d[1] probability of choosing this cluster
                                            (measure of free energy barriers/surface)
        seed        - [int] d[None] random seed for reproducibility
        less_mem    - [int] d[None] if less memory (maybe tlength are too large), any positive integer number
        
    OUTPUTS::
        data        - (ntrajs, tlength(i), ncv) trajectory data
        
    '''
    if type(seed)==int:
        np.random.seed(seed)
        
    if type(tlength)==int:
        tlength = [ tlength for i in range(ntrajs) ]
    elif not len(tlength)==ntrajs:
        raise ValueError('len(tlength) != ntrajs')

        
    maxi=np.max(tlength)
    if type(less_mem)==int:
        maxi=less_mem
    if ctype == 'random':
        if type(centers) == int:
            centers = np.random.randint(0, 2*centers, centers)

        if isinstance(centers_var, (float,int)) or ( isinstance(centers_var, (list,np.ndarray)) and len(centers_var) == ncv ):
            centers_var = [centers_var for i in centers ]

        dcenters = [
            np.random.normal( centers[i], centers_var[i], (maxi, ncv) )
        for i in range(len(centers))]
    
    
    elif ctype == 'covar':
        if not centers.shape[1]==ncv or not any(i.shape == (ncv,ncv) for i in centers_var):
            raise ValueError('problem with centers and/or centers_var')
        dcenters = [
            np.random.multivariate_normal(mean=centers[i], cov=centers_var[i], size=maxi)
        for i in range(len(centers))]
    
    
    else:
        raise ValueError('ctype E [random, covar]')
    
        
    if type(residence)==int:
        residence = [residence for i in centers]
    elif isinstance(residence, (list,np.ndarray)):
        if not len(residence)==len(centers): raise ValueError('residence != centers')
            
    if isinstance(var, (float,int)) and 0 <= var <= 1:
        var = [var for i in centers]
    elif isinstance(var, (list,np.ndarray)):
        if not len(var) == len(centers): raise ValueError('var != centers')
            
    if isinstance(probs, (float,int)) and 0 <= probs <= 1:
        probs = np.array([probs for i in centers]).astype(float)
    elif isinstance(probs, (list,np.ndarray)):
        if not len(probs)==len(centers): raise ValueError('probs != centers')
    probs /= np.sum(probs)
    
        
    
    
    data = []
    indices = np.arange(maxi)
    
    if type(start)==int and start < len(centers):
        n=start
    else:
        n = np.random.choice( np.arange(len(centers)), p=probs )
    
    for t in range(ntrajs):
        
        dd = [ dcenters[n][np.random.choice(indices)] ]
        
        while len(dd) < tlength[t]:
            if np.random.random() > 1-var[n]:
                n = np.random.choice( np.arange(len(centers)), p=probs )
                
            ddd = dcenters[n][ np.random.choice(indices, residence[n]) ]
            for k in range(len(ddd)):
                dd.append( dd[-1]+ddd[k] )
            
        data.append(dd[:tlength[t]])
        
     
    
    return data
                
       
    
    
    
    
def random_additive_walk(ntrajs, tlength, ncv,
               center=0, center_var=1,
               less_mem=None, seed=None):
    '''
    generate trajectories of random walk on a multi-dimensional surface
    
    THEORY::
            x(t+i) = x(t) + r(j)
            r(j) ~ N
                    N is d dimensional univariate distribution
                            
    INPUTS::
        ntrajs     - [int] no of ntrajs
        tlength    - [int,1d-arr] no of timesteps (datapoints) in each traj
        ncv        - [int] no of dimensions
        center     - [float,1d-arr] d[0] center of N (1d-arr of (ncv,) )
                                        FOR ADDITIVE - SUGGESTED == 0
        center_var - [float,1d-arr] d[1] var of N (1d-arr of (ncv,) )
        less_mem   - [int] d[None] any positive integer for less memory
        seed       - [int] d[None] random seed
        
    OUTPUTS::
        data       - (ntrajs, tlength(i), ncv) trajectory data
        
    '''
    if type(seed)==int:
        np.random.seed(seed)
        
    if type(tlength)==int:
        tlength = [tlength for i in range(ntrajs)]
        
    maxi=np.max(tlength)
    if less_mem==int:
        maxi=less_mem
    dcenter = np.random.normal(center, center_var, (maxi,ncv))     
    
    data = []
    indices = np.arange(maxi)
    for t in range(ntrajs):
        
        dd = [ dcenter[np.random.choice(indices)] ]
        while len(dd)<tlength[t]:
            dd.append( dd[-1] + dcenter[np.random.choice(indices)] )
            
        data.append(dd)
        
    return data



    
def random_walk(ntrajs, tlength, ncv,
               center=0, center_var=1,
               less_mem=None, seed=None):
    '''
    generate trajectories of random walk on a multi-dimensional surface
    
    THEORY::
            x(t+i) = r(j)
            r(j) ~ N
                    N is d dimensional univariate distribution
                            
    INPUTS::
        ntrajs     - [int] no of ntrajs
        tlength    - [int,1d-arr] no of timesteps (datapoints) in each traj
        ncv        - [int] no of dimensions
        center     - [float,1d-arr] d[0] center of N (1d-arr of (ncv,) )
        center_var - [float,1d-arr] d[1] var of N (1d-arr of (ncv,) )
        less_mem   - [int] d[None] any positive integer for less memory
        seed       - [int] d[None] random seed
        
    OUTPUTS::
        data       - (ntrajs, tlength(i), ncv) trajectory data
        
    '''
    if type(seed)==int:
        np.random.seed(seed)
        
    if type(tlength)==int:
        tlength = [tlength for i in range(ntrajs)]
        
    maxi=np.max(tlength)
    if less_mem==int:
        maxi=less_mem
    dcenter = np.random.normal(center, center_var, (maxi,ncv))     
    
    data = []
    indices = np.arange(maxi)
    for t in range(ntrajs):
        
        dd = [ dcenter[np.random.choice(indices)] ]
        while len(dd)<tlength[t]:
            dd.append( dcenter[np.random.choice(indices)] )
            
        data.append(dd)
        
    return data


        



def uncorrelated_cluster_walk(ntrajs, tlength, ncv,
                              centers=1, centers_var=10, start=None,
                              residence=1, var=0.3, probs=1, 
                              seed=None, less_mem=None, lout=None):
    '''
    generates trajectories of d dimensional cluster walk, with difference 
    from 'random_cluster_walk' as the d dimensions independently sample different clusters.
    
    Since, at any timestep t, the d dimensions can be at different clusters, its difficult to 
    label the output data. Nevertheless, a (t,d) labels can be generated or labels corr to 
    particular d or most common cluster can be generated. see lout.
    
    INPUTS::
        ntrajs      - [int] no. of timelines to generate
        tlength     - [int,arr] timestep lengths of ntrajs (int, or int 1d-arr of len(ntrajs) )
        ncv         - [int] no of dimensions
        centers     - [int,arr] integer (no of centers), 1d-arr (center positions in uniform ncv space),
                                 2d-arr (center positions, shape[1]==ncv)
        centers_var - [int,arr] center variance, same as centers, 
                                (+ 1d-arr (len=ncv)) - corr to feature importances
                                WARNING: if len(centers) = ncv: the above option is preferred
        start       - [int] d[None] the starting center 
        residence   - [int, ndarray] residence of centers (no of steps)
        var         - [float, ndarray] coefficient of jumping to another cluster
        probs       - [arr] [0-1] d[1] probability of choosing a cluster
                        NOTE: residence, var and probs: if array, then len() == no of centers
        seed        - [int] d[None] random seed
        less_mem    - [int] d[None] any positive integer for less memory
        lout        - [str,int] d[None] label the output trajectory
                        all - d dimensional
                        int - the dth label
                        -1  - most common label (with d > no of centers)
        
    OUTPUTS::
        data        - (ntrajs, tlength(i), ncv) trajectory data with added noise
        labels      - [ndarr] labels
        
    '''
    if seed != None:
        np.random.seed(seed)
    
    if type(tlength)==int:
        tlength = [ tlength for i in range(ntrajs) ]
    elif not len(tlength)==ntrajs:
        raise ValueError('len(tlength) != ntrajs')
    
    
    maxi=np.max(tlength)
    if isinstance(less_mem, int):
        maxi = less_mem
        
    if isinstance(centers, int):
        centers = np.random.randint(-10,10,centers)
        
    if isinstance(centers_var, (int,float)) or ( isinstance(centers_var, (list,np.ndarray)) and len(centers_var)==ncv ):
        centers_var = [centers_var for i in centers]
        
    dcenters = np.array([
        np.random.normal( centers[i], centers_var[i], (maxi,ncv) )
    for i in range(len(centers)) ])
    
    
    if isinstance(residence, int): residence = [residence for i in centers]
    if isinstance(var, (int,float)): var = [var for i in centers]
    if isinstance(probs, (float,int)): probs = np.array([probs for i in centers]).astype(float) 
    probs /= np.sum(probs)
    
    
    data = []
    indices = np.arange(maxi)
    labels = []
    if isinstance(start, int) and start < len(centers):
        n = np.array([start for i in range(ncv)])
    else:
        n = np.random.choice(np.arange(len(centers)), p=probs, size=ncv)
    
    for t in range(ntrajs):
        
        ddd = []
        lll = []
        for v in range(ncv):
            
            dd = dcenters[ n[v] ][np.random.choice(indices, residence[n[v]])][:,v]
            ll = np.zeros((residence[n[v]])).astype(int) + n[v]
            
            while len(dd) < tlength[t]:
                
                if np.random.normal() > 1 - var[n[v]]:
                    n[v] = np.random.choice(np.arange(len(centers)), p=probs)
                    
                dd = np.concatenate((dd, dcenters[n[v]][np.random.choice(indices, residence[n[v]])][:,v]))
                ll = np.concatenate((ll, np.zeros((residence[n[v]])).astype(int)+n[v]))
                
            ddd.append(dd)
            lll.append(ll)
            
        data.append(np.array(ddd).T)
        labels.append(np.array(lll).T)
        
    if lout == 'all':
        return data, labels
    
    elif lout == -1:
        labels = [i[np.arange(len(i)), np.argmax(i,axis=1)] for i in labels]
        return data, labels
        
    elif isinstance(lout, int) and 0<=lout<ncv:
        labels = [ i[:,lout] for i in labels ]
        return data, labels
        
    else:
        return data

    
    
    
def guided_cluster_walk(ntrajs, ncv,
                        centers, centers_var, ctype='random',
                        seed=None, less_mem=None):
    '''
    generate random d-dimensional datapoints for a given multi-cluster walk
    
    INPUTS::
        ntrajs      - [list of arr] input trajectories (integer labels [0..n])
        ncv         - [int] no of dimensions
        centers     - [arr] 1d-arr (center positions in uniform ncv space),
                            2d-arr (center positions, shape[1]==ncv)
        centers_var - [int,arr] center variance, same as centers, 
                                (+ 1d-arr (len=ncv)) - corr to feature importances
        ctype       - [str] E [random, covar] type of centers
                                random - the d dimensions has no correlation
                                covar  - with a particular d dimensional covariance
                                        for covar: 
                                            centers     - [2d-arr] of (n,cv) defining centers
                                            centers_var - [3d-arr] of (n,cv,cv) defining covariance
        seed        - [int] d[None] random seed for reproducibility
        less_mem    - [int] d[None] if less memory (maybe tlength are too large), any positive integer number
        
    OUTPUTS::
        data        - (ntrajs, tlength(i), ncv) trajectory data
        
    '''
    if type(seed)==int:
        np.random.seed(seed)
        
    tlength = [ len(i) for i in ntrajs ]
        
    maxi=np.max(tlength)
    if type(less_mem)==int:
        maxi=less_mem
    if ctype == 'random':
        if len(centers) != np.max(np.concatenate((ntrajs))):
            raise ValueError('centers != centers in ntrajs')

        if isinstance(centers_var, (float,int)) or ( isinstance(centers_var, (list,np.ndarray)) and len(centers_var) == ncv ):
            centers_var = [centers_var for i in centers ]

        dcenters = [
            np.random.normal( centers[i], centers_var[i], (maxi, ncv) )
        for i in range(len(centers))]
    
    
    elif ctype == 'covar':
        if not centers.shape[1]==ncv or not any(i.shape == (ncv,ncv) for i in centers_var):
            raise ValueError('problem with centers and/or centers_var')
        dcenters = [
            np.random.multivariate_normal(mean=centers[i], cov=centers_var[i], size=maxi)
        for i in range(len(centers))]
    
    
    else:
        raise ValueError('ctype E [random, covar]')
    
    
        
    data = []
    indices = np.arange(maxi)
    
    for t in range(len(ntrajs)):
        
        dd=[]
        for i in range(len(ntrajs[t])):
            
            dd.append( dcenters[ntrajs[t][i]][np.random.choice(indices)] )
            
        data.append(np.array(dd))
        
    
    return data
                    
    
    
    




def generate_variable_timeline(ntrajs, ncv, tlength, centers, centers_var, var, maintain, seed=None):
    '''
    ntrajs - int
    ncv - int
    tlength - int arr of length ntrajs
    centers - arr of single for ncv floats
    centers_var - arr of single for ncv floats
    var - float [0-1]
    maintain = int
    '''
    if type(seed) == int:
        np.random.seed(seed)
        
    dcenters = [
        np.random.normal( centers[i], centers_var[i], (np.max(tlength), ncv) )
    for i in range(len(centers))]
    
    data = []
    n = np.random.randint(0, len(centers))
    stick = 0
    for i in range(ntrajs):
        
        dd = []
        for j in range(tlength[i]):
            
            if stick >= maintain:
                if np.random.random() > 1-var:
                    n = np.random.randint(0, len(centers))
                stick = 0
            else:
                stick += 1
                
            dd.append( dcenters[n][np.random.randint(0, tlength[i])] )
            
        data.append(dd)
        
        
    return data


