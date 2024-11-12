import numpy as np
import sklearn as skl
import copy as cp
from tqdm import tqdm
from multiprocessing import Pool
import sys
sys.path.append('./')
import utils 
import proximity_matrix as pmt
import random
from sklearn.ensemble import RandomForestClassifier as rfc, RandomForestRegressor as rfr
import fastcluster
import gc
import time
from numba import njit, prange, set_num_threads




def _kmed_euclidean(data, 
                     centers=None, nclus=None, init='random', 
                     max_iter=None, min_dp_re=1, min_cl_re=1,
                     random_seed=42):
    '''
    Perform k-medoids clustering on given data.
        Based on euclidean distances
        RUN ON SINGLE CPU ONLY

    #INPUTS::
        data        - [arr] a 2-dimensional array of datapoint of shape (n,f), 
                                n=no of datapoints(dp), f=variables per dp
        centers     - [arr-int] d[None] array of indexes of datapoints, taken as 
                                        initial cluster centers
        nclus       - [int] d[None] no of cluster centers, if centers==None, 
                                    if nclus==None: nclus = sqrt(n)
        init        - [str] d[random] strategy to identify inital cluster centers
                            random - nclus dp choosen randomly 
                            pp_like - nclus dp choosen via strategy similar to kmeans++ 
                                        (well separated datapoints as initial cluster centers)
        max_iter    - [int] d[None] no of maximum iterations to be performed
                                    if None: max_iter = max[ 5000, floor(sqrt(n)) ]
        min_dp_re   - [int] d[1] no of minimum dp re-assigned to continue iteration   ***PREFFERED 
        min_cl_re   - [int] d[1] no of minimum cluster centers re-assigned to continue iteration
        random_seed - [int] d[42] random seed for reproducibity, numpy.random.seed()

    #OUTPUTS::
        centers     - [arr] optimized indices of cluster centers
        dtraj       - [arr] final assigned cluster labels

    '''
    np.random.seed(random_seed)

    nobs = data.shape[0]

    if centers != None:
        if isinstance(centers, np.ndarray) and centers.dtype == int and np.max(centers) < nobs:
            nclus = centers.shape[0]
            centers_old = cp.deepcopy(centers)
        else:
            raise ValueError('problem with centers')

    else:

        if nclus == None:
            nclus = int( np.sqrt(nobs) )
        elif not isinstance(nclus, int) and not nclus < nobs:
            raise ValueError('problem with nclus')


        if init == 'random':
            centers_old = np.random.randint(0, nobs, nclus)


        elif init == 'pp_like':

            centers_old = np.zeros(nclus).astype(int)
            centers_old[0] = np.random.choice(nobs)
            dist_old = np.linalg.norm( data[centers_old[0]] - data, axis=1)

            for i in range(1, nclus):
                probs = np.square(dist_old)
                probs = probs / np.sum(probs)
                
                centers_old[i] = np.random.choice(nobs, p=probs)

                dist_new = np.linalg.norm( data[centers_old[i]] - data, axis=1)
                dist_old = np.minimum( dist_old, dist_new)

        else:
            raise ValueError('init E [random, pp_like]')



    if max_iter == None:
        max_iter = np.max( [ 5000, int(np.sqrt(nobs)) ] )
    else:
        max_iter = int(max_iter)

    
    dtraj_old = np.zeros(( nobs )).astype(int)
    centers_old = centers_old[centers_old.argsort()]

    for i in tqdm(range(max_iter), desc='Iterating'):

        # Assign Data points to centers
        dtraj_new = np.zeros(( nobs )).astype(int)
        dcenters = data[centers_old]
        for i in range(nobs):
            dtraj_new[i] = centers_old[ np.argmin( np.linalg.norm( data[i] - dcenters, axis=1) ) ]


        dp_re = np.sum( dtraj_old != dtraj_new )
        if dp_re >= min_dp_re:
            dtraj_old = cp.deepcopy(dtraj_new)
        else:
            break



        # recalculates centers
        centers_new = []
        for i in np.unique(dtraj_new):
            indices = np.where( dtraj_new == i )[0]
            dists = np.zeros(( len(indices) ))
            for j in range(len(indices)):
                members = np.delete(indices, j)
                dists[j] = np.mean( np.linalg.norm(data[indices[j]] - data[members], axis=1) )

            centers_new.append( indices[np.argmin(dists)] )


        centers_new = np.array(centers_new)[np.argsort(centers_new)]
        cl_re = np.sum(centers_old != centers_new)
        if cl_re >= min_cl_re:
            centers_old = cp.deepcopy(centers_new)
        else:
            break

    _relabel(dtraj_new)

    return centers_new, dtraj_new





def _kmed_pmt(nobs, pmt, 
                     centers=None, nclus=None, init='random', 
                     max_iter=None, min_dp_re=1, min_cl_re=1,
                     random_seed=42):
    '''
    Perform k-medoids clustering on given data.
        Based on user provided distances (proximity matrix), as condenced matrix
        RUN ON SINGLE CPU ONLY

    #INPUTS::
        nobs        - [int] number of datapoints (n)
        pmt         - [arr] pre-calculated distances, condensed matrix
                                pmt.shape == ( (n*(n-1))/2 , )
        centers     - [arr-int] d[None] array of indexes of datapoints, taken as 
                                        initial cluster centers
        nclus       - [int] d[None] no of cluster centers, if centers==None, 
                                    if nclus==None: nclus = sqrt(n)
        init        - [str] d[random] strategy to identify inital cluster centers
                            random - nclus dp choosen randomly 
                            pp_like - nclus dp choosen via strategy similar to kmeans++ 
                                        (well separated datapoints as initial cluster centers)
                                        (not preferred :: if error - change random state)
        max_iter    - [int] d[None] no of maximum iterations to be performed
                                    if None: max_iter = max[ 5000, floor(sqrt(n)) ]
        min_dp_re   - [int] d[1] no of minimum dp re-assigned to continue iteration 
        min_cl_re   - [int] d[1] no of minimum cluster centers re-assigned to continue iteration
        random_seed - [int] d[42] random seed for reproducibity, numpy.random.seed()

    #OUTPUTS::
        centers     - [arr] optimized indices of cluster centers
        dtraj       - [arr] final assigned cluster labels

    '''

    np.random.seed(random_seed)

    if pmt.shape[0] != (nobs*(nobs-1))/2:
        raise ValueError('problem with nobs or pmt')

    if centers != None:
        if isinstance(centers, np.ndarray) and centers.dtype == int and np.max(centers) < nobs:
            nclus = centers.shape[0]
            centers_old = cp.deepcopy(centers)
        else:
            raise ValueError('problem with centers')

    else:

        if nclus == None:
            nclus = int( np.sqrt(nobs) )
        elif not isinstance(nclus, int) and not nclus < nobs:
            raise ValueError('problem with nclus')


        if init == 'random':
            centers_old = np.random.randint(0, nobs, nclus)


        elif init == 'pp_like':

            centers_old = np.zeros(nclus).astype(int)
            centers_old[0] = np.random.choice(nobs)
            dist_old = _get_all_pmts(centers_old[0], pmt, nobs)

            for i in range(1, nclus):
                probs = np.square(dist_old)
                probs = probs / np.sum(probs) 
                centers_old[i] = np.random.choice(nobs, p=probs)

                dist_new = _get_all_pmts( centers_old[i], pmt, nobs )
                dist_old = np.minimum( dist_old, dist_new)

        else:
            raise ValueError('init E [random, pp_like]')



    if max_iter == None:
        max_iter = np.max( [ 5000, int(np.sqrt(nobs)) ] )
    else:
        max_iter = int(max_iter)

    
    dtraj_old = np.zeros(( nobs )).astype(int)
    centers_old = centers_old[centers_old.argsort()]

    for i in tqdm(range(max_iter), desc='iterating:'):

        # Assign Data points to centers
        dtraj_new = np.zeros(( nobs )).astype(int)
        for i in range(nobs):
            try:
                dtraj_new[i] = centers_old[ np.argmin( pmt[ utils.get_condenced_indices(i, centers_old, nobs) ] ) ]
            except IndexError:
                dtraj_new[i] = i


        dp_re = np.sum( dtraj_old != dtraj_new )
        if dp_re >= min_dp_re:
            dtraj_old = cp.deepcopy(dtraj_new)
        else:
            break



        # recalculates centers
        centers_new = []
        for i in np.unique(dtraj_new):
            indices = np.where( dtraj_new == i )[0]
            dists = np.zeros(( len(indices) ))
            for j in range(len(indices)):
                members = np.delete(indices, j)
                dists[j] = np.mean( pmt[utils.get_condenced_indices(indices[j], members, nobs)] )

            centers_new.append( indices[np.argmin(dists)] )


        centers_new = np.array(centers_new)[np.argsort(centers_new)]
        cl_re = np.sum(centers_old != centers_new)
        if cl_re >= min_cl_re:
            centers_old = cp.deepcopy(centers_new)
        else:
            break

    _relabel(dtraj_new)

    return centers_new, dtraj_new





def _get_all_pmts(row, pmt, nobs):
    '''
    get all distances (columns) of a given row
    '''
    dists = np.arange(nobs)
    col = np.delete( dists, row )
    cind = utils.get_condenced_indices(row, col, nobs)
    dists[row] = 0
    dists[col] = pmt[cind]
    return dists


def _relabel(dtrj):
    '''
    relabel the cluster centers from 0 to n
    '''
    n=0
    for i in np.unique(dtrj):
        dtrj[np.where(dtrj == i)[0]] = n
        n+=1





def kmed_pmt(nobs, pmt, 
                     centers=None, nclus=None, init='random', 
                     max_iter=None, min_dp_re=1, min_cl_re=1,
                     random_seed=42, n_jobs=2):
    '''
    Perform k-medoids clustering on given data.
        Based on user provided distances (proximity matrix), as condenced matrix

    #INPUTS::
        nobs        - [int] number of datapoints (n)
        pmt         - [arr] pre-calculated distances, condensed matrix
                                pmt.shape == ( (n*(n-1))/2 , )
        centers     - [arr-int] d[None] array of indexes of datapoints, taken as 
                                        initial cluster centers
        nclus       - [int] d[None] no of cluster centers, if centers==None, 
                                    if nclus==None: nclus = sqrt(n)
        init        - [str] d[random] strategy to identify inital cluster centers
                            random - nclus dp choosen randomly 
                            pp_like - nclus dp choosen via strategy similar to kmeans++ 
                                        (well separated datapoints as initial cluster centers)
                                        (not preferred :: if error - change random state)
        max_iter    - [int] d[None] no of maximum iterations to be performed
                                    if None: max_iter = max[ 5000, floor(sqrt(n)) ]
        min_dp_re   - [int] d[1] no of minimum dp re-assigned to continue iteration  ***PREFERRED 
        min_cl_re   - [int] d[1] no of minimum cluster centers re-assigned to continue iteration
        random_seed - [int] d[42] random seed for reproducibity, numpy.random.seed()
        n_jobs      - [int] d[2] no of cpus to be used

    #OUTPUTS::
        centers     - [arr] optimized indices of cluster centers
        dtraj       - [arr] final assigned cluster labels

    '''

    np.random.seed(random_seed)

    if pmt.shape[0] != (nobs*(nobs-1))/2:
        raise ValueError('problem with nobs or pmt')

    if centers != None:
        if isinstance(centers, np.ndarray) and centers.dtype == int and np.max(centers) < nobs:
            nclus = centers.shape[0]
            centers_old = cp.deepcopy(centers)
        else:
            raise ValueError('problem with centers')

    else:

        if nclus == None:
            nclus = int( np.sqrt(nobs) )
        elif not isinstance(nclus, int) and not nclus < nobs:
            raise ValueError('problem with nclus')


        if init == 'random':
            centers_old = np.random.randint(0, nobs, nclus)


        elif init == 'pp_like':

            centers_old = np.zeros(nclus).astype(int)
            centers_old[0] = np.random.choice(nobs)
            dist_old = _get_all_pmts(centers_old[0], pmt, nobs)

            for i in range(1, nclus):
                probs = np.square(dist_old)
                probs = probs / np.sum(probs) 
                centers_old[i] = np.random.choice(nobs, p=probs)

                dist_new = _get_all_pmts( centers_old[i], pmt, nobs )
                dist_old = np.minimum( dist_old, dist_new)

        else:
            raise ValueError('init E [random, pp_like]')



    if max_iter == None:
        max_iter = np.max( [ 5000, int(np.sqrt(nobs)) ] )
    else:
        max_iter = int(max_iter)


    global _get_dtraj_
    def _get_dtraj_(i):
        try:
            return centers_old[ np.argmin( pmt[ utils.get_condenced_indices(i, centers_old, nobs) ] ) ]
        except IndexError:
            return i

    global _get_centers_
    def _get_centers_(c):
        indices = np.where( dtraj_new == c )[0]
        dists = np.zeros(( len(indices) ))
        for j in range(len(indices)):
            members = np.delete(indices, j)
            dists[j] = np.mean( pmt[utils.get_condenced_indices(indices[j], members, nobs)] )
        return indices[np.argmin(dists)]
    

    dtraj_old = np.zeros(( nobs )).astype(int)
    centers_old = centers_old[centers_old.argsort()]

    for i in tqdm(range(max_iter), desc='iterating:'):

        # Assign Data points to centers
        with Pool(processes=n_jobs) as pool:
            dtraj_new = np.array( list( pool.imap(_get_dtraj_, range(nobs) ) ) )


        dp_re = np.sum( dtraj_old != dtraj_new )
        if dp_re >= min_dp_re:
            dtraj_old = cp.deepcopy(dtraj_new)
        else:
            break


        # recalculates centers
        with Pool(processes=n_jobs) as pool:
            centers_new = np.array( list( pool.imap(_get_centers_, np.unique(dtraj_new) ) ) )


        centers_new = centers_new[np.argsort(centers_new)]
        cl_re = np.sum(centers_old != centers_new)
        if cl_re >= min_cl_re:
            centers_old = cp.deepcopy(centers_new)
        else:
            break


    _relabel(dtraj_new)

    return centers_new, dtraj_new







def kmed_euclidean(data, 
                     centers=None, nclus=None, init='random', 
                     max_iter=None, min_dp_re=1, min_cl_re=1,
                     random_seed=42, n_jobs=2):
    '''
    Perform k-medoids clustering on given data.
        Based on euclidean distances

    #INPUTS::
        data        - [arr] a 2-dimensional array of datapoint of shape (n,f), 
                                n=no of datapoints(dp), f=variables per dp
        centers     - [arr-int] d[None] array of indexes of datapoints, taken as 
                                        initial cluster centers
        nclus       - [int] d[None] no of cluster centers, if centers==None, 
                                    if nclus==None: nclus = sqrt(n)
        init        - [str] d[random] strategy to identify inital cluster centers
                            random - nclus dp choosen randomly 
                            pp_like - nclus dp choosen via strategy similar to kmeans++ 
                                        (well separated datapoints as initial cluster centers)
        max_iter    - [int] d[None] no of maximum iterations to be performed
                                    if None: max_iter = max[ 5000, floor(sqrt(n)) ]
        min_dp_re   - [int] d[1] no of minimum dp re-assigned to continue iteration   ***PREFFERED 
        min_cl_re   - [int] d[1] no of minimum cluster centers re-assigned to continue iteration
        random_seed - [int] d[42] random seed for reproducibity, numpy.random.seed()
        n_jobs      - [int] d[2] no of cpus to be used

    #OUTPUTS::
        centers     - [arr] optimized indices of cluster centers
        dtraj       - [arr] final assigned cluster labels

    '''

    np.random.seed(random_seed)

    nobs = data.shape[0]

    if centers != None:
        if isinstance(centers, np.ndarray) and centers.dtype == int and np.max(centers) < nobs:
            nclus = centers.shape[0]
            centers_old = cp.deepcopy(centers)
        else:
            raise ValueError('problem with centers')

    else:

        if nclus == None:
            nclus = int( np.sqrt(nobs) )
        elif not isinstance(nclus, int) and not nclus < nobs:
            raise ValueError('problem with nclus')


        if init == 'random':
            centers_old = np.random.randint(0, nobs, nclus)


        elif init == 'pp_like':

            centers_old = np.zeros(nclus).astype(int)
            centers_old[0] = np.random.choice(nobs)
            dist_old = np.linalg.norm( data[centers_old[0]] - data, axis=1)

            for i in range(1, nclus):
                probs = np.square(dist_old)
                probs = probs / np.sum(probs)
                
                centers_old[i] = np.random.choice(nobs, p=probs)

                dist_new = np.linalg.norm( data[centers_old[i]] - data, axis=1)
                dist_old = np.minimum( dist_old, dist_new)

        else:
            raise ValueError('init E [random, pp_like]')



    if max_iter == None:
        max_iter = np.max( [ 5000, int(np.sqrt(nobs)) ] )
    else:
        max_iter = int(max_iter)

    

    global _get_dtraj_
    def _get_dtraj_(i):
        return centers_old[ np.argmin( np.linalg.norm( data[i] - dcenters, axis=1) ) ]

    global _get_centers_
    def _get_centers_(i):
        return np.mean( np.linalg.norm( data[i] - data[indices[dtraj_new[i]]], axis=1 ) )

    dtraj_old = np.zeros(( nobs )).astype(int)
    centers_old = centers_old[centers_old.argsort()]
    for i in tqdm(range(max_iter), desc='Iterating'):

        # Assign Data points to centers
        dcenters = data[centers_old]
        with Pool(processes=n_jobs) as pool:
            dtraj_new = np.array( list( pool.imap(_get_dtraj_, range(nobs) ) ) )


        dp_re = np.sum( dtraj_old != dtraj_new )
        if dp_re >= min_dp_re:
            dtraj_old = cp.deepcopy(dtraj_new)
        else:
            break


        # recalculates centers

        indices = {}
        for i in np.unique(dtraj_new):
            indices[i] = np.where( dtraj_new == i )[0]

        with Pool(processes=n_jobs) as pool:
            dists = np.array( list( pool.imap(_get_centers_, range(nobs) ) ) )

        centers_new = []
        for i in indices.keys():
            centers_new.append( indices[i][np.argmin(dists[indices[i]])] )


        centers_new = np.array(centers_new)[np.argsort(centers_new)]
        cl_re = np.sum(centers_old != centers_new)
        if cl_re >= min_cl_re:
            centers_old = cp.deepcopy(centers_new)
        else:
            break


    _relabel(dtraj_new)

    return centers_new, dtraj_new






def _kmed_check(nobs, pmt, 
                     centers=None, nclus=None, init='random', 
                     max_iter=None, min_dp_re=1, min_cl_re=1,
                     random_seed=42):
    np.random.seed(random_seed)

    if pmt.shape[0] != (nobs*(nobs-1))/2:
        raise ValueError('problem with nobs or pmt')

    if centers != None:
        if isinstance(centers, np.ndarray) and centers.dtype == int and np.max(centers) < nobs:
            nclus = centers.shape[0]
            centers_old = cp.deepcopy(centers)
        else:
            raise ValueError('problem with centers')

    else:

        if nclus == None:
            nclus = int( np.sqrt(nobs) )
        elif not isinstance(nclus, int) and not nclus < nobs:
            raise ValueError('problem with nclus')


        if init == 'random':
            centers_old = np.random.randint(0, nobs, nclus)


        elif init == 'pp_like':

            centers_old = np.zeros(nclus).astype(int)
            centers_old[0] = np.random.choice(nobs)
            dist_old = _get_all_pmts(centers_old[0], pmt, nobs)

            for i in range(1, nclus):
                probs = np.square(dist_old)
                probs = probs / np.sum(probs) 
                centers_old[i] = np.random.choice(nobs, p=probs)

                dist_new = _get_all_pmts( centers_old[i], pmt, nobs )
                dist_old = np.minimum( dist_old, dist_new)

        else:
            raise ValueError('init E [random, pp_like]')



    if max_iter == None:
        max_iter = np.max( [ 5000, int(np.sqrt(nobs)) ] )
    else:
        max_iter = int(max_iter)

    
    dtraj_old = np.zeros(( nobs )).astype(int)
    centers_old = centers_old[centers_old.argsort()]

    for i in tqdm(range(max_iter), desc='iterating:'):

        # Assign Data points to centers
        dtraj_new = np.zeros(( nobs )).astype(int)
        for i in range(nobs):
            try:
                dtraj_new[i] = centers_old[ np.argmin( pmt[ utils.get_condenced_indices(i, centers_old, nobs) ] ) ]
            except IndexError:
                dtraj_new[i] = i


        dp_re = np.sum( dtraj_old != dtraj_new )
        if dp_re >= min_dp_re:
            dtraj_old = cp.deepcopy(dtraj_new)
        else:
            break



        # recalculates centers
        centers_new = []
        for i in np.unique(dtraj_new):
            indices = np.where( dtraj_new == i )[0]
            dists = np.zeros(( len(indices) ))
            for j in range(len(indices)):
                members = np.delete(indices, j)
                dists[j] = np.mean( pmt[utils.get_condenced_indices(indices[j], members, nobs)] )

            centers_new.append( indices[np.argmin(dists)] )

#####
        indices = {}
        for i in np.unique(dtraj_new):
            indices[i] = np.where(dtraj_new == i)[0]

        dists = np.zeros((nobs))
        for i in range(nobs):
            members = indices[dtraj_new[i]]
            members = members[members != i]
            dists[i] = np.mean



####


        centers_new = np.array(centers_new)[np.argsort(centers_new)]
        cl_re = np.sum(centers_old != centers_new)
        if cl_re >= min_cl_re:
            centers_old = cp.deepcopy(centers_new)
        else:
            break

    _relabel(dtraj_new)

    return centers_new, dtraj_new


##########################################################################################################################################
##########################################################################################################################################
########                                                                                                                      ############
########                                        HC CLUSTERING                                                                 ############  
########                                                                                                                      ############
##########################################################################################################################################
##########################################################################################################################################




class proximity_based_hierarchy:
    '''
    performs hierarchial clustering on data, using proximities ,derived from trained random forest models,
    as a distance metric.
    This protocol utilises nearest neighbour chain algorithm in combination with RF-proximities calculation.
    The proximity estimation occurs on the fly, hence preventing the saving of 2 copies of ^nC_2 length arrays.
    '''

    def __init__(self, models, data, algorithm=2, n_jobs=1):

        #checking trained classifiers
        if isinstance(models, (rfc, rfr) ) and hasattr(models, 'estimator_'):
            models = [models]
        elif isinstance(models, (list, np.ndarray) ) and all( hasattr(i, 'estimator_') for i in models ):
            pass
        else:
            raise TypeError('models should be trained Random Forest model or list of it')

        #checking data
        if not isinstance(data, np.ndarray) or len(data.shape) != 2 :
            raise ValueError('data should 2-dim array similar/same as training data for model')

        self.nobs = data.shape[0]

        try:
            forest = np.concatenate(([rf.estimators_ for rf in models]))
            self.ntrees = len(forest)
            self.leafs = np.column_stack(([tree.apply(data) for tree in forest]))
        except:
            raise ValueError('incompatibility between models and data')

        self.algorithm = algorithm
        self.training = False
        self.n_jobs = int(n_jobs)



    def cluster(self):
        set_num_threads(self.n_jobs)
        self.dendrogram = leafs_hc(self.leafs)
        self.training = True
            


    def dtraj(self, nids):
        if not self.training:
            return 'training is notperformed [ obj.cluster() ]'
        if isinstance(nids, int):
            return utils.get_hc_dtraj(self.dendrogram, nids=nids)
        else:
            return 'nids should be int'



    def get_output(self):
        if self.training:
            return self.dendrogram
        else:
            return None



    def close(self):
        del self.leafs
        del self.dendrogram
        gc.collect()
        



def leafs_hc(leafs):
    nobs = np.shape(leafs)[0]
    consider = np.ones(nobs, dtype=np.bool_)
    clusters = list( np.reshape( np.arange(nobs), (nobs,1) ) )

    nchain = np.empty(nobs, dtype=int)
    nlength = 0

    dgm = np.empty((nobs-1, 4))

    #hierarchy
    for k in tqdm( range(nobs-1), desc='RF-HC: ' ):

        #initiation
        if nlength == 0:
            nlength = 1
            nchain[0] = np.where(consider == True)[0][0]

        
        #extending nearest neighbour chain
        while True:
            x = nchain[nlength-1]
            y, dist = _min_true_partner2(x, consider, clusters, leafs)
            #y, dist = _min_true_partner1(x, consider, clusters, leafs)  #putting so many if

            if nlength > 1 and  y == nchain[nlength-2]:
                break

            nchain[nlength] = y
            nlength += 1

        nlength -= 2

        #update dendrogram
        if x>y:
            x,y = y,x

        dgm[k,0] = x
        dgm[k,1] = y
        dgm[k,2] = dist
        dgm[k,3] = len(clusters[x]) + len(clusters[y])

        #update clusters
        clusters[y] = np.concatenate((clusters[x], clusters[y]))
        consider[x] = False

    dgm = _rearrange(dgm)
    _relabel(dgm)

    return dgm



@njit(parallel=True)
def _min_true_partner1(x, consider, clusters, leafs):
    trues = np.where(consider == True)[0]
    trues = trues[~np.equal(trues,x)]
    nobs = np.shape(trues)[0]
    dists = np.empty(nobs)
    for k in range(nobs):
        dists[k] = _get_pmt_many_to_many(leafs, clusters[x], clusters[trues[k]])
    index = np.argmin(dists)
    return trues[index], dists[index]

@njit
def _get_pmt_many_to_many(leafs, di, dj):
    ntrees = np.shape(leafs)[1]
    nobs = len(di)
    pmt = 0
    for x in range(nobs):
        pmt += np.mean( np.sqrt( 1 - np.sum( np.equal(leafs[di[x]], leafs[dj]), axis=1)/ntrees ) )
    return pmt/nobs


@njit(parallel=True)
def _min_true_partner2(x, consider, clusters, leafs):
    di = np.shape(clusters[x])[0]
    nobs, ntrees = np.shape(leafs)
    dists = np.zeros((nobs))
    for k in prange(di):
        dists += np.sqrt( 1 - np.sum( np.equal(leafs[clusters[x][k]], leafs), axis=1)/ntrees )
    dists /= di

    trues = np.where(consider == True)[0]
    trues = trues[~np.equal(trues,x)]
    nobs = np.shape(trues)[0]
    pmt = np.empty(nobs)
    for k in prange(nobs):
        pmt[k] = np.mean(dists[clusters[trues[k]]])
    index = np.argmin(pmt)

    return trues[index], pmt[index]


def _rearrange(dgm):
    index = np.argsort(dgm[:,2])
    return dgm[index]



class linkage_union_find:
    '''
    for re-labelling the dendrogram data indices
    taken and modified from _hierarchy.pyx [LinkageUnionFold] function of scipy.cluster
    '''
    def __init__(self, n):
        self.parent = np.arange(2*n-1)
        self.next = n

    def merge(self, x, y):
        self.parent[x] = self.next
        self.parent[y] = self.next
        self.next += 1


    def find(self, x):
        p = x

        while self.parent[x] != x:
            x = self.parent[x]

        while self.parent[p] != x:
            p, self.parent[p] = self.parent[p], x

        return x


def _relabel(dgm):
    nobs = np.shape(dgm)[0]+1
    uf = linkage_union_find(nobs)

    for k in range(nobs-1):

        x, y = int(dgm[k,0]), int(dgm[k,1])
        xr, yr = uf.find(x), uf.find(y)

        if xr < yr:
            dgm[k,0], dgm[k,1] = xr, yr
        else:
            dgm[k,0], dgm[k,1] = yr, xr

        uf.merge(xr, yr)






class fit_predict_proximity_based_hierarchy:
    '''
    '''
    def __init__(self, models, data, data_size,
            random_state=0, n_jobs=1): 

        #checking trained classifiers
        if isinstance(models, (rfc, rfr) ) and hasattr(models, 'estimator_'):
            self.models = [models]
        elif isinstance(models, (list, np.ndarray) ) and all( hasattr(i, 'estimator_') for i in models ):
            self.models = models
        else:
            raise TypeError('models should be trained Random Forest model or list of it')

        #checking data
        if not isinstance(data, np.ndarray) or len(data.shape) != 2 :
            raise ValueError('data should 2-dim array similar/same as training data for model')

        self.data = data
        self.nobs = data.shape[0]

        #checking data_size
        if not 0.01 <= data_size <= 0.99:
            raise ValueError('0.01 <= data_size <= 0.99')
        self.data_size = data_size

        random.seed(random_state)
        np.random.seed(random_state)
        self.inds1 = np.random.permutation(self.nobs)
        self.inds2 = self.inds1[int(self.nobs*self.data_size):]
        self.inds1 = self.inds1[:int(self.nobs*self.data_size)]

        self.n_jobs = n_jobs



    def fit(self):
        prox = pmt.calculate_condenced_pmt_(self.models, self.data[self.inds1], self.n_jobs)
        self.dng = fastcluster.linkage(prox, method='average', preserve_input=False)


    def predict(self, nhc, nn=1):
        if not isinstance(nn, int) or nn < 1:
            raise ValueError('nn >= 1')
        self.nn = nn
        if not isinstance(nhc, int) or nhc < 2:
            raise ValueError('nhc >= 2')

        leafs1 = _get_leafs(self.models, self.data, self.n_jobs)
        leafs2 = leafs1[self.inds2]
        leafs1 = leafs1[self.inds1]

        labels1 = utils.get_hc_dtraj(self.dng, nids=nhc)
        self.labels = np.zeros((self.nobs)).astype(int)
        self.labels[self.inds1] = labels1

        set_num_threads(self.n_jobs)
        _ = _get_nn_labels(leafs1[:10], leafs2, labels1, self.nn) #warmup for njit
        self.labels[self.inds2] = _get_nn_labels(leafs1, leafs2, labels1, self.nn)

        return self.labels


    def close(self):
        for var in ['models', 'data', 'inds1', 'inds2', 'dng', 'labels']:
            if hasattr(self, var):
                delattr(self, var)
        gc.collect()


def _get_leafs(models, data, n_jobs):
    forest = np.concatenate(([model.estimators_ for model in models]))
    ntrees = len(forest)

    global operate_tree
    def operate_tree(i):
        return forest[i].apply(data)

    with Pool(processes=n_jobs) as pool:
        leafs = list( pool.imap(operate_tree, range(ntrees)) )

    return np.column_stack((leafs))


@njit(parallel=True)
def _get_nn_labels(leafs1, leafs2, labels1, nn):
    nobs, ntrees = np.shape(leafs2)
    labels2 = np.zeros((nobs)).astype(np.int64)

    for k in prange(nobs):
        neighbours = labels1[ np.argsort( np.sqrt( 1 - np.sum(np.equal(leafs2[k], leafs1), axis=1)/ntrees ) )[:nn] ]
        ids = np.unique(neighbours)
        pops = np.zeros(len(ids))
        for p in range(len(ids)):
            pops[p] = np.shape( np.where(neighbours==ids[p])[0] )[0]
        labels2[k] = ids[np.argmax(pops)]

    return labels2


