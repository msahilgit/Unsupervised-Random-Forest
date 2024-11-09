import copy as cp
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier as rfc, RandomForestRegressor as rfr
import fastcluster
import gc
from tqdm import tqdm
from multiprocessing import Pool
import sys
sys.path.append('./')
import extras
import proximity_matrix as pmt
import time
from numba import njit, prange, set_num_threads



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
            return extras.get_hc_dtraj(self.dendrogram, nids=nids)
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
    def __init__(self, models, data, data_size, nn=1,
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

        if not 0.01 <= data_size <= 0.99:
            raise ValueError('0.01 <= data_size <= 0.99')
        self.data_size = data_size

        random.seed(random_state)
        np.random.seed(random_state)
        self.inds1 = np.random.permutation(self.nobs)
        self.inds2 = self.inds1[int(self.nobs*self.data_size):]
        self.inds1 = self.inds1[:int(self.nobs*self.data_size)]

        self.n_jobs = n_jobs

        if not isinstance(nn, int) or nn < 1:
            raise ValueError('nn >= 1')
        self.nn = nn


    def fit(self):
        prox = pmt.calculate_condenced_pmt_(self.models, self.data[self.inds1], self.n_jobs)
        self.dng = fastcluster.linkage(prox, method='average', preserve_input=False)


    def predict(self, nhc):
        if not isinstance(nhc, int) or nhc < 2:
            raise ValueError('nhc >= 2')

        leafs1 = _get_leafs(self.models, self.data, self.n_jobs)
        leafs2 = leafs1[self.inds2]
        leafs1 = leafs1[self.inds1]

        labels1 = extras.get_hc_dtraj(self.dng, nids=nhc)
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



