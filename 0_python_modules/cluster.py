import numpy as np
import sklearn as skl
import copy as cp
from tqdm import tqdm
from multiprocessing import Pool
import extras 



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
                dtraj_new[i] = centers_old[ np.argmin( pmt[ extras.get_condenced_indices(i, centers_old, nobs) ] ) ]
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
                dists[j] = np.mean( pmt[extras.get_condenced_indices(indices[j], members, nobs)] )

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
    cind = extras.get_condenced_indices(row, col, nobs)
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
            return centers_old[ np.argmin( pmt[ extras.get_condenced_indices(i, centers_old, nobs) ] ) ]
        except IndexError:
            return i

    global _get_centers_
    def _get_centers_(c):
        indices = np.where( dtraj_new == c )[0]
        dists = np.zeros(( len(indices) ))
        for j in range(len(indices)):
            members = np.delete(indices, j)
            dists[j] = np.mean( pmt[extras.get_condenced_indices(indices[j], members, nobs)] )
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
                dtraj_new[i] = centers_old[ np.argmin( pmt[ extras.get_condenced_indices(i, centers_old, nobs) ] ) ]
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
                dists[j] = np.mean( pmt[extras.get_condenced_indices(indices[j], members, nobs)] )

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


