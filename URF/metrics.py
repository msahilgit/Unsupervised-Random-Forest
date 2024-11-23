import numpy as np
import sklearn as skl
import copy as cp
from tqdm import tqdm
from multiprocessing import Pool
import sys
sys.path.append('./')
import utils 
from sklearn.metrics import confusion_matrix as cfm
from sklearn.ensemble import RandomForestClassifier as rfc




def distortion_euclidean(data, dtraj, centers):
    '''
    '''
    centers = centers[dtraj]
    dist = np.mean( np.linalg.norm( data - centers, axis=0) )
    return dist


def distortion2_euclidean(data, dtraj, n_jobs=1):
    '''
    '''
    global get_average_cluster_dist
    def get_average_cluster_dist(i):
        members = np.where( dtraj == dtraj[i] )[0]
        members = data[members]
        dist = np.linalg.norm( data[i] - members, axis=0)
        return np.mean(dist)

    with Pool(processes=n_jobs) as pool:
        dists = list( tqdm( pool.imap( get_average_cluster_dist, range(len(dtraj)) ), total=len(dtraj), desc='calculating: ' ) )

    return np.mean(dists)
    





def distortion2_pmt(pmt, dtraj, n_jobs=1):
    '''
    '''
    nobs = dtraj.shape[0]

    global get_average_cluster_pmt
    def get_average_cluster_pmt(i):
        members = np.where( dtraj == dtraj[i] )[0]
        members = np.delete(dtraj, i)
        members = [ utils.get_condenced_index(i, j, nobs) for j in members ]
        members = pmt[members]
        return np.mean(members)

    with Pool(processes=n_jobs) as pool:
        dists = list( tqdm( pool.imap( get_average_cluster_pmt, range(len(dtraj)) ), total=len(dtraj), desc='calculating:' ) )

    return np.mean(dists)

     


def silhouette_score_eucledian(data, dtraj, output_type='samples', n_jobs=1):
    '''
    CODING::
	1. intra-cluster distanes
	2. nearest neighbours
	3. inter-cluster distances
	4. Silhouette score calculation
    '''
    
    global find_intra
    intra = np.zeros_like(dtraj).astype(float)
    def find_intra(i):
        indices = np.where( dtraj == i )[0]
        for j in range(len(indices)):
            members = np.delete(indices, j)
            intra[indices[j]] = np.mean(  np.linalg.norm( data[indices[j]] - data[members], axis=1 )  )

    global find_neighbour
    aindices = np.arange( len(dtraj) )
    neighbours = np.zeros_like(dtraj)
    def find_neighbour(i):
        indices = np.delete(aindices, i)
        neighbour = np.argmin( np.linalg.norm( data[i], data[indices], axis=1 ) )
        if neighbour >= i:
            neighbour += 1
        neighbours[i] = dtraj[neighbour]

    global find_inter
    inter = np.zeros_like(dtraj).astype(float)
    def find_inter(i):
        members = np.where(dtraj == neighbours[i])[0]
        inter[i] = np.mean(  np.linalg.norm( data[i] - data[members], axis=1 )  )


    with Pool(processes=n_jobs) as pool:
        tqdm( pool.imap( find_intra, np.unique(dtraj) ), total=len(np.unique(dtraj)), desc='intra-cluster:' ) 

    with Pool(processes=n_jobs) as pool:
        tqdm( pool.imap( find_neighbour, range(len(dtraj)) ), total=len(dtraj), desc='neighbouring :' ) 

    with Pool(processes=n_jobs) as pool:
        tqdm( pool.imap( find_intra, range(len(dtraj)) ), total=len(dtraj), desc='inter-cluster:' ) 


    silhouette = (inter - intra) / np.max( np.column_stack((inter,intra)), axis=1 )

    if output_type == 'samples':
        return silhouette
    elif output_type == 'score':
        return np.mean(silhouette)
    else:
        print('output_type E [samples, score]')
        return silhouette

#    intra = np.zeros_like(dtraj)
#    for  i in np.unique(dtraj):
#        indices = np.where( dtraj == i)[0]
#        for j in range(len(indices)):
#            members = np.delete(indices, j)
#            intra[ indices[j] ] = np.mean(  np.linalg.norm( data[indices[j]] - data[members], axis=1 )  )
#
#
#    neighbours = np.zeros_like(dtraj)
#    aindices = np.arange( len(dtraj) )
#    for i in range(len(dtraj)):
#        indices = np.delete( aindices, i )
#        neighbours[i] = dtraj[ np.argmin( np.linalg.norm( data[i] - data[indices], axis=1) ) ]
#
#    inter = np.zeros_like(dtraj)
#    for i in range(len(dtraj)):
#        members = np.where(dtraj == neighbours[i] )[0]
#        inter[i] = np.mean(  np.linalg.norm( data[i] - data[members], axis=1 )  )
#
#    silhouette = np.zeros_like(dtraj)
#    inter - intra / max(inter,intra)
#


def time_lagged_covariance_matrix(trajs, lag, 
                                  tweighted=True, mean_free=False, symmeterized=True):
    '''
    calculates time lagged covariance matrix on a MD trajectory data
    
    INPUTS::
    trajs        - [l-2darr] input time profiles, list of 2d arrays
                            WARNING: for 1 dimensional data, convert to 2 dimensions
    lag          - [int] lag time in terms of timesteps
    tweighted    - [bool] d[True] weighted the covar by trajecotry weights
    mean_free    - [bool] d[false] if the data is already mean free
    symmeterized - [bool] d[True] if symmetric covariance matrix is to be obtained
    
    OUTPUTS::
    covar        - [2d-arr] covariance matrix
    
    '''
    
    ntrajs = len(trajs)
    ndim = trajs[0].shape[1]
    
    
    if any( len(i)-1 <= lag for i in trajs):
        raise ValueError('lag is too high')
        
    total = np.concatenate((trajs)).shape[0]
    if tweighted:
        tweighted = np.array([ i.shape[0]/total for i in trajs ])
    else:
        tweighted = np.array([ 1 for i in range(ntrajs) ])
    total = np.sum(tweighted)
    
    if not mean_free:
        trajs = [ i-np.mean(i, axis=0) for i in trajs ]
        
        
    covar = np.zeros((ndim, ndim))
    for i in range(ntrajs):
        trj = trajs[i]
        
        for d1 in range(ndim):
            for d2 in range(ndim):
                
                t1 = trj[:,d1][:len(trj)-lag]
                t2 = trj[:,d2][lag:]
                
                cv = np.sum(t1 * t2) / (len(trj)-lag-1)
                
                if d1==d2:
                    covar[d1,d2] = 0.5 * cv * tweighted[i]
                else:
                    covar[d1,d2] = cv * tweighted[i]
               
            
    covar = covar / total
    if symmeterized:
        covar = 0.5 * (covar + covar.T)
        
    return covar


    

def learning_coefficient(dtraj, model, xtest, ytest, 
        cfactor=2, b=0, a=1, 
        penalty=0):
    '''
    given hierarchial clustering labels (dtraj) and confusion matrix on test data (cft),
    this function estimates an internal metric named as learning coefficient(lc), predictable
    of functional efficiency of unsupervised random forest.

    **It assumes that draj (including ytest) is labelled as 0-n**

    MATHS:

        lc = ( cl*a + (1-cl)*b )* <f1>

        cl - hierarchial clustering efficiency

             |  1,  if card( n_i(dtraj)>cutoff ) >= 2   ; n_i(dtraj) - number of labels belonging to ith class
        cl = |
             |  0,  else
                                     cutoff = cfactor * sqrt(nobs/n_dtraj); nobs - no of datapoints, n_dtraj - no of class labels

        f1_i - f1 score on test data for class i
        p_i = precision
        r_i = recall
        f1_i = 2*(p_i + r_i) / (p_i + r_i)
        <f1> = <f1_i>|i, i are class labels with n_i(dtraj)>cutoff

        a, b - upper and lower bounds for lc, default are 1, 0

        penalty can be imposed, given the large number of under_cutoff label classes
        if cl=1
        lc = lc - n(under_cutoff)*penalty

    INPUTS:
        dtraj           - [1d-arr] integer numpy array of hirarchial clustering labels
        model           - [rfc] a trained random forest classifier
        xtest           - [1d-arr] test data
        ytest           - [1d-arr] test labels
        cfactor         - [float] d[2] cutoff factor
        a               - [float] d[1] upper bound for lc
        b               - [float] d[0] lower bound for lc
        penalty         - [float] d[0] impose penalty on under_cutoff labels, only if good clustering

    OUTPUTS:
        lc              - [float] the learning coefficient

    '''

    nobs = dtraj.shape[0]   #no of datapoitns
    cls = np.unique(dtraj)   # label classes
    nc = len(cls)            # no of label classes
    if nc < 2: raise ValueError('only one type of labels')
    pops = np.array([np.where(dtraj==i)[0].shape[0] for i in cls])  #datapoints belonging to each label class

    cutoff = cfactor * np.sqrt(nobs/nc)
    nc_above = np.where(pops > cutoff)[0]
    nc_below = np.where(pops <= cutoff)[0]

    if len(nc_above) >= 2:
        cle = 1              #hierarchial clustering efficiency
    else:
        cle = 0


    pred = model.predict(xtest)
    cft = cfm(ytest, pred)
    f1s = utils.get_f1_score(cft, output_type='all')
    f1s[np.isnan(f1s)] = 0

    if nc > len(f1s):
        missing = np.setdiff1d( cls, np.unique(np.concatenate((ytest,pred))) )
        f1s = np.insert( f1s, missing-np.arange(len(missing)), 0)

    f1s = np.mean(f1s[nc_above])

    lc = (cle*a + (1-cle)*b) * f1s - penalty*len(nc_below)

    return lc


    




