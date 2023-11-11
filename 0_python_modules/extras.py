import numpy as np
import copy as cp
import warnings


def get_two_hierarchial_classes(hc):
    """
    This function returns top two classes, given a hierarchail clustering object.
    
    INPUT
        hc - [ndarray - (n-1,4)] hierarchial clustering object
    OUTPUT
        cl1, cl2 - [1d array] - members in class 1 and class2
    """
    
    try:
        if hc.shape[1] != 4:
            raise ValueError('problem with hc')
    except:
        return TypeError('hierarchial clustering object should have shape (n-1,4)')
    
    nobs = len(hc)+1        # number of datapoints
    
    d1, d2 = hc[-1][[0,1]].astype(int)        # ids of top two classes
    
    ncl1, ncl2 = get_size(hc,d1,nobs), get_size(hc,d2,nobs)    # expected number of members in top two classes
    if ncl1+ncl2 != nobs:
        return ValueError('members in two classes {a,b} did not equal nobs {c}'.format(a=ncl1,b=ncl2,c=nobs))
    
    cl1, cl2 = get_members(hc, d1, nobs, ncl1), get_members(hc, d2, nobs, ncl2)    # members in top two classes
    
    return cl1, cl2


def get_size(h,d,n):
    """
    this function_to_be_used calculates expected number of members in class (d)
    of hierarchial clustering object (h) with number of datapoints (n).
    """
    if d < n:
        return 1
    else:
        return h[d-n][3]

def get_members(h,d,n,exp):
    """
    This function_to_be_used calculates members in class id (d) in hierarchail clustering object (h)
    with number of datapoints (n) and validate with expected number of members (exp).
    """
    members = []
    if d < n:                                 # if less than n - then its the data index
        members.append(d)
    else:                                     # else - make to_iter list to search for other indexes
        to_iter = list(h[d-n][[0,1]].astype(int))
        for i in to_iter:
            if i < n:
                members.append(i)
            else:
                to_iter.extend(list(h[i-n][[0,1]].astype(int)))

    members = np.array(members)
    if len(members) == exp:
        return members
    else:
        return AssertionError('members found {a} are not equal to expected {b}'.format(a=len(members), b=exp))

    
def get_two_class_labels(cl1,cl2, l1=0, l2=1):
    """
    This function return class labels to be used for data labelling or representation.
    INPUTS
        cl1, cl2 - [1d-array] ids of class members (as given by function get_two_hierarchial_classes)
        l1, l2 - [int] d[0,1] - labels for cl1 and cl2
    OUTPUT
        labels [1d-array] class labels 0 (cl1) and 1 (cl2)
    """
    cl1 = np.zeros_like(cl1) + l1
    cl2 = cl2[cl2.argsort()]
    for i in cl2:
        cl1 = np.insert(cl1, i, l2)
    return cl1


def get_hc_dtraj(hc, nids=2, labels=None):
    '''
    this function assign labels to data, based on hierarchial clustering.
    #INPUTS::
	hc     - [n,4 arr] hierarchial clustering
	nids   - [int] d[2] number of classes to divide
	labels - [arr] d[None] list of labels to be assigned
    #OUTPUTS::
	dtraj  - [arr] linearized labels for data points
    '''
    if labels == None:
        labels = np.arange(nids)

    n = len(hc)+1

    ids = get_top_n_hc_ids(hc, nids)
    sizes = np.array([ get_size(hc, ids[i], n) for i in range(len(ids)) ])

    members = [ get_members(hc, ids[i], n, sizes[i]) for i in range(len(ids)) ]
    dtraj = [ np.zeros(( len(members[i]) ))+labels[i] for i in range(len(ids)) ]
    members = np.concatenate(( members )).argsort()
    dtraj = np.concatenate(( dtraj ))[members]

    return dtraj.astype(int)
   
    

def get_dtraj_outliers(dtraj, cutoff=None, return_new=False, display=True):
    '''
    this function removed/detect the dtraj outliers generally occured in hierarchial clustering (get_hc_dtraj function).
    #INPUTS::
	dtraj      - [arr]  the dtraj labels
	cutoff     - [int] d[None] cutoff number of observations - labels below cutoff are removed
				default = sqrt(data_length / n_clusters)
	return_new - [bool] d[F] also return a new dtraj label
	display    - [bool] d[T] display removed labels
    #OUTPUTS::
	oinds      - [arr] indices of outliers
	ndtraj     - [arr] new labels after removing outliers and renumbering the integer labels

    '''
    if not isinstance(cutoff, int):
        cutoff = int( np.sqrt( len(dtraj) / len(np.unique(dtraj)) ) )

    oinds = np.array([])
    retained = np.array([])
    removed = np.array([])
    for i in np.unique(dtraj):
        inds = np.where( dtraj == i )[0]
        if len(inds) < cutoff:
            oinds = np.append(oinds, inds)
            removed = np.append(removed, i)
        else:
            retained = np.append(retained, i)

    if display:
        print(f'removed labels: {removed}')

    oinds = oinds.astype(int)

    if return_new:
        ndtraj = cp.deepcopy(dtraj)

        p = 0
        for i in np.argsort(retained):
            ndtraj[ np.where(ndtraj == retained[i])[0] ] = p
            p += 1
   
        ndtraj = np.delete(ndtraj, oinds)
        
        return oinds, ndtraj.astype(int)

    else:
        return oinds


def get_dtraj_renumbered_outliers(dtraj, cutoff=None, display=True):
    '''
    this function replaces the outliers with non-outlier labels(based on their probabilities), in opposed to
    get_hc_dtraj which reduces the dtraj length.
    '''
    oinds = get_dtraj_outliers(dtraj, None, False, False)

    removed = np.unique(dtraj[oinds])
    if display:
        print(f'following labels are removed: {removed}')

    remaining = np.unique( np.delete(dtraj, oinds) )
    probs = np.zeros(( remaining.shape[0] ))
    for i in range(len(remaining)):
        probs[i] = ( np.where(dtraj == remaining[i])[0].shape[0] )
    probs = probs / np.sum(probs)

    ndtraj = cp.deepcopy(dtraj)
    ndtraj[oinds] = np.random.choice(remaining, oinds.shape[0], replace=True, p=probs)

    return ndtraj




def get_correct_two_class_labels(model, data, actual, predicted):
    """
    This function_in_comb_"get_two_class_labels" return the corrected class labels;
        in case the first function wrongly labelled the classes.
    This function takes a correctly predicted instance of "data" by "model" as per "actual"
    and used it to correct the "predicted" (if wrong) as given by "get_two_class_labels".
    """
    for i in range(len(data)):
        pred = model.predict([data[i]])
        
        if pred == actual[i]:
            if actual[i] != predicted[i]:
                predicted = 1 - predicted
            
            break
            
    return predicted
    

def get_correct_multi_class_labels(model, data, actual, predicted):
    '''
    This function_in_comb_"get_hc_dtraj" return the corrected class labels;
        in case the first function wrongly labelled the classes.
    This function takes a correctly predicted instance of "data" by "model" as per "actual"
    and used it to correct the "predicted" (if wrong).
    '''
    alabels = np.unique(actual)
    for i in alabels:
        indices = np.where( actual == i )[0]

        found = False
        for j in indices:
            pred = model.predict([data[j]])

            if pred == actual[j]:

                if pred != predicted[j]:
                    predicted[indices] = i

                found = True
                break

        if not found:
            raise AssertionError(f'model cannot predict all labels correctly: label-{i}')

    return predicted.astype(int)



def get_relabelled_class_labels(actual, predicted, dcut=0.5):
    '''
    this function relabels the predicted labels,
    assuming that actual and predicted dtrajs are generated by different methods and may be numbered differently,
    such that predicted is relabelled to match the actual...
    The procedure assume that predicted do represent actual, except different numbering -- hence tries to find out 
        which predicted labels best describe the actual labels.
        This may create artifact by randomly overfitting (to some extent) the predicted to actual. To check this, 
        some prelimnary but inadequate quality had been applied. 0<dcut<=1 is one such metric.
    '''
    if actual.shape[0] != predicted.shape[0]: raise ValueError('')

    alabels = np.unique(actual)
    plabels = np.unique(predicted)
    if alabels.shape[0] != plabels.shape[0]: raise AssertionError('actual and predicted should have same number of class labels')

    probs = []
    for i in alabels:
        indices = np.where(actual == i)[0]
        n_i = indices.shape[0]

        pbs = []
        for j in plabels:
            pbs.append(  np.where( predicted[indices] == j )[0].shape[0] / n_i  )

        probs.append(pbs)

    probs = np.array(probs)
    check_probs(probs, dcut)
    probs = np.argmax(probs, axis=1)
    plabels = plabels[probs]

    nlabels = np.zeros(( predicted.shape[0] ))
    for i in range(len(plabels)):
        nlabels[ np.where( predicted == plabels[i] )[0] ] = alabels[i]

    return nlabels.astype(int)

def check_probs(probs, dcut):
    maxi = np.argmax(probs, axis=1)
    if np.unique(maxi).shape[0] != maxi.shape[0]:
        raise ValueError('One-to-one relationship b/w actual and predicted cannot be established.')
    maxi = probs[ np.arange(len(probs)), maxi ]
    if any(maxi < dcut):
        print('some class labels are represented by lower than dcut')






def get_top_n_hc_ids(hc, nids):
    '''
    this function returns the top n ids in the hc clustering.
    '''
    n = len(hc)+1
    ids = hc[-1][[0,1]]
    
    while len(ids) < nids:
        imax = np.argmax(ids)
        vmax = ids[imax]
        ids = np.delete(ids, imax)
        ids = np.concatenate(( ids, hc[int(vmax-n)][[0,1]] ))
        
    return ids.astype(int)





def get_f1_score(cmt, output_type='min'):
    '''
    this function returns the f1 score given a confusion matrix (cmt).
    the f1 score is calculated from the perspective of all labels and returned based on
    output_type.
    '''
    if len(cmt) == 1:
        print('WARNING: only 1 label in cmt')
        return '1'
    else:
        f1s = np.zeros(( len(cmt) ))
        for i in range(len(cmt)):
            f1 = cmt[i,i] / np.sum(cmt[i])
            f2 = cmt[i,i] / np.sum(cmt[:,i])
            f1s[i] = ( 2 * f1 * f2 ) / ( f1 + f2 )

        if output_type == 'min':
            return np.min(f1s)
        elif output_type == 'nanmin':
            return np.nanmin(f1s)
        elif output_type == 'max':
            return np.max(f1s)
        elif output_type == 'mean':
            return np.mean(f1s)
        elif output_type == 'all':
            return f1s
        elif output_type == 'range':
            return [np.min(f1s), np.max(f1s)]
        else:
            raise ValueError('output_type E [min, nanmin, max, mean, all, range]')
    





def get_condenced_index(row, col, nobs):
    '''
    This function return return the condenced index number given the row and col indices
    of square matrix composed of nobs data point.
    '''
    if any(i >= nobs for i in [row,col]):
        raise ValueError('row, col < nobs')
    elif row == col:
        raise ValueError('row != col')
    elif row > col:
        row, col = col, row

    return int( np.sum([nobs - (i+1) for i in range(row)]) + (col-row) - 1 ), int( (row/2) * (2*nobs - row - 3) + col - 1 )


def get_condenced_indices(row, col, nobs):
    '''
    this function returns condenced indices corresponding to given row and multiple col numbers
    of squared matrix composed of nobs datapoints.

    ONLY INTEGER VALUES ALLOWED -- WRONG RESULT ON FLOATS

    MATHS:
        cind = \Sum{i=1}{row}{nobs-i} + (col-row) - 1
        cind = (row/2) * (2*nobs - row - 3) + col - 1

    INPUTS::
        row     - [int] row index of squared matrix (axis=0)
        col     - [arr] col indices of squared matrix (axis=1)
        nobs    - [int] dimension of squared matrix
    OUTPUTS::
        indices - [arr] output condenced indices

    '''

    col = np.array(col).astype(int)

    if row >= nobs or any(i >= nobs for i in col):
        raise ValueError('row, col < nobs')

    if np.where( col == row )[0].shape[0] > 0:
        raise IndexError('row != col')

    ci = np.where(col>row)[0]
    col[ci] = (row/2) * (2*nobs - row - 3) + col[ci] - 1
    
    ci = np.where(col<row)[0]
    col[ci] = (col[ci]/2) * (2*nobs - col[ci] - 3) + row - 1

    return col.astype(int)


    



def get_matrix_index(i, nobs):
    '''
    this function return the indices of square matric composed of nobs given condenced index i.
    '''
    if i >= ( (nobs-1) * (nobs-2) ) / 2:
        raise ValueError('large i - not possible')
    vals = np.cumsum( [nobs - (i+1) for i in range(nobs-1)] )
    row = np.where( vals <= i )[0][-1]
    col = i - vals[row] + row + 1
    return row+1, col+1




def get_label_correlation(l1, l2, 
                                otype='sankey', weighted=True):
    '''
    Calculates relationship of class labels from one labelling (l1) to another (l2) 
    via two algorithms i.e., sankey and gini impurity.
    class labels are output-ordered in ascending order.

    INPUTS::
        l1       - [arr] label-1, an array of 1-d labels
        l2       - [arr] label-2
        otype    - [str] type of correlation b/w labels
                         sankey - l1 weighted relationship to l2
                                  to be used to plot sankey plots, (used as links)
                         gini   - gini impurity of l1 classes, in terms of l2 classes
        weighted - [bool] d[True] reweighing of probabilities
                         sankey - probability reweighing such that correl represent the
                                    relative lengths of nodes in sankey plot
                         gini   - in case, the l2 is skewed, the probabilities are re-weighted as:
                                    MATHS:
                                        wp = (e * p / w) / \Sum(e * p / w) , e and w are equals and weights

    OUTPUTS::
        out      - [arr] 
                         sankey - (n,3) links for sankey plots replresenting source, target and weights
                         gini   - (n, ) gini impurities of l1 classes as per l2 classes.

    '''
    if not l1.shape == l2.shape or not len(l1.shape) == 1: raise ValueError('l1 and l2 to be of same shape (n,1)')

    u1, u2 = np.unique(l1), np.unique(l2)
    c1, c2 = len(u1), len(u2)
    w1, w2 = np.zeros(u1.shape), np.zeros(u2.shape)
    pis = np.zeros(( len(u1), len(u2) ))

    for i in range(c1):
        ind1 = np.where( l1 == u1[i] )[0]
        w1[i] = ind1.shape[0]/l1.shape[0]

        for j in range(c2):
            pis[i,j] = np.where( l2[ind1] == u2[j] )[0].shape[0] / ind1.shape[0]


    if otype == 'sankey':

        if weighted:
            pis = pis * np.column_stack( [w1 for i in range(c2)] )

        out = np.zeros(( c1*c2, 3 ))
        for i in range(c1):
            out[ i*c2 : (i+1)*c2, 0 ] = i
            out[ i*c2 : (i+1)*c2, 1 ] = np.arange(c2)+c1
            out[ i*c2 : (i+1)*c2, 2 ] = pis[i]


    elif otype =='gini':
        
        if weighted:
            for j in range(c2):
                w2[j] = np.where(l2 == u2[j])[0].shape[0] / l2.shape[0]

            w2 = np.vstack( [w2 for i in range(c1)] )
            equals = np.vstack( [np.ones((c2))/c2 for i in range(c1)] )
            pis = (pis * equals) / w2
            total = np.sum(pis, axis=1)
            total = np.column_stack( [total for i in range(c2)] )
            pis = pis / total

        out = np.array([ 1 - np.sum(np.square(i)) for i in pis ])


    else:
        raise ValueError('otype E [sankey, gini]')


    return out





