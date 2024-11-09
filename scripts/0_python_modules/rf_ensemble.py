import numpy as np
import copy as cp
import tqdm as tq
import sys
sys.path.append('./')
import synthetic_data as syn
from sklearn.model_selection import train_test_split as splt
from sklearn.ensemble import RandomForestClassifier as rfc, RandomForestRegressor as rfr




class rf_clean:
    """
    """

    def init(self, data, dtype='single', 
            syn_data='permute', syn_size=1,
            syn_test = 0.3, syn_rstate=0,
            n_classes=3, min_class='sqrt', single_fold=1,   n_jobs=1):

        if dtype == 'single':
            if not len(data.shape) == 2: raise ValueError('problem with data')

            np.random.seed(syn_rstate)
            create_syn = syn.synthetic_data(data, size=syn_size)
            getattr(create_syn, syn_data)()
            pfeatures, plabels = create_syn.get_output()

            clfs = []
            for i in range(single_fold):
                xtrain, xtest, ytrain, ytest = splt(pfeatures, plabels, test_size=syn_test, random_state=syn_rstate)






        elif dtype == 'multi':
            if not is_3d(data): raise ValueError('problem with data')
            self.labels = np.concatenate(( [np.zeros(( data[i].shape[0]+i for i in range(len(data)) ))] ))
            self.features = np.concatenate(( data ))



        else:
            raise ValueError('dtype E [single, multi]')







def is_3d(data):
    if not isinstance(data, (list, np.ndarray)):
        return False
    if not all(len(i.shape)==2 for i in data):
        return False
    return True



def get_rf_clean_ensembles(model, data,
                        tree_weighted=False, op_type='accu', cutoff=0.9):
    '''
    this function predicts the class labels of given data (data) as per trained RF classifier (model)
    along with outliers defined by op_type and cutoff.

    INPUTS::
        model         - [rfc, list] single of list(array) of trained RF classifier models
        data          - [2d-arr] input data to predict class labels
        tree_weighted - [bool] d[False] in case, type(model)=list and different models have different number of trees
                                        this can average out the tree differences
        op_type       - [str] d[accu], 
                                    accu  - predict class labels based on "predict_proba >= cutoff"
                                    dsize - based on data size to be retained (0-1)
        cutoff        - [int] d[0.9] cutoff to be used in op_type

    OUTPUTS::
        labels        - [arr] the predicted labels of data based on given trained model
                                the labels are 0-n, nclasses trained on.
                                the outliers are labelled as NaN

    '''
    if not isinstance(data, np.ndarray) or len(data.shape) != 2 :
        raise ValueError('data(arg2) should 2-dim array similar/same as training data for model')
    nobs = len(data)

    if isinstance(model, rfc) and hasattr(model, 'estimators_'):
        model = [model]
    elif isinstance(model, (list, np.ndarray) ) and all( hasattr(i, 'estimators_') for i in model ):
        pass
    else:
        raise ValueError('problem with model')


    probs = np.array([ i.predict_proba(data) for i in model ])

    if tree_weighted:
        weights = [i.n_estimators for i in model]
    else:
        weights = [1 for i in model]
        
    probs = np.average(probs, axis=0, weights=weights)


    if op_type == 'accu':
        if not 1 > cutoff > 1/probs.shape[1]:
            raise ValueError('1 > cutoff > 1/n_labels')

        labels = probs >= cutoff
        amax = np.argmax(labels, axis=1)
        amin = np.argmin(labels, axis=1)

        labels = cp.deepcopy(amax)
        labels[ amax == amin ] = np.nan
        

    elif op_type == 'dsize':
        if not 1 > cutoff > 0:
            raise ValueError('1 > cutoff > 0')

        amax = np.argmax(probs, axis=1)
        amin = np.argmin(probs, axis=1)
        labels = cp.deepcopy(amax)
        labels[ amax == amin ] = np.nan

        vmax = probs[ np.arange(nobs), amax ]
        smax = np.argsort(vmax)[::-1]
        rval = int(cutoff * nobs)
        rval = smax[rval:]

        labels[rval] = np.nan


    else:
        raise ValueError('op_type E [accu, dsize]')


    return labels







