import os
#import tables as tb
import numpy as np
from tqdm import tqdm 
import sklearn as skl
import pickle as pkl
import sys
import scipy.spatial as scp
from multiprocessing import Pool
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.ensemble import RandomForestRegressor as rfr



def calculate_condenced_pmt_fast(model, data, output_type='horvath'):
    '''
    This function calculates proximity matrix of a given data
    #INPUTS::
        model       - [rfr] trained RF classifier
        data        - [arr] input data to calculate proximity - same/subset/full data used to train model
        output_type - [str] type of output
                            similarity    - probability of residing in same leaf node, measure of closeness
                            dissimilarity - math(1 - similarity) , measure of distance
                            horvath (def) - math( sqrt(1 - similarity) ), measure of distance-cite1

    #OUTPUT::
        pmt        - [arr] a condensed array of proximity distances, numbering based on sequence in input data

    cite1 - 10.1198/106186006X94072
    '''
    nobs = len(data)
    forest = model.estimators_
    ntrees = len(forest)
    pmt = np.zeros((nobs, nobs))
    
    for tree in tqdm(forest):
        dd = tree.apply(data)
        pmt += np.equal.outer(dd,dd) * 1
    
    pmt = pmt/ntrees
    
    if output_type == 'horvath':
        pmt = np.sqrt(1 - pmt)
    elif output_type == 'dissimilarity':
        pmt = 1 - pmt
    elif output_type == 'similarity':
        for i in range(len(pmt)):
            pmt[i,i] = 0
    else:
        raise ValueError('output_type E [horvath, dissimilarity, similarity]')
    
    pmt = scp.distance.squareform(pmt)
    
    return pmt




def calculate_condenced_pmt_slow(model, data, output_type='horvath'):
    '''
    This function calculates proximity matrix of a given data
    This function calculates proximities for individual datapoints, hence requiring less memory but more time.
    #INPUTS::
        model       - [rfr] trained RF classifier
        data        - [arr] input data to calculate proximity - same/subset/full data used to train model
        output_type - [str] type of output
                            similarity    - probability of residing in same leaf node, measure of closeness
                            dissimilarity - math(1 - similarity) , measure of distance
                            horvath (def) - math( sqrt(1 - similarity) ), measure of distance-cite1

    #OUTPUT::
        pmt        - [arr] a condensed array of proximity distances, numbering based on sequence in input data

    cite1 - 10.1198/106186006X94072
    '''
    forest = model.estimators_
    pmt = np.array([])
    nobs = len(data)
    
    for row in tqdm(range( nobs - 1 )):
        
        dists = np.zeros(( nobs - (row+1) ))
        
        for tree in forest:
            
            leafs = tree.apply(data[row:])
            
            for col in range(1, len(leafs)):
                
                if leafs[0] == leafs[col]:
                    dists[col-1] += 1
                    
        if output_type == 'horvath':
            dists = np.sqrt( 1 - dists/len(forest) )
        elif output_type == 'dissimilarity':
            dists = 1 - dists/len(forest)
        elif output_type == 'similarity':
            dists = dists/len(forest)
        else:
            raise ValueError('output_type E [horvath, dissimilarity, similarity]')
                
        pmt = np.concatenate((pmt, dists))
        
    return pmt


def calculate_condenced_pmt_multi(model, data, output_type='horvath', n_jobs=1):
    '''
    This function calculates proximity matrix of a given data
    This function calculates proximities as `calculate_condenced_pmt_slow` but on multiple cpus, hence faster and requires low memory: 
    still slower than `calculated_concenced_pmt_fast` unless 116 cpu cores are used.
    #INPUTS::
        model       - [---] trained RF classifier,regressor or list of it, or leaf nodes
        data        - [arr] input data to calculate proximity - same/subset/full data used to train model
        output_type - [str] type of output
                            similarity    - probability of residing in same leaf node, measure of closeness
                            dissimilarity - math(1 - similarity) , measure of distance
                            horvath (def) - math( sqrt(1 - similarity) ), measure of distance-cite1
        n_jobs      - [int] d[1] number of cpu cores to be used.

    #OUTPUT::
        pmt        - [arr] a condensed array of proximity distances, numbering based on sequence in input data

    cite1 - 10.1198/106186006X94072
    '''

    # checking data
    if not isinstance(data, np.ndarray) or len(data.shape) != 2 : 
        raise ValueError('data(arg2) should 2-dim array similar/same as training data for model')
    nobs = len(data)

    #checking models and or leafs
    found_aleafs = False
    if isinstance(model, (rfc, rfr) ) and hasattr(model, 'estimator_') :
        model = [model]
    elif isinstance(model, (list, np.ndarray) ) and all( hasattr(i, 'estimator_') for i in model ):
        pass
    elif isinstance(model, np.ndarray) and np.issubdtype(model.dtype, np.integer) and model.shape[0] == data.shape[0]:
        aleafs = model
        ntrees = model.shape[1]
        found_aleafs = True
    else:
        raise ValueError('model(arg1) should be trained RF model(s) or column_stacked leaf nodes')


    # getting leafs if not given
    if not found_aleafs:
        forest = []
        for i in model:
            forest = np.concatenate(( forest, i.estimators_ ))

        ntrees = len(forest)

        global operate_tree
        def operate_tree(i):
            return forest[i].apply(data)

        with Pool(processes=n_jobs) as pool:
            aleafs = list( tqdm( pool.imap(operate_tree, range(ntrees)), total=ntrees, desc='getting leafs' ) )

        aleafs = np.column_stack(( aleafs ))


    # calculating proximities
    global operate_row
    def operate_row(row):
        dists = aleafs[row] == aleafs[row+1:]
        dists = np.sum( dists * 1, axis=1)

        if output_type == 'horvath':
            dists = np.sqrt(1 - dists/ntrees)
        elif output_type == 'dissimilarity':
            dists = 1 - dists/ntrees
        elif output_type == 'similarity':
            dists = dists/ntrees
        else:
            raise ValueError('output_type E [horvath, dissimilarity, similarity]')

        return dists

    
    with Pool(processes=n_jobs) as pool:
        pmt = list( tqdm( pool.imap(operate_row, range(nobs-1) ), total=nobs-1, desc='getting dists') )

    #pmt = np.concatenate(pmt)

    return np.concatenate(( pmt ))



def get_leafs(model_path, nmodel, data, suffix='model_'):
    for i in range(nmodel):
        model_ = pkl.load(open(f'{model_path}/{suffix}{i}.pkl','rb'))
        dd = model_.apply(data)
        if i == 0:
            aleafs = cp.deepcopy(dd)
        else:
            aleafs = np.column_stack(( aleafs, dd ))

    return aleafs





def calculate_pmt_individual(model, di, dj, output_type='horvath'):
    '''
    this function calculates RF based proximity between two datapoints ( di, dj ).
         see calculate_condenced_pmt_multi for details.
    '''
    leafs = model.apply([di,dj])
    dists = leafs[0] == leafs[1]
    dists = np.sum(dists * 1)

    if output_type == 'horvath':
        dists = np.sqrt( 1 - dists/leafs.shape[1])
    elif output_type == 'dissimilarity':
        dists = 1 - dists/leafs.shape[1]
    elif output_type == 'similarity':
        dists = dists/leafs.shape[1]

    return dists


def calculate_pmt_one_to_many(model, di, dj, output_type='horvath'):
    '''
    this function calculates RF based proximities from one data point (di) to multiple others (dj).
         see calculate_condenced_pmt_multi for details.
    '''
    leafs_i = model.apply([di])
    leafs_j = model.apply(dj)
    dists = leafs_i == leafs_j
    dists = np.sum( dists * 1, axis=1)

    if output_type == 'horvath':
        dists = np.sqrt( 1 - dists/leafs.shape[1])
    elif output_type == 'dissimilarity':
        dists = 1 - dists/leafs.shape[1]
    elif output_type == 'similarity':
        dists = dists/leafs.shape[1]

    return dists



        
def calculate_large_pmt(model, data, output_type='horvath', 
                        output_file='pmat.h5', suffix='row', compression_level=9, compression_type='blosc',
                       output=False):
    '''
    This function calculates proximity matrix (preferred for large datasets) and saved in
    condensed format in .h5 pytables file format.
    
    #INPUTS
        model       - [clf ] trained random forest model
        data        - [nd-a] (2dim) 2dimensional data
        output_type - [str] d[horvath] type of proximity matrix calculation
                           similarity    - tree with datapoints i, j in same leaf node / total trees in forest
                           dissimilarity - 1 - similarity
                           horvath       - sqrt(1-similarity)
        output_file - [str] d[pmat.h5]
        suffix      - [str] d[row] suffix to name data rows in output_file
        compression_level - [str] d[9]
        compression_type  - [str] d[blosc] same as implemented in pytables
        
    #OUTPUTS
        output_file - [output_file] a proximity matrix saved in condensed form in pytables format
                                    readable by pmt_read() or pmt_update().
    
    '''
    
    if type(model) != skl.ensemble._forest.RandomForestClassifier: raise TypeError('model not RF classifier')
    if len(data.shape) != 2: raise ValueError('data is not 2dim array')
    
    if os.path.exists(output_file):
        raise ValueError('output file "{a}" already exists: Not preferable'.format(a=output_file))
    else:
        hfile = tb.open_file(output_file, 'w')
        hfilters = tb.Filters(complevel=compression_level, complib=compression_type)
        
    
    nobs = len(data)
    for row in tqdm(range(nobs-1)):
        elements = nobs - (row+1)
        dd = np.zeros((elements))
        hput = hfile.create_carray(hfile.root, suffix+str(row), tb.Float32Atom(),
                                  shape=(1,elements), filters=hfilters)
        
        forest = model.estimators_
        for tree in forest:
            leafs = tree.apply(data)
            p=0
            for col in range(row+1, nobs):
                if leafs[row] == leafs[col]:
                    dd[p] += 1
                p += 1
                
        if output_type == 'horvath':
            dd = np.sqrt(1 - dd/len(forest))
        elif output_type == 'dissimilarity':
            dd = 1 - dd/len(forest)
        elif output_type == 'similarity':
            dd = dd/len(forest)
        else:
            raise ValueError('output_type E [horvath, dissimilarity, similarity]')
            
        hput[0] = dd
    hfile.close()
    if output == True:
        print('proximity matrix has been saved in file '+str(output_file))







class pmt_read:
    """
    pmt_read class reads the h5 file (proximity matrix) written on hard disc
        and return the value specific to given indices.
        This class largely followed the format used to save proximity matrix by function calculate_large_pmt().
        
    # INPUTS::
        hfile - [str]           input h5 file
        loc   - [str]  d[root]  data location on hfile
        dname - [str]  d[row]   initials of rows saved on hfile.loc
        warn  - [bool] d[True]  warning - if file remained unclosed - cn interfere with further opening/writing
                                with similar names
        
    # OPERATIONS
        get    - get the value at specified index (defined by row, col)
        update - update the value at specified index (defined by row, col)
        close  - close the object
        
    # EXAMPLE::
        object = pmt_read(hfile)
        object.get(row, col)
        object.update(row, col, val)
        object.close()   DON'T FORGET TO CLOSE THE FILE AFTER EXTRACTING/UPDATING THE VALUES
    
    """
    
    def __init__(self, hfile, loc='root', dname='row', warn=True):
        if os.path.exists(hfile) == True:
            self.hfile = hfile
            self.opened_hfile = tb.open_file(self.hfile, 'r+')
        else:
            raise ValueError('hfile not found')
        
        self.loc = str(loc)
        self.dname = str(dname)
        
        if self.loc == 'root': self.lname = '/'
        else: self.lname = self.loc
        self.child_nodes = []
        self.child_sizes = []
        for i in self.opened_hfile.iter_nodes(self.lname):
            self.child_nodes.append(i.name)
            self.child_sizes.append(i.shape[1])
        self.child_nodes = np.array(self.child_nodes)
        self.child_sizes = np.array(self.child_sizes)
        
        
        if warn:
            print("DON'T FORGET TO CLOSE THE FILE AFTER EXTRACTING/UPDATING THE VALUES")
       
    
        
    def get(self, row=0, col=None, 
            ignore=False, check=True):
        '''
        extract the value for matrix indexes row, col i.e., pmat[row,col]
        
        #INPUTS::
            row    - [int]  d[0]     row entry of matrix
            col    - [int]  d[None]  col entry of matrix, None - entire row will be extracted
            ignore - [bool] d[False] raise necessary warnings
            check  - [bool] d[True]  error handling - False for high speed
                        To skip check:
                        > row, col if provided should be int
                        > row < col
                        > row should be in hfile
                        
        #OUTPUTS::
            output entry/row extracted from hfile
        
        '''
        
        if check:
            if type(row) != int:
                raise ValueError('row should be int')
            if f'{self.dname}{row}' not in self.child_nodes:
                raise ValueError(f'row not found in file {self.child_nodes}')

            if col != None:
                if type(col) != int:
                    raise ValueError('col should be int')

                if row == col:
                    if ignore == False:
                        print('row and col should not be equal')
                    return 0

                elif row > col:
                    if ignore == False:
                        print('row should be less than col. pmt format')
                    tmp = row
                    row = col
                    col = tmp
                elif row < col:
                    pass
                else:
                    raise ValueError('problem with row and col values')
                
        try:
            rout = eval('self.opened_hfile.{loc}.{dname}{row}'.format(loc=self.loc, dname=self.dname, row=row)).read()[0]
        except:
            raise AttributeError('given h5 file did not have {loc}.{dname}{row}'.format(loc=self.loc, dname=self.dname, row=row))
        
        if col == None:
            return rout
        elif col-(row+1) < len(rout):
            return rout[col-(row+1)]
        else:
            raise ValueError('col is larger than data')
            
            
    def update(self, row, col, val,
               check=True):
        '''
        Update the hfile at specific row and col i.e., pmat[row,col]
        
        #INPUTS::
            row   - [int] row entry of hfile
            col   - [int] col entry of hfile , -1 to update entire row
            val   - [int/list,array] value to be updated
                        list,array if col = -1
                        else : int
            check - [bool] d[True] Error handling - False for higher speed
                        To skip check:
                        > row and col should be int
                        > row < col
                        > row should be in hfile
                        > val should be of proper size in case col = -1
                        
        '''
        
        if check:  
            if any(type(i) != int for i in [row,col]) or row == col:
                raise ValueError('problem with row and col')
                
            if col != -1:
                if all(type(val) != i for i in [int,float]):
                    raise ValueError('val E [int,float]')
                if row > col:
                    row, col = col, row
                    
            elif col == -1:
                if any(type(val) == i for i in [list,np.ndarray]):
                    if len(val) != self.child_sizes[np.where(self.child_nodes == f'{self.dname}{row}')[0][0]]: 
                        raise ValueError('problem with val size')
                else: raise ValueError('val E [list,array]')
                    
                    
        if col == -1:
            eval(f'self.opened_hfile.{self.loc}.{self.dname}{row}')[0] = val
        else:
            eval(f'self.opened_hfile.{self.loc}.{self.dname}{row}')[0,col - (row+1)] = val
                
                
                
    def close(self):
        self.opened_hfile.close()


def calculate_direct_pmt(data, distance_type='euclidean', n_jobs=1):
    '''
    this function calculates proximity matrix of given data, based on simple distance.

    #INPUTS::
        data          - [2d-a] data point - 2 dimensional
        distance_type - [str ] d[eucledian] distance type as implemented in scipy.spatial.distance.cdist
                                            { specifics are taken as default }
        n_jobs        - [int ] d[1] number of cpu cores to be used - for multiprocessing

    #OUTPUT::
        pmt           - [arr] condenced proximity matrix

    '''
    pmt = np.array([])
    nobs = len(data)

    global operate_row
    def operate_row(row):
        dists = scp.distance.cdist( [data[row]], data[row+1:], metric=distance_type )
        return dists[0]

    with Pool(processes=n_jobs) as p:
        results = list( tqdm( p.imap(operate_row, range(nobs-1) ), total=nobs-1) )

    pmt = np.concatenate((results))

    return pmt
         

