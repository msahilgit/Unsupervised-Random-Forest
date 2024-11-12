import numpy as np
import math
import copy as c
import random

class synthetic_data:
    """
    This class create synthetic data, extracted from the given data and 
    return the concated data with original plus synthetic data.
    
    #
    INPUTS::
        rdata    = [n-d array; max-2] original data
        size     = [0-1]   size of synthetic data to be created
                                overwritten in case of mean/pd
        replace  = [bool]  whether to omit the synthetic data from original data
        nclasses = [int>1] number of classes to be created. Currently applicable for 2 classes only
        seed     = [int] random seed for reproducibility
        
    #
    EXAMPLE::
        dobj = synthetic_data(rdata, size=1, replace=False, nclasses=2, seed=42)
        dobj.marginal() 
                        random   - create data randomly
                        marginal - create data with same marginal distribution as rdata
                        mean     - create random data with data[:,feature] larger than mean
                        permute  - create a random version of multi-dim-data - different features permuted randomly
                        nonsense - create a random data with no meaning perse 
                        ....
        dout = dobj.get_output(randomized=True)
            randomized = [bool] d[True] - randomized the data
    
    #
    OUTPUTS::
        fdata   - [n-d array] combined original and synthetic data
        flabels - [1d array]  class labels of fdata; 0-original 1-synthetic data
    
    #
    """
    # generic attributes
    data = 'synthetic_data'
    msg = ''
#     for line in open('readme_sahil.dat','r'):
#         msg += line
    sahil = msg
    #
    def __init__(self, rdata, size=1, replace=False, nclasses=2, seed=42):
        self.rdata = rdata
        self.dim = len(rdata.shape)
        if self.dim >= 3:
            raise TypeError('Only 1 or 2 dimensional data can be processed.')
        elif self.dim == 2:
            self.features = rdata.shape[1]
            self.instances = rdata.shape[0]
        elif self.dim == 1:
            self.features = self.dim
            self.instances = rdata.shape[0]
            self.rdata = rdata[:,None]
        #
        self.size = size
        self.synthetic_size = math.floor(len(rdata)*size)
        if self.synthetic_size == 0:
            raise ValueError('Synthetic data size of 0 data points cannot be created: \n check size')
        #
        self.replace = replace
        if self.replace == True and self.size == 1:
            raise ValueError('replace=True and size=1 are incompatible')
        #
        self.nclasses = nclasses
        if self.nclasses <= 1:
            raise ValueError('nclasses should be greater than or equal to 2')

        self.seed = seed
    
    
    #
    # creating synthetic data
    def random(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.operation_type = 'random'
        indices = np.random.randint(self.synthetic_size, size=self.synthetic_size)
        self.indices = indices
        
    #
    def marginal(self, repeat=False, ignore=False):
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.operation_type = 'marginal'
        if repeat == False and self.size == 1 and ignore == False:
            print('marginal: with repreat=False, size=1: synthetic data is same copy of original')
        indices = np.random.choice(np.arange(self.synthetic_size), size=self.synthetic_size, replace=repeat)
        self.indices = indices
    
    #
    def mean(self, fnum='random', stype='median'):
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.operation_type = 'mean'
        if fnum == 'random':
            self.fnum = np.random.randint(self.features)
        elif fnum == 'max_variable':
            self.fnum = get_variable_feature(self.rdata)
        elif fnum < self.features:
            self.fnum = fnum
        else:
            raise ValueError('problem with fnum')
            
        if stype == 'median':
            self.sval = np.median(self.rdata[:,self.fnum])
        elif stype == 'mean':
            self.sval = np.mean(self.rdata[:,self.fnum])
        else:
            raise ValueError('problem with stype')
        
        indices = np.where(self.rdata[:,self.fnum] > self.sval)[0]
        self.indices = indices
    
    #
    def permute(self):
        """
        Randomly permute data - independently of different features.
        To break the internal structural in data (if present)
        See permute_data_features(data).
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        self.operation_type = 'permute'
        if self.replace == True:
            print('replace=True did not work for operation_type permute')
        self.tdata = self.rdata
        
        dd = c.deepcopy(self.rdata)
        self.sdata = permute_data_features(dd, size=self.size)


    def nonsense(self, complete=True):
        """
        randomly creates non-sensical data - not matching with given rdata in any sense (perse)
        though, by using complete d[True](True/False); position of synthetic data in hyperspace can be make
        nearby to rdata - not guaranteed (using False).
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.operation_type = 'nonsense'
        if complete:
            maxi = np.max(self.rdata)
            if maxi <0:
                maxi=10
            means = np.random.random() * np.random.randint(maxi)
            stds = means + 0
        else:
            means = np.mean(self.rdata)
            stds = np.std(self.rdata)

        means = np.random.normal( means, stds, self.features )
        stds = np.random.normal( stds, stds, self.features )
        stds[np.where(stds < 0)[0]] = np.random.random()

        self.sdata = np.random.normal( means, stds, size=(self.synthetic_size, self.features) )
        self.tdata = self.rdata

        
        
    #
    # getiing putput
    def get_output(self, randomized=True):
        
        if any(self.operation_type == i for i in ['permute', 'nonsense']):
            try:
                self.sdata = self.sdata
            except:
                raise AttributeError('synthetic data has not been created. use OBJECT.permute(etc) b4 get_output')


        else:

            try:
                self.indices = self.indices     # synthetic data
            except:
                raise AttributeError('synthetic data has not been created. use OBJECT.marginal(etc) b4 get_output')
            self.sdata = self.rdata[self.indices]

            #
            if self.replace == True:
                self.tdata = np.delete(self.rdata, self.indices, axis=0)     # truncated data
            elif self.replace == False:
                self.tdata = self.rdata
                
        
        
        #
        self.cdata = np.concatenate((self.tdata,self.sdata))     # concated data
        self.clabels = np.concatenate((np.zeros((len(self.tdata))), np.ones((len(self.sdata)))))
        
        #
        self.randomized = randomized
        if self.randomized == True:
            findices = np.random.permutation(len(self.cdata))
            self.fdata = self.cdata[findices]                   # final data
            self.flabels = self.clabels[findices]
        elif self.randomized == False:
            self.fdata = self.cdata
            self.flabels = self.clabels
        else:
            raise ValueError('randomized E boolean')
        
        #
        return self.fdata, self.flabels
    


def get_variable_feature(data, nd=3, output_type='max'):
    """
    This function_to_be_used finds the feature id from nd-array (data) exhibiting minimum/maximum (min/max)
    variability, as measured by removing outliers (mean +- nd*std):
    """
    diffs = np.empty((data.shape[1]))
    for i in range(data.shape[1]):
        mean = np.mean(data[:,i])
        std = np.std(data[:,i]) * nd
        non_outliers = data[:,i][np.unique(np.where((data[:,i] > mean-std ) & (data[:,i] < mean+std)))]
        diffs[i] = np.max(non_outliers) - np.min(non_outliers)
    
    if output_type == 'max':
        return np.argmax(diffs)
    elif output_type == 'min':
        return np.argmin(diffs)
    else:
        raise ValueError('output_type E [min, max]')


        
def permute_data_features(data, size=1):
    if len(data.shape) != 2:
        raise TypeError('Data should be 2 dimensional number array')
    
    if size > 1 or size <= 0:
        raise ValueError('size E (0-1]')
    else:
        size = int(len(data)*size)
        
    for i in range(data.shape[1]):
        data[:,i] = data[:,i][np.random.permutation(len(data[:,i]))]
    
    data = data[:size]
    
    return data
