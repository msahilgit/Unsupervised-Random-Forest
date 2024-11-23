import numpy as np
import random
from sklearn.model_selection import train_test_split as splt
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.metrics import confusion_matrix as cfm
import scipy.cluster as scc
import fastcluster
import pickle as pkl
from tqdm import tqdm
import sys
import os
import sys
sys.path.append('./')
import utils
import synthetic_data as syn
import proximity_matrix as pmt
import cluster 
import metrics




class unsupervised_random_forest:
    '''
    a feature selection pipeline based on unsupervised (self-supervised) random forest as 
    defined in citation_here.

    usage:
        from model import unsupervised_random_forest as urf
        dobj = urf(**args)
        dobj.fit(data)
        lc, fimp = dobj.get_output()

    INPUTS::
        random_state    - [int] d[42] a random integer to set the random seed

        syn_data_type   - [str] d[permute]
                            type of synthetic data to generate
                            random   - randomly selects some of the datapoints from original data
                                        and labelled them as synthetic data
                            marginal - also randomly selects data but tries to maintain the marginal
                                        distribution similar to original data
                                        its the same copy as original data with syn_dat_size=1
                            permute  - it randomly permute the columns (features) of original data
                                        and labelled it as synthetic data
                                        NOTE:: THIS IS PREFERRED CHOICE
                            mean     - create synthetic data with original_data[:,feature] larger than mean/median
                            nonsense - also referred as [fictitious], create a random data with no meaning perse

        syn_data_size   - [float] d[1] size of the synthetic data with respect to original data
                            NOTE: 1.0 [same size as original size] IS IDEAL AND BALANCED
                            not for mean type of synthetic data generation

        syn_replace     - [bool] d[False] to replace the synthetic data from original data
                            not for permute or nonsense type of synthetic data generation
        syn_marg_repeat - [bool] d[False] whether to repeat the data in marginal type of data generation
        syn_mean_fnum   - [str, int] E ['random', 'variable'] which feature (feature type) to consider in mean type of data generation

        rf1_cv          - [int] d[5] number of cross validation replicates to perform on RF-I
                            for each iteration, the input features (original+synthetic) were randomly (controlled by seed)
                            selected as per train-test split

        rf1_est         - [int] d[1000] number of decision tree estimators in RF-I
                            IMPORTANT NOTE: this parameter controls the resolution of proximities,
                            hence larger is better. It does not have any upper limit.
                            This is not from the perspective of predictive learning

        rf1_criterion   - [str] d[gini] the node splitting criteria for RF-I,
                            as defined in sklearn.ensemble.RandomForestClassifier
                            rf1_criterion E [gini, entropy, log_loss]

        rf1_test_size   - [float] d[0.3] the size of test data for RF-I, defined as fraction of total data
                            and must be between 0.01 to 0.99

        pmt_alg         - [str] d[efficient] algorithm used to estimate proximity matrix and hierarchial clustering
                            different algorithms tradeoff in terms of memory, time usage or calculative assumptions:
    
                            efficient (ef)   - pre-calculates a condenced proximity matrix, followed by
                                                estimation of average dendrogram via fastcluster library
                                                memory requirement - moderate to high, depending on data size [control via pmt_data_size]
                                                for n data point:
                                                    memory = (^n C_2 * 8) / (1024^3) gb  [[ plus additional for processing ]]
                                                overall; for ~150,000 datapoint, roughly ~100 gb working RAM is required
                                                time requirement - low to moderate, depending on data size

                            low-mem (lm)     - combines proximity estimation and hierarchical clustering, hence avoiding the
                                                need to save a large proximity matrix.
                                                memory requirement - low
                                                time requirement - high, depending on data size [control via pmt_data_size]
                                                it performs on-the-fly proximity calculation while performing nearest-neighbour-chain
                                                algorithm based hierarchical clustering
                                                Though parallelized, but slow

                            fit-predict (fp) - combines proximity estimation and hierarchical clustering, hence memory efficient
                                                it takes random-uniform sample from data [pmt_data_size] and perform proximity based hierarchy 
                                                as for efficient algorithm, the clustering for remaining data is estimated via 
                                                proximity-based-k-nearest-neighbour [controlled via nn] algorithm.
                                                memory requirement - low to moderate depending on pmt_data_size
                                                time requirement - low to moderate depending on data size
                                                pros - did not reduce data on pmt and subsequent steps, memory requirement
                                                cons - actual calculation on small data subset, rest is estimated [the deviation in results is small ]

        pmt_data_size   - [float] d[1] the fraction of original data to be used for proximity matrix,
                            hc and RF-II stages.
                            This parameter can avoid large memory usage, as required in " pmt_alg = efficient "
                            or other pmt_alg options
                            0.01 <= pmt_data_size <= 1
                            1 means taking all data (original)
                            0.5 means skipping a data point i.e., [::2] and so on.

                            if pmt_alg is 'fit-predict': 
                                then pmt_data_size is size of data used in fit part.
                                if pmt_data_size = 1:
                                    then pmt_data_size = (3*rf1_est*rf1_cv)/nobs

        nn              - [int] d[None] the number of nearest neighbours to be used in predict part
                            for fit-predict hierarchical clustering
                            nn >= 1
                            if None:
                                nn = int[ cfactor * sqrt(nobs/nhc) ]

        nhc             - [int] or [list-int] d[None] number of labels in hierarchial dendrogram
                            lower limit - 2 (strict)
                            upper limit - 10 (adviced but not neccesarily)
                            if int - nhc labels are used
                            if list-int - best one is taken from the given options
                            if None - estimated based on pre-learning_coefficient calculation (nhc=2-10)

        cfactor         - [float] d[2] a cutoff factor to define outliers in learning coefficient estimation
                                        other params of Lc: a=1, b=0, penalty=0

        rf2_cv          - [int] d[5] number of cross validation replicates to perform on RF-II
                            for each iteration, the original data is randomly (controlled by seed)
                            selected as per train-test split

        rf2_est         - [int] d[1000] number of decision tree estimators in RF-II

        rf2_criterion   - [str] d[gini] the node splitting criteria for RF-II,
                            as defined in sklearn.ensemble.RandomForestClassifier
                            rf2_criterion E [gini, entropy, log_loss]

        rf2_depth       - [int] d[None] max depth of decision trees in RF-II,
                            to reduce overfitt if any
        rf2_test_size   - [float] d[0.3] the size of test data for RF-I, defined as fraction of total data
                            and must be between 0.01 to 0.99

        n_jobs          - [int] d[1] number of parallel cpus to be utilized

        save_syn        - [str] d[None] save synthetic data (+original data and labels)
                            as pfeatures_{name}.npy and plabels_{name}.npy

        save_rf1        - [str] d[None] save RF-I models in pickle format
                            to open back: rf1_i = pickle.load( open('rf1_{name}_{cv}.pkl', 'rb') )

        save_pmt        - [str] d[None] save condenced proximity matrix in numpy format
                            only for 'efficient' pmt_alg

        save_dng        - [str] d[None] saved hierarchial clustering in dendrogram format

        save_dtrj       - [str] d[None] saved labels of hieracrchial clustering

        save_rf2        - [str] d[None] saved pickle objects of trained RF-II models as rf2_{name}_{cv}.pkl

    OUTPUTS::
        fimp            - feature importance scores for each feature in original data
        save            - any saved object/file given specified save_syn, save_rf1, save_pmt, save_dng, save_dtrj, save_rf2
        exp_fnum        - an expected (suggested) number of features to consider
        lc              - learning coefficient

    '''
    def __init__(self,
            random_state=42,
            syn_data_type='permute', syn_data_size=1, syn_replace=False, syn_marg_repeat=False, syn_mean_fnum='random',  #synthetic data args
            rf1_cv=5, rf1_est=1000, rf1_criterion='gini', rf1_test_size=0.3,                                             #RF-I args
            pmt_alg='efficient', pmt_data_size=1, nn=None, nhc=None, cfactor=2,                                          #pmt-hc args
            rf2_cv=5, rf2_est=1000, rf2_criterion='gini', rf2_depth=None, rf2_test_size=0.3,                             #RF-II args
            n_jobs=1,                                                                                                    #
            save_syn=None, save_rf1=None, save_pmt=None, save_dng=None, save_dtrj=None, save_rf2=None                    #saving args
            ):

        options_syn = ['random', 'marginal', 'permute', 'mean', 'nonsense']
        options_cri = ['gini', 'entropy', 'log_loss']
        options_pmt = ['efficient', 'ef', 'low-mem', 'lm', 'fit-predict', 'fp']

        self.rstate = _check_type(random_state, 0, None, int)

        if syn_data_type == 'fictitious':
            syn_data_type = 'nonsense'
        if syn_data_type in options_syn:
            self.stype = syn_data_type
        else:
            raise ValueError(f'syn_data_type E {options_syn}')

        self.ssize = _check_type(float(syn_data_size), 0.01, None, float)
        self.sreplace = bool(syn_replace)
        self.s_marg_repeat = bool(syn_marg_repeat)      
        self.s_mean_fnum = syn_mean_fnum                # error - if any - shall raise by synthetic data class

        self.rf1_cv = _check_type(rf1_cv, 1, None, int)
        self.rf1_est = _check_type(rf1_est, 1, None, int)
        self.rf1_test_size = _check_type(rf1_test_size, 0.01, 0.99, float)
        if rf1_criterion in options_cri:
            self.rf1_cri = rf1_criterion
        else:
            raise ValueError(f'rf1_criterion E {options_cri}')

        self.pmt_data_size = _check_type(float(pmt_data_size), 0.01, 1, float)
        if pmt_alg in options_pmt:
            self.pmt_alg = pmt_alg
        else:
            raise ValueError(f'pmt_alg E {options_pmt}')
        if nn is None:
            self.nn = nn
        else:
            self.nn = _check_type(nn, 1, None, int)
        self.nhc = nhc
        self.cfactor = _check_type( float(cfactor), 1, None, float)

        self.rf2_cv = _check_type(rf2_cv, 1, None, int)
        self.rf2_est = _check_type(rf2_est, 1, None, int)
        self.rf2_test_size = _check_type(rf2_test_size, 0.01, 0.99, float)
        self.rf2_depth = rf2_depth   #if error - raise by sklearn-rfc class
        if rf2_criterion in options_cri:
            self.rf2_cri = rf2_criterion
        else:
            raise ValueError(f'rf2_criterion E {options_cri}')

        self.n_jobs = _check_type(n_jobs, 1, None, int)
    
        self.save_syn = save_syn
        self.save_rf1 = save_rf1
        self.save_pmt = save_pmt
        self.save_dng = save_dng
        self.save_dtrj = save_dtrj
        self.save_rf2 = save_rf2



    def fit(self, data):
        '''
        INPUTS::
            data - [np.ndarray] a 2d input data, with rows as #observations and cols as #features
        '''
        nobs, nofs = np.shape(data)
        random.seed(self.rstate)
        np.random.seed(self.rstate)
        self.randoms = np.random.randint(0, 1000, np.max([self.rf1_cv, self.rf2_cv]))


        #expected memory usage and available memory 
        if self.pmt_alg in ['efficient', 'ef']:
            _print_mem_usage( int(nobs*self.pmt_data_size) )    


        
        #synthetic data
        dsyn = syn.synthetic_data(data, 
                size=self.ssize, replace=self.sreplace, seed=self.rstate)

        if self.stype == 'marginal':
            dsyn.marginal(repeat=self.syn_marg_repeat)

        elif self.stype == 'mean':
            dsyn.mean(fnum=self.syn_mean_fnum)

        else:
            getattr(dsyn, self.stype)()

        pfeatures, plabels = dsyn.get_output()

        if self.save_syn is not None:
            np.save(f'pfeatures_{self.save_syn}.npy', pfeatures)
            np.save(f'plabels_{self.save_syn}.npy', plabels)



        #RF-I
        clfs = []
        for i in range(self.rf1_cv):
            clf = rfc(n_estimators=self.rf1_est, random_state=self.randoms[i], 
                                n_jobs=self.n_jobs, criterion=self.rf1_cri)
            xtrain, _, ytrain, _ = splt(pfeatures, plabels, 
                                test_size=self.rf1_test_size, random_state=self.randoms[i])
            clf.fit(xtrain, ytrain)
            clfs.append(clf)
            if self.save_rf1 is not None:
                pkl.dump(clf, open(f'rf1_{self.save_rf1}_{i}.pkl', 'wb') )



        #pmt-hc
        if self.pmt_alg in ['efficient', 'ef']:
            np.random.seed(self.rstate)
            self.indices = np.random.permutation(nobs)[:int(nobs*self.pmt_data_size)]

            prox = pmt.calculate_condenced_pmt_(clfs, data[self.indices], n_jobs=self.n_jobs)

            if self.save_pmt is not None:
                np.save(f'{self.save_pmt}.npy', prox)

            self.dng = fastcluster.linkage( prox,
                    method='average', preserve_input=False)


        elif self.pmt_alg in ['low-mem', 'lm']:
            np.random.seed(self.rstate)
            self.indices = np.random.permutation(nobs)[:int(nobs*self.pmt_data_size)]

            hobj = cluster.proximity_based_hierarchy(clfs, data[self.indices], algorithm=2, n_jobs=self.n_jobs)
            hobj.cluster()
            self.dng = hobj.get_output()
            hobj.close()
            

        elif self.pmt_alg in ['fit-predict', 'fp']:
            self.indices = np.arange(nobs)

            if self.pmt_data_size == 1:
                self.pmt_data_size = (3*self.rf1_cv*self.rf1_est)/nobs

            self.hobj = cluster.fit_predict_proximity_based_hierarchy(clfs, data, self.pmt_data_size,
                                                random_state=self.rstate, n_jobs=self.n_jobs)
            self.hobj.fit()
            self.dng = self.hobj.dng


        if self.save_dng is not None:
            np.save(f'{self.save_dng}.npy', self.dng)


        # check pre-lc
        if isinstance(self.nhc, int):
            self.nhc = _check_type(self.nhc, 2, None, int)
            _raise_pre_lc_warn(self.dng, self.nhc, self.cfactor)


        elif isinstance(self.nhc, (list,np.ndarray)):
            found_best = False
            self.nhc = np.array(self.nhc)[np.argsort(self.nhc)]      #increasing order, so to select the lowest one
            for nids in self.nhc:
                nids = _check_type(nids, 2, None, int)
                if _is_outlier_free(self.dng, nids, self.cfactor):
                    found_best = True
                    self.nhc = nids
                    break

            if not found_best:
                print('Pre-LC WARNING: the learning may not be good:::')
                self.nhc = np.max(self.nhc)
                print(f'proceeding with {self.nhc}')


        elif self.nhc is None:
            found_best = False
            for nids in range(2,11):
                if _is_outlier_free(self.dng, nids, self.cfactor):
                    found_best = True
                    self.nhc = nids
                    break

            if not found_best:
                raise AssertionError('best nhc cound not be found within 2-10 \n you may try restart_fit with defined nhc')


        else:
            raise ValueError('wrong input for nhc')


        # RF-II
        self.restart_fit(data, self.nhc)





    def restart_fit(self, data, nhc):
        '''
        this function restarts the fitting process if it gets terminated at pre-lc step of the fit step
        INPUTS::
            data - input data, same as given in .fit step
            nhc - [int, or list of int] #label classes to try on
        '''
        if isinstance(nhc, (list,np.ndarray)):
            nhc = np.array(nhc)[np.argsort(nhc)]
            found_best = False
            for nids in nhc:
                nids = _check_type(nids, 2, None, int)
                if _is_outlier_free(self.dng, nids, self.cfactor):
                    found_best = True
                    nhc = nids
                    break

            if not found_best:
                print('Pre-LC WARNING: the learning may not be good:::')
                nhc = np.max(nhc)
                print(f'proceeding with {nhc}')

        else:
            nhc = _check_type(nhc, 2, None, int)


        nobs, nofs = np.shape(data)

        if self.pmt_alg in ['efficient', 'ef', 'low-mem', 'lm']:
            dtrj = utils.get_hc_dtraj(self.dng, nhc)

        else:
            if self.nn is None:
                self.nn = int( self.cfactor * np.sqrt( (nobs*self.pmt_data_size)/nhc ) )
            dtrj = self.hobj.predict(nhc, self.nn)

        if self.save_dtrj is not None:
            np.save(f'{self.save_dtrj}.npy', dtrj)

        self.lc = np.zeros(self.rf2_cv)
        self.fimp = np.zeros((self.rf2_cv, nofs))
        
        for i in range(self.rf2_cv):
            xtrain, xtest, ytrain, ytest = splt(data[self.indices], dtrj, test_size=self.rf2_test_size, random_state=self.randoms[i])

            clf = rfc(n_estimators=self.rf2_est, random_state=self.randoms[i],
                            n_jobs=self.n_jobs, criterion=self.rf2_cri, max_depth=self.rf2_depth)
            clf.fit(xtrain, ytrain)

            if self.save_rf2 is not None:
                pkl.dump(clf, open(f'rf2_{self.save_rf2}_{i}.pkl', 'wb'))

            self.lc[i] = metrics.learning_coefficient(dtrj, clf, xtest, ytest, self.cfactor)
            self.fimp[i] = clf.feature_importances_
            


    def supervised_fit(self, data, labels):
        '''
        this class performs supervised random forest based feature selection, given some labels
        all the hyperparameters of RF-II applies here.
        '''
        nobs, nofs = np.shape(data)
        randoms = np.random.randint(0, 1000, self.rf2_cv)
        self.fimp = np.zeros((self.rf2_cv, nofs))

        accu = np.zeros(self.rf2_cv)
        for i in range(self.rf2_cv):
            xtrain, xtest, ytrain, ytest = splt(data, labels, test_size=self.rf2_test_size, random_state=randoms[i])

            clf = rfc(n_estimators=self.rf2_est, random_state=randoms[i],
                    n_jobs=self.n_jobs, criterion=self.rf2_cri, max_depth=self.rf2_depth)
            clf.fit(xtrain, ytrain)

            if self.save_rf2:
                pkl.dump(clf, open(f'sup-rf2_{self.save_rf2}_{i}.pkl', 'wb'))
            
            pred = clf.predict(xtest)
            accu[i] = utils.get_f1_score( cfm(ytest, pred) )
            self.fimp[i] = clf.feature_importances_

        return accu



    def get_output(self):
        return self.lc, self.fimp


############ finish ######################################################################################################



def _check_type(val, mini, maxi, vtype):
    if maxi == None:
        maxi = np.inf
    if isinstance(val, vtype) and mini <= val <= maxi:
        return val
    else:
        raise ValueError(f'val({val}) should be {vtype} and {mini} <= val <= {maxi}')




def _print_mem_usage(nobs, factor=1.4):
    '''
    a quick check if enough memory is available on the machine or not
    '''
    if os.path.exists('/proc/meminfo'):

        for line in open('/proc/meminfo', 'r'):
            if 'MemAvailable' in line:
                mem = line.strip().split()[1]

                mem = np.round( float(mem) / ( 1024 ** 2), 4)

                req = np.round( factor * (8*nobs*(nobs-1)/2) / (1024**3), 4)

                if req > mem:
                    print(f'there may not be enough memory ({mem} gb) on the machine to run this calculation which may required around {req} gb memory \n CONSIDER reducing pmt_data_size ')

                break



def _is_outlier_free(dng, nhc, cfactor):
    dtrj = utils.get_hc_dtraj(dng, nids=nhc)
    pops = [ np.where(dtrj==i)[0].shape[0] for i in np.unique(dtrj) ]
    rcut = cfactor * np.sqrt(dtrj.shape[0]/nhc)
    non_out = np.where(pops > rcut)[0]
    if len(non_out) >= 2:
        return True
    else:
        return False

def _raise_pre_lc_warn(dng, nhc, cfactor):
    if not _is_outlier_free(dng, nhc, cfactor):
        print('Pre-LC WARNING: the learning may not be good:::')
