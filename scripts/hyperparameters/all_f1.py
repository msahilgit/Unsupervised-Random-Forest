import numpy as np
np.bool = np.bool_
import copy as cp
import sys
import matplotlib as mt
import matplotlib.pyplot as plt
import pickle as pkl
import sklearn as skl
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.model_selection import train_test_split as splt
from sklearn.metrics import accuracy_score as acc, confusion_matrix
import scipy as sc
import scipy.cluster as scc
from tqdm import tqdm
import time
import pyemma
import pyemma.coordinates as coor
import pyemma.plots as mplt

sys.path.append('../0_python_modules/')

import proximity_matrix as pmt
import synthetic_data as syn
import extras
from navjeet_hist import hist_raw_data
import navjeet_hist as nh

features = np.load('../1_datasets/t4l/features.npy')
randoms = np.loadtxt('../1_datasets/randoms.txt', dtype=int)


# permute - default

accu = []
for l in range(2,11):

    hlabels_ = np.load(f'saved_furf/hlabels_{l}.npy')
    aa = []
    for i in range(len(randoms)):

        _, xtest, _, ytest = splt(features, hlabels_, test_size=0.3, random_state=randoms[i])
        clf_ = pkl.load(open(f'saved_furf/clf_hlabel{l}_{i}.pkl', 'rb'))

        pred_ = clf_.predict(xtest)
        cfm = extras.get_confusion_matrix(ytest, pred_, l)
        aa.append( extras.get_f1_score(cfm, output_type='all') )

    accu.append(aa)

pkl.dump(accu, open('saved_all_f1/accu_spermute.pkl', 'wb'))


# marginal
hc_ = np.load('saved_synthetic_marginal/hc_marginal.npy')
accu = []
for l in range(2,11):

    hlabels_ = extras.get_hc_dtraj(hc_, nids=l)
    aa = []
    for i in range(len(randoms)):

        _, xtest, _, ytest = splt(features, hlabels_, test_size=0.3, random_state=randoms[i])
        clf_ = pkl.load(open(f'saved_synthetic_marginal/clf_hlabel{l}_{i}.pkl', 'rb'))

        pred_ = clf_.predict(xtest)
        cfm = extras.get_confusion_matrix(ytest, pred_, l)
        aa.append( extras.get_f1_score(cfm, output_type='all') )
    accu.append(aa)
pkl.dump( accu, open('saved_all_f1/accu_smarginal.pkl', 'wb'))



# random
hc_ = np.load('saved_synthetic_random/hc_random.npy')
accu = []
for l in range(2,11):

    hlabels_ = extras.get_hc_dtraj(hc_, nids=l)
    aa = []
    for i in range(len(randoms)):

        _, xtest, _, ytest = splt(features, hlabels_, test_size=0.3, random_state=randoms[i])
        clf_ = pkl.load(open(f'saved_synthetic_random/clf_hlabel{l}_{i}.pkl', 'rb'))

        pred_ = clf_.predict(xtest)
        cfm = extras.get_confusion_matrix(ytest, pred_, l)
        aa.append( extras.get_f1_score(cfm, output_type='all') )
    accu.append(aa)
pkl.dump( accu, open('saved_all_f1/accu_srandom.pkl', 'wb'))



# nonsense
hc_ = np.load('saved_synthetic_nonsense/hc_nonsense.npy')
accu = []
for l in range(2,11):

    hlabels_ = extras.get_hc_dtraj(hc_, nids=l)
    aa = []
    for i in range(len(randoms)):

        _, xtest, _, ytest = splt(features, hlabels_, test_size=0.3, random_state=randoms[i])
        clf_ = pkl.load(open(f'saved_synthetic_nonsense/clf_hlabel{l}_{i}.pkl', 'rb'))

        pred_ = clf_.predict(xtest)
        cfm = extras.get_confusion_matrix(ytest, pred_, l)
        aa.append( extras.get_f1_score(cfm, output_type='all') )
    accu.append(aa)
pkl.dump( accu, open('saved_all_f1/accu_snonsense.pkl', 'wb'))



# hc
hc_ = np.load('saved_hc/hc_average.npy')
accu = []
for l in range(2,11):

    hlabels_ = extras.get_hc_dtraj(hc_, nids=l)
    aa = []
    for i in range(len(randoms)):

        _, xtest, _, ytest = splt(features, hlabels_, test_size=0.3, random_state=randoms[i])
        clf_ = pkl.load(open(f'saved_hc/clf_hc{l}_{i}.pkl', 'rb'))

        pred_ = clf_.predict(xtest)
        cfm = extras.get_confusion_matrix(ytest, pred_, l)
        aa.append( extras.get_f1_score(cfm, output_type='all') )
    accu.append(aa)
pkl.dump( accu, open('saved_all_f1/accu_hc.pkl', 'wb'))



