import numpy as np
import tqdm




def get_feature_diff(mdtrj, features, norm=False):
    if norm==True:
        features = ( features - np.min(features, axis=0) ) / ( np.max(features, axis=0) - np.min(features, axis=0) )

    diffs = []
    upairs = np.unique(mdtrj)

    for i in range(len(upairs)-1):

        fi = features[np.where(mdtrj == upairs[i])[0]]
        fi = np.mean(fi, axis=0)

        for j in range(i+1, len(upairs)):

            fj = features[np.where(mdtrj == upairs[j])[0]]
            fj = np.mean(fj, axis=0)

            fj = fi - fj
            fj = np.sqrt(np.mean(np.square(fj)))
            diffs.append(fj)

    return np.mean(diffs)


def get_contact_diffs(mdtrj, features, cutoff=0.5):
    diffs = [[], [], [], []]
    upairs = np.unique(mdtrj)

    for i in range(len(upairs)-1):

        fi = features[np.where(mdtrj == upairs[i])[0]]
        fi = np.mean(fi, axis=0), np.median(fi,axis=0), np.std(fi, axis=0)

        for j in range(i+1, len(upairs)):

            fj = features[np.where(mdtrj == upairs[j])[0]]
            fj = np.mean(fj, axis=0), np.median(fj,axis=0), np.std(fj, axis=0)

            fmean = np.abs(fj[0]-fi[0])
            fmedian = np.abs(fj[1]-fi[1])

            fmean_loose = fmean >= cutoff
            fmean_strict = fmean - (fi[2]+fj[2])/2 >= cutoff
            fmedian_loose = fmedian >= cutoff
            fmedian_strict = fmedian - (fi[2]+fj[2])/2 >= cutoff

            diffs[0].append(np.sum(fmean_loose))
            diffs[1].append(np.sum(fmean_strict))
            diffs[2].append(np.sum(fmedian_loose))
            diffs[3].append(np.sum(fmedian_strict))
           
    diffs = np.array(diffs)
    return np.mean(diffs, axis=1)

                








def get_str_composition(mdtrj, str_data, spairs):
    upairs = np.unique(mdtrj)
    scomp = np.zeros(( str_data.shape[1], len(upairs), len(spairs) ))
    
    for i in range(len(upairs)):
        ss_i = str_data[ np.where(mdtrj == upairs[i])[0] ]

        for f in range(str_data.shape[1]):

            for s in range(len(spairs)):

                scomp[f,i,s] = np.where(ss_i[:,f] == spairs[s])[0].shape[0] / ss_i.shape[0]

    return scomp







