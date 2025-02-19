import numpy as np
import numpy.fft as fft
import pandas as pd
from claspy.segmentation import BinaryClaSPSegmentation
import time
#downsamplers
from tsdownsample import MinMaxLTTBDownsampler
from tsdownsample import MinMaxDownsampler
from tsdownsample import M4Downsampler
from tsdownsample import LTTBDownsampler
from tsdownsample import EveryNthDownsampler
from claspy.tests.evaluation import covering
import ltd

from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
import tsfel

def prepare_ts(ts):
    if ts.ndim == 1:
        ts = np.expand_dims(ts, axis=0)
    else:
        ts = np.transpose(ts)
    return ts


def get_significant_features(ts_list,domain):
    cfg_file = tsfel.get_features_by_domain(domain)
    features = tsfel.time_series_features_extractor(cfg_file, ts_list)
    # remove highly correlated features
    corr_features, features = tsfel.correlated_features(features, drop_correlated=True)
    # remove low variance features
    selector = VarianceThreshold()
    features = selector.fit_transform(features)
    # scale
    scaler = preprocessing.StandardScaler()
    nfeatures = scaler.fit_transform(features)
    return selector.get_feature_names_out(), nfeatures


def z_ED_fft(ts1, ts2):

    max_length = max(len(ts1), len(ts2))

    # Pad time series to equal lengths if not equal
    if len(ts1) != len(ts2):
        ts1 = np.pad(ts1, (0, max_length - len(ts1)), 'constant')
        ts2 = np.pad(ts2, (0, max_length - len(ts2)), 'constant')

    n = len(ts1)
    
    # Fourier-Transformation of time series
    fft1, fft2 = fft.rfft(ts1), fft.rfft(ts2)
    
    # Mean values and standard deviations in frequency space
    mu_fft1, mu_fft2 = np.mean(fft1), np.mean(fft2)
    sigma_fft1, sigma_fft2 = np.std(fft1), np.std(fft2)
    
    dot_product_fourier = np.dot(fft1, np.conj(fft2)).real
    
    # Calculation of the z-normalised Euclidean distance in frequency space
    znorm_dist_squared = 2 * (1 - (dot_product_fourier - n * mu_fft1 * np.conj(mu_fft2).real) / (n * sigma_fft1 * sigma_fft2))
    
    return np.sqrt(znorm_dist_squared).real


def z_normalized_euclidean_distance(v1,v2):
    zv1 = (v1 - np.mean(v1)) / np.std(v1)
    zv2 = (v2 - np.mean(v2)) / np.std(v2)
    return np.sqrt(np.sum((zv1-zv2)**2))


def evaluate_clasp(time_series, change_points, window_size=None, pred_cps=[]):
    runtime = time.process_time()
    if window_size:
        clasp = BinaryClaSPSegmentation(window_size=window_size)
    else:
        clasp = BinaryClaSPSegmentation()
    clasp = clasp.fit(time_series)
    runtime = time.process_time() - runtime
    if len(pred_cps)==0:
        score = covering({0: change_points}, clasp.change_points, time_series.shape[0])
    else:
        score = covering({0: change_points}, pred_cps, time_series.shape[0])
    return clasp.window_size, clasp.change_points, np.round(score, 3), np.round(runtime, 3)
	

def adjust_nout(algo, num):
    if num < 3:
        num = 3
    if algo == "M4":
        rest = num % 4
        if rest != 0:
            return num+(4-rest)
        else:
            return num
        if num % 4 != 0:
            return num+1
        else:
            return num
    else:
        if num % 2 != 0:
            return num+1
        else:
            return num

			
def split_ts(ts, cps):
    if len(cps)==0:
        return np.expand_dims(ts,axis=0)
    splits_ts = []
    cps = np.append(cps, 0)
    for t in ts:
        lst_idx = 0
        splits_t = []
        for idx, cp in enumerate(cps):
            if idx == len(cps)-1:
                split = t[lst_idx:]
            else:
                split = t[lst_idx:cp]
            splits_t.append(split)
            lst_idx = cp + 1
        splits_ts.append(np.array(splits_t, dtype=object))
    return np.array(splits_ts, dtype=object)
	
	
def downsample_splits(algo, splits, cf):
    ts = []
    cps = []
    for split in splits:
        split = np.ascontiguousarray(split)
        #get idx of final set after downsampling
        n_out = adjust_nout(algo, int(cf*len(split)))
        s_ds = downsample(algo, split, n_out)
        downsampled_split = split[s_ds]
        ts = np.concatenate((ts,downsampled_split), axis=0)
        cps.append(len(ts))
    # return downsampled ts by splits and offsets as new change points
    return ts, np.array(cps[:-1])


def downsample_multivariate(ds_algo, data, n_out):
    ts_array = prepare_ts(data)
    ds_ts_list=[]
    for ts in ts_array:
        ds_ts_i = downsample(ds_algo, ts, n_out)
        ds_ts_list.append((ds_ts_i))

    if data.ndim == 1:
        ds_ts_is = np.array(ds_ts_list).squeeze()
    else:
        ds_ts_is = np.transpose(np.array(ds_ts_list))
    return ds_ts_is


def downsample(ds_algo, data, n_out):
    #returns the indexes of the original ts that should be used for downsampled ts
    data = np.ascontiguousarray(data)
    match ds_algo:
        case "MinMaxLTTB": 
            return MinMaxLTTBDownsampler().downsample(data, n_out=n_out)
        case "MinMax":
            return MinMaxDownsampler().downsample(data, n_out=n_out)
        case "M4":
            return M4Downsampler().downsample(data, n_out=n_out)
        case "LTTB":
            return LTTBDownsampler().downsample(data, n_out=n_out)
        case "EveryNth":
            return EveryNthDownsampler().downsample(data, n_out=n_out)
        case "LTD":
            return ltd.ltd(data, n_out)
        case _:
            return data
			
def get_ds_ts_ndim(ts_arr, ds_ts_i_arr):
    out = []
    ds_ts_i_arr = np.transpose(ds_ts_i_arr)
    for i, ts in enumerate(np.transpose(ts_arr)):
        ds_ts = ts[ds_ts_i_arr[i]]
        out.append(ds_ts)
    out = np.transpose(np.array(out))
    return out
	
def prune_cps(cps, threshold = 400):
    cps = sorted(cps)
    i=0
    while i < len(cps)-1:
        if cps[i+1]-cps[i]<threshold:
            cps.pop(i+1)
        else:
            i+=1
    return [int(cp) for cp in cps]
	