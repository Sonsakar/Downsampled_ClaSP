from claspy.window_size import suss
import numpy as np
import mt_utils as ut
import numpy.fft as fft
from numba import njit

def get_window_size_multivariate(ts_array):
    ts_array = ut.prepare_ts(ts_array)
    window_sizes=[]
    for ts in ts_array:
        window_sizes.append(suss(ts))
    return int(max(window_sizes)/2)


def find_cr_based_on_w(ts, w_threshold, scale, algo=None):
    crs=np.round(np.arange(0.2,0.8,0.1),1)
    crs = np.delete(crs, np.where(crs == 0.8))
    orig_w = get_window_size_multivariate(ts)
    for cr in crs:
        if scale:
            curr_w = int(orig_w*cr)
        else:
            n_out = ut.adjust_nout(algo, int(ts.shape[0]*cr))
            ds_ts_i = ut.downsample_multivariate(algo, ts, n_out)
            if ts.ndim == 1:
                ds_ts = ts[ds_ts_i]
            else:
                ds_ts = ut.get_ds_ts_ndim(ts, ds_ts_i)
            curr_w = get_window_size_multivariate(ds_ts)
        if curr_w>=w_threshold:
            return cr
    return max(crs)
	
	
def find_cr_based_on_w_ubound(ts, w_threshold, scale, algo=None):
    #find cr as high as possible but below ubound
    crs = np.flip(np.round(np.arange(0.2,0.8,0.1),1))
    crs = np.delete(crs, np.where(crs == 0.8))
    orig_w = get_window_size_multivariate(ts)
    for cr in crs:
        if scale:
            curr_w = int(orig_w*cr)
        else:
            n_out = ut.adjust_nout(algo, int(ts.shape[0]*cr))
            ds_ts_i = ut.downsample_multivariate(algo, ts, n_out)
            if ts.ndim == 1:
                ds_ts = ts[ds_ts_i]
            else:
                ds_ts = ut.get_ds_ts_ndim(ts, ds_ts_i)
            curr_w = get_window_size_multivariate(ds_ts)
        if curr_w<=w_threshold:
            return cr
    return min(crs)
	
	
def upscale_cps_linear(ds_cps, cf):
    return (ds_cps*(1/cf)).astype(int)
	

@njit(fastmath=True)
def define_bucket(cp,ds_algo,cf,ts,downscaled,bucket_range=0):
    #all ts in multivariate have same length, so this can simply be done using any ts in the set
    # B = bucket size
    n = ts.shape[0]
    match ds_algo:
        case 'MinMax':
            B = 2/cf
            P = 2
        case 'M4':
            B = 4/cf
            P = 4
        case 'MinMaxLTTB':
            B = int((n-2)/((cf*n)-2))
            P = 1
        case 'LTTB':
            B = int((n-2)/((cf*n)-2))
            P = 1
        case 'LTD':
            #bucket size is dynamic but mean must be the same as lttb bucket size
            B = int((n-2)/((cf*n)-2))
            P = 1
        case 'EveryNth':
            B = 1/cf
            P = 1
    if downscaled:
        bucket_index = np.ceil(cp/P)
    else:
        bucket_index = np.ceil(cp/B)
        
    lbucket=bucket_index-bucket_range
    ubucket=bucket_index+bucket_range
    index_range = [int((lbucket-1)*B), int(ubucket*B)]
    
    # check lrange or rrange is out of bounds
    if index_range[1] > ts.shape[0]:
        index_range[1]=ts.shape[0]-1
    if index_range[0]<0:
        index_range[0]=0
    # can happen with upper bound restriction
    if index_range[0]>index_range[1]:
        index_range[0]=int(index_range[1]-(bucket_range*B))
        
    return int(bucket_index), index_range
