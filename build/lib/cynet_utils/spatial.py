import pandas as pd
import numpy as np
import math
import json
from tqdm import tqdm
from time import time
from datetime import datetime, timedelta 
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

import matplotlib.pyplot as plt
import matplotlib.cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import geopy.distance
import geopandas as gpd
import contextily as ctx



def saveFIG(filename='tmp.pdf'):
    import pylab as plt
    
    plt.subplots_adjust(
        top=1, 
        bottom=0, 
        right=1, 
        left=0, 
        hspace=0, 
        wspace=0)

    plt.margins(0, 0)
    
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())

    plt.savefig(filename, dpi=300, bbox_inches=0, transparent=True) 
    return


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def lighten_color(color, amount=0.5):
    
    """
    By Ian Hincks from stack overflow
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])



# =========================== DF column names ==========================START
day_col = 'day'
actual_event_col = 'actual_event'
variable_col = 'target'
source_col = 'source'
predictin_col = 'predictions'
lon_col = 'lon2'
lat_col = 'lat2'
source = None
grace = 1
# =========================== DF column names ==========================END


def df_intersect(df1, df2, columns=[]):
    df1__ = df1[columns]
    df2__ = df2[columns]

    df1__m = df1__.apply(lambda x: hash(tuple(x)), axis=1)
    df2__m = df2__.apply(lambda x: hash(tuple(x)), axis=1)

    df_ = df1[df1__m.isin(df2__m)]

    return df_


def df_setdiff(df1, df2, columns=[]):
    df1__ = df1[columns]
    df2__ = df2[columns]

    df1__m = df1__.apply(lambda x: hash(tuple(x)), axis=1)
    df2__m = df2__.apply(lambda x: hash(tuple(x)), axis=1)

    df_ = df1[~df1__m.isin(df2__m)]

    return df_


def df_union(df_1, df_2, columns=[], count_only=False):

    dfh_1 = df_1[columns].apply(lambda x: hash(tuple(x)), axis=1)
    dfh_2 = df_2[columns].apply(lambda x: hash(tuple(x)), axis=1)
    
    diff = df_1[~dfh_1.isin(dfh_2)]
    union = pd.concat([diff, df_2], axis=0, sort=False)
    if count_only:
        return len(union)
    else:
        return union
    

def transCMAP(cmap=plt.cm.RdBu,linear=True):
    cmap1 = cmap(np.arange(cmap.N))
    if linear:
        cmap1[:,-1] = np.linspace(0, 1, cmap.N)
    else:
        cmap1[:,-1] = np.logspace(0, 1, cmap.N)
    return ListedColormap(cmap1)


def getHausdorf(coord,pt):
    return np.min([geopy.distance.distance(pt,i).miles for i in coord])


def getHausdorf_df(df, pt, EPS=0.0001):
    if len(df) == 0:
        return np.inf, []
    
    while True:
        T = [tuple(i) for i in df[(np.abs(df.lat-pt[0])<EPS) 
              & (np.abs(df.lon-pt[1])<EPS)].values]
        if len(T)>0:
            break
        else:
            EPS=2*EPS
    return getHausdorf(T,tuple(pt)),T


def get_intensity(intensity,lon_mesh,lat_mesh,pt_,sigma=3,radius=2):
    '''
    single point spread calculation with Gaussian diffusion
    '''
    lon_del=lon_mesh[0,:]
    lat_del=lat_mesh[:,0]
    lon_index=np.arange(len(lon_del))[(pt_[1]-lon_del<radius)*(pt_[1]-lon_del>-radius)]
    lat_index=np.arange(len(lat_del))[(pt_[0]-lat_del<radius)*(pt_[0]-lat_del>-radius)]
    mu=np.mean(lon_index)
    bins=lon_index
    intensity_lon=1/(sigma*np.sqrt(2*np.pi))*np.exp(-(bins - mu)**2/(2 * sigma**2))

    mu=np.mean(lat_index)
    bins=lat_index
    intensity_lat=1/(sigma*np.sqrt(2*np.pi))*np.exp(-(bins - mu)**2/(2 * sigma**2))
    for i in np.arange(len(lon_index)):
        for j in np.arange(len(lat_index)):
            intensity[lat_index[j],lon_index[i]]=intensity[lat_index[j],lon_index[i]]                +intensity_lon[i]*intensity_lat[j]
    return intensity


def get_mesh(df0,lat_min,lat_max,lon_min,lon_max,radius=2,detail=0.25):
    coord_=df0[[lat_col,lon_col]].values
    lon_grid=np.arange(lon_min-radius,lon_max+radius,detail)
    lat_grid=np.arange(lat_min-radius,lat_max+radius,detail)
    lon_mesh,lat_mesh=np.meshgrid(lon_grid,lat_grid)
    return lon_mesh,lat_mesh,coord_


def get_prediction(
        df,
        days,
        types,
        lat_min,
        lat_max,
        lon_min,
        lon_max,
        sigma=3.5, #=======YI made sigma a parameter
        radius=0.01,
        detail=0.2,
        Z=1.0,
        miles=50 #=======YI made miles in spatial relaxation a paramter
    ):

    # =========================== DF column names ==========================START
    day_col = 'day'
    actual_event_col = 'actual_event'
    variable_col = 'target'
    source_col = 'source'
    predictin_col = 'predictions'
    lon_col = 'lon2'
    lat_col = 'lat2'
    source = None
    grace = 1
    # =========================== DF column names ==========================END

    df = df[df[day_col].between(days - grace,days + grace)]
    df = df[df[variable_col].isin(types)]
    # df = df[df[source_col] == source]
    
    df_gnd = df[(df[day_col]==days) & (df[actual_event_col]==1)]
    df_prd0 = df[(df[day_col]==days) & (df[predictin_col]==1)]
    df_prd1 = df[(df[day_col]==days - grace) & (df[predictin_col]==1)]
    df_prd2 = df[(df[day_col]==days + grace) & (df[predictin_col]==1)]
    
    df_prd0_tp = df_prd0[df_prd0[actual_event_col]==1]


    # UPDXX calculate tp
    df_gndB = df[(df[day_col]==days-grace) & (df[actual_event_col]==1)]
    df_gndF = df[(df[day_col]==days+grace) & (df[actual_event_col]==1)]    
    df_tpB = df_intersect(df_prd0,df_gndB, columns=[lat_col, lon_col])
    df_tpF = df_intersect(df_prd0,df_gndF, columns=[lat_col, lon_col])
    df_tp = df_union(
        df_union(df_prd0_tp, df_tpB, columns=[lat_col, lon_col]),
        df_tpF,
        columns=[lat_col, lon_col])
    tp = df_tp.index.size
    
    df_fp = df_setdiff(df_prd0,df_tp,columns=[lat_col, lon_col])
    fp = df_fp.index.size
    
    df_fn0 = df[(df[day_col]==days) & (df[actual_event_col]==1) & (df[predictin_col]==0)]
    df_fn1 = df[(df[day_col]==days - grace)  & (df[predictin_col]==0)]
    df_fn2 = df[(df[day_col]==days + grace)  & (df[predictin_col]==0)]
    df_fn = df_intersect(df_intersect(df_fn0,df_fn1,columns=[lat_col, lon_col]),
                      df_fn2,columns=[lat_col, lon_col])
    fn= df_fn.index.size
    
    print('tmporal comp: --> ', 'tp ',tp, ' fp ', fp, ' fn ',fn)
        
    # SPATIAL ADJUSTMENT
    lon_grid = np.arange(lon_min - radius, lon_max + radius, detail)
    lat_grid = np.arange(lat_min - radius, lat_max + radius, detail)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid,lat_grid)
    
    lon_mesh0, lat_mesh0, coord_= get_mesh(
        df_prd0,
        lat_min,
        lat_max,
        lon_min,
        lon_max,
        radius=radius,
        detail=detail)
    
    intensity = np.zeros(lat_mesh0.shape)
    for i in coord_:
        intensity = get_intensity(
            intensity,
            lon_mesh0,
            lat_mesh0,
            i,
            sigma=sigma,
            radius=radius)
        
    intensity0 = np.multiply(intensity, (intensity > Z))
    intensity0=(1. / intensity0.max()) * intensity0
    
    lon_del=lon_mesh0[0,:]
    lat_del=lat_mesh0[:,0]
    A=(intensity0>Z).nonzero()
    coordNZ=[(lat_del[A[0][i]],lon_del[A[1][i]]) for i in np.arange(len(A[0]))]
    df_cnz=pd.DataFrame(coordNZ,columns=['lat','lon'])

    xgfp = np.array([getHausdorf_df(df_cnz,tuple(i),EPS=0.01)[0] for i in (df_fp[[lat_col,lon_col]].drop_duplicates().values)])
    fp = np.sum(xgfp < miles)
    
    xgfn = np.array([getHausdorf_df(df_cnz, tuple(i), EPS=0.01)[0] for i in (df_fn[[lat_col,lon_col]].drop_duplicates().values)])
    fn = np.sum(xgfn > 2 * miles)
    
    df_tp_0 = df_intersect(df_tp, df_prd0,columns=[lat_col, lon_col])
    return fn, tp, fp, tp/(tp+fp), tp/(tp+fn), lon_mesh0, lat_mesh0, intensity, intensity0, df_gnd, df_fn,df_tp,df_fp,df_tp_0