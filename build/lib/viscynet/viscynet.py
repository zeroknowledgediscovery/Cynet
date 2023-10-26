"""
Visualization library for cynet
@author zed.uchicago.edu
"""
import pandas as pd
import numpy as np
import json
import os
import warnings

try:
    import cartopy.crs as ccrs
    import cartopy as crt
    import cartopy.io.shapereader as shpreader
    import cartopy.feature as cfeature
    from cartopy.io.shapereader import Reader
    from cartopy.feature import ShapelyFeature
    import cartopy.io.img_tiles as cimgt
except ImportError:
    raise ImportError('Error: Please ensure cartopy is installed.\
    Due to failing builds of cartopy, this package cannot guarantee\
    correct installation of the cartopy dependency.')
import matplotlib as mpl
mpl.use('Agg')

from matplotlib import pyplot as plt
plt.ioff()

import matplotlib.cm as cm
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch
import matplotlib.colors as colors
from scipy.spatial import ConvexHull
from cynet.cynet import uNetworkModels
import glob
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
from sklearn import metrics
import geopy.distance

def _scaleforsize(a):
    """
    normalize array for plotting
    @author zed.uchicago.edu

    Inputs -
        a (ndarray): input array
    Output -
        a (ndarray): output array
    """

    mx=np.percentile(a,98)
    mn=np.percentile(a,2)

    if mx > mn:
        a=np.array([(i-mn)/(mx-mn+0.0) for i in a])

    for index in np.arange(len(a)):
        if a[index] > 1:
            a[index]=1
        if a[index] < 0.001:
            a[index]=0.001

    return a


def draw_screen_poly(lats,lons,m,ax,val,cmap,ALPHA=0.6):
    """
    utility function to draw polygons on basemap
    @author zed.uchicago.edu

    Inputs -
        lats (list of floats): mpl_toolkits.basemap lat parameters
        lons (list of floats): mpl_toolkits.basemap lon parameters
        m (mpl.mpl_toolkits.Basemap): mpl instance for plotting
        ax (axis parent handle)
        cax (colorbar parent handle)
        val (Matplotlib color)
        cmap (string): colormap cmap parameter
        ALPHA (float): alpha value to use for plot

    Outputs -
        (No outputs - modifies objects in place)
    """
    norm = mpl.colors.Normalize(vmin=.75, vmax=.82)
    col = cm.ScalarMappable(norm=norm, cmap=cmap)
    x, y = m( lons, lats )
    xy = zip(x,y)
    poly = Polygon( xy, facecolor=col.to_rgba(val),
                   alpha=ALPHA, zorder=20,lw=0)
    ax.add_patch(poly)


def getalpha(arr,index,F=.9,M=0):
    """
    utility function to normalize transparency of quiver
    @author zed.uchicago.edu

    Inputs -
        arr (iterable): list of input values
        index (int): index position from which alpha value should be taken from
        F (float): multiplier
        M (float): minimum alpha value

    Outputs -
        v (float): alpha value
    """

    mn=np.min(arr)
    mx=np.max(arr)
    val=arr[index]
    v=(val-F*mn)/(mx-mn)
    if (v>1):
        v=1
    if (v<=M):
        v=M
    return v


def viz(unet,jsonfile=False,colormap='autumn',res='c',
        drawpoly=False,figname='fig',BGIMAGE=None,BGIMGNAME='BM',
        IMGRES='high',WIDTH=0.007,SHPPATH=None,
        OCEAN_FACECOLOR='.3',LAND_FACECOLOR='k',LAKE_FACECOLOR='.3',
        FIGSIZE=None,EXTENT=None,ASPECTAUTO=False):
    """
    utility function to visualize spatio temporal interaction networks
    @author zed.uchicago.edu

    Inputs -
            unet (string): json filename
            unet (python dict):
            jsonfile (bool): True if unet is string  specifying json filename
            colormap (string): colormap
            res (string): 'c' or 'f'
            drawpoly (bool): if True draws transparent patch showing srcs
            figname  (string): prefix of pdf image file
    Returns -
            m (Basemap handle)
            fig (figure handle)
            ax (axis handle)
            cax (colorbar handle)
    """
    if BGIMAGE is not None:
        os.environ['CARTOPY_USER_BACKGROUNDS'] = BGIMAGE

    f=lambda x: x[:-1] if len(x)%2==1  else x
    if jsonfile:
         with open(unet) as data_file:
            unet_ = json.load(data_file)
    else:
        unet_=unet

    colormap=colormap
    colormap1='hot_r'

    latsrc=[]
    lonsrc=[]
    lattgt=[]
    lontgt=[]
    gamma=[]
    delay=[]
    NUM=None
    for key,value in unet_.iteritems():
        src=[float(i) for i in f(value['src'].replace('#',' ').split())]
        tgt=[float(i) for i in f(value['tgt'].replace('#',' ').split())]
        if NUM is None:
            NUM=len(src)/2
        latsrc.append(np.mean(src[0:NUM]))
        lonsrc.append(np.mean(src[NUM:]))
        lattgt.append(np.mean(tgt[0:NUM]))
        lontgt.append(np.mean(tgt[NUM:]))
        gamma.append(value['gamma'])
        delay.append(value['delay'])

    if EXTENT is None:
        margin = 0.02 # buffer to add to the range
        lat_min = min(min(latsrc),min(lattgt)) - margin
        lat_max = max(max(latsrc),max(lattgt)) + margin
        lon_min = min(min(lonsrc),min(lontgt)) - margin
        lon_max = max(max(lonsrc),max(lontgt)) + margin
    else:
        lat_min = EXTENT[0]
        lat_max = EXTENT[1]
        lon_min = EXTENT[2]
        lon_max = EXTENT[3]
        
        
    if FIGSIZE is None:
        fig=plt.figure(figsize=(10,10*np.round((lon_max-lon_min)/(lat_max-lat_min))))
    else:
        fig=plt.figure(figsize=(FIGSIZE[0],FIGSIZE[1]))
    PROJ=ccrs.PlateCarree()#ccrs.LambertConformal()

    ax = plt.axes([0, 0, 1, 1],
                  projection=PROJ)

    ax.set_extent([lon_min,lon_max,lat_min,lat_max],crs=PROJ)
    if BGIMAGE is not None:
        ax.background_img(name=BGIMGNAME, resolution=IMGRES)
    else:
        #ax.stock_img()
        #ax.gridlines()
        ax.add_feature(crt.feature.LAND,facecolor=LAND_FACECOLOR)
        ax.add_feature(crt.feature.OCEAN,facecolor=OCEAN_FACECOLOR)
        ax.add_feature(crt.feature.COASTLINE)
        ax.add_feature(crt.feature.LAKES,
                       facecolor=LAKE_FACECOLOR,alpha=0.95)
        ax.add_feature(crt.feature.BORDERS,edgecolor='w',
                       linewidth=2,linestyle='-', alpha=.1)

    if SHPPATH is not None:
        shpf = shpreader.Reader(SHPPATH)
        #request = cimgt.OSM()
        #ax.add_image(request, 10, interpolation='spline36', regrid_shape=2000)

        for record, ward in zip(shpf.records(), shpf.geometries()):
            try:
                colorNormalized = '.3'
                ax.add_geometries([ward],PROJ,
                                  facecolor=colorNormalized,
                                  alpha=0.5,
                                  edgecolor='.4', linewidth=1,)
            except KeyError:
                ax.add_geometries([ward],PROJ,
                                  facecolor='.3',
                                  alpha=0.5,
                                  edgecolor=None, linewidth=0)


    y_o=latsrc
    x_o=lonsrc
    y=lattgt
    x=lontgt

    norm = mpl.colors.LogNorm(vmin=(np.min(np.array(gamma))),
                          vmax=(np.max(np.array(gamma))))
    colx = cm.ScalarMappable(norm=norm,
                         cmap=colormap)

    #WIDTH=0.007      # arrow width (scaled by gamma)
    ALPHA=1  # opacity for arrows (scaled by gamma)
    for index in np.arange(len(x)):
        plt.quiver(x_o[index], y_o[index],x[index]-x_o[index],
                   y[index]-y_o[index],color=colx.to_rgba(gamma[index]),
                   alpha=ALPHA*getalpha(gamma,index,F=.1,M=.3),
                   scale_units='xy',angles='xy',scale=1.05,
                   width=WIDTH*getalpha(gamma,index,F=.7,M=.4),
                   headwidth=4,headlength=5,zorder=10,transform=PROJ)

    if ASPECTAUTO:
        ax.set_aspect('auto')
    cax, _ = mpl.colorbar.make_axes(ax, shrink=0.5)
    cbar = mpl.colorbar.ColorbarBase(cax, cmap=colormap,
                                     norm=mpl.colors.Normalize(vmin=np.min(np.array(gamma)),
                                                               vmax=(np.max(np.array(gamma)))))
    pos1=cax.get_position()
    pos2 = [pos1.x0-.03, pos1.y0+.15 ,pos1.width, pos1.height/4]
    cax.set_position(pos2) # set a new position
    plt.setp(plt.getp(cax, 'yticklabels'), color='k')
    plt.setp(plt.getp(cax, 'yticklabels'), fontweight='bold')
    cax.set_title('$\gamma$',fontsize=18,color='k',y=1.05)

    plt.savefig(figname+'.pdf',dpi=300,bbox_inches='tight',transparent=True)

    return fig,ax,cax



def render_network(model_path,MAX_DIST,MIN_DIST,MAX_GAMMA,MIN_GAMMA,
                    COLORMAP,Horizon,model_nums, newmodel_name='newmodel.json',
                    figname='fig2'):
    '''
    For use after model.json files are produced via XgenESeSS. Will produce a
    network interaction map of all the models. Requires vizcynet import
    Inputs:
        model_path(str)- path to the model.json files.
        MAX_DIST(int)- max distance cutoff in render network.
        MIN_DIST(int)- min distance cutoff in render network.
        MAX_GAMMA(float)- max gamma cutoff in render network.
        MIN_GAMMA(float)- min gamma cutoff in render network.
        COLORMAP(str)- colormap in render network.
        Horizon(int)- prediction horizons to test in unit of temporal
            quantization.
        model_nums(int)- number of models to use in prediction.
        newmodel_name(str): Name to save the newmodel as. This new model
            will be loaded in by viz.
        figname(str)-Name of figure drawn)
    '''
    first=True
    for jfile  in tqdm(glob.glob(model_path)):
        if first:
            M=uNetworkModels(jfile)
            M.setVarname()
            M.augmentDistance()
            M.select(var='distance',low=MIN_DIST,inplace=True)
            M.select(var='distance',high=MAX_DIST,inplace=True)
            M.select(var='delay',inplace=True,low=Horizon)
            M.select(var='distance',n=model_nums,
                     reverse=False,inplace=True)
            M.select(var='gamma',high=MAX_GAMMA,low=MIN_GAMMA,inplace=True)
            first=False
        else:
            mtmp=uNetworkModels(jfile)
            if mtmp.models:
                mtmp.setVarname()
                mtmp.augmentDistance()
                mtmp.select(var='distance',high=MAX_DIST,inplace=True)
                mtmp.select(var='distance',low=MIN_DIST,inplace=True)
                mtmp.select(var='delay',inplace=True,low=Horizon)
                mtmp.select(var='distance',n=model_nums,reverse=False,
                            inplace=True)
                mtmp.select(var='gamma',high=MAX_GAMMA,low=MIN_GAMMA,inplace=True)
                M.append(mtmp.models)

    M.to_json(newmodel_name)
    viz(newmodel_name,jsonfile=True,colormap=COLORMAP,res='c',
           drawpoly=False,figname=figname,BGIMAGE=None,BGIMGNAME=None,WIDTH=0.005)


def render_network_parallel(model_path,MAX_DIST,MIN_DIST,MAX_GAMMA,MIN_GAMMA,
                    COLORMAP,Horizon,model_nums, newmodel_name='newmodel.json',
                    figname='fig2',workers=4,rendered_glob='models/*_rendered.json'):
    '''
    This function aims to achieve the same thing as render_network but in
    parallel.
    Inputs:
        model_path(str)- path to the model.json files.
        MAX_DIST(int)- max distance cutoff in render network.
        MIN_DIST(int)- min distance cutoff in render network.
        MAX_GAMMA(float)- max gamma cutoff in render network.
        MIN_GAMMA(float)- min gamma cutoff in render network.
        COLORMAP(str)- colormap in render network.
        Horizon(int)- prediction horizons to test in unit of temporal
            quantization.
        model_nums(int)- number of models to use in prediction.
        newmodel_name(str): Name to save the newmodel as. This new model
            will be loaded in by viz.
        figname(str)-Name of figure drawn)
    '''
    arguments = []
    counter = 1
    for jfile in glob.glob(model_path):
        arguments.append([jfile,MIN_DIST,MAX_DIST,MIN_GAMMA,MAX_GAMMA,Horizon,model_nums,counter])
        counter += 1

    pool = Pool(workers)
    pool.map(individual_render, arguments)

    combined_dict = {}
    for rendered_jfile in glob.glob(rendered_glob):
        with open(rendered_jfile) as entry_dict:
            combined_dict.update(json.load(entry_dict))

    with open(newmodel_name,'w') as outfile:
        json.dump(combined_dict, outfile)

    viz(newmodel_name,jsonfile=True,colormap=COLORMAP,res='c',
           drawpoly=False,figname=figname,BGIMAGE=None,BGIMGNAME=None,WIDTH=0.005)


def individual_render(arguments):
    '''
    A rendering for a single file. This function is called by render_network_parallel.
    arguments(list)-list of arguments for the rendering.
        arguments[0]:model_path(str)- path to the model.json files.
        arguments[1]:MIN_DIST(int)- min distance cutoff in render network.
        arguments[2]:MAX_DIST(int)- max distance cutoff in render network.
        arguments[3]:MIN_GAMMA(float)- min gamma cutoff in render network.
        arguments[4]:MAX_GAMMA(float)- max gamma cutoff in render network.
        arguments[5]:Horizon(int)- prediction horizons to test in unit of temporal
            quantization.
        arguments[6]:model_nums(int)- number of models to use in prediction.
        arguments[7]:counter(int)- a number to keep track of model progress.
    '''
    jfile = arguments[0]
    MIN_DIST = arguments[1]
    MAX_DIST = arguments[2]
    MIN_GAMMA = arguments[3]
    MAX_GAMMA = arguments[4]
    Horizon = arguments[5]
    model_nums = arguments[6]
    counter = arguments[7]
    print "Rendering a model {}".format(counter)
    new_jfile =  jfile.replace('.json','_rendered.json')
    M=uNetworkModels(jfile)
    M.setVarname()
    M.augmentDistance()
    M.select(var='distance',low=MIN_DIST,inplace=True)
    M.select(var='distance',high=MAX_DIST,inplace=True)
    M.select(var='delay',inplace=True,low=Horizon)
    M.select(var='distance',n=model_nums,
                reverse=False,inplace=True)
    M.select(var='gamma',high=MAX_GAMMA,low=MIN_GAMMA,inplace=True)

    with open(new_jfile, 'w') as outfile:
        json.dump(M.models, outfile)

    return


def getHausdorf(coord,pt):
    '''
    Uses the geopy library distance function to find
    the minimal distance(on the spherical globe) from a point
    given by a lat and lon coordinates and other points
    in coord.
    Inputs
    '''
    return np.min([geopy.distance.distance(pt,i).miles for i in coord])


def getHausdorf_df(df,pt,EPS=0.0001):
    '''
    Uses getHausdorf function to find the smallest distance between
    pt and a tile in df. Will be used to find the closest tile with
    intensity greater than Z. Used to allow for distance grace.
    '''
    while True:
        T=[tuple(i) for i in df[(np.abs(df.lat-pt[0])<EPS) 
                                          & (np.abs(df.lon-pt[1])<EPS)].values]
        if len(T)>0:
            break
        else:
            EPS=2*EPS
    return getHausdorf(T,tuple(pt)),T

def get_intensity(intensity,lon_mesh,lat_mesh,pt_,sigma=3,radius=2):
    '''
    single point spread calculation with Gaussian diffusion. Used
    to find the intensity of a tile.
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
            intensity[lat_index[j],lon_index[i]]=intensity[lat_index[j],lon_index[i]]\
                                                +intensity_lon[i]*intensity_lat[j]

    return intensity


def get_mesh(df0,lat_min,lat_max,lon_min,lon_max,radius=2,detail=0.25,lat_col='lat2',lon_col='lon2'):
    '''
    Returns the mesh grid and the coordinates used to generate them.
    '''
    coord_=df0[[lat_col,lon_col]].values

    lon_grid=np.arange(lon_min-radius,lon_max+radius,detail)
    lat_grid=np.arange(lat_min-radius,lat_max+radius,detail)
    lon_mesh,lat_mesh=np.meshgrid(lon_grid,lat_grid)
    return lon_mesh,lat_mesh,coord_

def df_intersect(df1,df2,columns=[]):
    '''
    Returns the intersection of dataframes. Specifically used here
    with tiles. So we use this to find the set of tiles in
    both dataframes.
    '''
    df1__=df1[columns]
    df2__=df2[columns]
    
    df1__m = df1__.apply(lambda x: hash(tuple(x)), axis=1)
    df2__m = df2__.apply(lambda x: hash(tuple(x)), axis=1)
    #bl=df1__['match'].isin(df2__['match']).values
    df_=df1[df1__m.isin(df2__m)]
    
    return df_


def df_setdiff(df1,df2,columns=[]):
    '''
    Returns the set of tiles in dataframe 1 that is not in 
    dataframe 2.
    '''
    df1__=df1[columns]
    df2__=df2[columns]
    
    df1__m = df1__.apply(lambda x: hash(tuple(x)), axis=1)
    df2__m = df2__.apply(lambda x: hash(tuple(x)), axis=1)
    #bl=df1__['match'].isin(df2__['match']).values
    df_=df1[~df1__m.isin(df2__m)]
    
    return df_


def get_prediction(df,days,lat_min,lat_max,lon_min,lon_max,source,types,
                   radius=0.01,detail=0.2,save=False,
                   startdate="12/31/2017",offset=1095,Z=1.0,SINGLE=True,
                   day_col='day',grace=1,variable_col='target',source_col='source',
                   actual_event_col='actual_event',prediction_col='predictions',
                   lat_col='lat2',lon_col = 'lon2'):
    '''
    For use with Chicago crime example only.
    Calculates the true positives, false positives, and false negatives
    of our predictions. We do allow for a grace of one day in our
    predictions. That is, if we are off by one day in our prediction,
    we count that as a correct prediction. Also allows for a small distance 
    grace in our predictions. 
    '''
    dt=pd.to_datetime(startdate) + pd.DateOffset(days=days-offset)
    dt=dt.strftime('%m-%d-%Y')
    
    df = df[df[day_col].between(days-grace,days+grace)]
    df = df[df[variable_col].isin(types)]
    df = df[df[source_col] == source]
    df_gnd = df[(df[day_col]==days) & (df[actual_event_col]==1)]
    df_prd0 = df[(df[day_col]==days) & (df[prediction_col]==1)]
    df_prd1 = df[(df[day_col]==days-grace) & (df[prediction_col]==1)]
    df_prd2 = df[(df[day_col]==days+grace) & (df[prediction_col]==1)]

    # true positives .. will need to add true positives from yesterday matches.
    df_prd0_tp = df_prd0[df_prd0[actual_event_col]==1]
    tp=df_prd0_tp.index.size

    # this is not quite false positives, because of possible matches in grace (false pos=_fp)
    df_prd0_fp = df_prd0[df_prd0[actual_event_col]==0]
    df_prd0_fp = df_setdiff(df_prd0_fp,df_prd0_tp,columns=[lat_col,lon_col])
    #Actual events from yesterday and tommorow.
    df_gnd1 = df[(df[day_col]==days-1) & (df[actual_event_col]==1)]
    df_gnd2 = df[(df[day_col]==days+1) & (df[actual_event_col]==1)]

    c1 = df_intersect(df_gnd1,df_prd0_fp,columns=[lat_col,lon_col])
    c2 = df_intersect(df_gnd2,df_prd0_fp,columns=[lat_col,lon_col])
    df_c = pd.concat([c1,c2], sort=True)
    #Predictions of today that did not match with events from today but
    #did match with either yesterday or tommorow.

    # Now we calculate correct false positives
    df_prd0_fp = df_setdiff(df_prd0_fp,df_c,columns=[lat_col,lon_col])
    #Predictions that do not match today, yesterday, or tommorow.

    df_gnd0=df_gnd.copy()
    df_gnd = df_intersect(df_gnd,df_prd1, columns=[lat_col,lon_col])
    df_prd0_ = df_intersect(df_prd1,df_gnd, columns=[lat_col,lon_col])
    #Predictions of yesterday that matched with today.

    df_prd0_fp = df_setdiff(df_prd0_fp,df_prd0_,columns=[lat_col,lon_col])
    #False positives that do not coincide with correct predictions made
    #yesterday either.

    #Include true positives from matching yesterday.
    tp=tp+df_prd0_.index.size
        
    #concat df_prd0_tp  df_prd0_fp df_prd0_
    df_prd0 = pd.concat([df_prd0_tp , df_prd0_fp, df_prd0_], sort=True)

    lon_mesh0,lat_mesh0,coord_=get_mesh(df_prd0,lat_min,lat_max,lon_min,lon_max,radius=radius,detail=detail)
    intensity0=np.zeros(lat_mesh0.shape)
    if SINGLE:
        print 'calculating intensity'
        for i in tqdm(coord_):
            intensity0=get_intensity(intensity0,lon_mesh0,lat_mesh0,i,sigma=3.5,radius=radius)
    else:
        for i in coord_:
            intensity0=get_intensity(intensity0,lon_mesh0,lat_mesh0,i,sigma=3.5,radius=radius)

    intensity0=np.multiply(intensity0,(intensity0>Z))    
    dfpu=df_prd0_fp[[lat_col,lon_col]].drop_duplicates()
    lon_del=lon_mesh0[0,:]
    lat_del=lat_mesh0[:,0]
    A=(intensity0>Z).nonzero()
    coordNZ=[(lat_del[A[0][i]],lon_del[A[1][i]]) for i in np.arange(len(A[0]))]
    df_cnz=pd.DataFrame(coordNZ,columns=['lat','lon'])
    #df_cnz is dataframe of coordinates of tiles whose intensity are greater than 0.
    if SINGLE:
        print 'calculating FP'
        xg=np.array([getHausdorf_df(df_cnz,tuple(i),EPS=0.0001)[0] for i in tqdm(dfpu.values)])
    else:
        xg=np.array([getHausdorf_df(df_cnz,tuple(i),EPS=0.0001)[0] for i in (dfpu.values)])
    #Tiles not close enough to other tiles with positive intensity are false positives.
    fp=np.sum(xg<0.01)
    
    
    dfnu=df_gnd0[[lat_col,lon_col]].drop_duplicates()
    if SINGLE:
        print 'Calculating FN'
        xgfn=np.array([getHausdorf_df(df_cnz,tuple(i),EPS=0.0001)[0] for i in tqdm(dfnu.values)])
    else:
        xgfn=np.array([getHausdorf_df(df_cnz,tuple(i),EPS=0.0001)[0] for i in (dfnu.values)])
    fn=np.sum(xgfn>0.02)
    #Tiles which we fail to predict and are not close enough to one of our predictions
    #is considered false negatives.
    df_gnd_augmented = pd.concat([df_prd0_,df_gnd0], sort=True)
    
    return dt,fp,fn,tp,df_gnd_augmented,lon_mesh0,lat_mesh0,intensity0
 

def getFigure(days,dt,fp,fn,tp,df_gnd_augmented,lon_mesh,lat_mesh,intensity,
                lat_col='lat2',lon_col='lon2',
                fname=None,cmap='terrain',
                save=True,PREFIX='fig'):
    '''
    Draws a heatmap. For use with the getPrediction function. Mainly used in
    the UChicago example.
    '''
    fig=plt.figure(figsize=(10,5.5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    if fname is not None:
        ax.add_geometries(Reader(fname).geometries(),ccrs.PlateCarree(),
                          edgecolor='w', lw=1.5, facecolor="w",alpha=.15)

    plt.plot(df_gnd_augmented[lon_col].values,df_gnd_augmented[lat_col].values,'ro',alpha=.4,ms=1)
    plt.pcolormesh(lon_mesh,lat_mesh,intensity,cmap=cmap,
                   alpha=1,edgecolor=None,linewidth=0)
    plt.xlim(lon_mesh.min(),lon_mesh.max())
    plt.ylim(lat_mesh.min(),lat_mesh.max())
    props = dict(boxstyle='round', facecolor='w', alpha=0.95)
    props1 = dict(boxstyle='round', facecolor=None,lw=0, edgecolor=None,alpha=0.05)

    ax.text(0.98, 0.9,dt, transform=ax.transAxes,fontweight='bold',fontsize=8,color='k',
            verticalalignment='top', horizontalalignment='right', bbox=props)

    ax.text(0.98, 0.98,'Horizon: 6-8 days',
            transform=ax.transAxes, fontsize=8,color='w',fontweight='bold',
            verticalalignment='top', horizontalalignment='right', bbox=props1)

    ax2 = plt.gcf().add_axes([0.325, 0.2, 0.07, 0.15])
    ax2.patch.set_alpha(0)

    plt.bar(['FN','TP','FP'],[fn,tp,fp],color='r',lw=0,alpha=.5)

    ax2.spines['bottom'].set_color('w')
    ax2.spines['top'].set_color('w') 
    ax2.spines['right'].set_visible(False) 
    ax2.spines['left'].set_visible(False) 

    ax2.tick_params(axis='x', colors='w')
    ax2.tick_params(axis='y', colors='w')
    ax2.grid(True)
    for label in ax2.get_yticklabels():
        label.set_color('w')
        label.set_fontsize(6)
        label.set_fontweight('bold')
    for label in ax2.get_xticklabels():
        label.set_color('w')
        label.set_fontsize(8)
        label.set_fontweight('bold')

    ax2.tick_params(axis=u'both', which=u'both',length=0)
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)

    if save:
        plt.savefig(PREFIX+str(days).zfill(5)+'.png',dpi=300, bbox_inches='tight',pad_inches = 0)
    return ax

def getFigure_detailed(days,dt,fp,fn,tp,df_gnd_augmented,lon_mesh,lat_mesh,intensity,
                types,short_type_names,spatial_resolution_lat,spatial_resolution_lon,
                lat_col='lat2',lon_col='lon2', fname=None,cmap='terrain',
                temporal_quantization='1 day',save=True,PREFIX='fig'):
    '''
    Draws a heatmap. For use with the getPrediction function. Mainly used in
    the UChicago example.
    '''
    fig=plt.figure(figsize=(10,5.5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    if fname is not None:
        ax.add_geometries(Reader(fname).geometries(),ccrs.PlateCarree(),
                          edgecolor='w', lw=1.5, facecolor="w",alpha=.15)

    
    plt.plot(df_gnd_augmented[lon_col].values,df_gnd_augmented[lat_col].values,'ro',alpha=.4,ms=1)
    plt.pcolormesh(lon_mesh,lat_mesh,intensity,cmap=cmap,
                   alpha=1,edgecolor=None,linewidth=0)
    plt.xlim(lon_mesh.min(),lon_mesh.max())
    plt.ylim(lat_mesh.min(),lat_mesh.max())
    props = dict(boxstyle='round', facecolor='w', alpha=0.95)
    props1 = dict(boxstyle='round', facecolor=None,lw=0, edgecolor=None,alpha=0.05)

    ax.text(0.98, 0.9,dt, transform=ax.transAxes,fontweight='bold',fontsize=8,color='k',
            verticalalignment='top', horizontalalignment='right', bbox=props)
    ax.text(0.02, 0.02,'zed.uchicago.edu', transform=ax.transAxes,fontweight='bold',
            fontsize=8,color='k',
            verticalalignment='bottom', horizontalalignment='left', bbox=props)
    ax.text(0.02, 0.06,'data source: City of Chicago', transform=ax.transAxes,fontweight='bold',
            fontsize=8,color='.7',
            verticalalignment='bottom', horizontalalignment='left', bbox=props1)

    sourcetype='('+', '.join([short_type_names[i] for i in types])+')'
    ax.text(0.98, 0.98,'Event Prediction '+sourcetype+'\nHorizon: 6-8 days',
            transform=ax.transAxes, fontsize=8,color='w',fontweight='bold',
            verticalalignment='top', horizontalalignment='right', bbox=props1)

    ax.text(0.98, 0.02,'Red dots: actual events\nLatitude Res.: '+\
            spatial_resolution_lat+'\nLongitude Res.: '+spatial_resolution_lon
            +'\nTemporal Res.: '+temporal_quantization+'\nRegions With Event Rate > 5% Considered',
            transform=ax.transAxes, fontsize=8,color='.7',fontweight='bold',
            verticalalignment='bottom', horizontalalignment='right', bbox=props1)

    ax2 = plt.gcf().add_axes([0.325, 0.2, 0.07, 0.15])
    ax2.patch.set_alpha(0)


    plt.bar(['FN','TP','FP'],[fn,tp,fp],color='r',lw=0,alpha=.5)

    ax2.spines['bottom'].set_color('w')
    ax2.spines['top'].set_color('w') 
    ax2.spines['right'].set_visible(False) 
    ax2.spines['left'].set_visible(False) 

    ax2.tick_params(axis='x', colors='w')
    ax2.tick_params(axis='y', colors='w')
    ax2.grid(True)
    for label in ax2.get_yticklabels():
        label.set_color('w')
        label.set_fontsize(6)
        label.set_fontweight('bold')
    for label in ax2.get_xticklabels():
        label.set_color('w')
        label.set_fontsize(8)
        label.set_fontweight('bold')

    ax2.tick_params(axis=u'both', which=u'both',length=0)
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)

    if save:
        plt.savefig(PREFIX+str(days).zfill(5)+'.png',dpi=300, bbox_inches='tight',pad_inches = 0)
    return ax