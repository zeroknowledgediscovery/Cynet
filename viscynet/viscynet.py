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

from multiprocessing import Pool

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
