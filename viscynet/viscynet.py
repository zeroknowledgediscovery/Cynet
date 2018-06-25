"""
Visualization library for cynet
@author zed.uchicago.edu
"""



import pandas as pd
import numpy as np
import json
import os
import warnings

# import subprocess as sp
# sp.call('source $HOME/miniconda/bin/activate', shell=True)

try:
    import cartopy.crs as ccrs
    import cartopy as crt
except ImportError:
    raise ImportError('Error: Please ensure cartopy is installed.\
    Due to failing builds of cartopy, this package cannot guarantee\
    correct installation of the cartopy dependency.')


import matplotlib as mpl
mpl.use('Agg')

from matplotlib import pyplot as plt
plt.ioff()

import matplotlib.cm as cm
# from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch
import matplotlib.colors as colors
from scipy.spatial import ConvexHull



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
        drawpoly=False,figname='fig',BGIMAGE=None,BGIMGNAME='BM',IMGRES='high',WIDTH=0.007):
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

    margin = 2 # buffer to add to the range
    lat_min = min(min(latsrc),min(lattgt)) - margin
    lat_max = max(max(latsrc),max(lattgt)) + margin
    lon_min = min(min(lonsrc),min(lontgt)) - margin
    lon_max = max(max(lonsrc),max(lontgt)) + margin

    fig=plt.figure(figsize=(20,15))

    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([lon_min,lon_max,lat_min,lat_max])
    if BGIMAGE is not None:
        ax.background_img(name=BGIMGNAME, resolution=IMGRES)
    else:
        #ax.stock_img()
    #ax.gridlines()
        ax.add_feature(crt.feature.LAND,facecolor='k')
        ax.add_feature(crt.feature.OCEAN,facecolor='.3')
        ax.add_feature(crt.feature.COASTLINE)
        ax.add_feature(crt.feature.LAKES,facecolor='.5', alpha=0.95)
    ax.add_feature(crt.feature.BORDERS,edgecolor='w',linewidth=2,linestyle='-', alpha=.1)
    y_o=latsrc
    x_o=lonsrc
    y=lattgt
    x=lontgt


#    CLS={}
#    for index in np.arange(len(lontgt)):
#        CLS[(lontgt[index],lattgt[index])]=[]

#    for index in np.arange(len(lontgt)):
#        CLS[(lontgt[index],
#             lattgt[index])].append((latsrc[index],
#                                     lonsrc[index]))

#    if drawpoly:
#        for key, value in CLS.iteritems():
#            a=[]
#            for i in value:
#                a.append(i[0])
#            b=[]
#            for i in value:
#                b.append(i[1])
#
#            a=np.array(a)
#            b=np.array(b)
#
#            zP=[[i[0],i[1]] for i in zip(a,b)]
#            hull = ConvexHull(zP)
#            aa=[a[i] for i in hull.vertices]
#            bb=[b[i] for i in hull.vertices]
#            draw_screen_poly(aa,bb,m,ax,0,colormap1,ALPHA=0.3)
#
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
                   headwidth=4,headlength=5,zorder=10)

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
