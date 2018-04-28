"""
Visualization library for cynet
@author zed.uchicago.edu
"""

import pandas as pd
import numpy as np

try:
    import cPickle as pickle
except ImportError: # python version will be 3 if this returns an error
    import pickle

from datetime import datetime
from datetime import timedelta
from tqdm import tqdm, tqdm_pandas
from haversine import haversine
import json
from sodapy import Socrata

import matplotlib as mpl
mpl.use('Agg')

from matplotlib import pyplot as plt
plt.ioff()

import matplotlib.cm as cm
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch
import matplotlib.colors as colors
from scipy.spatial import ConvexHull
import seaborn as sns

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, LogColorMapper, LinearColorMapper, Segment, Circle
from bokeh.models.tools import HoverTool, BoxSelectTool, TapTool
from bokeh.models.widgets import DataTable, TableColumn, NumberFormatter
from bokeh.io import show, output_file
from bokeh.palettes import RdYlBu11 as palette
from bokeh.tile_providers import CARTODBPOSITRON
from bokeh.layouts import widgetbox, gridplot


def showGlobalPlot(coords,ts=None,fsize=[14,14],cmap='jet',
                   m=None,figname='fig',F=2):
    """
    plot global distribution of events
    within time period specified
    @author zed.uchicago.edu

    Inputs -
        coords (string): filename with coord list as lat1#lat2#lon1#lon2
        ts (string): time series filename with data in rows, space separated
        fsize (list): figure dimensions
        cmap (string): Colormap parameter
        m (mpl.mpl_toolkits.Basemap): mpl instance for plotting
        figname (string): Name of the Plot
        F (int)

    Output -
        num (np.array): data values
        fig (mpl.figure): heatmap of events from fitted data
        ax (axis handler): output axis handler
        cax (colorbar axis handler): output colorbar axis handler
    """

    num=None
    if ts is not None:
        num=pd.read_csv(ts,header=None,sep=" ").sum(axis=1).values
        coords_=pd.read_csv(coords,header=None,sep=" ")[0].values

    if num is not None:
        num=np.array(_scaleforsize(num))

    fig=plt.figure(figsize=(14,14))
    ax=fig.add_subplot(111)

    A=[]
    for value in coords_:
        A.append(np.array(value.replace("#"," ").split()[0:4]).astype(float))
    B=np.array(A).reshape(len(A),4)
    lat = (B[:,0]+B[:,1])/2
    lon = (B[:,2]+B[:,3])/2

    margin = 2 # buffer to add to the range
    lat_min = min(lat) - margin
    lat_max = max(lat) + margin
    lon_min = min(lon) - margin
    lon_max = max(lon) + margin

    if m is None:
        m = Basemap(llcrnrlon=lon_min,
                    llcrnrlat=lat_min,
                    urcrnrlon=lon_max,
                    urcrnrlat=lat_max,
                    lat_0=(lat_max - lat_min)/2,
                    lon_0=(lon_max-lon_min)/2,
                    projection='merc',
                    resolution = 'h',
                    area_thresh=10000.,
        )
        m.drawcountries()
        m.drawstates()
        m.drawmapboundary(fill_color='#acbcec')
        m.fillcontinents(color = 'k',lake_color='#acbcec')

    lons, lats = m(lon, lat)
    if num is not None:
        m.scatter(lons, lats,s=4**(4.25+F*(num)),
                  c=num,
                  cmap=cmap,alpha=.7,
                  norm=colors.LogNorm(vmin=np.min(num),
                                      vmax=np.max(num)),linewidth=2,edgecolor='w',
                  zorder=5)
    else:
        m.scatter(lons, lats,s=100,zorder=5)

    if num is not None:
        cax, _ = mpl.colorbar.make_axes(ax, shrink=0.5, orientation='horizontal')
        cbar = mpl.colorbar.ColorbarBase(cax,
                                         cmap=cmap,orientation='horizontal',
                                         norm=mpl.colors.Normalize(vmin=np.min(np.array(num)),
                                                                   vmax=(np.max(np.array(num)))))
        pos1=cax.get_position()
        pos2 = [pos1.x0+.35, pos1.y0+.225 ,pos1.width/2, pos1.height/2.5]
        cax.set_position(pos2) # set a new position
        #plt.setp(plt.getp(cax, 'yticklabels'), color='k',fontsize=16)
        #plt.setp(plt.getp(cax, 'yticklabels'), fontweight='bold')
        cax.set_title('intensity',fontsize=18,color='w',y=1.05)

        plt.savefig(figname+'.pdf',dpi=300,bbox_inches='tight',transparent=True)

    return num,fig,ax,cax



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
    x, y = m(lons, lats)
    xy = zip(x,y)

    poly = Polygon(xy, facecolor=col.to_rgba(val),
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
        drawpoly=False,figname='fig'):
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
    Output -
        m (Basemap handle)
        fig (figure handle)
        ax (axis handle)
        cax (colorbar handle)
    """

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
        src=[float(i) for i in value['src'].replace('#',' ').split()]
        tgt=[float(i) for i in value['tgt'].replace('#',' ').split()]
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

    m = Basemap(llcrnrlon=lon_min,
                llcrnrlat=lat_min,
                urcrnrlon=lon_max,
                urcrnrlat=lat_max,
                lat_0=(lat_max - lat_min)/2,
                lon_0=(lon_max-lon_min)/2,
                projection='merc',
                resolution = res,
                area_thresh=1.)


    fig=plt.figure(figsize=(20,15))
    ax      = fig.add_subplot(111)
    m.drawcountries(color='w',linewidth=1)
    #m.drawstates(color='w')
    m.drawmapboundary(fill_color='#554433')#5645ec
    m.fillcontinents(color = 'k',lake_color='#554433')

    x_o, y_o = m(lonsrc,latsrc)
    x,y = m(lontgt, lattgt)

    CLS={}
    for index in np.arange(len(lontgt)):
        CLS[(lontgt[index],lattgt[index])]=[]

    for index in np.arange(len(lontgt)):
        CLS[(lontgt[index],
             lattgt[index])].append((latsrc[index],
                                     lonsrc[index]))

    if drawpoly:
        for key, value in CLS.iteritems():
            a=[]
            for i in value:
                a.append(i[0])
            b=[]
            for i in value:
                b.append(i[1])

            a=np.array(a)
            b=np.array(b)

            zP=[[i[0],i[1]] for i in zip(a,b)]
            hull = ConvexHull(zP)
            aa=[a[i] for i in hull.vertices]
            bb=[b[i] for i in hull.vertices]
            draw_screen_poly(aa,bb,m,ax,0,colormap1,ALPHA=0.3)

    norm = mpl.colors.LogNorm(vmin=(np.min(np.array(gamma))),
                          vmax=(np.max(np.array(gamma))))
    colx = cm.ScalarMappable(norm=norm,
                         cmap=colormap)

    WIDTH=0.007      # arrow width (scaled by gamma)
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
    plt.savefig(figname+'.pdf',dpi=300,bbox_inches='tight',transparent=False)

    return m,fig,ax,cax



def bokeh_plot(filepath):
    '''
    This is the first implementation of our Bokeh plot. The function takes the filepath
    of the data and opens the bokeh plot in a browser. Google Chrome seems to be the
    best browser for bokeh plots. The datafile must be a csv file in the correct format.
    See the file 'crime_filtered_data.csv' for an example. Each row represents a point,
    all the lines(sources) connected to it and the gammas and delays associated with
    the lines. The current implementation results in the bokeh plot, and a linked
    table of the data. IMPORTANT: Points are in MERCATOR Coordinates. This is because
    the current tileset for the map is in mercator coordinates.
    @author zed.uchicago.edu

    Inputs -
        filepath (string) - path to file

    Outputs -
        (No outputs, will open up external browser window)
    '''
    ########################################################
    #We define all our constants here. Change them as you please to adjust the plots.
    #N is the maximum number of sources a point can have. The crime_filtered_data.csv
    # is derived from a dataset which filters with N=20.

    FILE = filepath

    N = 20

    palette = ['darkblue','mediumblue','lime','yellow','orange','crimson','darkred']
    legend = ['darkblue','mediumblue','lime','yellow','orange','crimson','darkred']
    color_mapper = LinearColorMapper(palette=palette, low = 0, high = 0.1)

    TOOLS = "pan,wheel_zoom"
    ##########################################################

    #Import data
    df = pd.read_csv(FILE)
    df.fillna(value = 0)
    source = ColumnDataSource(df)
    #Create the figure. The x_range and y_range are defined as such because they are the
    #the mercator coordinates of Chicago.
    p = figure(x_range=(-9800000, -9700000), y_range=(5100000, 5200000), tools = TOOLS,
    x_axis_type = 'mercator', y_axis_type = 'mercator')
    p.add_tile(CARTODBPOSITRON)

    #Define the glyphs.
    circle_glyph = Circle(x='lontgt',y='lattgt', size = 3, fill_color = 'skyblue', line_width = 1)
    circle_render = p.add_glyph(source_or_glyph = source, glyph = circle_glyph)
    circle_box = BoxSelectTool(renderers = [circle_render])
    circle_tap = TapTool(renderers = [circle_render])
    p.add_tools(circle_box)
    p.add_tools(circle_tap)

    #A glyph for every line of each point.
    for n in range(1, N + 1):
        segment_glyph = Segment(x0 ='lonsrc' + str(n), y0='latsrc'+ str(n) , x1='lontgt',y1='lattgt', line_alpha = 0)
        segment_render = p.add_glyph(source_or_glyph = source, glyph = segment_glyph)
        nonselected_segment = Segment( line_alpha = 0,)
        selected_segment = Segment(line_alpha = 1, line_color = {'field':'gamma' + str(n), 'transform':color_mapper})
        segment_render. nonselection_glyph = nonselected_segment
        segment_render. selection_glyph = selected_segment

    #Here we begin defining our table.
    columns = []
    for n in range(1, N + 1):
        n_str = str(n)
        gamma_field = 'gamma' + n_str
        gamma_title = 'Gamma' + ' ' + n_str
        delay_field = 'delay' + n_str
        delay_title = 'Delay' + ' ' + n_str

        gamma_column = TableColumn(field = gamma_field, title = gamma_title,
                        formatter = NumberFormatter(format = '0.0000'), width = 70)
        delay_column = TableColumn(field = delay_field, title = delay_title, width = 50)
        columns.append(gamma_column)
        columns.append(delay_column)

    data_table = DataTable(source = source, columns = columns, scroll_to_selection = True, selectable = True,
        sortable = True, fit_columns = False,)

    #Please see bokeh's gridplot documentation to find out more on how the page layout works.
    #Essentially the following line says to place the table to the right of the figure on the same row.

    page = gridplot([[p, data_table]])

    show(page)
