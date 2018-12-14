'''
After generating the integrated csv, we can use our heatmap utilities to draw
them.
'''
import numpy as np
import math
import cartopy.crs as ccrs
import cartopy.feature as crt
import matplotlib.pyplot as plt
import json
import pandas as pd
import multiprocessing
plt.style.use('ggplot')

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
            intensity[lat_index[j],lon_index[i]]=intensity[lat_index[j],lon_index[i]]\
                                                +intensity_lon[i]*intensity_lat[j]
    return intensity

def get_mesh(df0,lat_min,lat_max,lon_min,lon_max,radius=2,detail=0.25):
    coord_=df0[[lat_col,lon_col]].values
    lon_grid=np.arange(lon_min-radius,lon_max+radius,detail)
    lat_grid=np.arange(lat_min-radius,lat_max+radius,detail)
    lon_mesh,lat_mesh=np.meshgrid(lon_grid,lat_grid)
    return lon_mesh,lat_mesh,coord_

def get_prediction(df,days,lat_min,lat_max,lon_min,lon_max,
                   radius=2,detail=0.2,save=False,
                   startdate="1/1/2017",offset=1462,
                   PREFIX='Q'):

    dt=pd.to_datetime(startdate) + pd.DateOffset(days=days-offset)
    dt=dt.strftime('%m-%d-%Y')

    df = df[df[day_col].between(days-grace,days+grace)]
    df = df[df[variable_col].isin(types)]
    df = df[df[source_col] == source]
    df_gnd = df[(df[day_col]==days) & (df[actual_event_col]==1)]
    df_prd0 = df[(df[day_col]==days) & (df[predictin_col]==1)]
    df_prd1 = df[(df[day_col]==days-grace) & (df[predictin_col]==1)]
    df_prd2 = df[(df[day_col]==days+grace) & (df[predictin_col]==1)]

    # true positives .. will show
    df_prd0_tp = df_prd0[df_prd0[actual_event_col]==1]
    tp=df_prd0_tp.index.size

    # this is not quite false positives, because of possible matches in grace (false pos=_fp)
    df_prd0_fp = df_prd0[df_prd0[actual_event_col]==0]
    df_gnd1 = df[(df[day_col]==days-1) & (df[actual_event_col]==1)]
    df_gnd2 = df[(df[day_col]==days+1) & (df[actual_event_col]==1)]


    df_gnd1 = df_gnd1[df_gnd1[lat_col].isin(df_prd0_fp[[lat_col,lon_col]].values[:,0])]
    df_gnd1 = df_gnd1[df_gnd1[lon_col].isin(df_prd0_fp[[lat_col,lon_col]].values[:,1])]
    c1 = df_gnd1[[lat_col,lon_col]].values

    df_gnd2 = df_gnd2[df_gnd2[lat_col].isin(df_prd0_fp[[lat_col,lon_col]].values[:,0])]
    df_gnd2 = df_gnd2[df_gnd2[lon_col].isin(df_prd0_fp[[lat_col,lon_col]].values[:,1])]
    c2 = df_gnd2[[lat_col,lon_col]].values

    # Now we calculate correct false positives
    df_prd0_fp = df_prd0_fp[(~df_prd0_fp[lat_col].isin(c1[:,0])) & (~df_prd0_fp[lon_col].isin(c1[:,1]))]
    df_prd0_fp = df_prd0_fp[(~df_prd0_fp[lat_col].isin(c2[:,0])) & (~df_prd0_fp[lon_col].isin(c2[:,1]))]
    fp=df_prd0_fp.index.size

    # account for grace tp from day before
    df_gnd = df_gnd[df_gnd[lat_col].isin(df_prd1[[lat_col,lon_col]].values[:,0])]
    df_gnd = df_gnd[df_gnd[lon_col].isin(df_prd1[[lat_col,lon_col]].values[:,1])]
    c0 = df_gnd[[lat_col,lon_col]].values
    df_prd0_ = df_prd1[(df_prd1[lat_col].isin(c0[:,0])) & (df_prd1[lon_col].isin(c0[:,1]))]

    tp=tp+df_prd0_.index.size

    #concat df_prd0_tp  df_prd0_fp df_prd0_
    df_prd0 = pd.concat([df_prd0_tp , df_prd0_fp, df_prd0_])

    # false negative
    df_gnd_fn = df_gnd[~((df_gnd[lat_col].isin(df_prd0[lat_col].values))
                       &(df_gnd[lon_col].isin(df_prd0[lon_col].values)))]

    fn=df_gnd_fn.index.size
    #print df_gnd_fn

    lon_grid=np.arange(lon_min-radius,lon_max+radius,detail)
    lat_grid=np.arange(lat_min-radius,lat_max+radius,detail)
    lon_mesh,lat_mesh=np.meshgrid(lon_grid,lat_grid)

    lon_mesh0,lat_mesh0,coord_=get_mesh(df_prd0,lat_min,lat_max,lon_min,lon_max,radius=radius,detail=detail)
    intensity0=np.zeros(lat_mesh0.shape)
    for i in coord_:
        intensity0=get_intensity(intensity0,lon_mesh0,lat_mesh0,i,sigma=3.5,radius=radius)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines('50m',color='.5',lw=1)
    ax.add_feature(crt.BORDERS,edgecolor='w',lw=.4, alpha=.15)
    #ax.add_feature(crt.OCEAN,color='w', alpha=.25)
    #ax.add_feature(crt.LAKES, alpha=0.95)

    plt.plot(df_gnd[lon_col].values,df_gnd[lat_col].values,'ko',alpha=.4,ms=4)
    plt.pcolormesh(lon_mesh0,lat_mesh0,intensity0,cmap='terrain',
                   alpha=1,edgecolor=None,linewidth=0)
    plt.xlim(lon_mesh0.min(),lon_mesh0.max())
    plt.ylim(lat_mesh0.min(),lat_mesh0.max())


    #Annotations
    props = dict(boxstyle='round', facecolor='w', alpha=0.95)
    props1 = dict(boxstyle='round', facecolor=None,lw=0, edgecolor=None,alpha=0.05)

    ax.text(0.98, 0.89,dt, transform=ax.transAxes,fontweight='bold',fontsize=8,color='k',
            verticalalignment='top', horizontalalignment='right', bbox=props)
    ax.text(0.01, 0.075,'zed.uchicago.edu\nData source:', transform=ax.transAxes,
            fontweight='bold',fontsize=8,color='w',alpha=.5,
            verticalalignment='bottom', horizontalalignment='left')

    sourcetype='('+', '.join([short_type_names[i] for i in types])+')'
    ax.text(0.98, 0.98,'Event Prediction '+sourcetype+'\nHorizon: 6-8 days',
            transform=ax.transAxes, fontsize=8,color='w',fontweight='bold',
            verticalalignment='top', horizontalalignment='right', bbox=props1)
    ax.text(0.98, 0.02,'Black circles: actual events\nLatitude Res.: '+\
            spatial_resolution_lat+'\nLongitude Res.: '+spatial_resolution_lon
            +'\nTemporal Res.: '+temporal_quantization+'\nRegions With Event Rate > 5% Considered',
            transform=ax.transAxes, fontsize=8,color='.7',fontweight='bold',
            verticalalignment='bottom', horizontalalignment='right', bbox=props1)

    #logo = plt.imread(datalogo)
    #ax.figure.figimage(logo, 20, 20, alpha=.5, zorder=1)
    #Annotate Country Names
    ax.text(0.2, 0.5,'Africa',
            transform=ax.transAxes, fontsize=16,color='.4',fontweight='bold',
            verticalalignment='bottom', horizontalalignment='right', bbox=props1)

    ax.text(0.3, 0.9,'Europe',
            transform=ax.transAxes, fontsize=16,color='.4',fontweight='bold',
            verticalalignment='bottom', horizontalalignment='right', bbox=props1)

    ax.text(0.5, 0.45,'Saudi\nArabia',
            transform=ax.transAxes, fontsize=10,color='.4',fontweight='bold',
            verticalalignment='bottom', horizontalalignment='left', bbox=props1)

    ax.text(0.9, 0.5,'India',
            transform=ax.transAxes, fontsize=10,color='.4',fontweight='bold',
            verticalalignment='bottom', horizontalalignment='right', bbox=props1)

    ax.text(0.65, 0.65,'Iran',
            transform=ax.transAxes, fontsize=10,color='.4',fontweight='bold',
            verticalalignment='bottom', horizontalalignment='right', bbox=props1)

    #New axes to show TP and FP
    ax2 = plt.gcf().add_axes([0.925, 0.725, 0.06, 0.075])
    ax2.patch.set_alpha(0)

    plt.bar(['FP','TP'],[fp,tp],color='r',lw=0,alpha=.5)
    ax2.spines['bottom'].set_color('w')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.tick_params(axis='x', colors='w')
    ax2.tick_params(axis='y', colors='w')
    ax2.grid('on')
    for label in ax2.get_yticklabels():
        label.set_color('w')
        label.set_fontsize(6)
        label.set_fontweight('bold')
    for label in ax2.get_xticklabels():
        label.set_color('w')
        label.set_fontsize(8)
        label.set_fontweight('bold')

    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    if save:
        plt.savefig(PREFIX+str(days).zfill(5)+'.png',dpi=200, bbox_inches='tight',pad_inches = 0)
    return ax,df_gnd,df_prd0

day_col='day'
actual_event_col='actual_event'
variable_col='target'
source_col='source'
predictin_col='predictions'
lon_col = 'lon2'
lat_col='lat2'
cmap='jet'
grace=1

source1 = 'Armed_Assault-Hostage_Taking_Barricade_Incident-Hijacking-Assassination-Hostage_Taking_Kidnapping_'
source2 = 'VAR'
source3 = 'Bombing_Explosion-Facility_Infrastructure_Attack'
source4 = 'ALL'
source = 'ALL'
types = [source3]
datasource = 'Global Terrorism Database'
datalogo = 'logogtd.png'
#---------------------------------------------------------------------
#note: source can be Armed_Assault.., Bombing.., VAR, or ALL This are the
#predicting variables
#types cannot be ALL, but can be a list of of one or more of the others
#These are the event types predicted
#------------------------------------------------------------------------
EPS=50
latres=np.linspace(-4,49,EPS)
lonres=np.linspace(-16,84,EPS)

spatial_resolution_lat=str(latres[1]-latres[0])[0:4]+r'$\degree$'
spatial_resolution_lon=str(lonres[1]-lonres[0])[0:4]+r'$\degree$'

temporal_quantization='1 day'
short_type_names={source1: 'Assaults', source2: 'Casualties',source3: 'Bombings','ALL':'ALL'}
[lat_min,lat_max,lon_min,lon_max]=[-2.91837, 49.0, -1.71429, 84.0]
#Load the dataframe
df = pd.read_csv('20modelsALL.csv')
df00=df.copy()

get_prediction(df00,900,lat_min,lat_max,lon_min,lon_max,radius=5,save=True,PREFIX='XX');
