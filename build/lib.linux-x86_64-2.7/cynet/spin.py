"""
Spatio temporal analysis for inferrence of statistical causality
@author zed.uchicago.edu
"""

import pandas as pd
import numpy as np
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
#from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch
import matplotlib.colors as colors
from scipy.spatial import ConvexHull
import seaborn as sns



__version__='1.0.0'
__DEBUG__=False

class spatioTemporal:
    """
    Utilities for spatio temporal analysis
    @author zed.uchicago.edu

    Attributes:
        log_store (Pickle): Pickle storage of class data & dataframes
        log_file (string): path to CSV of legacy dataframe
        ts_store (string): path to CSV containing most recent ts export
        DATE (string):
        EVENT (string): column label for category filter
        coord1 (string): first coordinate level type; is column name
        coord2 (string): second coordinate level type; is column name
        coord3 (string): third coordinate level type;
                         (z coordinate)
        end_date (datetime.date): upper bound of daterange
        freq (string): timeseries increments; e.g. D for date
        columns (list): list of column names to use;
            required at least 2 coordinates and event type
        types (list of strings): event type list of filters
        value_limits (tuple): boundaries (magnitude of event;
                              above threshold)
        grid (dict): dict with coord and eps (see example)
        threshold (float): significance threshold
    """

    def __init__(self,
            log_store='log.p',
            log_file=None,
            ts_store=None,
            DATE='Date',
            year=None,
            month=None,
            day=None,
            EVENT='Primary Type',
            coord1='Latitude',
            coord2='Longitude',
            coord3=None,
            init_date=None,
            end_date=None,
            freq=None,
            columns=None,
            types=None,
            value_limits=None,
            grid=None,
            threshold=None):

        # either types is specified
        # or value limits are specified, not both
        assert not ((types is not None)
                    and (value_limits is not None)), "Either types can be specified \
                    or value_limits: not both."

        # either a DATE column is specified, or separate
        # columns for year month day are specified, not both
        # NOTE: could fail if only year is specified but not the other two, etc.
        assert not ((DATE is not None)
                    and ((year is not None)
                         or (month is not None)
                         or (day is not None )))

        # if log_file is specified, then read
        # else read log_store pickle
        if log_file is not None:
            # if date is not specified, then year month and day are individually specified
            if year is not None and month is not None and day is not None:
                df=pd.read_csv(log_file, parse_dates={DATE: [year, month, day]} )
                # Line originally read df[DATE] = pd.to_datetime(df['DATE'], errors='coerce'), changed
                # column name to match for consistency
                df[DATE] = pd.to_datetime(df[DATE], errors= 'coerce')
            else: # DATE variable was renamed or could be 'Date'
                df = pd.read_csv(log_file)
                df[DATE] = pd.to_datetime(df[DATE])
            # will be stored in logfile called "log.p" from csv
            df.to_pickle(log_store)
        # at this point the column name corresponding to date will be stored in the variable DATE
        else:
            # but all bets are off here, b/d date column can be called anything
            # Assuming that date column will be 'Date' as per the DATE variable, but must either have
            # user confirm or force 'Date'
            df = pd.read_pickle(log_store)

        self._logdf = df
        self._spatial_tiles = None
        self._dates = None
        self._THRESHOLD=threshold

        if freq is None:
            self._FREQ = 'D'
        else:
            self._FREQ=freq

        self._DATE = DATE

        if init_date is None:
            self._INIT = '1/1/2001'
        else:
            self._INIT = init_date

        if end_date is not None:
            self._END = end_date
        else:
            self._END=None

        self._EVENT = EVENT
        self._coord1 = coord1
        self._coord2 = coord2
        self._coord3 = coord3

        if columns is None:
            self._columns = [EVENT, coord1, coord2, DATE]
        else:
            self._columns = columns

        self._types = types
        self._value_limits = value_limits

        # pandas timeseries will be stored as separate entries in a dict with the filter as the name
        self._ts_dict = {}

        # grid stores directions on how to create the grid indexes for the final pandas df
        self._grid = {}
        if grid is not None:
            assert(self._coord1 in grid)
            assert(self._coord2 in grid)
            assert('Eps' in grid)

            # constructing private variable self._grid in the desired format with the values taken from
            # the input grid
            self._grid[self._coord1]=grid[self._coord1]
            self._grid[self._coord2]=grid[self._coord2]
            self._grid['Eps']=grid['Eps']

        self._trng = pd.date_range(start=self._INIT,
                                   end=self._END,freq=self._FREQ)


    def getTS(self,_types=None,tile=None):
        """
        Utilities for spatio temporal analysis
        @author zed.uchicago.edu

        Utilities for spatio temporal analysis
        @author zed.uchicago.edu

        Given location tile boundaries and type category filter, creates the
        corresponding timeseries as a pandas DataFrame
        (Note: can reassign type filter, does not have to be the same one
        as the one initialized to the dataproc)

        Inputs:
            _types (list of strings): list of category filters
            tile (list of floats): location boundaries for tile

        Outputs:
            pd.Dataframe of timeseries data to corresponding grid tile
            pd.DF index is stringified LAT/LON boundaries
            with the type filter  included
        """

        assert(self._END is not None)
        TS_NAME = ('#'.join(str(x) for x in tile))+"#"+stringify(_types)

        lat_ = tile[0:2]
        lon_ = tile[2:4]

        if self._value_limits is None:
            df = self._logdf[self._columns]\
                     .loc[self._logdf[self._EVENT].isin(_types)]\
                     .sort_values(by=self._DATE).dropna()
        else:
            df = self._logdf[self._columns]\
                     .loc[self._logdf[self._EVENT]\
                          .between(self._value_limits[0],
                                   self._value_limits[1])]\
                     .sort_values(by=self._DATE).dropna()

        df = df.loc[(df[self._coord1] > lat_[0])
                    & (df[self._coord1] <= lat_[1])
                    & (df[self._coord2] > lon_[0])
                    & (df[self._coord2] <= lon_[1])]
        df.index = df[self._DATE]
        df=df[[self._EVENT]]

        ts = [df.loc[self._trng[i]:self._trng[i + 1]].size for i in
              np.arange(self._trng.size - 1)]

        return pd.DataFrame(ts, columns=[TS_NAME],
                            index=self._trng[:-1]).transpose()

    def timeseries(self,LAT,LON,EPS,_types,CSVfile='TS.csv',THRESHOLD=None):
        """
        Utilities for spatio temporal analysis
        @author zed.uchicago.edu

        Creates DataFrame of location tiles and their
        respective timeseries from
        input datasource with
        significance threshold THRESHOLD
        latitude, longitude coordinate boundaries given by LAT, LON
        calls on getTS for individual tile then concats them together

        Input:
            LAT (float):
            LON (float):
            EPS (float): coordinate increment ESP
            _types (list): event type filter; accepted event type list
            CSVfile (string): path to output file

        Output:
            (None): grid pd.Dataframe written out as CSV file
                    to path specified
        """
        if THRESHOLD is None:
            if self._THRESHOLD is None:
                THRESHOLD=0.1
            else:
                THRESHOLD=self._THRESHOLD

        if self._trng is None:
            self._trng = pd.date_range(start=self._INIT,
                                       end=self._END,freq=self._FREQ)

        _TS = pd.concat([self.getTS(tile=[i, i + EPS, j, j + EPS],
                                    _types=_types) for i in tqdm(LAT)
                         for j in tqdm(LON)])

        LEN=pd.date_range(start=self._INIT,
                          end=self._END,freq=self._FREQ).size+0.0

        statbool = _TS.astype(bool).sum(axis=1) / LEN
        _TS = _TS.loc[statbool > THRESHOLD]
        self._ts_dict[repr(_types)] = _TS

        if CSVfile is not None:
            _TS.to_csv(CSVfile, sep=' ')

        return

    def fit(self,grid=None,INIT=None,END=None,THRESHOLD=None,csvPREF='TS'):
        """
        Utilities for spatio temporal analysis
        @author zed.uchicago.edu

        Fit dataproc with specified grid parameters and
        create timeseries for
        date boundaries specified by INIT, THRESHOLD,
        and END which do not have
        to match the arguments first input
        to the dataproc

        Inputs:
            grid (pd.DataFrame): dataframe of location
            timeseries data
            INIT (datetime.date): starting timeseries date
            END (datetime.date): ending timeseries date
            THRESHOLD (float): significance threshold

        Outputs:
            (None)
        """

        if INIT is not None:
            self._INIT=INIT
        if END is not None:
            self._END=END
        if grid is not None:
            self._grid=grid

        assert(self._END is not None)
        assert(self._coord1 in self._grid)
        assert(self._coord2 in self._grid)
        assert('Eps' in self._grid)

        if self._types is not None:
            for key in self._types:
                self.timeseries(self._grid[self._coord1],
                                self._grid[self._coord2],
                                self._grid['Eps'],
                                key,
                                CSVfile=csvPREF+stringify(key)+'.csv',
                                THRESHOLD=THRESHOLD)
            return
        else:
            assert(self._value_limits is not None)
            self.timeseries(self._grid[self._coord1],
                            self._grid[self._coord2],
                            self._grid['Eps'],
                            None,
                            CSVfile=csvPREF+'.csv',
                            THRESHOLD=THRESHOLD)
            return


    def pull(self, domain="data.cityofchicago.org",dataset_id="crimes",\
        token="ZIgqoPrBu0rsvhRr7WfjyPOzW",store=True, out_fname="pull_df.p",
        pull_all=False):
        """
        Utilities for spatio temporal analysis
        @author zed.uchicago.edu

        Pulls new entries from datasource
        NOTE: should make flexible but for now use city of Chicago data

        Input -
            domain (string): Socrata database domain hosting data
            dataset_id (string): dataset ID to pull
            token (string): Socrata token for increased pull capacity
            store (boolean): whether or not to write out new dataset
            pull_all (boolean): pull complete dataset
            instead of just updating

        Output -
            None (writes out files if store is True and modifies inplace)
        """

        client = Socrata(domain, token)
        if domain == "data.cityofchicago.org" and dataset_id=="crimes":
            self._coord1 = "latitude"
            self._coord2 = "longitude"
            self._EVENT = "primary_type"

        if pull_all:
            new_data = client.get(dataset_id)
            pull_df = pd.DataFrame(new_data).dropna(\
                subset=[self._coord1, self._coord2, self._DATE, self._EVENT],\
                axis=1).sort_values(self._DATE)
            self._logdf = pull_df
        else:
            self._logdf.sort_values(self._DATE)
            pull_after_date = "'"+str(self._logdf[self._DATE].iloc[-1]).replace(\
            " ", "T")+"'"
            new_data = client.get(dataset_id, where=\
                ("date > "+pull_after_date))
            if domain == "data.cityofchicago.org" and dataset_id=="crimes":
                self._DATE = "date"
            pull_df = pd.DataFrame(new_data).dropna(\
                subset=[self._coord1, self._coord2, self._DATE, self._EVENT],\
                axis=1).sort_values(self._DATE)
            self._logdf.append(pull_df)

        if store:
            assert out_fname is not None, "Out filename not specified"
            self._logdf.to_pickle(out_fname)


def stringify(List):
    """
    Utility function
    @author zed.uchicago.edu

    Converts list into string separated by dashes
             or empty string if input list
             is not list or is empty

    Input:
        List (list): input list to be converted

    Output:
        (string)
    """
    if List is None:
        return ''
    if not List:
        return ''

    return '-'.join(str(elem) for elem in List)


def readTS(TSfile,csvNAME='TS1',BEG=None,END=None):
    """
    Utilities for spatio temporal analysis
    @author zed.uchicago.edu

    Reads in output TS logfile into pd.DF
        and then outputs necessary
        CSV files in XgenESeSS-friendly format

    Input -
        TSfile (string): filename input TS to read
        csvNAME (string)
        BEG (string): start datetime
        END (string): end datetime

    Returns -
        dfts (pandas.DataFrame)
    """

    dfts=pd.read_csv(TSfile,sep=" ",index_col=0)
    dfts.columns = pd.to_datetime(dfts.columns)

    cols=dfts.columns[np.logical_and(dfts.columns >= pd.to_datetime(BEG),
                                     dfts.columns <= pd.to_datetime(END))]

    dfts=dfts[cols]

    dfts.to_csv(csvNAME+'.csv',sep=" ",header=None,index=None)
    np.savetxt(csvNAME+'.columns', cols, delimiter=',',fmt='%s')
    np.savetxt(csvNAME+'.coords', dfts.index.values, delimiter=',',fmt='%s')


    return dfts



def splitTS(TSfile,csvNAME='TS1',dirname='./',prefix="@",
            BEG=None,END=None):
    """
    Utilities for spatio temporal analysis
    @author zed.uchicago.edu

    Writes out each row of the pd.DataFrame as a separate CSVfile
    For XgenESeSS binary

    No I/O
    """

    dfts=pd.read_csv(TSfile,sep=" ",index_col=0)
    dfts.columns = pd.to_datetime(dfts.columns)

    cols=dfts.columns[np.logical_and(dfts.columns >= pd.to_datetime(BEG),
                                     dfts.columns <= pd.to_datetime(END))]

    dfts=dfts[cols]


    for row in dfts.index:
        dfts.loc[[row]].to_csv(dirname+"/"+prefix+row,header=None,index=None,sep=" ")

    return



class uNetworkModels:
    """
    Utilities for storing and manipulating XPFSA models
    inferred by XGenESeSS
    @author zed.uchicago.edu

    Attributes:
        jsonFile (string): path to json file containing models
    """

    def __init__(self,
                 jsonFILE):
        with open(jsonFILE) as data_file:
            self._models = json.load(data_file)


    @property
    def models(self):
         return self._models

    @property
    def df(self):
         return self._df


    def append(self,pydict):
        """
        append models
        @author zed.uchicago.edu
        """
        self._models.update(pydict)

    def select(self,var="gamma",n=None,
               reverse=False, store=None,
               high=None,low=None,inplace=False):
        """
        Utilities for storing and manipulating XPFSA models
        inferred by XGenESeSS
        @author zed.uchicago.edu

        Selects the N top models as ranked by var specified value
        (in reverse order if reverse is True)

        Inputs -
            var (string): model parameter to rank by
            n (int): number of models to return
            reverse (boolean): return in ascending order (True)
                or descending (False) order
            store (string): name of file to store selection json
            high (float): higher cutoff
            low (float): lower cutoff
            inplace (bool): update models if true
        Returns -
            (dictionary): top n models as ranked by var
                         in ascending/descending order
        """

        #assert var in self._models.keys(), "Error: Model parameter specified not valid"

        this_dict={value[var]:key
                   for (key,value) in self._models.iteritems() }

        if low is not None:
            this_dict={key:this_dict[key] for key in this_dict.keys() if key >= low }
        if high is not None:
            this_dict={key:this_dict[key] for key in this_dict.keys() if key <= high }

        if n is None:
            n=len(this_dict)
        if n > len(this_dict):
            n=len(this_dict)

        out = {this_dict[k]:self._models[this_dict[k]]
                for k in sorted(this_dict.keys(),
                                reverse=reverse)[0:n]}

        if inplace:
            self._models=out

        if store is not None:
            with open(store, 'w') as outfile:
                json.dump(out, outfile)

        return out


    def augmentDistance(self):
        """
        Utilities for storing and manipulating XPFSA models
        inferred by XGenESeSS
        @author zed.uchicago.edu

        Calculates the distance between all models and stores
        them under the
        distance key of each model;

        No I/O
        """

        for key,value in self._models.iteritems():
            src=[float(i) for i in value['src'].replace('#',' ').split()]
            tgt=[float(i) for i in value['tgt'].replace('#',' ').split()]

            dist = haversine((np.mean(src[0:2]),np.mean(src[2:])),
                           (np.mean(tgt[0:2]),np.mean(tgt[2:])),
                           miles=True)
            self._models[key]['distance'] = dist

        return


    def to_json(self,outFile):
        """
        Utilities for storing and manipulating XPFSA models
        inferred by XGenESeSS
        @author zed.uchicago.edu

        Writes out updated models json to file

        Input -
            outFile (string): name of outfile to write json to

        Returns -
            Nonexs
        """

        with open(outFile, 'w') as outfile:
            json.dump(self._models, outfile)

        return


    def setDataFrame(self,scatter=None):
        """
        Generate dataframe representation of models
        @author zed.uchicago.edu

        Input -
            scatter (string) : prefix of filename to plot 3X3 regression
            matrix between delay, distance and coefficiecient of causality
        Returns -
            Dataframe with columns
            ['latsrc','lonsrc','lattgt',
             'lontgtt','gamma','delay','distance']
        """

        latsrc=[]
        lonsrc=[]
        lattgt=[]
        lontgt=[]
        gamma=[]
        delay=[]
        distance=[]
        NUM=None
        for key,value in self._models.iteritems():
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
            distance.append(value['distance'])

        self._df = pd.DataFrame({'latsrc':latsrc,
                                 'lonsrc':lonsrc,
                                 'lattgt':lattgt,
                                 'lontgt':lontgt,
                                 'gamma':gamma,
                                 'delay':delay,
                                 'distance':distance})

        if scatter is not None:
            sns.set_style('darkgrid')
            fig=plt.figure(figsize=(12,12))
            fig.subplots_adjust(hspace=0.25)
            fig.subplots_adjust(wspace=.25)
            ax = plt.subplot2grid((3,3), (0,0), colspan=1,rowspan=1)
            sns.distplot(self._df.gamma,ax=ax,kde=True,color='#9b59b6');
            ax = plt.subplot2grid((3,3), (0,1), colspan=1,rowspan=1)
            sns.regplot(ax=ax,x="gamma", y="distance", data=self._df);
            ax = plt.subplot2grid((3,3), (0,2), colspan=1,rowspan=1)
            sns.regplot(ax=ax,x="gamma", y="delay", data=self._df);

            ax = plt.subplot2grid((3,3), (1,0), colspan=1,rowspan=1)
            sns.regplot(ax=ax,x="distance", y="gamma", data=self._df);
            ax = plt.subplot2grid((3,3), (1,1), colspan=1,rowspan=1)
            sns.distplot(self._df.distance,ax=ax,kde=True,color='#9b59b6');
            ax = plt.subplot2grid((3,3), (1,2), colspan=1,rowspan=1)
            sns.regplot(ax=ax,x="distance", y="delay", data=self._df);

            ax = plt.subplot2grid((3,3), (2,0), colspan=1,rowspan=1)
            sns.regplot(ax=ax,x="delay", y="gamma", data=self._df);
            ax = plt.subplot2grid((3,3), (2,1), colspan=1,rowspan=1)
            sns.regplot(ax=ax,x="delay", y="distance", data=self._df);
            ax = plt.subplot2grid((3,3), (2,2), colspan=1,rowspan=1)
            sns.distplot(self._df.delay,ax=ax,kde=True,color='#9b59b6');

            plt.savefig(scatter+'.pdf',dpi=300,bbox_inches='tight',transparent=False)


        return self._df


    def iNet(self,init=0):
        """
        Utilities for storing and manipulating XPFSA models
        inferred by XGenESeSS
        @author zed.uchicago.edu

        Calculates the distance between all models and stores
        them under the
        distance key of each model;

        No I/O
        """

        pass


def to_json(pydict,outFile):
    """
        Writes dictionary json to file
        @author zed.uchicago.edu

        Input -
            pydict (dict): ditionary to store
            outFile (string): name of outfile to write json to

        Returns -
            Nonexs
    """

    with open(outFile, 'w') as outfile:
        json.dump(pydict, outfile)

    return


def showGlobalPlot(coords,ts=None,fsize=[14,14],cmap='jet',
                   m=None,figname='fig',F=2):
    """
        plot global distribution of events
        within time period specified

        Inputs -
            coords (string): filename with coord list as lat1#lat2#lon1#lon2
            ts (string): time series filename with data in rows, space separated
            fsize (list):
            cmap (string):
            m (mpl.mpl_toolkits.Basemap): mpl instance for plotting
            figname (string): Name of the Plot

        Returns -
            m (mpl.mpl_toolkits.Basemap): mpl instance of heat map of
                crimes from fitted data
        """
    num=None
    if ts is not None:
        num=pd.read_csv(ts,header=None,sep=" ").sum(axis=1).values
        coords_=pd.read_csv(coords,header=None,sep=" ")[0].values

    if num is not None:
        num=np.array(_scaleforsize(num))

    fig=plt.figure(figsize=(14,14))
    ax      = fig.add_subplot(111)

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

        Inputs -
            a (ndarray):
        Returns -
            a (ndarray):
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


def draw_screen_poly( lats, lons, m,ax,val,cmap,ALPHA=0.6):
    """
      utility function to draw polygons on basemap
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
      utility function to visualize spatio temporal
      interaction networks
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
