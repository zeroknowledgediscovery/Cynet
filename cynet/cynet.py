"""
Spatio temporal analysis for inferrence of statistical causality
@author zed.uchicago.edu
"""

import numpy as np
import pandas as pd
import random

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
import operator
import warnings
import os
import sys
import uuid
import glob
import subprocess
from joblib import Parallel , delayed
import yaml
import shlex

import matplotlib as mpl
mpl.use('Agg')

from matplotlib import pyplot as plt
plt.ioff()

from scipy.spatial import Delaunay
import seaborn as sns
import pylab as plt

__DEBUG__=False
PRECISION=5

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
        grid (dictionary or list of lists): coordinate dictionary with
                respective ranges and EPS value OR custom list of lists
                of custom grid tiles as [coord1_start, coord1_stop,
                coord2_start, coord2_stop]
        grid_type (string): parameter to determine if grid should be built up
                            from a coordinate start/stop range ('auto') or be
                            built from custom tile coordinates ('custom')
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
        '''
        assert not ((DATE is not None)
                    and ((year is not None)
                         or (month is not None)
                         or (day is not None)))
        '''

        # if log_file is specified, then read
        # else read log_store pickle
        if log_file is not None:
            # if date is not specified, then year month and day are individually specified
            if year is not None and month is not None and day is not None:
                df=pd.read_csv(log_file, parse_dates={DATE: [year, month, day]})
                if not types is None:
                    for filter_subset in types:
                        for a_filter in filter_subset:
                            if not a_filter in df[EVENT].unique():
                                warnings.warn("{} filter not in dataset, will produce empty dataframe".format(filter_subset))
                # Line originally read df[DATE]
                #        = pd.to_datetime(df['DATE'], errors='coerce'), changed
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

        # pandas timeseries will be stored as separate entries in a dict with
        # the filter as the name
        self._ts_dict = {}

        # grid stores directions on how to create the grid indexes for the
        # final pandas df
        self._grid = None
        assert grid is not None, "Error: no grid parameter specified."
        if isinstance(grid, dict):
            self._grid = {}
            assert(self._coord1 in grid)
            assert(self._coord2 in grid)
            assert('Eps' in grid)
            # constructing private variable self._grid in the desired format
            # with the values taken from the input grid
            self._grid[self._coord1]=grid[self._coord1]
            self._grid[self._coord2]=grid[self._coord2]
            self._grid['Eps']=grid['Eps']
            self._grid_type = "auto"
        elif isinstance(grid, list):
            self._grid = grid
            self._grid_type = "custom"
        else:
            raise TypeError("Unsupported grid type.")

        self._trng = pd.date_range(start=self._INIT,
                                   end=self._END,freq=self._FREQ)



    def getTS(self,_types=None,tile=None,freq=None,poly_tile=False):
        """
        Utilities for spatio temporal analysis
        @author zed.uchicago.edu

        Given location tile boundaries and type category filter, creates the
        corresponding timeseries as a pandas DataFrame
        (Note: can reassign type filter, does not have to be the same one
        as the one initialized to the dataproc)

        Inputs:
            _types (list of strings): list of category filters
            tile (list of floats): location boundaries for tile OR
            if poly_tile is TRUE, tile is a list of tuples defining the polygon
            freq (string): intervals of time between timeseries columns
            poly_tile (boolean): whether or not input for tiles defines
            a polygon filter

        Outputs:
            pd.Dataframe of timeseries data to corresponding grid tile
            pd.DF index is stringified LAT/LON boundaries
            with the type filter included
        """

        assert(self._END is not None)
        TS_NAME = ('#'.join(str(x) for x in tile))+"#"+stringify(_types)

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

        if poly_tile:
                poly_lat = [point[0] for point in tile]
                poly_lon = [point[1] for point in tile]
                hull_pts = np.column_stack((poly_lat, poly_lon))
                pred_pt = np.column_stack((df[self._coord1], df[self._coord2]))

                if not isinstance(hull_pts, Delaunay):
                    hull = Delaunay(hull_pts)
                mask = hull.find_simplex(pred_pt)>=0
                df = df[mask]

        else:
            lat_ = tile[0:2]
            lon_ = tile[2:4]

            df = df.loc[(df[self._coord1] > lat_[0])
                        & (df[self._coord1] <= lat_[1])
                        & (df[self._coord2] > lon_[0])
                        & (df[self._coord2] <= lon_[1])]

        # make the DATE the index and keep only the event col
        df.index = df[self._DATE]
        df=df[[self._EVENT]]

        if freq is None:
            ts = [df.loc[self._trng[i]:self._trng[i + 1]].size for i in
                  np.arange(self._trng.size - 1)]
            out = pd.DataFrame(ts, columns=[TS_NAME],
                            index=self._trng[:-1]).transpose()
        else:
            trng = pd.date_range(start=self._INIT,
                                       end=self._END,freq=freq)
            ts = [df.loc[trng[i]:trng[i + 1]].size for i in
                  np.arange(trng.size - 1)]
            out = pd.DataFrame(ts, columns=[TS_NAME],
                            index=trng[:-1]).transpose()

        return out


    def get_rand_tile(self,tiles=None,LAT=None,LON=None,EPS=None,_types=None,poly_tile=False):
        '''
        Utilities for spatio temporal analysis
        @author zed.uchicago.edu

        Picks random tile from options fed into timeseries method which maps to a
        non-empty subset within the larger dataset

        Inputs -
            LAT (float or list of floats): singular coordinate float or list of
                                           coordinate start floats
            LON (float or list of floats): singular coordinate float or list of
                                           coordinate start floats
            EPS (float): coordinate increment ESP
            _types (list): event type filter; accepted event type list
            tiles (list of lists): list of tiles to build where tile can be
            a (list of floats i.e. [lat1 lat2 lon1 lon2]) or tuples (i.e. [(x1,y1),(x2,y2)])
            defining polygons
            poly_tile (boolean): whether input for tile specifies a polygon

        Outputs -
            tile dataframe (pd.DataFrame)
        '''

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
        TS_NAME=None
        stop = False

        if tiles is not None:
            while not stop:
                tile = random.choice(tiles)
                TS_NAME = ('#'.join(str(x) for x in tile))+"#"+stringify(_types)
                if not poly_tile:
                    lat_ = tile[0:2]
                    lon_ = tile[2:4]

                    test = df.loc[(df[self._coord1] > lat_[0])
                                & (df[self._coord1] <= lat_[1])
                                & (df[self._coord2] > lon_[0])
                                & (df[self._coord2] <= lon_[1])]

                    if test.shape[0] > 0:
                        stop = True
                else: #poly_tile is true
                    poly_lat = [point[0] for point in tile]
                    poly_lon = [point[1] for point in tile]
                    hull_pts = np.column_stack((poly_lat, poly_lon))
                    pred_pt = np.column_stack((df[self._coord1], df[self._coord2]))

                    if not isinstance(hull_pts, Delaunay):
                        hull = Delaunay(hull_pts)
                    mask = hull.find_simplex(pred_pt)>=0
                    if df[mask].shape[0] > 0:
                        stop = True
        else:
            for i in LAT:
                if stop:
                    break
                for j in LON:
                    tile = [i, i + EPS, j, j + EPS]

                    TS_NAME = ('#'.join(str(x) for x in tile))+"#"+stringify(_types)
                    lat_ = tile[0:2]
                    lon_ = tile[2:4]

                    test = df.loc[(df[self._coord1] > lat_[0])
                                & (df[self._coord1] <= lat_[1])
                                & (df[self._coord2] > lon_[0])
                                & (df[self._coord2] <= lon_[1])]

                    if test.shape[0] > 0:
                        stop=True
                        break

        # make the DATE the index and keep only the event col
        df.index = df[self._DATE]
        df=df[[self._EVENT]]

        return df,TS_NAME


    def get_opt_freq(self,df,TS_NAME,incr=6,max_incr=24):
        '''
        Utilities for spatio temporal analysis
        @author zed.uchicago.edu

        Returns the optimal frequency for timeseries based on highest non-zero
        to zero timeseries event count

        Input -
            df (pd.DataFrame): filtered subset of dataset corresponding to
            random tile from get_rand_tile
            incr (int): frequency increment
            max_incr (int): user-specified maximum increment
            TS_NAME

        Output -
            (string) to pass to pd.date_range(freq=) argument
        '''

        ratios = []
        for i in range(max_incr/incr):
            curr_incr = ((i+1)*incr)
            curr_freq = str(curr_incr)+'H'
            curr_trng = pd.date_range(start=self._INIT,
                                       end=self._END,freq=curr_freq)
            ts = [df.loc[curr_trng[i]:curr_trng[i + 1]].size for i in
                  np.arange(curr_trng.size - 1)]
            out = pd.DataFrame(ts, columns=[TS_NAME],
                                index=curr_trng[:-1]).transpose()
            tot = curr_trng.size+0.0
            non_zero_ratio = out.astype(bool).sum(axis=1)/tot
            ratios.append((curr_freq, non_zero_ratio[0]))

        ratios.sort(key=operator.itemgetter(1))

        return ratios[-1][0]


    def timeseries(self,LAT=None,LON=None,EPS=None,_types=None,CSVfile='TS.csv',
                   THRESHOLD=None,tiles=None,auto_adjust_time=False,incr=6,
                   max_incr=24,poly_tile=False):
        """
        Utilities for spatio temporal analysis
        @author zed.uchicago.edu

        Creates DataFrame of location tiles and their
        respective timeseries from
        input datasource with significance threshold THRESHOLD
        latitude, longitude coordinate boundaries given by LAT, LON and EPS
        or the custom boundaries given by tiles
        calls on getTS for individual tile then concats them together

        Input:
            LAT (float or list of floats): singular coordinate float or list of
                                           coordinate start floats
            LON (float or list of floats): singular coordinate float or list of
                                           coordinate start floats
            EPS (float): coordinate increment ESP
            _types (list): event type filter; accepted event type list
            CSVfile (string): path to output file
            tiles (list of lists): list of tiles to build where tile can be
            a (list of floats i.e. [lat1 lat2 lon1 lon2]) or tuples (i.e. [(x1,y1),(x2,y2)])
            defining polygons
            auto_adjust_time (boolean): if True, within increments specified (6H default),
                determine optimal temporal frequency for timeseries data
            incr (int): frequency increment
            max_incr (int): user-specified maximum increment
            poly_tile (boolean): whether or tiles define polygons

        Output:
            No Output grid pd.Dataframe written out as CSV file to path specified
        """

        if THRESHOLD is None:
            if self._THRESHOLD is None:
                THRESHOLD=0.1
            else:
                THRESHOLD=self._THRESHOLD

        if self._trng is None:
            self._trng = pd.date_range(start=self._INIT,
                                       end=self._END,freq=self._FREQ)

        assert (LAT is not None and LON is not None and EPS is not None) or\
               (tiles is not None),\
                "Error: (LAT, LON, EPS) or tiles not defined."

        if tiles is not None:
            if auto_adjust_time:
                rand_tile_df,ts_name=self.get_rand_tile(\
                tiles=tiles,_types=_types,poly_tile=poly_tile)
                opt_freq = self.get_opt_freq(df=rand_tile_df,incr=incr,\
                                             max_incr=max_incr,TS_NAME=ts_name)
                _TS = pd.concat([self.getTS(tile=coord_set,_types=_types,\
                                freq=opt_freq,poly_tile=poly_tile) for coord_set in tqdm(tiles)])
            else:
                _TS = pd.concat([self.getTS(tile=coord_set,_types=_types,poly_tile=poly_tile)\
                                for coord_set in tqdm(tiles)])
        else: # note custom coordinate boundaries takes precedence
            if auto_adjust_time:
                rand_tile_df,ts_name=self.get_rand_tile(LAT=LAT,LON=LON,EPS=EPS,_types=_types)
                opt_freq = self.get_opt_freq(df=rand_tile_df,incr=incr,\
                                             max_incr=max_incr,TS_NAME=ts_name)
                _TS = pd.concat([self.getTS(tile=[i,i+EPS,j,j+EPS],freq=opt_freq,\
                                            _types=_types) for i in tqdm(LAT)
                                            for j in tqdm(LON)])
            else: # note custom coordinate boundaries takes precedence
                _TS = pd.concat([self.getTS(tile=[i,i+EPS,j,j+EPS],\
                                            _types=_types) for i in tqdm(LAT)
                                            for j in tqdm(LON)])

        LEN = pd.date_range(start=self._INIT,
                          end=self._END,freq=self._FREQ).size+0.0

        statbool = _TS.astype(bool).sum(axis=1) / LEN
        _TS = _TS.loc[statbool > THRESHOLD]
        self._ts_dict[repr(_types)] = _TS

        self._TS=_TS

        if CSVfile is not None:
            _TS.to_csv(CSVfile, sep=' ')

        return


    def fit(self,grid=None,INIT=None,END=None,THRESHOLD=None,csvPREF='TS',
            auto_adjust_time=False,incr=6,max_incr=24,poly_tile=False):
        """
        Utilities for spatio temporal analysis
        @author zed.uchicago.edu

        Fit dataproc with specified grid parameters and
        create timeseries for
        date boundaries specified by INIT, THRESHOLD,
        and END or by the input list of custom coordinate boundaries
        which do NOT have to match the arguments first input to the dataproc

        Inputs:
            grid (dictionary or list of lists): coordinate dictionary with
                respective ranges and EPS value OR custom list of lists
                of custom grid tiles as [coord1_start, coord1_stop,
                coord2_start, coord2_stop]
            INIT (datetime.date): starting timeseries date
            END (datetime.date): ending timeseries date
            THRESHOLD (float): significance threshold
            auto_adjust_time (boolean): if True, within increments specified (6H default),
                determine optimal temporal frequency for timeseries data
            incr (int): frequency increment
            max_incr (int): user-specified maximum increment
            poly_tile (boolean): whether or not tiles define polygons

        Outputs:
            (No output) grid pd.Dataframe written out as CSV file
                    to path specified
        """

        if INIT is not None:
            self._INIT=INIT
        if END is not None:
            self._END=END
        if grid is not None:
            if isinstance(grid, dict):
                assert(self._coord1 in grid)
                assert(self._coord2 in grid)
                assert('Eps' in grid)
                # constructing private variable self._grid in the desired format
                # with the values taken from the input grid
                self._grid[self._coord1]=grid[self._coord1]
                self._grid[self._coord2]=grid[self._coord2]
                self._grid['Eps']=grid['Eps']
                self._grid_type = "auto"
            elif isinstance(grid, list):
                self._grid = grid
                self._grid_type = "custom"
            else:
                raise TypeError("Unsupported grid type.")

        assert(self._END is not None)

        if self._types is not None:
            for key in self._types:
                if self._grid_type == "auto":
                    self.timeseries(LAT=self._grid[self._coord1],
                                    LON=self._grid[self._coord2],
                                    EPS=self._grid['Eps'],
                                    _types=key,
                                    CSVfile=csvPREF+stringify(key)+'.csv',
                                    THRESHOLD=THRESHOLD,
                                    auto_adjust_time=auto_adjust_time,
                                    incr=incr,max_incr=max_incr,poly_tile=poly_tile)
                else:
                    self.timeseries(tiles=self._grid,
                                    _types=key,
                                    CSVfile=csvPREF+stringify(key)+'.csv',
                                    THRESHOLD=THRESHOLD,
                                    auto_adjust_time=auto_adjust_time,
                                    incr=incr,max_incr=max_incr,poly_tile=poly_tile)
            return
        else:
            assert(self._value_limits is not None), \
            "Error: Neither value_limits nor _types has been defined."
            if self._grid_type == "auto":
                self.timeseries(LAT=self._grid[self._coord1],
                                LON=self._grid[self._coord2],
                                EPS=self._grid['Eps'],
                                _types=None,
                                CSVfile=csvPREF+'.csv',
                                THRESHOLD=THRESHOLD,
                                auto_adjust_time=auto_adjust_time,
                                incr=incr,max_incr=max_incr,poly_tile=poly_tile)
            else:
                self.timeseries(tiles=self._grid,
                                _types=None,
                                CSVfile=csvPREF+'.csv',
                                THRESHOLD=THRESHOLD,
                                auto_adjust_time=auto_adjust_time,
                                incr=incr,max_incr=max_incr,poly_tile=poly_tile)
            return


    def getGrid(self):
        '''
        Returns the tile coordinates of the working as a list of lists

        Input -
            (No inputs)
        Output -
            TILE (list of lists): the grid tiles
        '''

        cols=self._TS.index.values
        f=lambda x: x[:-1] if len(x)%2==1  else x
        TILE=None

        for word in cols:
            tile=[float(i) for i in f(word.replace('#',' ').split())]
            if TILE is None:
                TILE=[tile]
            else:
                TILE.append(tile)
        # get grid from TSfile tiles
        # and return in list form
        return TILE


    def pull(self, domain="data.cityofchicago.org",dataset_id="crimes",\
        token=None, store=True, out_fname="pull_df.p",
        pull_all=False):
        """
        Utilities for spatio temporal analysis
        @author zed.uchicago.edu

        Pulls new entries from datasource
        NOTE: should make flexible but for now use city of Chicago data

        Input -
            domain (string): Socrata database domain hosting data
            dataset_id (string): dataset ID to pull
            token (string): Socrata token for increased pull capacity;
                Note: Requires Socrata account
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
        return 'VAR'
    if not List:
        return ''
    whole_string = '-'.join(str(elem) for elem in List)
    return whole_string.replace(' ','_').replace('/','_').replace('(','').replace(')','')


def readTS(TSfile,csvNAME='TS1',BEG=None,END=None):
    """
    Utilities for spatio temporal analysis
    @author zed.uchicago.edu

    Reads in output TS logfile into pd.DF
        and then outputs necessary
        CSV files in XgenESeSS-friendly format

    Input -
        TSfile (string or list of strings): filename of input TS to read
            or list of filenames to read in and concatenate into one TS
        csvNAME (string)
        BEG (string): start datetime
        END (string): end datetime

    Output -
        dfts (pandas.DataFrame)
    """
    # adjustment to make readTS oeprate on a list of input files
    # instead of just one
    dfts = None
    if isinstance(TSfile, list):
        for tsfile in TSfile:
            if dfts is None:
                dfts=pd.read_csv(tsfile,sep=" ",index_col=0)
            else:
                dfts=dfts.append(pd.read_csv(tsfile,sep=" ",index_col=0))
    else:
        dfts=pd.read_csv(TSfile,sep=" ",index_col=0)


    dfts.columns = pd.to_datetime(dfts.columns)

    cols=dfts.columns[np.logical_and(dfts.columns >= pd.to_datetime(BEG),
                                     dfts.columns <= pd.to_datetime(END))]

    dfts=dfts[cols]
    indices = []
    for index, row in dfts.iterrows():
        if len(row.unique()) != 1:
            indices.append(index)
    dfts = dfts.loc[indices]

    dfts.to_csv(csvNAME+'.csv',sep=" ",header=None,index=None)
    np.savetxt(csvNAME+'.columns', cols, delimiter=',',fmt='%s')
    np.savetxt(csvNAME+'.coords', dfts.index.values, delimiter=',',fmt='%s')

    return dfts


def splitTS(TSfile,dirname='./',prefix="@",
            BEG=None,END=None,VARNAME=''):
    """
    Utilities for spatio temporal analysis
    @author zed.uchicago.edu

    Writes out each row of the pd.DataFrame as a separate CSVfile
    For XgenESeSS binary

    Inputs -
        TSfile (pd.DataFrame): DataFrame to write out
        csvNAME (string): output filename
        dirname (string): directory for output file
        prefix (string): prefix for files
        BEG (datetime): start date
        END (datetime): end date
        VARNAME (string): specifer variable for row

    Outputs -
        (No output)
    """

    dfts = None
    if isinstance(TSfile, list):
        for tsfile in TSfile:
            if dfts is None:
                dfts=pd.read_csv(tsfile,sep=" ",index_col=0)
            else:
                dfts=dfts.append(pd.read_csv(tsfile,sep=" ",index_col=0))
    else:
        dfts=pd.read_csv(TSfile,sep=" ",index_col=0)
    dfts.columns = pd.to_datetime(dfts.columns)

    cols=dfts.columns[np.logical_and(dfts.columns >= pd.to_datetime(BEG),
                                     dfts.columns <= pd.to_datetime(END))]
    dfts=dfts[cols]

    for row in dfts.index:
        dfts.loc[[row]].to_csv(dirname+"/"+prefix+row+VARNAME,
                               header=None,index=None,sep=" ")

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
               high=None,low=None,equal=None,inplace=False):
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

        Output -
            (dictionary): top n models as ranked by var
                         in ascending/descending order
        """

        #assert var in self._models.keys(), "Error: Model parameter specified not valid"

        if equal is not None:
            out={key:value
                 for (key,value) in self._models.iteritems() if value[var]==equal }
        else:

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
        if dict:
            if inplace:
                self._models=out
            if store is not None:
                with open(store, 'w') as outfile:
                    json.dump(out, outfile)
        else:
            warnings.warn("Selection creates empty model dict")

        return out


    def setVarname(self):
        """
        Utilities for storing and manipulating XPFSA models
        inferred by XGenESeSS
        @author zed.uchicago.edu

        Extracts the varname for src and tgt of
        each model and stores under src_var and tgt_var
        keys of each model;

        No I/O
        """

        VARNAME='var'
        f=lambda x: x[-1] if len(x)%2==1  else VARNAME

        for key,value in self._models.iteritems():
            self._models[key]['src_var']=f(value['src'].replace('#',' ').split())
            self._models[key]['tgt_var']=f(value['tgt'].replace('#',' ').split())

        return


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

        f=lambda x: x[:-1] if len(x)%2==1  else x

        for key,value in self._models.iteritems():
            src=[float(i) for i in f(value['src'].replace('#',' ').split())]
            tgt=[float(i) for i in f(value['tgt'].replace('#',' ').split())]

            dist = haversine((np.mean(src[0:2]),np.mean(src[2:])),
                           (np.mean(tgt[0:2]),np.mean(tgt[2:])),
                           unit = 'mi')
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

        Output -
            No output
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

        Output -
            Pandas.DataFrame with columns
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
        src_var=[]
        tgt_var=[]

        NUM=None
        f=lambda x: x[:-1] if len(x)%2==1  else x

        for key,value in self._models.iteritems():
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
            distance.append(value['distance'])
            src_var.append(value['src_var'])
            tgt_var.append(value['tgt_var'])

        self._df = pd.DataFrame({'latsrc':latsrc,
                                 'lonsrc':lonsrc,
                                 'lattgt':lattgt,
                                 'lontgt':lontgt,
                                 'gamma':gamma,
                                 'delay':delay,
                                 'distance':distance,
                                 'src':src_var,
                                 'tgt':tgt_var})

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

        Output -
            No output
    """

    with open(outFile, 'w') as outfile:
        json.dump(pydict, outfile)

    return


class simulateModel:
    '''
    Use the subprocess library to call cynet on a model to process
    it and then run flexroc on it to obtain statistics: auc, tpr, fuc.
    Input -
        MODEL_PATH(string)- The path to the model being processed.
        DATA_PATH(string)- Path to the split file.
        RUNLEN(integer)- Length of the run.
        READLEN(integer)- Length of split data to read from begining
        CYNET_PATH - path to cynet binary.
        FLEXROC_PATH - path to flexroc binary.
    '''

    def __init__(self, MODEL_PATH,
                 DATA_PATH,
                 RUNLEN,
                 CYNET_PATH,
                 FLEXROC_PATH,
                 READLEN=None):

        assert os.path.exists(CYNET_PATH), "cynet binary cannot be found."
        assert os.path.exists(FLEXROC_PATH), "roc binary cannot be found."
        assert os.path.exists(MODEL_PATH), "model file cannot be found."
        assert any(glob.iglob(DATA_PATH+"*")), \
            "split data files cannot be found."

        self.MODEL_PATH = MODEL_PATH
        self.DATA_PATH = DATA_PATH
        self.RUNLEN = RUNLEN
        self.CYNET_PATH = CYNET_PATH
        self.FLEXROC_PATH = FLEXROC_PATH
        self.RUNLEN = RUNLEN

        if READLEN is None:
            self.READLEN = RUNLEN
        else:
            self.READLEN = READLEN

    def run(self, LOG_PATH=None,
            PARTITION=0.5,
            DATA_TYPE='continuous',
            FLEXWIDTH=1,
            FLEX_TAIL_LEN=100,
            POSITIVE_CLASS_COLUMN=5,
            EVENTCOL=3,
            tpr_threshold=0.85,
            fpr_threshold=0.15):

        '''
        This function is intended to replace the cynrun.sh shell script. This
        function will use the subprocess library to call cynet on a model to process
        it and then run flexroc on it to obtain statistics: auc, tpr, fuc.
        Input -
           LOG_PATH (string)- Logfile from cynet run
           PARTITION (string)- Partition to use on split data
           FLEXWIDTH (int)-  Parameter to specify flex in flwxroc
           FLEX_TAIL_LEN (int)- tail length of input file to consider [0: all]
           POSITIVE_CLASS_COLUMN (int)- positive class column
           EVENTCOL (int)- event column
           tpr_threshold (float)- minimum tpr threshold
           fpr_threshold (float)- maximum fpr threshold

        Output -
            auc (float)- Area under the curve
            tpr (float)- True positive rate at specified maximum false positive rate
            fpr (float)- False positive rate at specified minimum true positive rate

        '''

        if LOG_PATH is None:
            LOG_PATH = self.MODEL_PATH + '-XX.log'
        cyrstr = self.CYNET_PATH + ' -J ' + self.MODEL_PATH\
            + ' -T ' + DATA_TYPE + ' -p ' + str(PARTITION) + ' -N '\
            + str(self.RUNLEN) + ' -x ' + str(self.READLEN)\
            + ' -l ' + LOG_PATH\
            + ' -w ' + self.DATA_PATH
        cyrstr_arg = shlex.split(cyrstr)
        #print cyrstr
        subprocess.check_call(cyrstr_arg, shell=False)
        flexroc_str = self.FLEXROC_PATH + ' -i ' + LOG_PATH\
            + ' -w ' + str(FLEXWIDTH) + ' -x '\
            + str(FLEX_TAIL_LEN) + ' -C '\
            + str(POSITIVE_CLASS_COLUMN) + ' -E ' + str(EVENTCOL)\
            + ' -t ' + str(tpr_threshold) + ' -f ' + str(fpr_threshold)
        #print flexroc_str
        flexstr_arg = shlex.split(flexroc_str)
        output_str = subprocess.check_output(flexstr_arg, shell=False)
        results = np.array(output_str.split())
        auc = float(results[1])
        tpr = float(results[7])
        fpr = float(results[13])

        return auc, tpr, fpr

    def get_threshold(self, cynet_logfile, tpr=None, fpr=None,
            FLEXWIDTH=1,
            FLEX_TAIL_LEN=100,
            POSITIVE_CLASS_COLUMN=5,
            EVENTCOL=3):
        '''
        Returns the desired threshold which will obtain
        a necessary threshold for a given tpr/fpr for a model
        logfile. Only one of tpr and fpr can be given.
        Inputs:
            cynet_logfile(str): path to the cynet_logfile
            tpr(float): desired tpr
            fpr(float): desired fpr
        '''
        assert (tpr is not None or fpr is not None), "Enter a fpr or tpr"
        if tpr:
            flexroc_str = self.FLEXROC_PATH + ' -i ' + cynet_logfile\
                + ' -w ' + str(FLEXWIDTH) + ' -x '\
                + str(FLEX_TAIL_LEN) + ' -C '\
                + str(POSITIVE_CLASS_COLUMN) + ' -E ' + str(EVENTCOL)\
                + ' -t ' + str(tpr)
        elif fpr:
            flexroc_str = self.FLEXROC_PATH + ' -i ' + cynet_logfile\
                + ' -w ' + str(FLEXWIDTH) + ' -x '\
                + str(FLEX_TAIL_LEN) + ' -C '\
                + str(POSITIVE_CLASS_COLUMN) + ' -E ' + str(EVENTCOL)\
                + ' -f ' + str(fpr)
        flexstr_arg = shlex.split(flexroc_str)
        output_str = subprocess.check_output(flexstr_arg, shell=False)
        threshold = float(output_str.split()[5])
        return threshold


    def single_cynet(self,LOG_PATH=None, DATA_TYPE='continuous', PARTITION=0.5):
        '''
        A single call of the cynet binary on a model file and producing a
        cynet logfile.
        Inputs-
            LOG_PATH(str): path to write cynet logfile to.
            LOG_PATH (string)- Logfile from cynet run
            PARTITION (string)- Partition to use on split data
        '''
        if LOG_PATH is None:
            LOG_PATH = self.MODEL_PATH + '-XX.log'
        cyrstr = self.CYNET_PATH + ' -J ' + self.MODEL_PATH\
            + ' -T ' + DATA_TYPE + ' -p ' + str(PARTITION) + ' -N '\
            + str(self.RUNLEN) + ' -x ' + str(self.READLEN)\
            + ' -l ' + LOG_PATH\
            + ' -w ' + self.DATA_PATH
        cyrstr_arg = shlex.split(cyrstr)
        subprocess.check_call(cyrstr_arg, shell=False)


    def parse_cynet(self, cynet_logfile,
                    varname,
                    tpr=None,
                    fpr=None,
                    FLEXWIDTH=1,
                    FLEX_TAIL_LEN=100,
                    coord_col=1,
                    day_col=2,
                    EVENTCOL=3,
                    NEGATIVE_CLASS_COLUMN=4,
                    POSITIVE_CLASS_COLUMN=5,
                    header=['lat1','lat2','lon1','lon2','target','day',\
                            'actual_event','negative_event','positive_event'],
                    positive_str='positive_event',
                    outfile=None):
        '''
        The function parses the cynet logs into a more python friendly format
        and returns it as a dataframe. We use flexroc to grab the necessary threshold.
        We also use a threshold to map the POSITIVE_CLASS_COLUMN to a column of
        predicted events.
            Inputs:
                cynet_logfile(str): path to the cynet_logfile
                tpr(float): desired true positive rate
                fpr(float): desired false positive rate
                coord_col(int): column number of the coords in the cynet log file
                day_col(int): column number of the day in the cynet log file
                EVENTCOL(int): column number of whether the actual event
                    (based on data)
                NEGATIVE_CLASS_COLUMN(int): column number of the model'sprediction
                    for a non event.
                POSITIVE_CLASS_COLUMN(int): column number of the model's prediction
                    of the event.
                header(list of str): Headers for dataframe
                positive_str(str): Column name of positive class column.
                outfile(str): file path to write dataframe to
        '''
        threshold = get_flexroc_threshold(cynet_logfile,self.FLEXROC_PATH,tpr,fpr,FLEXWIDTH,FLEX_TAIL_LEN,POSITIVE_CLASS_COLUMN,EVENTCOL)
        with open(cynet_logfile,'r') as file:
            content = file.readlines()

        lines = []
        for line in content:
            elements = line.split()
            #Getting location
            location = elements[coord_col].split("#")
            lat1,lat2,lon1,lon2,variable = \
            float(location[0]),float(location[1]),float(location[2]),float(location[3]),\
            location[4]
            #Get Day
            day = int(elements[day_col])
            #Get actual event.
            actual_event = int(elements[EVENTCOL])
            negative_event = float(elements[NEGATIVE_CLASS_COLUMN])
            positive_event = float(elements[POSITIVE_CLASS_COLUMN])

            lines.append([lat1,lat2,lon1,lon2,variable,day,actual_event,negative_event,positive_event])

        df = pd.DataFrame(lines, columns=header)
        df['predictions'] = (df[positive_str] > threshold).astype(int)
        df['source'] = varname
        df['threshold'] = threshold

        if outfile:
            outfile = outfile + '#' + variable + '.csv'
            df.to_csv(outfile, index=False)
        return df

def flexroc_only(arguments):
    '''
    Similar to parse_cynet but utilized in the flexroc only pipeline. Called by
    flexroc_only_parallel. For details on the arguments, see flexroc_only_parallel
    '''
    logfile,tpr_threshold,fpr_threshold,FLEXWIDTH,FLEX_TAIL_LEN,coord_col,day_col,EVENTCOL,\
    NEGATIVE_CLASS_COLUMN,POSITIVE_CLASS_COLUMN,header,positive_str,FLEXROC_PATH = \
    arguments[0],arguments[1],arguments[2],arguments[3],arguments[4],arguments[5],\
    arguments[6],arguments[7],arguments[8],arguments[9],arguments[10],arguments[11],arguments[12]

    threshold = get_flexroc_threshold(logfile, FLEXROC_PATH,tpr_threshold,fpr_threshold,FLEXWIDTH,FLEX_TAIL_LEN,POSITIVE_CLASS_COLUMN,EVENTCOL)
    with open(logfile,'r') as file:
        content = file.readlines()

    lines = []
    for line in content:
        elements = line.split()
        #Getting location
        location = elements[coord_col].split("#")
        lat1,lat2,lon1,lon2,variable = \
        float(location[0]),float(location[1]),float(location[2]),float(location[3]),\
        location[4]
        #Get Day
        day = int(elements[day_col])
        #Get actual event.
        actual_event = int(elements[EVENTCOL])
        negative_event = float(elements[NEGATIVE_CLASS_COLUMN])
        positive_event = float(elements[POSITIVE_CLASS_COLUMN])

        lines.append([lat1,lat2,lon1,lon2,variable,day,actual_event,negative_event,positive_event])

    df = pd.DataFrame(lines, columns=header)
    varname = logfile.split('#')[1].strip('.log')
    df['predictions'] = (df[positive_str] > threshold).astype(int)
    df['source'] = varname
    df['threshold'] = threshold

    log_prefix = logfile.strip('.log')
    outfile = log_prefix + '#' + variable + '.csv'
    df.to_csv(outfile, index=False)


def flexroc_only_parallel(glob_path,
                cores = 4,
                FLEXWIDTH=1,
                FLEX_TAIL_LEN=100,
                POSITIVE_CLASS_COLUMN=5,
                EVENTCOL=3,
                tpr_threshold=0.85,
                fpr_threshold=0.15,
                coord_col=1,
                day_col=2,
                NEGATIVE_CLASS_COLUMN=4,
                header=['lat1','lat2','lon1','lon2','target','day',\
                        'actual_event','negative_event','positive_event'],
                positive_str='positive_event'):
    '''
    In the event that we do not need to rerun cynet, but only flexroc on some files
    we will use this pipeline. This pipeline assumes that run_pipeline has already
    been run and thus the cynet logfiles already exist. This function will apply
    flexroc to all of the cynet files and produce csvs from them. These csvs are
    pandas friendly.
    Input-
        glob_path(str)- glob path matching cynet log files.
        cores(int)-number of cores to use in multiprocessing.
        FLEXWIDTH(int)-grace period. flexroc argument.
        FLEX_TAIL_LEN(int)- prediction length. flexroc argument.
        X_Column(int)- column of cynet logfile where X is.
        tpr_threshold(float)-desired tpr. Only tpr or fpr may be specified.
        fpr_threshold(float)-desired fpr. Only tpr or fpr may be specified.
        header(list)-column names of the csvs.
        positive_str(str)- name of the positive event column.
    '''
    cynet_logfile = glob.glob(glob_path)

    FLEXROC_PATH = os.path.dirname(sys.modules['cynet'].__file__) + '/bin/flexroc'
    args = []
    for logfile in cynet_logfile:
        args.append([logfile,tpr_threshold,fpr_threshold,FLEXWIDTH,FLEX_TAIL_LEN,coord_col,day_col,EVENTCOL,NEGATIVE_CLASS_COLUMN,POSITIVE_CLASS_COLUMN,header,positive_str,FLEXROC_PATH])

    Parallel(n_jobs = cores, verbose = 1, backend = 'threading')\
    (map(delayed(flexroc_only), args))


def get_flexroc_threshold(cynet_logfile,FLEXROC_PATH, tpr=None, fpr=None,
        FLEXWIDTH=1,
        FLEX_TAIL_LEN=100,
        POSITIVE_CLASS_COLUMN=5,
        EVENTCOL=3):
    '''
    Returns the desired threshold which will obtain
    a necessary threshold for a given tpr/fpr for a model
    logfile. Only one of tpr and fpr can be given.
    Inputs:
        cynet_logfile(str): path to the cynet_logfile
        tpr(float): desired tpr
        fpr(float): desired fpr
    '''
    assert (tpr is not None or fpr is not None), "Enter a fpr or tpr"
    if tpr:
        flexroc_str = FLEXROC_PATH + ' -i ' + cynet_logfile\
            + ' -w ' + str(FLEXWIDTH) + ' -x '\
            + str(FLEX_TAIL_LEN) + ' -C '\
            + str(POSITIVE_CLASS_COLUMN) + ' -E ' + str(EVENTCOL)\
            + ' -t ' + str(tpr)
    elif fpr:
        flexroc_str = FLEXROC_PATH + ' -i ' + cynet_logfile\
            + ' -w ' + str(FLEXWIDTH) + ' -x '\
            + str(FLEX_TAIL_LEN) + ' -C '\
            + str(POSITIVE_CLASS_COLUMN) + ' -E ' + str(EVENTCOL)\
            + ' -f ' + str(fpr)
    flexstr_arg = shlex.split(flexroc_str)
    output_str = subprocess.check_output(flexstr_arg, shell=False)
    threshold = float(output_str.split()[5])
    return threshold


def single_map(arguments):
    '''
    A single call for a single model which will map the events predicted by
    said model.
    Inputs:
        arguments(list) - a list of arguments necessary for the function:
            arguments[0]-FILE(str): path to the model being processed.
            arguments[1]-model_nums(int): Number of models to use in prediction
            arguments[2]-Horizon(int): prediction horizon.
            arguments[3]-DATA_PATH: path to split file.
                Ex: './split/1995-01-01_1999-12-31'
            arguments[4]-RUNLEN(int): the runlength
            arguments[5]-VARNAME(list)-Variable names to be considering.
            arguments[6]-RESSUFIX- suffix to add to the end of results.
            arguments[7]-CYNET_PATH- path to cynet binary.
            arguments[8]-FLEXROC_PATH- path to flexroc binary.
            other arguments are for cynet and flexroc. See simulateModel for
            description.
    '''
    FILE = arguments[0]
    model_nums = arguments[1]
    Horizon = arguments[2]
    DATA_PATH = arguments[3]
    RUNLEN = arguments[4]
    VARNAME = arguments[5]
    RESSUFIX = arguments[6]
    CYNET_PATH = arguments[7]
    FLEXROC_PATH = arguments[8]
    LOG_PATH = arguments[9]
    PARTITION = arguments[10]
    DATA_TYPE = arguments[11]
    FLEXWIDTH = arguments[12]
    FLEX_TAIL_LEN = arguments[13]
    POSITIVE_CLASS_COLUMN = arguments[14]
    EVENTCOL = arguments[15]
    tpr_threshold = arguments[16]
    fpr_threshold = arguments[17]
    coord_col = arguments[18]
    day_col = arguments[19]
    NEGATIVE_CLASS_COLUMN = arguments[20]
    header = arguments[21]
    positive_str = arguments[22]
    outfile = arguments[23]

    for varname in VARNAME:
        stored_model=FILE+'_sel_'+str(uuid.uuid4())+'.json'

        M=uNetworkModels(FILE + '.json')
        M.setVarname()
        M.augmentDistance()

        if varname is not 'ALL':
            M.select(var='src_var',equal=varname,inplace=True)

        M.select(var='delay',inplace=True,low=Horizon)
        #M.select(var='distance',n=model_nums,store=stored_model,reverse=False,inplace=True)
        M.select(var='gamma',n=model_nums,store=stored_model,reverse=True,inplace=True)

        if M.models:
            outfile = FILE + 'use{}models'.format(model_nums) + '#' + varname
            LOG_PATH = FILE + 'use{}models'.format(model_nums) + '#' + varname + '.log'
            simulation = simulateModel(stored_model, DATA_PATH, RUNLEN, CYNET_PATH=CYNET_PATH,FLEXROC_PATH=FLEXROC_PATH)
            simulation.single_cynet(LOG_PATH=LOG_PATH, DATA_TYPE=DATA_TYPE, PARTITION=PARTITION)
            simulation.parse_cynet(LOG_PATH,
                                   varname,
                                   tpr=tpr_threshold,
                                   fpr=fpr_threshold,
                                   FLEXWIDTH=FLEXWIDTH,
                                   FLEX_TAIL_LEN=FLEX_TAIL_LEN,
                                   coord_col=coord_col,
                                   day_col=day_col,
                                   EVENTCOL=EVENTCOL,
                                   NEGATIVE_CLASS_COLUMN=NEGATIVE_CLASS_COLUMN,
                                   POSITIVE_CLASS_COLUMN=POSITIVE_CLASS_COLUMN,
                                   header=header,
                                   positive_str=positive_str,
                                   outfile=outfile)


def map_events_parallel(glob_path,model_nums,horizon, DATA_PATH, RUNLEN, VARNAME,RES_PATH,
                RESSUFIX = '.res', cores = 4,
                LOG_PATH=None,
                PARTITION=0.5,
                DATA_TYPE='continuous',
                FLEXWIDTH=1,
                FLEX_TAIL_LEN=100,
                POSITIVE_CLASS_COLUMN=5,
                EVENTCOL=3,
                tpr_threshold=0.85,
                fpr_threshold=0.15,
                coord_col=1,
                day_col=2,
                NEGATIVE_CLASS_COLUMN=4,
                header=['lat1','lat2','lon1','lon2','target','day',\
                        'actual_event','negative_event','positive_event'],
                positive_str='positive_event'):
    '''
    The parallel function which will parallelize cynet and flexroc to map model
    predictions to actual events. Either happens or does not happen.
    Inputs:
        Glob_path(str)-The glob string to be used to find all models. EX: 'models/*model.json'
        model_nums(list of ints)- The model numbers to use. Ex; [10,15,20,25]
        Horizon(int)- prediction horizons to test in unit of temporal quantization (using cynet binary)
        DATA_PATH(str)-Path to the split files. Ex: './split/1995-01-01_1999-12-31'
        RUNLEN(int)-Length of run. Ex: 2291.
        VARNAME(list of str)- List of variables to consider.
        RES_PATH(str)- glob string for glob to locate all result files. Ex:'./models/*model*res'
        RESUFFIX(str)- suffix to add to the end of results.Ex:'.res'
        cores(int)-cores to use for parrallel processing.
        kwargs- other arguments for cynet and flexroc. See simulateModel class.
    '''
    models_files = glob.glob(glob_path)
    models_files = [m.split('.')[0] for m in models_files]

    CYNET_PATH = os.path.dirname(sys.modules['cynet'].__file__) + '/bin/cynet'
    FLEXROC_PATH = os.path.dirname(sys.modules['cynet'].__file__) + '/bin/flexroc'

    args = []
    for model in models_files:
        for num in model_nums:
            LOG_PATH = model + 'use_{}models'.format(num)
            outfile = model + 'use_{}models'.format(num)
            args.append([model, num, horizon, DATA_PATH, RUNLEN, VARNAME, RESSUFIX, \
            CYNET_PATH, FLEXROC_PATH,
            LOG_PATH, #Here onwards are the run parameters of the pipeline.
            PARTITION,
            DATA_TYPE,
            FLEXWIDTH,
            FLEX_TAIL_LEN,
            POSITIVE_CLASS_COLUMN,
            EVENTCOL,
            tpr_threshold,
            fpr_threshold,
            coord_col,
            day_col,
            NEGATIVE_CLASS_COLUMN,
            header,
            positive_str,
            outfile])
    Parallel(n_jobs = cores, verbose = 1, backend = 'threading')\
    (map(delayed(single_map), args))


def parallel_process(arguments):
    '''
    This function takes a model and produces statistics on them. The output is
    saved to a result file with the suffix defined by RESUFFIX. We note that
    arguments needs to be a list of various arguments (detailed below) due to
    the nature of joblib. We expect this function to be called by a parallel
    processing library such as joblib.
    Inputs:
        arguments(list) - a list of arguments necessary for the function:
            arguments[0]-FILE(str): path to the model being processed.
            arguments[1]-model_nums(int): Number of models to use in prediction
            arguments[2]-Horizon(int): prediction horizon.
            arguments[3]-DATA_PATH: path to split file.
                Ex: './split/1995-01-01_1999-12-31'
            arguments[4]-RUNLEN(int): the runlength
            arguments[5]-VARNAME(list)-Variable names to be considering.
            arguments[6]-RESSUFIX- suffix to add to the end of results.
            arguments[7]-CYNET_PATH- path to cynet binary.
            arguments[8]-FLEXROC_PATH- path to flexroc binary.
            other arguments are for cynet and flexroc. See simulateModel for
            description.
    '''
    FILE = arguments[0]
    model_nums = arguments[1]
    Horizon = arguments[2]
    DATA_PATH = arguments[3]
    RUNLEN = arguments[4]
    VARNAME = arguments[5]
    RESSUFIX = arguments[6]
    CYNET_PATH = arguments[7]
    FLEXROC_PATH = arguments[8]
    LOG_PATH = arguments[9]
    PARTITION = arguments[10]
    DATA_TYPE = arguments[11]
    FLEXWIDTH = arguments[12]
    FLEX_TAIL_LEN = arguments[13]
    POSITIVE_CLASS_COLUMN = arguments[14]
    EVENTCOL = arguments[15]
    tpr_threshold = arguments[16]
    fpr_threshold = arguments[17]
    gamma = arguments[18]
    distance = arguments[19]
    testing = arguments[20]
    sample_num = arguments[21]
    RESULT = []
    header=['loc_id','lattgt1','lattgt2','lontgt1','lontgt2','varsrc','vartgt','num_models','auc','tpr','fpr','horizon']

    for varname in VARNAME:
        stored_model=FILE+'_sel_'+str(uuid.uuid4())+'.json'


        M=uNetworkModels(FILE + '.json')
        M.setVarname()
        M.augmentDistance()

        if varname is not 'ALL':
            M.select(var='src_var',equal=varname,inplace=True)

        M.select(var='delay',inplace=True,low=Horizon)
        if distance:
            M.select(var='distance',n=model_nums,store=stored_model,reverse=False,inplace=True)
        if gamma:
            M.select(var='gamma',n=model_nums,store=stored_model,reverse=True,inplace=True)

        if M.models:
            if testing:
                LOG_PATH = FILE + 'use{}models'.format(model_nums) + '#' + varname + 'sample' + str(sample_num) + 'test.log'
            else:
                LOG_PATH = FILE + 'use{}models'.format(model_nums) + '#' + varname + '.log'
            simulation = simulateModel(stored_model, DATA_PATH, RUNLEN, CYNET_PATH=CYNET_PATH,FLEXROC_PATH=FLEXROC_PATH)
            [auc, tpr, fpr] = simulation.run(LOG_PATH = LOG_PATH,
            PARTITION = PARTITION,
            DATA_TYPE = DATA_TYPE,
            FLEXWIDTH = FLEXWIDTH,
            FLEX_TAIL_LEN = FLEX_TAIL_LEN,
            POSITIVE_CLASS_COLUMN = POSITIVE_CLASS_COLUMN,
            EVENTCOL = EVENTCOL,
            tpr_threshold = tpr_threshold,
            fpr_threshold = fpr_threshold)

            f=lambda x: x[:-1] if len(x)%2==1  else x
            tgt=[float(j) for j in f((M.models).itervalues().next()['tgt'].replace('#',' ').split())]

            varnametgt=(M.models).itervalues().next()['tgt_var']
            result=[FILE]+list(tgt)+[varname,varnametgt,model_nums,auc,tpr,fpr,Horizon]
            RESULT.append(result)

    pd.DataFrame(RESULT,columns=header).to_csv(FILE+'_'+str(model_nums)+'_'+str(Horizon)+ '_' +str(sample_num) + RESSUFIX,index=None)
    #print pd.DataFrame(RESULT,columns=header)[['lattgt1','lattgt2','varsrc','vartgt','auc']]


def run_pipeline(glob_path,model_nums,horizon, DATA_PATH, RUNLEN, VARNAME,RES_PATH,
                RESSUFIX = '.res', cores = 4,
                LOG_PATH=None,
                PARTITION=0.5,
                DATA_TYPE='continuous',
                FLEXWIDTH=1,
                FLEX_TAIL_LEN=100,
                POSITIVE_CLASS_COLUMN=5,
                EVENTCOL=3,
                tpr_threshold=0.85,
                fpr_threshold=0.15,
                gamma=False,
                distance=False):
    '''
    This function is intended to take the output models from midway, process
    them, and produce graphs. This will call the parallel_process function
    in parallel using joblib. Eventually stores the result as 'res_all.csv'.
    Cynet and flexroc are binaries written in C++.
    Inputs:
        Glob_path(str)-The glob string to be used to find all models. EX: 'models/*model.json'
        model_nums(list of ints)- The model numbers to use. Ex; [10,15,20,25]
        Horizon(int)- prediction horizons to test in unit of temporal quantization (using cynet binary)
        DATA_PATH(str)-Path to the split files. Ex: './split/1995-01-01_1999-12-31'
        RUNLEN(int)-Length of run. Ex: 2291.
        VARNAME(list of str)- List of variables to consider.
        RES_PATH(str)- glob string for glob to locate all result files. Ex:'./models/*model*res'
        RESUFFIX(str)- suffix to add to the end of results.Ex:'.res'
        cores(int)-cores to use for parrallel processing.
        gamma(bool)- Whether to sort by gamma.
        distance(bool)- Whether to sort by distance.
        kwargs- other arguments for cynet and flexroc. See simulateModel class.

    Outputs: Produces graphs of statistics.
    '''
    models_files = glob.glob(glob_path)
    models_files = [m.split('.')[0] for m in models_files]

    CYNET_PATH = os.path.dirname(sys.modules['cynet'].__file__) + '/bin/cynet'
    FLEXROC_PATH = os.path.dirname(sys.modules['cynet'].__file__) + '/bin/flexroc'

    args = []
    for model in models_files:
        for num in model_nums:
            args.append([model, num, horizon, DATA_PATH, RUNLEN, VARNAME, RESSUFIX, \
            CYNET_PATH, FLEXROC_PATH,
            LOG_PATH, #Here onwards are the run parameters of the pipeline.
            PARTITION,
            DATA_TYPE,
            FLEXWIDTH,
            FLEX_TAIL_LEN,
            POSITIVE_CLASS_COLUMN,
            EVENTCOL,
            tpr_threshold,
            fpr_threshold,
            gamma,
            distance,
            False,
            0])
    Parallel(n_jobs = cores, verbose = 1, backend = 'threading')\
    (map(delayed(parallel_process), args))
    df=pd.concat([pd.read_csv(i) for i in glob.glob(RES_PATH)])
    df.to_csv('res_all.csv',index=None)


def test_model_nums(sample_size,glob_path,model_nums,horizon, DATA_PATH, RUNLEN, VARNAME,RES_PATH,
                RESSUFIX = '.res', cores = 4,
                LOG_PATH=None,
                PARTITION=0.5,
                DATA_TYPE='continuous',
                FLEXWIDTH=1,
                FLEX_TAIL_LEN=100,
                POSITIVE_CLASS_COLUMN=5,
                EVENTCOL=3,
                tpr_threshold=0.85,
                fpr_threshold=0.15,
                gamma=False,
                distance=False,
                resamples=1):
    '''
    This function is intended to help test for the best model numbers to use for
    highest auc results. This will largely run the same processes as run_pipeline
    but on a sample of the models to get through all the model numbers quickly.
    Inputs:
        sample_size(int)- Number of models to be sampled and used in the test.
        Glob_path(str)-The glob string to be used to find all models. EX: 'models/*model.json'
        model_nums(list of ints)- The model numbers to use. Ex; [10,15,20,25]
        Horizon(int)- prediction horizons to test in unit of temporal quantization (using cynet binary)
        DATA_PATH(str)-Path to the split files. Ex: './split/1995-01-01_1999-12-31'
        RUNLEN(int)-Length of run. Ex: 2291.
        VARNAME(list of str)- List of variables to consider.
        RES_PATH(str)- glob string for glob to locate all result files. Ex:'./models/*model*res'
        RESUFFIX(str)- suffix to add to the end of results.Ex:'.res'
        cores(int)-cores to use for parrallel processing.
        gamma(bool)- Whether to sort by gamma.
        distance(bool)- Whether to sort by distance.
        resamples(int)- Number of times to resample
        kwargs- other arguments for cynet and flexroc. See simulateModel class.
    Outputs: Sampled auc results.
    '''
    args = []
    models_files = glob.glob(glob_path)
    models_files = [m.split('.')[0] for m in models_files]
    CYNET_PATH = os.path.dirname(sys.modules['cynet'].__file__) + '/bin/cynet'
    FLEXROC_PATH = os.path.dirname(sys.modules['cynet'].__file__) + '/bin/flexroc'

    for num in model_nums:
        for sample_num in range(1,resamples + 1):
            random_sample = [models_files[i] for i in random.sample(range(len(models_files)), sample_size)]
            for model in random_sample:
                args.append([model, num, horizon, DATA_PATH, RUNLEN, VARNAME, RESSUFIX, \
                CYNET_PATH, FLEXROC_PATH,
                LOG_PATH, #Here onwards are the run parameters of the pipeline.
                PARTITION,
                DATA_TYPE,
                FLEXWIDTH,
                FLEX_TAIL_LEN,
                POSITIVE_CLASS_COLUMN,
                EVENTCOL,
                tpr_threshold,
                fpr_threshold,
                gamma,
                distance,
                True,
                sample_num])
    print len(args)
    Parallel(n_jobs = cores, verbose = 1, backend = 'threading')\
    (map(delayed(parallel_process), args))
    df=pd.concat([pd.read_csv(i) for i in glob.glob(RES_PATH)])
    df.to_csv('testres_all.csv',index=None)
    performance = df.groupby('num_models').mean()['auc'].sort_values(ascending=False)
    best_model_num = performance.index[0]
    print "The best number of models is {}".format(best_model_num)
    return args


def get_var(res_csv, coords,varname='auc',VARNAMES=None):
    '''
    This function outputs graphs of the results produced by run_pipeline. The
    graphs concern auc, fpr, and tpr statistics.
    Inputs:
        res_csv(str)- path to 'res_all.csv' file produced by run_pipeline.
        coords(list of str)- the coords to consider.
            Ex:['lattgt1','lattgt2','lontgt1','lontgt2']
        varname(str)-the variable name to consider. Ex: 'auc'.
        VARNAMES(str)- List of the variable name from the dataset to consider.
            Ex: VARNAMES=['Personnel','Infrastructure','Casualties']
    '''
    plt.figure()
    df = pd.read_csv(res_csv)
    df1=df.groupby(coords,squeeze=True)[varname].max().reset_index()
    df1=df1[df1[varname].between(0.01,0.99)]
    if len(coords)%2 == 0:
        ax=sns.distplot(df1[varname])
        ax.set_xlabel(varname,fontsize=18,fontweight='bold');
        Type=''
    else:
        Type='vartgt'
        print coords
        print varname
        print df1
        ax=sns.violinplot(x=coords[-1],y=varname,data=df1,cut=0)
        if VARNAMES is not None:
            ax.set_xticklabels(VARNAMES)
        ax.set_xlabel('Event Type',fontsize=18,fontweight='bold')
        ax.set_ylabel(varname,fontsize=18,fontweight='bold');
    df1.to_csv(varname+Type+'.csv',sep=" ",index=None)
    plt.savefig(varname+Type+'.pdf',dpi=600, bbox_inches='tight',transparent=False)


class xgModels:
    '''
    Utility class for running XgenESeSS. This class will either run XgenESeSS
    locally or produce the list of commands to run on a cluster. We note that
    you may set the path of XgenESeSS in the yaml file. If running on a cluster
    then the commands will use the path use the XgenESeSS path in the yaml. If
    running on
    Attributes -
        TS_PATH(string)- path to file which has the rowwise multiline
            time series data
        NAME_PATH(string)-path to file with name of the variables
        LOG_PATH(string)-path to log file for xgenesess inference
        BEG(int) & END(int)- xgenesses run parameters (not hyperparameters,
            Beg is 0, End is whatever tempral memory is)
        NUM(int)-number of restarts (20 is good)
        PARTITION(float)-partition sequence
        XgenESeSS_PATH(str)-path to XgenESeSS
        RUN_LOCAL(bool)- whether to run XgenESeSS locally or produce a list of
        commands to run on a cluster.
    '''
    def __init__(self, TS_PATH,
                NAME_PATH,
                LOG_PATH,
                FILEPATH,
                BEG,
                END,
                NUM,
                PARTITION,
                XgenESeSS_PATH,
                RUN_LOCAL):

        assert os.path.exists(TS_PATH), "Time series file not found"
        assert os.path.exists(NAME_PATH), "Name file not found"

        self.TS_PATH = TS_PATH
        self.NAME_PATH = NAME_PATH
        self.LOG_PATH = LOG_PATH
        self.FILEPATH = FILEPATH
        self.BEG = BEG
        self.END = END
        self.NUM = NUM
        self.PARTITION = PARTITION
        self.RUN_LOCAL = RUN_LOCAL

        if self.RUN_LOCAL:
            #Find the local copy of XgenESeSS binary
            self.XgenESeSS_PATH = os.path.dirname(sys.modules['cynet'].__file__) \
            + '/bin/XgenESeSS'
            assert os.path.exists(self.XgenESeSS_PATH),"XgenESeSS binary not found"
        else:
            self.XgenESeSS_PATH = XgenESeSS_PATH

    def run_oneXG(self,command):
        '''
        This function is intended to be called by the run method in xgModels. This
        function uses the subprocess module to execute a XgenESeSS command
        and wait for its completion.
        Input-
            command(str): the XgenESeSS command to be executed.
            command_count(int): the command number of this command.
        '''
        print "XgenESeSS Command {} has started".format(command[1])
        args = shlex.split(command[0])
        subprocess.check_output(args, shell=False)
        print "XgenESeSS Command {} has finished".format(command[1])


    def run(self, calls_name='program_calls.txt', workers = 4):
        '''
        Here we run XgenESeSS. This either happens locally or this function
        will output the program calls text file to run on a cluster.
        Input-
            calls_name(str)-Name of file containing program_calls. Only used if
                RUN_LOCAL = 0.
            workers(int)- Number of workers to use in pool. If none, then will
            default to number of cores in the system.
        '''
        INDICES=sum([1 for i in open(self.TS_PATH,"r").readlines() if i.strip()])

        if self.RUN_LOCAL:
            commands = []
            command_count = 0
            for INDEX in np.arange(INDICES):
                xgstr= self.XgenESeSS_PATH +' -f ' + self.TS_PATH\
                 + " -k \"  :" + str(INDEX) +  " \"  -B " + str(self.BEG)\
                 + "  -E " +str(self.END) + ' -n ' +str(self.NUM)+ ' -p '\
                 + " ".join(str(x) for x in self.PARTITION) + ' -S -N '\
                 + self.NAME_PATH + ' -l ' + self.FILEPATH+str(INDEX)\
                 + self.LOG_PATH + ' -m -g 0.01 -G 10000 -v 0 -A .5 -q -w '\
                 + self.FILEPATH+str(INDEX)
                command_count += 1
                commands.append([xgstr,command_count])
            #Parallel of XgenESeSS
            Parallel(n_jobs = workers, verbose = 1, backend = 'threading')\
            (map(delayed(self.run_oneXG), commands))
            print "Processing on XgenESeSS finished."

        else:
            with open(calls_name, 'wr') as file:
                for INDEX in np.arange(INDICES):
                    xgstr= self.XgenESeSS_PATH +' -f ' + self.TS_PATH\
                     + " -k \"  :" + str(INDEX) +  " \"  -B " + str(self.BEG)\
                     + "  -E " +str(self.END) + ' -n ' +str(self.NUM)+ ' -p '\
                     + " ".join(str(x) for x in self.PARTITION) + ' -S -N '\
                     + self.NAME_PATH + ' -l ' + self.FILEPATH+str(INDEX)\
                     + self.LOG_PATH + ' -m -g 0.01 -G 10000 -v 0 -A .5 -q -w '\
                     + self.FILEPATH+str(INDEX)
                    file.write(xgstr + '\n')


class mapped_events:
    '''
    A class which inclues utility for combining the mapped events
    produced by simulateModel's parse_cynet class.
    '''
    def __init__(self, csv_glob):
        self.csv_glob = csv_glob

    def map_dataframes(self, outfile,predictions_col='predictions',variable_col='variable',
                       lat1_col='lat1', lat2_col='lat2',lon1_col='lon1', lon2_col='lon2',
                       day_col='day'):
        '''
        Creates a dictionary of accumulated events.
        Inputs:
            outfile(str): path to write json file to
            csv_glob(str):glob string which will match the csv files containing
                the mapped events dataframes.
            predictions_col(str)- name of the predictions' column.
            lat_col/lon_col(str)- column name of those coordinates
            day_col(str)- name of day column
        Returns:
            a dictionary with the day number as the keys and a list of coordinates
            where events took place as keys.
        '''
        mapped_events = {}

        for file in glob.glob(self.csv_glob):
            df = pd.read_csv(file)

            positive_predictiondf = df[df[predictions_col] == 1]
            if not positive_predictiondf.empty:
                first_row = positive_predictiondf.iloc[0]
                lat1 = first_row[lat1_col]
                lat2 = first_row[lat2_col]
                lon1 = first_row[lon1_col]
                lon2 = first_row[lon2_col]
                variable = first_row[variable_col]
                if not mapped_events.has_key(variable):
                    mapped_events[variable] = {}
                for day in positive_predictiondf[day_col]:
                    if not mapped_events[variable].has_key(day):
                        mapped_events[variable][day] = []
                    mapped_events[variable][day].append((lat1,lat2,lon1,lon2))
        with open(outfile,'w') as file:
            json.dump(mapped_events,file)

        return mapped_events

    def concat_dataframes(self, outfile):
        '''
        A simple function which concatenates all dataframe csvs matching the glob path.
        Inputs-
            outfile(str): path to write dataframe to.
        '''
        print("Concating {} files.".format(len(glob.glob(self.csv_glob))))
        df = pd.concat([pd.read_csv(csv) for csv in glob.glob(self.csv_glob)])
        df.to_csv(outfile,index=False)
