cynet
=====

 **cynet is a spatial-temporal analysis library for inferrence of statistical causality**

**NOTE:** if issues arise with dependencies in python3, be sure that tkinter is installed  
if not, please run:

``` {.sourceCode .}
sudo apt-get install python3-tk
```

**Usage:**

 ``` {.sourceCode .python}
 from cynet import cynet
 from cynet.cynet import uNetworkModels as models
 from viscynet import viscynet as vcn
 ```

cynet module includes:  
-   cynet
-   viscynet
-   bokeh\_pipe

cynet library classes:
----------------------

-   spatioTemporal
-   uNetworkModels
-   simulateModels

**class spatioTemporal**  
Utilities for spatial-temporal analysis

**Attributes:**  
-   log\_store (Pickle): Pickle storage of class data & dataframes
-   log\_file (string): path to CSV of legacy dataframe
-   ts\_store (string): path to CSV containing most recent ts export
-   DATE (string):
-   EVENT (string): column label for category filter
-   coord1 (string): first coordinate level type; is column name
-   coord2 (string): second coordinate level type; is column name
-   coord3 (string): third coordinate level type; (z coordinate)
-   end\_date (datetime.date): upper bound of daterange
-   freq (string): timeseries increments; e.g. D for date
-   columns (list): list of column names to use; requires at least 2 coordinates and event type
-   types (list of strings): event type list of filters
-   value\_limits (tuple): boundaries (magnitude of event above threshold)
-   grid (dictionary or list of lists): coordinate dictionary with respective ranges and EPS value OR custom list of lists of custom grid tiles as [coord1\_start, coord1\_stop, coord2\_start, coord2\_stop]
-   grid\_type (string): parameter to determine if grid should be built up from a coordinate start/stop range ('auto') or be built from custom tile coordinates ('custom')
-   threshold (float): significance threshold

**Methods:**

``` 
  __init__(self, log_store='log.p', log_file=None, ts_store=None, DATE='Date', year=None, month=None, day=None, EVENT='Primary Type', coord1='Latitude', coord2='Longitude', coord3=None, init_date=None, end_date=None, freq=None, columns=None, types=None, value_limits=None, grid=None, threshold=None)}
 fit(self, grid=None, INIT=None, END=None, THRESHOLD=None, csvPREF='TS',
     auto_adjust_time=False,incr=6,max_incr=24):

     Fit dataproc with specified grid parameters and
     create timeseries for
     date boundaries specified by INIT, THRESHOLD,
     and END or input list of custom coordinate boundaries which do NOT have
     to match the arguments first input to the dataproc

     Inputs -
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

     Outputs -
         (No output) grid pd.Dataframe written out as CSV file
                 to path specified


 getTS(self, _types=None, tile=None, freq=None)
     Given location tile boundaries and type category filter, creates the
     corresponding timeseries as a pandas DataFrame
     (Note: can reassign type filter, does not have to be the same one
     as the one initialized to the dataproc)

     Inputs:
         _types (list of strings): list of category filters
         tile (list of floats): location boundaries for tile
         freq (string): intervals of time between timeseries columns

     Outputs:
         pd.Dataframe of timeseries data to corresponding grid tile
         pd.DF index is stringified LAT/LON boundaries
         with the type filter  included


 get_rand_tile(tiles=None,LAT=None,LON=None,EPS=None,_types=None)
     Picks random tile from options fed into timeseries method which maps to a
     non-empty subset within the larger dataset

     Inputs -
         LAT (float or list of floats): singular coordinate float or list of
                                        coordinate start floats
         LON (float or list of floats): singular coordinate float or list of
                                        coordinate start floats
         EPS (float): coordinate increment ESP
         _types (list): event type filter; accepted event type list
         tiles (list of lists): list of tiles to build (list of [lat1 lat2 lon1 lon2])

     Outputs -
         tile dataframe (pd.DataFrame)


 get_opt_freq(df,incr=6,max_incr=24):
     Returns the optimal frequency for timeseries based on highest non-zero
     to zero timeseries event count

     Input -
         df (pd.DataFrame): filtered subset of dataset corresponding to
         random tile from get_rand_tile
         incr (int): frequency increment
         max_incr (int): user-specified maximum increment

     Output -
         (string) to pass to pd.date_range(freq=) argument


 getGrid(self):
     Returns the tile coordinates of the working as a list of lists

     Input -
         (No inputs)
     Output -
         TILE (list of lists): the grid tiles


 pull(self, domain='data.cityofchicago.org', dataset_id='crimes', token=None, store=True, out_fname='pull_df.p', pull_all=False)
     Pulls new entries from datasource

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


 timeseries(self, LAT=None, LON=None, EPS=None,_types=None,CSVfile='TS.csv',THRESHOLD=None,tiles=None,incr=6,max_incr=24):
     Creates DataFrame of location tiles and their
     respective timeseries from input datasource with
     significance threshold THRESHOLD
     latitude, longitude coordinate boundaries given by LAT, LON and EPS
     or the custom boundaries given by tiles
     calls on getTS for individual tile then concats them together

     Input -
         LAT (float or list of floats): singular coordinate float or list of
                                        coordinate start floats
         LON (float or list of floats): singular coordinate float or list of
                                        coordinate start floats
         EPS (float): coordinate increment ESP
         _types (list): event type filter; accepted event type list
         CSVfile (string): path to output file
         tiles (list of lists): list of tiles to build (list of [lat1 lat2 lon1 lon2])
         auto_adjust_time (boolean): if True, within increments specified (6H default),
             determine optimal temporal frequency for timeseries data
         incr (int): frequency increment
         max_incr (int): user-specified maximum increment

     Output:
         No Output grid pd.Dataframe written out as CSV file to path specified
```

**Utility functions:**

``` 
 splitTS(TSfile, csvNAME='TS1', dirname='./', prefix='@', BEG=None, END=None, VARNAME='')
   Utilities for spatio temporal analysis

   Writes out each row of the pd.DataFrame as a separate CSVfile
   For XgenESeSS binary

   Inputs -
       TSfile (pd.DataFrame): DataFrame to write out
       csvNAME (string): output filename
       dirname (string): directory for output file
       prefix (string): prefix for files
       VARNAME (string): string to append to file names
       BEG (datetime): start date
       END (datetime): end date

   Outputs -
       (No output)


 stringify(List)
   Utility function

   Converts list into string separated by dashes
   or empty string if input list
        is not list or is empty

   Input:
       List (list): input list to be converted

   Output:
       (string)


 to_json(pydict, outFile)
   Writes dictionary json to file

   Input -
       pydict (dict): ditionary to store
       outFile (string): name of outfile to write json to

   Output -
       (No output but writes out files)


 readTS(TSfile,csvNAME='TS1',BEG=None,END=None):
    Utilities for spatio temporal analysis

    Reads in output TS logfile into pd.DF and outputs necessary
    CSV files in XgenESeSS-friendly format

    Input -
        TSfile (string or list of strings): filename of input TS to read
            or list of filenames to read in and concatenate into one TS
        csvNAME (string)
        BEG (string): start datetime
        END (string): end datetime

    Output -
        dfts (pandas.DataFrame)
```
