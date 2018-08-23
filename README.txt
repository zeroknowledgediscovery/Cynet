Help on module cynet:

NAME
    cynet

FILE
    /home/ishanu/Dropbox/ZED/Research/Cynet/cynet/cynet.py

DESCRIPTION
    Spatio temporal analysis for inferrence of statistical causality
    @author zed.uchicago.edu

CLASSES
    simulateModel
    spatioTemporal
    uNetworkModels
    
    class simulateModel
     |  Use the subprocess library to call cynet on a model to process
     |  it and then run flexroc on it to obtain statistics: auc, tpr, fuc.
     |  Input -
     |      MODEL_PATH(string)- The path to the model being processed.
     |      DATA_PATH(string)- Path to the split file.
     |      RUNLEN(integer)- Length of the run.
     |      READLEN(integer)- Length of split data to read from begining
     |      CYNET_PATH - path to cynet binary.
     |      FLEXROC_PATH - path to flexroc binary.
     |  
     |  Methods defined here:
     |  
     |  __init__(self, MODEL_PATH, DATA_PATH, RUNLEN, CYNET_PATH, FLEXROC_PATH, READLEN=None)
     |  
     |  run(self, LOG_PATH=None, PARTITION=0.5, DATA_TYPE='continuous', FLEXWIDTH=1, FLEX_TAIL_LEN=100, POSITIVE_CLASS_COLUMN=5, EVENTCOL=3, tpr_thrshold=0.85, fpr_threshold=0.15)
     |      This function is intended to replace the cynrun.sh shell script. This
     |      function will use the subprocess library to call cynet on a model to process
     |      it and then run flexroc on it to obtain statistics: auc, tpr, fuc.
     |      Input -
     |         LOG_PATH (string)- Logfile from cynet run
     |         PARTITION (string)- Partition to use on split data
     |         FLEXWIDTH (int)-  Parameter to specify flex in flwxroc
     |         FLEX_TAIL_LEN (int)- tail length of input file to consider [0: all]
     |         POSITIVE_CLASS_COLUMN (int)- positive class column
     |         EVENTCOL (int)- event column
     |         tpr_thershold (float)- minimum tpr threshold
     |         fpr_threshold (float)- maximum fpr threshold
     |      
     |      Output -
     |          auc (float)- Area under the curve
     |          tpr (float)- True positive rate at specified maximum false positive rate
     |          fpr (float)- False positive rate at specified minimum true positive rate
    
    class spatioTemporal
     |  Utilities for spatio temporal analysis
     |  @author zed.uchicago.edu
     |  
     |  Attributes:
     |      log_store (Pickle): Pickle storage of class data & dataframes
     |      log_file (string): path to CSV of legacy dataframe
     |      ts_store (string): path to CSV containing most recent ts export
     |      DATE (string):
     |      EVENT (string): column label for category filter
     |      coord1 (string): first coordinate level type; is column name
     |      coord2 (string): second coordinate level type; is column name
     |      coord3 (string): third coordinate level type;
     |                       (z coordinate)
     |      end_date (datetime.date): upper bound of daterange
     |      freq (string): timeseries increments; e.g. D for date
     |      columns (list): list of column names to use;
     |                      required at least 2 coordinates and event type
     |      types (list of strings): event type list of filters
     |      value_limits (tuple): boundaries (magnitude of event;
     |                            above threshold)
     |      grid (dictionary or list of lists): coordinate dictionary with
     |              respective ranges and EPS value OR custom list of lists
     |              of custom grid tiles as [coord1_start, coord1_stop,
     |              coord2_start, coord2_stop]
     |      grid_type (string): parameter to determine if grid should be built up
     |                          from a coordinate start/stop range ('auto') or be
     |                          built from custom tile coordinates ('custom')
     |      threshold (float): significance threshold
     |  
     |  Methods defined here:
     |  
     |  __init__(self, log_store='log.p', log_file=None, ts_store=None, DATE='Date', year=None, month=None, day=None, EVENT='Primary Type', coord1='Latitude', coord2='Longitude', coord3=None, init_date=None, end_date=None, freq=None, columns=None, types=None, value_limits=None, grid=None, threshold=None)
     |  
     |  fit(self, grid=None, INIT=None, END=None, THRESHOLD=None, csvPREF='TS', auto_adjust_time=False, incr=6, max_incr=24, poly_tile=False)
     |      Utilities for spatio temporal analysis
     |      @author zed.uchicago.edu
     |      
     |      Fit dataproc with specified grid parameters and
     |      create timeseries for
     |      date boundaries specified by INIT, THRESHOLD,
     |      and END or by the input list of custom coordinate boundaries
     |      which do NOT have to match the arguments first input to the dataproc
     |      
     |      Inputs:
     |          grid (dictionary or list of lists): coordinate dictionary with
     |              respective ranges and EPS value OR custom list of lists
     |              of custom grid tiles as [coord1_start, coord1_stop,
     |              coord2_start, coord2_stop]
     |          INIT (datetime.date): starting timeseries date
     |          END (datetime.date): ending timeseries date
     |          THRESHOLD (float): significance threshold
     |          auto_adjust_time (boolean): if True, within increments specified (6H default),
     |              determine optimal temporal frequency for timeseries data
     |          incr (int): frequency increment
     |          max_incr (int): user-specified maximum increment
     |          poly_tile (boolean): whether or not tiles define polygons
     |      
     |      Outputs:
     |          (No output) grid pd.Dataframe written out as CSV file
     |                  to path specified
     |  
     |  getGrid(self)
     |      Returns the tile coordinates of the working as a list of lists
     |      
     |      Input -
     |          (No inputs)
     |      Output -
     |          TILE (list of lists): the grid tiles
     |  
     |  getTS(self, _types=None, tile=None, freq=None, poly_tile=False)
     |      Utilities for spatio temporal analysis
     |      @author zed.uchicago.edu
     |      
     |      Given location tile boundaries and type category filter, creates the
     |      corresponding timeseries as a pandas DataFrame
     |      (Note: can reassign type filter, does not have to be the same one
     |      as the one initialized to the dataproc)
     |      
     |      Inputs:
     |          _types (list of strings): list of category filters
     |          tile (list of floats): location boundaries for tile OR
     |          if poly_tile is TRUE, tile is a list of tuples defining the polygon
     |          freq (string): intervals of time between timeseries columns
     |          poly_tile (boolean): whether or not input for tiles defines
     |          a polygon filter
     |      
     |      Outputs:
     |          pd.Dataframe of timeseries data to corresponding grid tile
     |          pd.DF index is stringified LAT/LON boundaries
     |          with the type filter included
     |  
     |  get_opt_freq(self, df, TS_NAME, incr=6, max_incr=24)
     |      Utilities for spatio temporal analysis
     |      @author zed.uchicago.edu
     |      
     |      Returns the optimal frequency for timeseries based on highest non-zero
     |      to zero timeseries event count
     |      
     |      Input -
     |          df (pd.DataFrame): filtered subset of dataset corresponding to
     |          random tile from get_rand_tile
     |          incr (int): frequency increment
     |          max_incr (int): user-specified maximum increment
     |          TS_NAME
     |      
     |      Output -
     |          (string) to pass to pd.date_range(freq=) argument
     |  
     |  get_rand_tile(self, tiles=None, LAT=None, LON=None, EPS=None, _types=None, poly_tile=False)
     |      Utilities for spatio temporal analysis
     |      @author zed.uchicago.edu
     |      
     |      Picks random tile from options fed into timeseries method which maps to a
     |      non-empty subset within the larger dataset
     |      
     |      Inputs -
     |          LAT (float or list of floats): singular coordinate float or list of
     |                                         coordinate start floats
     |          LON (float or list of floats): singular coordinate float or list of
     |                                         coordinate start floats
     |          EPS (float): coordinate increment ESP
     |          _types (list): event type filter; accepted event type list
     |          tiles (list of lists): list of tiles to build where tile can be
     |          a (list of floats i.e. [lat1 lat2 lon1 lon2]) or tuples (i.e. [(x1,y1),(x2,y2)])
     |          defining polygons
     |          poly_tile (boolean): whether input for tile specifies a polygon
     |      
     |      Outputs -
     |          tile dataframe (pd.DataFrame)
     |  
     |  pull(self, domain='data.cityofchicago.org', dataset_id='crimes', token=None, store=True, out_fname='pull_df.p', pull_all=False)
     |      Utilities for spatio temporal analysis
     |      @author zed.uchicago.edu
     |      
     |      Pulls new entries from datasource
     |      NOTE: should make flexible but for now use city of Chicago data
     |      
     |      Input -
     |          domain (string): Socrata database domain hosting data
     |          dataset_id (string): dataset ID to pull
     |          token (string): Socrata token for increased pull capacity;
     |              Note: Requires Socrata account
     |          store (boolean): whether or not to write out new dataset
     |          pull_all (boolean): pull complete dataset
     |          instead of just updating
     |      
     |      Output -
     |          None (writes out files if store is True and modifies inplace)
     |  
     |  timeseries(self, LAT=None, LON=None, EPS=None, _types=None, CSVfile='TS.csv', THRESHOLD=None, tiles=None, auto_adjust_time=False, incr=6, max_incr=24, poly_tile=False)
     |      Utilities for spatio temporal analysis
     |      @author zed.uchicago.edu
     |      
     |      Creates DataFrame of location tiles and their
     |      respective timeseries from
     |      input datasource with significance threshold THRESHOLD
     |      latitude, longitude coordinate boundaries given by LAT, LON and EPS
     |      or the custom boundaries given by tiles
     |      calls on getTS for individual tile then concats them together
     |      
     |      Input:
     |          LAT (float or list of floats): singular coordinate float or list of
     |                                         coordinate start floats
     |          LON (float or list of floats): singular coordinate float or list of
     |                                         coordinate start floats
     |          EPS (float): coordinate increment ESP
     |          _types (list): event type filter; accepted event type list
     |          CSVfile (string): path to output file
     |          tiles (list of lists): list of tiles to build where tile can be
     |          a (list of floats i.e. [lat1 lat2 lon1 lon2]) or tuples (i.e. [(x1,y1),(x2,y2)])
     |          defining polygons
     |          auto_adjust_time (boolean): if True, within increments specified (6H default),
     |              determine optimal temporal frequency for timeseries data
     |          incr (int): frequency increment
     |          max_incr (int): user-specified maximum increment
     |          poly_tile (boolean): whether or tiles define polygons
     |      
     |      Output:
     |          No Output grid pd.Dataframe written out as CSV file to path specified
    
    class uNetworkModels
     |  Utilities for storing and manipulating XPFSA models
     |  inferred by XGenESeSS
     |  @author zed.uchicago.edu
     |  
     |  Attributes:
     |      jsonFile (string): path to json file containing models
     |  
     |  Methods defined here:
     |  
     |  __init__(self, jsonFILE)
     |  
     |  append(self, pydict)
     |      append models
     |      @author zed.uchicago.edu
     |  
     |  augmentDistance(self)
     |      Utilities for storing and manipulating XPFSA models
     |      inferred by XGenESeSS
     |      @author zed.uchicago.edu
     |      
     |      Calculates the distance between all models and stores
     |      them under the
     |      distance key of each model;
     |      
     |      No I/O
     |  
     |  iNet(self, init=0)
     |      Utilities for storing and manipulating XPFSA models
     |      inferred by XGenESeSS
     |      @author zed.uchicago.edu
     |      
     |      Calculates the distance between all models and stores
     |      them under the
     |      distance key of each model;
     |      
     |      No I/O
     |  
     |  select(self, var='gamma', n=None, reverse=False, store=None, high=None, low=None, equal=None, inplace=False)
     |      Utilities for storing and manipulating XPFSA models
     |      inferred by XGenESeSS
     |      @author zed.uchicago.edu
     |      
     |      Selects the N top models as ranked by var specified value
     |      (in reverse order if reverse is True)
     |      
     |      Inputs -
     |          var (string): model parameter to rank by
     |          n (int): number of models to return
     |          reverse (boolean): return in ascending order (True)
     |              or descending (False) order
     |          store (string): name of file to store selection json
     |          high (float): higher cutoff
     |          low (float): lower cutoff
     |          inplace (bool): update models if true
     |      
     |      Output -
     |          (dictionary): top n models as ranked by var
     |                       in ascending/descending order
     |  
     |  setDataFrame(self, scatter=None)
     |      Generate dataframe representation of models
     |      @author zed.uchicago.edu
     |      
     |      Input -
     |          scatter (string) : prefix of filename to plot 3X3 regression
     |          matrix between delay, distance and coefficiecient of causality
     |      
     |      Output -
     |          Pandas.DataFrame with columns
     |          ['latsrc','lonsrc','lattgt',
     |           'lontgtt','gamma','delay','distance']
     |  
     |  setVarname(self)
     |      Utilities for storing and manipulating XPFSA models
     |      inferred by XGenESeSS
     |      @author zed.uchicago.edu
     |      
     |      Extracts the varname for src and tgt of
     |      each model and stores under src_var and tgt_var
     |      keys of each model;
     |      
     |      No I/O
     |  
     |  to_json(self, outFile)
     |      Utilities for storing and manipulating XPFSA models
     |      inferred by XGenESeSS
     |      @author zed.uchicago.edu
     |      
     |      Writes out updated models json to file
     |      Input -
     |          outFile (string): name of outfile to write json to
     |      
     |      Output -
     |          No output
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  df
     |  
     |  models

FUNCTIONS
    readTS(TSfile, csvNAME='TS1', BEG=None, END=None)
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
    
    splitTS(TSfile, dirname='./', prefix='@', BEG=None, END=None, VARNAME='')
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
    
    stringify(List)
        Utility function
        @author zed.uchicago.edu
        
        Converts list into string separated by dashes
                 or empty string if input list
                 is not list or is empty
        
        Input:
            List (list): input list to be converted
        
        Output:
            (string)
    
    to_json(pydict, outFile)
        Writes dictionary json to file
        @author zed.uchicago.edu
        
        Input -
            pydict (dict): ditionary to store
            outFile (string): name of outfile to write json to
        
        Output -
            No output

DATA
    PRECISION = 5
    __DEBUG__ = False


Help on module viscynet:

NAME
    viscynet

FILE
    /home/ishanu/Dropbox/ZED/Research/Cynet/viscynet/viscynet.py

DESCRIPTION
    Visualization library for cynet
    @author zed.uchicago.edu

FUNCTIONS
    draw_screen_poly(lats, lons, m, ax, val, cmap, ALPHA=0.6)
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
    
    getalpha(arr, index, F=0.9, M=0)
        utility function to normalize transparency of quiver
        @author zed.uchicago.edu
        
        Inputs -
            arr (iterable): list of input values
            index (int): index position from which alpha value should be taken from
            F (float): multiplier
            M (float): minimum alpha value
        
        Outputs -
            v (float): alpha value
    
    viz(unet, jsonfile=False, colormap='autumn', res='c', drawpoly=False, figname='fig', BGIMAGE=None, BGIMGNAME='BM', IMGRES='high', WIDTH=0.007)
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


