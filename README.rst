# Spatio temporal analysis for inferrence of statistical causality

@author zed.uchicago.edu

cynet library classes: spatioTemporal, uNetworkModels

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
     |          required at least 2 coordinates and event type
     |      types (list of strings): event type list of filters
     |      value_limits (tuple): boundaries (magnitude of event;
     |                            above threshold)
     |      grid (dictionary or list of lists): coordinate dictionary with
     |            respective ranges and EPS value OR custom list of lists
     |            of custom grid tiles as [coord1_start, coord1_stop,
     |            coord2_start, coord2_stop]
     |      grid_type (string): parameter to determine if grid should be built up
     |                         from a coordinate start/stop range ('auto') or be
     |                         built from custom tile coordinates ('custom')
     |      threshold (float): significance threshold
     |
     |  Methods defined here:
     |
     |  __init__(self, log_store='log.p', log_file=None, ts_store=None, DATE='Date', year=None, month=None, day=None, EVENT='Primary Type', coord1='Latitude', coord2='Longitude', coord3=None, init_date=None, end_date=None, freq=None, columns=None, types=None, value_limits=None, grid=None, threshold=None)
     |
     |
     |  fit(self, grid=None, INIT=None, END=None, THRESHOLD=None, csvPREF='TS')
     |      Utilities for spatio temporal analysis
     |      @author zed.uchicago.edu
     |
     |      Fit dataproc with specified grid parameters and
     |      create timeseries for
     |      date boundaries specified by INIT, THRESHOLD,
     |      and END or input list of custom coordinate boundaries which do NOT have
     |      to match the arguments first input to the dataproc
     |
     |      Inputs:
     |          grid (dictionary or list of lists): coordinate dictionary with
     |              respective ranges and EPS value OR custom list of lists
     |              of custom grid tiles as [coord1_start, coord1_stop,
     |              coord2_start, coord2_stop]
     |          INIT (datetime.date): starting timeseries date
     |          END (datetime.date): ending timeseries date
     |          THRESHOLD (float): significance threshold
     |
     |      Outputs:
     |          (None)
     |
     |
     |  getTS(self, _types=None, tile=None)
     |      Utilities for spatio temporal analysis
     |      @author zed.uchicago.edu
     |
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
     |          tile (list of floats): location boundaries for tile
     |
     |      Outputs:
     |          pd.Dataframe of timeseries data to corresponding grid tile
     |          pd.DF index is stringified LAT/LON boundaries
     |          with the type filter  included
     |
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
     |
     |  timeseries(self, LAT=None, LON=None, EPS=None, _types=None, CSVfile='TS.csv', THRESHOLD=None,
     |             tiles=None)
     |      Utilities for spatio temporal analysis
     |      @author zed.uchicago.edu
     |
     |      Creates DataFrame of location tiles and their
     |      respective timeseries from
     |      input datasource with
     |      significance threshold THRESHOLD
     |      latitude, longitude coordinate boundaries given by LAT, LON and EPS
     |      or the custom boundaries given by tiles
     |      calls on getTS for individual tile then concats them together
     |
     |      Input:
     |          LAT (float):
     |          LON (float):
     |          EPS (float): coordinate increment ESP
     |          _types (list): event type filter; accepted event type list
     |          CSVfile (string): path to output file
     |
     |      Output:
     |          (None): grid pd.Dataframe written out as CSV file
     |                  to path specified



    Utility functions:
    |  splitTS(TSfile, csvNAME='TS1', dirname='./', prefix='@', BEG=None, END=None)
    |     Utilities for spatio temporal analysis
    |     @author zed.uchicago.edu
    |
    |     Writes out each row of the pd.DataFrame as a separate CSVfile
    |     For XgenESeSS binary
    |
    |     Inputs -
    |         TSfile (pd.DataFrame): DataFrame to write out
    |         csvNAME (string): output filename
    |         dirname (string): directory for output file
    |         prefix (string): prefix for files
    |         BEG (datetime): start date
    |         END (datetime): end date
    |
    |     Outputs -
    |         (No output)
    |
    |
    |  stringify(List)
    |     Utility function
    |     @author zed.uchicago.edu
    |
    |     Converts list into string separated by dashes
    |     or empty string if input list
    |          is not list or is empty
    |
    |     Input:
    |         List (list): input list to be converted
    |
    |     Output:
    |         (string)
    |
    |
    |  to_json(pydict, outFile)
    |     Writes dictionary json to file
    |     @author zed.uchicago.edu
    |
    |     Input -
    |         pydict (dict): ditionary to store
    |         outFile (string): name of outfile to write json to
    |
    |     Output -
    |         (No output but writes out files)
    |
    |
    |  readTS(TSfile,csvNAME='TS1',BEG=None,END=None):
    |      Utilities for spatio temporal analysis
    |      @author zed.uchicago.edu
    |
    |      Reads in output TS logfile into pd.DF and outputs necessary
    |      CSV files in XgenESeSS-friendly format
    |
    |      Input -
    |          TSfile (string): filename input TS to read
    |          csvNAME (string)
    |          BEG (string): start datetime
    |          END (string): end datetime
    |
    |      Output -
    |          dfts (pandas.DataFrame)



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
     |
     |  append(self,pydict):
     |      Utilities for storing and manipulating XPFSA models
     |      inferred by XGenESeSS
     |      @author zed.uchicago.edu
     |
     |      append models to internal dictionary
     |
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
     |
     |  select(self, var='gamma', n=None, reverse=False, store=None)
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
     |
     |      Returns -
     |          (dictionary): top n models as ranked by var
     |                       in ascending/descending order
     |
     |
     |  to_json(outFile)
     |      Utilities for storing and manipulating XPFSA models
     |      inferred by XGenESeSS
     |      @author zed.uchicago.edu
     |
     |      Writes out updated models json to file
     |
     |      Input -
     |          outFile (string): name of outfile to write json to
     |
     |      Output -
     |          (No output but writes out files)
     |





viscynet library:
    Functions
     |
     |  draw\_screen\_poly(lats, lons, m, ax, val, cmap, ALPHA=0.6)
     |      utility function to draw polygons on basemap
     |
     |      Inputs -
     |          lats (list of floats): mpl_toolkits.basemap lat parameters
     |          lons (list of floats): mpl_toolkits.basemap lon parameters
     |          m (mpl.mpl_toolkits.Basemap): mpl instance for plotting
     |          ax (axis parent handle)
     |          cax (colorbar parent handle)
     |          val (Matplotlib color)
     |          cmap (string): colormap cmap parameter
     |          ALPHA (float): alpha value to use for plot
     |
     |      Outputs -
     |          (No outputs - modifies objects in place)
     |
     |
     |  getalpha(arr, index, F=0.9)
     |      utility function to normalize transparency of quiver
     |
     |      Inputs -
     |          arr (iterable): list of input values
     |          index (int): index position from which alpha value should be taken from
     |          F (float): multiplier
     |          M (float): minimum alpha value
     |
     |      Outputs -
     |          v (float): alpha value
     |
     |
     |  showGlobalPlot(coords, ts=None, fsize=[14, 14], cmap='jet', m=None, figname='fig', F=2)
     |      plot global distribution of events within time period specified
     |
     |      Inputs -
     |          coords (string): filename with coord list as lat1#lat2#lon1#lon2
     |          ts (string): time series filename with data in rows, space separated
     |          fsize (list):
     |          cmap (string):
     |          m (mpl.mpl_toolkits.Basemap): mpl instance for plotting
     |          figname (string): Name of the Plot
     |          F (int)
     |
     |      Output -
     |         num (np.array): data values
     |         fig (mpl.figure): heatmap of events from fitted data
     |         ax (axis handler): output axis handler
     |         cax (colorbar axis handler): output colorbar axis handler
     |
     |
     |  viz(unet, jsonfile=False, colormap='autumn', res='c', drawpoly=False, figname='fig')
     |      utility function to visualize spatio temporal
     |      interaction networks
     |
     |      Inputs -
     |          unet (string): json filename
     |          unet (python dict):
     |          jsonfile (bool): True if unet is string  specifying json filename
     |          colormap (string): colormap
     |          res (string): 'c' or 'f'
     |          drawpoly (bool): if True draws transparent patch showing srcs
     |          figname  (string): prefix of pdf image file
     |      Outputs -
     |          m (Basemap handle)
     |          fig (figure handle)
     |          ax (axis handle)
     |          cax (colorbar handle)
     |
     |
     |  _scaleforsize(a)
     |      normalize array for plotting
     |
     |      Inputs -
     |          a (ndarray): input array
     |      Output -
     |          a (ndarray): output array
     |
     |
     |  bokeh_plot(filepath)
     |      This function takes filepath given by data and produces bokeh plot in
     |      browser (Google Chrome recommended). Each row represents a point,
     |      all the lines(sources) connected to it and the gammas and delays associated with
     |      the lines. The current implementation results in the bokeh plot, and a linked
     |      table of the data. IMPORTANT: Points are in MERCATOR Coordinates. This is because
     |      the current tileset for the map is in mercator coordinates.
     |
     |      Inputs -
     |          filepath (string) - path to file
     |
     |      Outputs -
     |          (No outputs, will open up external browser window)


DATA **DEBUG** = False **version** = '1.0.8'

VERSION 1.0.8
