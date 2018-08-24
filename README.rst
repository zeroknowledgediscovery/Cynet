===============
cynet
===============
   
.. image:: http://zed.uchicago.edu/logo/logozed1.png
   :height: 400px
   :scale: 50 %
   :alt: alternate text
   :align: center


.. class:: no-web no-pdf

:Info: See <https://arxiv.org/abs/1406.6651> for theoretical background
:Author: ZeD@UChicago <zed.uchicago.edu>
:Description: Implementation of the Deep Granger net inference algorithm, described in https://arxiv.org/abs/1406.6651, for learning spatio-temporal stochastic processes (*point processes*). **cynet** learns a network of generative local models, without assuming any specific model structure.

.. NOTE:: If issues arise with dependencies in python3, be sure that *tkinter* is installed
  
  .. code-block::

    sudo apt-get install python3-tk

**Usage:**

  .. code-block:: python

    from cynet import cynet
    from cynet.cynet import uNetworkModels as models
    from viscynet import viscynet as vcn


**cynet module includes:**
  * cynet
  * viscynet
  * bokeh_pipe


cynet library classes:
~~~~~~~~~~~~~~~~~~~~~~
* spatioTemporal
* uNetworkModels
* simulateModels

**class spatioTemporal**
  Utilities for spatial-temporal analysis

  **Attributes:**
      * log_store (Pickle): Pickle storage of class data & dataframes
      * log_file (string): path to CSV of legacy dataframe
      * ts_store (string): path to CSV containing most recent ts export
      * DATE (string):
      * EVENT (string): column label for category filter
      * coord1 (string): first coordinate level type; is column name
      * coord2 (string): second coordinate level type; is column name
      * coord3 (string): third coordinate level type; (z coordinate)
      * end_date (datetime.date): upper bound of daterange
      * freq (string): timeseries increments; e.g. D for date
      * columns (list): list of column names to use; requires at least 2 coordinates and event type
      * types (list of strings): event type list of filters
      * value_limits (tuple): boundaries (magnitude of event above threshold)
      * grid (dictionary or list of lists): coordinate dictionary with respective ranges
        and EPS value OR custom list of lists
        of custom grid tiles as [coord1_start, coord1_stop, coord2_start, coord2_stop]
      * grid_type (string): parameter to determine if grid should be built up
        from a coordinate start/stop range ('auto') or be
        built from custom tile coordinates ('custom')
      * threshold (float): significance threshold

  **Methods:**

    .. code-block:: python

        __init__(self, log_store='log.p', log_file=None, ts_store=None, DATE='Date', year=None, month=None, day=None, EVENT='Primary Type', coord1='Latitude', coord2='Longitude', coord3=None, init_date=None, end_date=None, freq=None, columns=None, types=None, value_limits=None, grid=None, threshold=None)


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


**Utility functions:**

    .. code:: python

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


**class uNetworkModels:**

    Utilities for storing and manipulating XPFSA models
    inferred by XGenESeSS


    Attributes:
        jsonFile (string): path to json file containing models

    Methods defined here:

    .. code:: python

      __init__(self, jsonFILE)


      append(self,pydict):
          Utilities for storing and manipulating XPFSA models
          inferred by XGenESeSS

          append models to internal dictionary


      augmentDistance(self)
          Utilities for storing and manipulating XPFSA models
          inferred by XGenESeSS

          Calculates the distance between all models and stores
          them under the
          distance key of each model;

          No I/O


      select(self,var="gamma",n=None,
          reverse=False, store=None,
          high=None,low=None,equal=None,inplace=False):
          Utilities for storing and manipulating XPFSA models
          inferred by XGenESeSS

          Selects the N top models as ranked by var specified value
          (in reverse order if reverse is True)

          Inputs -
              var (string): model parameter to rank by
              n (int): number of models to return
              reverse (boolean): return in ascending order (True)
                  or descending (False) order
              store (string): name of file to store selection json
              high (float): higher cutoff
              equal (float): choose models with selection values
                  equal to the given value
              low (float): lower cutoff
              inplace (bool): update models if true
          Output -
              (dictionary): top n models as ranked by var
                           in ascending/descending order


      setVarname(self):
          Utilities for storing and manipulating XPFSA models
          inferred by XGenESeSS

          Extracts the varname for src and tgt of
          each model and stores under src_var and tgt_var
          keys of each model;

          No I/O


      to_json(outFile)
          Utilities for storing and manipulating XPFSA models
          inferred by XGenESeSS

          Writes out updated models json to file

          Input -
              outFile (string): name of outfile to write json to

          Output -
              (No output but writes out files)


      setDataFrame(self,scatter=None):
          Generate dataframe representation of models

          Input -
              scatter (string) : prefix of filename to plot 3X3 regression
              matrix between delay, distance and coefficiecient of causality
          Output -
              Dataframe with columns
              ['latsrc','lonsrc','lattgt',
               'lontgtt','gamma','delay','distance']

**class simulateModel**
    Utilities for generating statistical analysis after processing models

    **Attributes:**
        * MODEL_PATH(string)- The path to the model being processed.
        * DATA_PATH(string)- Path to the split file.
        * RUNLEN(integer)- Length of the run.
        * READLEN(integer)- Length of split data to read from begining
        * CYNET_PATH - path to cynet binary.
        * FLEXROC_PATH - path to flexroc binary.

  **Methods:**

    .. code-block:: python

        run(self, LOG_PATH=None,
            PARTITION=0.5,
            DATA_TYPE='continuous',
            FLEXWIDTH=1,
            FLEX_TAIL_LEN=100,
            POSITIVE_CLASS_COLUMN=5,
            EVENTCOL=3,
            tpr_thrshold=0.85,
            fpr_threshold=0.15):


        This function is intended to replace the cynrun.sh shell script. This
        function will use the subprocess library to call cynet on a model to process
        it and then run flexroc on it to obtain statistics: auc, tpr, fuc.
        Inputs:
           LOG_PATH(string)- Logfile from cynet run
           PARTITION(string)- Partition to use on split data
           FLEXWIDTH(int)-  Parameter to specify flex in flwxroc
           FLEX_TAIL_LEN(int)- tail length of input file to consider [0: all]
           POSITIVE_CLASS_COLUMN(int)- positive class column
           EVENTCOL(int)- event column
           tpr_thershold(float)- tpr threshold
           fpr_threshold(float)- fpr threshold
        Returns:
        auc, tpr, and fpr statistics from flexroc.



viscynet library classes:
~~~~~~~~~~~~~~~~~~~~~~~~~
  * viscynet

  **viscynet library:**

  visualization library for Network Models produced by uNetworkModels based on
  matplotlib

  Functions:
    .. code:: python

      draw_screen_poly(lats, lons, m, ax, val, cmap, ALPHA=0.6)
          utility function to draw polygons on basemap

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


      getalpha(arr, index, F=0.9)
          utility function to normalize transparency of quiver

          Inputs -
              arr (iterable): list of input values
              index (int): index position from which alpha value should be taken from
              F (float): multiplier
              M (float): minimum alpha value

          Outputs -
              v (float): alpha value


      showGlobalPlot(coords, ts=None, fsize=[14, 14], cmap='jet', m=None, figname='fig', F=2)
          plot global distribution of events within time period specified

          Inputs -
              coords (string): filename with coord list as lat1.lat2.lon1.lon2
              ts (string): time series filename with data in rows, space separated
              fsize (list):
              cmap (string):
              m (mpl.mpl_toolkits.Basemap): mpl instance for plotting
              figname (string): Name of the Plot
              F (int)

          Output -
             num (np.array): data values
             fig (mpl.figure): heatmap of events from fitted data
             ax (axis handler): output axis handler
             cax (colorbar axis handler): output colorbar axis handler


      viz(unet,jsonfile=False,colormap='autumn',res='c',
        drawpoly=False,figname='fig',BGIMAGE=None,BGIMGNAME='BM',IMGRES='high',WIDTH=0.007):

          Utility function to visualize spatio temporal interaction networks

          Inputs -
              unet (string): json filename
              unet (python dict):
              jsonfile (bool): True if unet is string  specifying json filename
              colormap (string): colormap
              res (string): 'c' or 'f'
              drawpoly (bool): if True draws transparent patch showing srcs
              figname  (string): prefix of pdf image file
          Outputs -
              m (Basemap handle)
              fig (figure handle)
              ax (axis handle)
              cax (colorbar handle)


      _scaleforsize(a)
          normalize array for plotting

          Inputs -
              a (ndarray): input array
          Output -
              a (ndarray): output array



bokeh_pipe library:
~~~~~~~~~~~~~~~~~~~
  visualization library for Network Models produced by uNetworkModels based on
  bokeh

  Process overview:
    This code starts from the point
    when the json data files have been obtained.

    To get the neighborhood plot:
        1. run json_to_csv on the batch of json files to get the batch of csv files.
        2. run combine_merc to combine the batch of csv files into one csv file in mercator coordinates.
        3. run neighbor_plot on the combined csv file to get the neighbor hood plot.


    To get the streamline plot:
        1. same as step 1 of neighborhood plot (can be skipped if already done)

        2. run streamheat_combine to combine the batch of csv files into one csv file. *THIS IS IN A FORMAT DIFFERENT FROM THAT OF THE NEIGHBORHOOD PLOT.*

        3. run crime_stream.py on the combined file.

    To get the heatplot:
        1. same as streamline plot.
        2. same as streamline plot.
        3. run heat_map on the combined file.

    We have provided two sample datasets for use. 'crime_filtered_data.csv' can be considered
    the combined file for the neighborhood plot. 'contourmerc.csv' can be considered
    the combined file for the streamline plot and the heatplot.

  Functions:
    .. code:: python

      json_to_csv(FILEPATH, DEST):
        This function takes a group of json data files and transforms
        them into csv files for use. Edit the selection variables as
        you see fit. It is very important that you initialize DEST to a folder,
        as it generates many csv files. WARNING: Run this function in
        python2. The rest of the code should use python3.
        THIS TAKES QUITE A BIT OF TIME.

        Inputs -
            FILEPATH (string): the filepath to the json files. Example: 'jsons/'
            DEST (string): the place for the csv files to be stored. Example: 'csvs/'


      combine_merc(DIR, filename, N = 20):
        This function combines the csv's into a single file. At the same time,
        this function will convert the format of the coordinates from longitude
        and latitude which is necessary to make our neighborhood plot. Our tileset
        accepts mercator coordinates. This generates one combined csv in the
        current directory. USE PYTHON 3.

        Inputs:
            DIR (string): The location(filepath) of the csvs to be combined. Example 'csvs/'
            filename (string): the desired name for the combined csv file. Example: 'combined.csv'
            N (int): the max number of sources selected for in json_to_csv:
                M.select(var='delay',high=20,reverse=False,inplace=True).
                high argument is N.


      neighbor_plot(filepath= 'crime_filtered_data.csv'):
        This is the first implementation of our Bokeh plot. The function takes the filepath
        of the data and opens the bokeh plot in a browser. Google Chrome seems to be the
        best browser for bokeh plots. The datafile must be a csv file in the correct format.
        See the file 'crime_filtered_data.csv' for an example. Each row represents a point,
        all the lines(sources) connected to it and the gammas and delays associated with
        the lines. The current implementation results in the bokeh plot, and a linked
        table of the data. IMPORTANT: Points are in MERCATOR Coordinates. This is because
        the current tileset for the map is in mercator coordinates.
        Example file is 'crime_filtered_data.csv'

        Inputs -
          filepath (string): input data file


      streamheat_combine(DIR, filename):
          We need to once again combine the csvs, into a format appropriate for the streamplots.
          This file will do that. This function will produce two files. File 1 will
          be in longitude and latitude. File 2 will be in mercator coordinates.
          We will be primiarily working with file 2

          Inputs -
              DIR (string): The filepath to the csvs. Ex: 'csvs/'
              filename (string): The filename for the combined csv file. 'contourmerc.csv'


      crime_stream(datafile='contourmerc.csv',density=4, npoints=10, output_name='streamplot.html', method = 'cubic'):
          This function takes a csv datafile of crime vectors, reads it into
          a pandas dataframe and plots the streamplot using Delanuay
          interpolation. Function will open the plot in a new browser. Use chrome.
          Inputs:
              datafile: name of the csv file. Example file is 'contourmerc.csv'
              density: desired line density of the plot. Ex: 4.
              npoints: The dimensions used for the streamplot. The grid will
                  have npoints**2 number of grids. It is not advised to have npoints > 200.
                  Reccommended: npoints =10.
              ouput_name: name to save plot to.
              method: method for interpolation. 'cubic','linear', or 'nearest'


      heat_map(datafile='contourmerc.csv', npoints=300, output_name='heatmap.html', method = 'linear'):
          Makes a heatmap from the same datafile that cimre_stream uses.
          datafile: name of the datafile. Example file is 'contourmerc.csv'.
          npoints: dimension for plot. number of squares = npoints**2.
              Recommended: 100-300

          Inputs -
            output_name (string): output file name for the plot.
            method (string): method for interpolation. 'cubic','linear', or 'nearest'


VERSION 1.0.50
