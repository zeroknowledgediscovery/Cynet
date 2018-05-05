#!/usr/bin/python
from __future__ import division

from spin import viz
import pandas as pd
import numpy as np
from spin import uNetworkModels as models

import os

import warnings

import csv
from pyproj import Proj, transform

from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, LogColorMapper, LinearColorMapper, Segment, Circle
from bokeh.models.tools import HoverTool, BoxSelectTool, TapTool
from bokeh.models.widgets import DataTable, TableColumn, NumberFormatter
from bokeh.io import show, output_file
from bokeh.palettes import RdYlBu11 as palette
from bokeh.tile_providers import CARTODBPOSITRON
from bokeh.layouts import widgetbox, gridplot
from bokeh.transform import linear_cmap, log_cmap
from bokeh.util.hex import hexbin

from scipy.interpolate import griddata
from scipy.spatial import Delaunay

from bokeh_utilities import in_hull, streamlines



'''
This file is a first implementation of the bokeh pipeline. It requires the bokeh_utilities.py
file, which is code borrowed from online. We will give an example
of how to use this file to get the plots. This code starts from the point
when the json data files have been obtained.

To get the neighborhood plot:
    1. run json_to_csv on the batch of json files to get the batch of csv files.
    2. run combine_merc to combine the batch of csv files into one csv file
        in mercator coordinates.
    3. run neighbor_plot on the combined csv file to get the neighbor hood plot.

To get the streamline plot:
    1. same as step 1 of neighborhood plot (can be skipped if already done)
    2. run streamheat_combine to combine the batch of csv files into one csv
    file. THIS IS IN A FORMAT DIFFERENT FROM THAT OF THE NEIGHBORHOOD PLOT.
    3. run crime_stream.py on the combined file.

To get the heatplot:
    1. same as streamline plot.
    2. same as streamline plot.
    3. run heat_map on the combined file.

We have provided two sample datasets for use. 'crime_filtered_data.csv' can be considered
the combined file for the neighborhood plot. 'contourmerc.csv' can be considered
the combined file for the streamline plot and the heatplot.

'''


def json_to_csv(FILEPATH, DEST):
    '''
    This function takes a group of json data files and transforms
    them into csv files for use. Edit the selection variables as
    you see fit. It is very important that you set DEST to a folder,
    as it generates many csv files. WARNING: Run this function in
    python2. The rest of the code should use python3.
    THIS TAKES QUITE A BIT OF TIME.

    Inputs:
        FILEPATH: the filepath to the json files. Example: 'jsons/'
        DEST: the place for the csv files to be stored. Example: 'csvs/'
    '''
    warnings.filterwarnings("ignore")
    maxdistance=3.5

    file_count=1
    for filename in os.listdir(FILEPATH):

        FILE=filename

        M=models(FILEPATH + FILE)
        M.augmentDistance()


        M.select(var='distance',high=maxdistance,reverse=False,inplace=True)
        M.select(var='delay',high=20,reverse=False,inplace=True)
        M.select(var='gamma',n=20,store='temp_short.json',reverse=True)


        M0=models('temp_short.json')
        #M1=models('../data/Q_75.json')
        #M1.augmentDistance()
        #M0.append(M1.select(var='gamma',n=5,store='tmp2.json',reverse=True))
        #viz('tmp1.json',jsonfile=True,figname='figx',res='c',drawpoly=True)
        #viz(M0.models,jsonfile=False,figname='figxxx',res='f',drawpoly=False)
        M0.setDataFrame().to_csv(DEST +str(file_count) +'.csv',index=False)
        file_count += 1


def combine_merc(DIR, filename, N = 20):
    '''
    This function combines the csv's into a single file. At the same time,
    this function will convert the format of the coordinates from longitude
    and latitude which is necessary to make our neighborhood plot. Our tileset
    accepts mercator coordinates. This generates one combined csv in the
    current directory. USE PYTHON 3.

    Inputs:
        DIR: The location(filepath) of the csvs to be combined. Example 'csvs/'
        filename: the desired name for the combined csv file. Example: 'combined.csv'
        N: the max number of sources selected for in json_to_csv:
            M.select(var='delay',high=20,reverse=False,inplace=True).
            high argument is N.

    '''
    lonlat = Proj(init = 'epsg:4326')
    mercator = Proj(init = 'epsg:3857')

    files = os.listdir(DIR)

    header = []

    header.append('lattgt')
    header.append('lontgt')

    for i in range(1,N+1):
        header.append('delay' + str(i))
        header.append('gamma' + str(i))
        header.append('latsrc' + str(i))
        header.append('lonsrc' + str(i))

    with open(filename, 'w') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(header)

        for file in files:
            first_row = False
            with open (DIR + file) as csvfile:
                new_row = []
                rows = csv.reader(csvfile)
                next(rows,None)
                for row in rows:
                    if first_row == False:
                        lon, lat = transform(lonlat,mercator, row[6], row[4])
                        new_row.append(lat)
                        new_row.append(lon)
                        first_row = True

                    new_row.append(row[0])
                    new_row.append(row[2])

                    lon, lat = transform(lonlat,mercator, row[5], row[3])
                    new_row.append(lat)
                    new_row.append(lon)
                wr.writerow(new_row)


def neighbor_plot(filepath='crime_filtered_data.csv'):
    '''
    This is the first implementation of our Bokeh plot. The function takes the filepath
    of the data and opens the bokeh plot in a browser. Google Chrome seems to be the
    best browser for bokeh plots. The datafile must be a csv file in the correct format.
    See the file 'crime_filtered_data.csv' for an example. Each row represents a point,
    all the lines(sources) connected to it and the gammas and delays associated with
    the lines. The current implementation results in the bokeh plot, and a linked
    table of the data. IMPORTANT: Points are in MERCATOR Coordinates. This is because
    the current tileset for the map is in mercator coordinates.
    Example file is 'crime_filtered_data.csv'
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

'''
From this point on, we define the functions necessary for a streamline plot and a
heat plot.
'''

def streamheat_combine(DIR, filename):
    '''
    We need to once again combine the csvs, into a format appropriate for the streamplots.
    This file will do that. This function will produce two files. File 1 will
    be in longitude and latitude. File 2 will be in mercator coordinates.
    We will be primiarily working with file 2
    Inputs:
        DIR: The filepath to the csvs. Ex: 'csvs/'
        filename: The filename for the combined csv file. 'contourmerc.csv'
    '''
    lonlat = Proj(init = 'epsg:4326')
    mercator = Proj(init = 'epsg:3857')

    files = os.listdir(DIR)

    merged = []
    for file in files:
        read = pd.read_csv(DIR + file)
        merged.append(read)

    results = pd.concat(merged)
    results.index.name = 'index'
    results.round(3)
    ###
    df = results

    new_lattgt = []
    new_lontgt = []
    new_latsrc = []
    new_lonsrc = []

    for index, row  in df.iterrows():
        lonsrc = row['lonsrc']
        latsrc = row['latsrc']
        lontgt = row['lontgt']
        lattgt = row['lattgt']
        n_lonsrc, n_latsrc = transform(lonlat,mercator, lonsrc, latsrc)
        n_lontgt, n_lattgt = transform(lonlat,mercator, lontgt, lattgt)

        new_lonsrc.append(n_lonsrc)
        new_latsrc.append(n_latsrc)

        new_lontgt.append(n_lontgt)
        new_lattgt.append(n_lattgt)

    df['lonsrc'] = new_lonsrc
    df['latsrc'] = new_latsrc
    df['lontgt'] = new_lontgt
    df['lattgt'] = new_lattgt


    df['theta'] = np.arctan((df['lattgt'] - df['latsrc']) / (df['lontgt'] - df['lonsrc']))

    df['uvector'] = df['gamma'] * np.cos(df['theta'])
    df['vvector'] = df['gamma'] * np.sin(df['theta'])

    new_u = []
    new_v = []

    for index,row in df.iterrows():
        if np.sign(row['lontgt'] - row['lonsrc']) != np.sign(row['uvector']):
            new_u.append(-1 * row['uvector'])
        else:
            new_u.append(row['uvector'])

        if np.sign(row['lattgt'] - row['latsrc']) != np.sign(row['vvector']):
            new_v.append(-1 * row['vvector'])
        else:
            new_v.append(row['vvector'])

    df['uvector'] = new_u
    df['vvector'] = new_v
    #We need these extra points inorder for Delanuay interpolation on a grid to work.
    extra_points = {
        'delay' : [0,0,0,0],
        'distance':[0,0,0,0],
        'gamma': [0,0,0,0],
        'latsrc':[5108694.47,5168472.14,5108694.47,5168472.14],
        'lattgt':[5108694.47,5168472.14,5108694.47,5168472.14],
        'lonsrc':[-9773851.29,-9773851.29,-9743795.029,-9743795.02],
        'lontgt':[-9773851.29,-9773851.29,-9743795.029,-9743795.02],
        'theta':[0,0,0,0],
        'uvector':[0,0,0,0],
        'vvector':[0,0,0,0]
    }
    extra_df = pd.DataFrame(extra_points)
    df = pd.concat([df, extra_df], ignore_index = True)
    df.index.name = 'index'
    df.to_csv(filename)


def crime_stream(datafile='contourmerc.csv',density=4, npoints=10, output_name='streamplot.html', method = 'cubic'):
    '''
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
    '''
    df = pd.read_csv(datafile)
    df = df.dropna()
    df = df.groupby(['lonsrc','latsrc']).mean()

    uvectors = np.array(df['uvector'])
    vvectors = np.array(df['vvector'])

    x_starts = np.array(df.index.get_level_values(0))
    y_starts = np.array(df.index.get_level_values(1))

    xw = np.linspace(-9773000,-9744021,npoints)
    yw = np.linspace(5109694,5165011,npoints)
    Xa, Ya = np.meshgrid(xw,yw)

    ud = griddata((x_starts,y_starts),uvectors, (Xa,Ya), method = method)
    vd = griddata((x_starts,y_starts),vvectors, (Xa,Ya), method = method)
    xs, ys = streamlines(xw, yw, ud.T, vd.T, density = 4)

    #Here we filter for convex hull.
    new_xs = []
    new_ys = []
    hull_points = np.column_stack((x_starts[2:-2],y_starts[2:-2]))
    for n in range(0, len(xs)):
        x_points = np.array(xs[n])
        y_points = np.array(ys[n])
        predict_points = np.column_stack((x_points,y_points))
        hull_array = in_hull(predict_points, hull_points)
        new_x_points = []
        new_y_points = []
        for n in range(0, len(hull_array)):
            if hull_array[n] == True:
                new_x_points.append(x_points[n])
                new_y_points.append(y_points[n])
        new_xs.append(new_x_points)
        new_ys.append(new_y_points)

    p2 = figure(x_range=(-9770511,-9746021), y_range=(5108694,5146011), x_axis_type = 'mercator', y_axis_type = 'mercator')
    p2.add_tile(CARTODBPOSITRON)

    p2.multi_line(new_xs, new_ys, color="#ee6666", line_width=2, line_alpha=0.8, legend = 'Stream')

    p2.circle(x_starts[2:-2],y_starts[2:-2], legend = 'Origins')

    p2.legend.location = 'top_left'
    p2.legend.click_policy = 'hide'

    output_file(output_name, title="Crime Stream")
    show(p2)


def heat_map(datafile='contourmerc.csv', npoints=300, output_name='heatmap.html', method = 'linear'):
    '''
    Makes a heatmap from the same datafile that cimre_stream uses.
    datafile: name of the datafile. Example file is 'contourmerc.csv'.
    npoints: dimension for plot. number of squares = npoints**2.
        Recommended: 100-300
    output_name: output file name for the plot.
    method: method for interpolation. 'cubic','linear', or 'nearest'
    '''

    df = pd.read_csv(datafile)
    df = df.dropna()
    df = df.groupby(['lonsrc','latsrc']).mean()

    x_starts = np.array(df.index.get_level_values(0))
    y_starts = np.array(df.index.get_level_values(1))
    gammas = np.array(df['gamma'])


    xw = np.linspace(-9773000,-9744021,npoints)
    yw = np.linspace(5109694,5165011,npoints)
    Xa, Ya = np.meshgrid(xw,yw)


    rect_x_length = xw[1] - xw[0]
    rect_y_length = yw[1] - yw[0]

    pgammas = griddata((x_starts,y_starts),gammas, (Xa,Ya), method = method)

    predict_x = []
    predict_y = []
    predict_g = []

    for x in range(0, npoints):
        for y in range(0, npoints):
            predict_x.append(xw[x])
            predict_y.append(yw[y])
            predict_g.append(pgammas[y][x])

    #Here we will do filtering for convex hull.
    predict_points = np.column_stack((np.array(predict_x),np.array(predict_y)))
    hull_points = np.column_stack((x_starts[2:-2],y_starts[2:-2]))
    hull_array = in_hull(predict_points, hull_points)

    filtered_x = []
    filtered_y = []
    filtered_g = []
    for n in range(0, npoints**2):
        if hull_array[n] == True:
            filtered_x.append(predict_x[n])
            filtered_y.append(predict_y[n])
            filtered_g.append(predict_g[n])

    predict_dic = {'predict_x':filtered_x,'predict_y':filtered_y, 'predict_g':filtered_g}
    predict_df = pd.DataFrame(predict_dic)
    source = ColumnDataSource(predict_df)

    TOOLS = ['pan','wheel_zoom','reset','hover','save']

    p2 = figure(x_range=(-9770511,-9746021), y_range=(5108694,5146011), tools = TOOLS,
        x_axis_type = 'mercator',
        y_axis_type = 'mercator')
    p2.add_tile(CARTODBPOSITRON)

    hover = p2.select_one(HoverTool)
    hover.point_policy = "follow_mouse"
    hover.tooltips = [
        ("Gamma", "@predict_g")
    ]

    #palette = [ '#410967', '#932567','#DC5039','#FBA40A', 'darkred']
    palette = ['#440154', '#30678D', '#35B778', '#FDE724']
    color_mapper = LogColorMapper(palette=palette, low = min(predict_g) , high = max(predict_g))

    p2.rect(x = 'predict_x', y = 'predict_y', width = rect_x_length, height = rect_y_length, source = source,
        fill_color = {'field':'predict_g', 'transform': color_mapper},
        fill_alpha = 0.5,
        line_alpha = 0)

    output_file("heat_map3.html", title="Heat Map")
    show(p2)
