from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, LogColorMapper, LinearColorMapper, Segment, Circle
from bokeh.models.tools import HoverTool, BoxSelectTool, TapTool
from bokeh.models.widgets import DataTable, TableColumn, NumberFormatter
from bokeh.io import show, output_file
from bokeh.palettes import RdYlBu11 as palette
from bokeh.tile_providers import CARTODBPOSITRON
from bokeh.layouts import widgetbox, gridplot

import pandas as pd

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