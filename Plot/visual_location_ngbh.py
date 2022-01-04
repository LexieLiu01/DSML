import folium
import pandas as pd
import webbrowser
import geopandas as gpd
# from branca.colormap import linear
import json
import pickle
import os
import folium
from folium import IFrame
import base64
from folium.plugins import MarkerCluster


plot_map_style = ' '

ngbh_geojson_path = '../data_set/NYC_geo/neighborhood_manhattan.geojson'
components = 'complete'
# components = 'part'

# theta_map_path = './visual_map/' + 'neighboring_regions_map' + '.html'
theta_map_path = './visual_map/' + 'ngbh_map' + components + '.html'
theta_map_html_path = './model_dml/ngbh_map' + components + '_html.html'


# Get center point at one region and geometry of current region
def get_center_point(ngbh_geojson_path):
    ngbh_geojson_gpd = gpd.read_file(ngbh_geojson_path)
    ngbh_polygon_gpd = ngbh_geojson_gpd[ngbh_geojson_gpd['BoroName'] == 'Manhattan']
    # ngbh_polygon_gpd = ngbh_polygon_gpd[ngbh_geojson_gpd['OBJECTID'] != 91]
    
    data_final_result_simplify_df = pd.DataFrame()
    data_final_result_simplify_df['same_color'] = [0] * len(ngbh_polygon_gpd)

    return ngbh_polygon_gpd, data_final_result_simplify_df

def plot_map(data_final_result_simplify_df, ngbh_polygon_gpd, plot_map_style):
    plot_map_style_list = plot_map_style.split('_')
    
    # plot map overview
    myscale = [i * 0.01 for i in range(-15, 1, 3)]
    m = folium.Map([40.730610, -73.935242], zoom_start=12, tiles=None)

    folium.raster_layers.TileLayer(
        tiles="https://tiles.stadiamaps.com/tiles/alidade_smooth/{z}/{x}/{y}{r}.png",
        attr='&copy; <a href="https://stadiamaps.com/">Stadia Maps</a>, &copy; <a href="https://openmaptiles.org/">OpenMapTiles</a> &copy; <a href="http://openstreetmap.org">OpenStreetMap</a> contributors'
    ).add_to(m)

    data_final_result_simplify_df['OBJECTID'] = ngbh_polygon_gpd['OBJECTID'].values.tolist()
    data_final_result_simplify_df = data_final_result_simplify_df.reset_index()
    
    ngbh_polygon_gpd = ngbh_polygon_gpd.reset_index()
    keys = ['same_color']
    for key in keys:
        ngbh_polygon_gpd[key] = data_final_result_simplify_df[key]
    
    add_choropleth = folium.Choropleth(geo_data=ngbh_polygon_gpd, data=data_final_result_simplify_df,
                                       columns=['OBJECTID', 'same_color'],
                                       key_on='feature.properties.OBJECTID', fill_color='Greys_r',
                                       fill_opacity=0.7, line_opacity=0.2,
                                       legend_name='Value of theta',
                                       highlight=True,
                                       threshold_scale=myscale,
                                       bins=10
                                       ).add_to(m)
    
    # show id
    add_choropleth.geojson.add_child(
        folium.features.GeoJsonTooltip(['OBJECTID', 'same_color']))
    m.add_child(folium.ClickForMarker(popup="Waypoint"))
    m.add_child(folium.map.LayerControl())
    m.add_child(folium.LatLngPopup())
    m.add_child(folium.LayerControl())
    
    return m


if __name__ == '__main__':
    ngbh_polygon_gpd, data_final_result_simplify_df = get_center_point(ngbh_geojson_path)
    
    m = plot_map(data_final_result_simplify_df, ngbh_polygon_gpd, plot_map_style)
    
    m.save(theta_map_path)
    webbrowser.open(theta_map_html_path)