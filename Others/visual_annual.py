import pickle
import folium
from folium import IFrame
import pandas as pd
import webbrowser
import geopandas as gpd
import sys
from visual_theta import *


data_for_training_path = './model_dml/training_data/training_data_dmsl_XWXMW_more_parameters2_3_4_16_17_18_19.pickle'

input_region_geo_Manhattan_path = './data_set/NYC_geo/NYC_Taxi_Zones_Manhattan.geojson'
center_point_region_df_path = './Shortest_path/center_point_region_df.pickle'
ids = [4, 12, 13, 24, 43, 45, 48, 50, 68, 75, 79, 87, 88, 90, 100, 107, 113, 114, 125, 137, 140, 141, 142, 143, 144,148, 151, 158, 161, 162, 163, 164, 170, 186, 209, 211, 224, 229, 230, 231, 232, 233, 234, 236, 237, 238, 239,246, 249, 261, 262, 263]
center_point_region_df, region_polygon_gpd = get_center_point(input_region_geo_Manhattan_path, center_point_region_df_path, ids)

theta_map_speed_path = './Visualization_plot/annual_peed.html'
theta_map_speed_html_path = './Visualization_plot/annual_speed_html.html'


def read_ave_annual(data_for_training_path, ids):
    with open(data_for_training_path, 'rb') as f:
        training_data_for_evening_df = pickle.load(f)
    
    training_data_for_evening_plot_speed_df = training_data_for_evening_df.groupby('region_id')['relative_speed'].mean()
    training_data_for_evening_plot_speed_df = training_data_for_evening_plot_speed_df.reset_index()
    training_data_for_evening_plot_speed_df['region_id'] = training_data_for_evening_plot_speed_df['region_id'].apply(
        lambda x: int(x))
    training_data_for_evening_plot_speed_df = training_data_for_evening_plot_speed_df[
        training_data_for_evening_plot_speed_df["region_id"].isin(ids)]
    
    training_data_for_evening_plot_pickdrop_df = training_data_for_evening_df.groupby('region_id')[
        'total_number'].mean()
    training_data_for_evening_plot_pickdrop_df = training_data_for_evening_plot_pickdrop_df.reset_index()
    training_data_for_evening_plot_pickdrop_df['region_id'] = training_data_for_evening_plot_pickdrop_df[
        'region_id'].apply(lambda x: int(x))
    training_data_for_evening_plot_pickdrop_df = training_data_for_evening_plot_pickdrop_df[
        training_data_for_evening_plot_pickdrop_df["region_id"].isin(ids)]
    
    return training_data_for_evening_plot_speed_df, training_data_for_evening_plot_pickdrop_df


def plot_map_speed(training_data_for_evening_plot_speed_df, region_polygon_gpd):
    # plot map overview
    #     myscale = [i * 0.01 for i in range(-15, 1, 3)]
    m = folium.Map([40.730610, -73.935242], zoom_start=12, tiles=None)
    # tiles = ['stamenwatercolor', 'cartodbpositron', 'openstreetmap', 'stamenterrain']

    training_data_for_evening_plot_speed_df = training_data_for_evening_plot_speed_df.reset_index()
    region_polygon_gpd = region_polygon_gpd.reset_index()
    keys = ['relative_speed']
    for key in keys:
        region_polygon_gpd[key] = training_data_for_evening_plot_speed_df[key]
    
    add_choropleth = folium.Choropleth(geo_data=region_polygon_gpd, data=training_data_for_evening_plot_speed_df,
                                       columns=['region_id', 'relative_speed'],
                                       key_on='feature.properties.region_id',
                                       #                                        fill_color='YlGnBu',
                                       fill_opacity=0.7, line_opacity=0.2,
                                       legend_name='Average of Speed in Regions',
                                       highlight=True
                                       #                                        threshold_scale=myscale,
                                       #                                        bins=10
                                       ).add_to(m)
    
    add_choropleth.geojson.add_child(folium.features.GeoJsonTooltip(['region_id', 'relative_speed']))
    m.add_child(folium.ClickForMarker(popup="Waypoint"))
    m.add_child(folium.map.LayerControl())
    m.add_child(folium.LatLngPopup())
    m.add_child(folium.LayerControl())
    
    m.save(theta_map_speed_path)
    webbrowser.open(theta_map_speed_html_path)
    
    return m


if __name__ == '__main__':
    training_data_for_evening_plot_speed_df, training_data_for_evening_plot_pickdrop_df = read_ave_annual(data_for_training_path, ids)
    
    m = plot_map_speed(training_data_for_evening_plot_speed_df, region_polygon_gpd)

