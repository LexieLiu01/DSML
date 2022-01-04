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
input_region_geo_Manhattan_path = '../data_set/NYC_geo/NYC_Taxi_Zones_Manhattan.geojson'
center_point_region_df_path = '../Shortest_path/center_point_region_df.pickle'

components = 'complete'
# components = 'part'

if components == 'part':
    ids = [4, 12, 13, 24, 43, 45, 48, 50, 68, 75, 79, 87, 88, 90, 100, 107, 113, 114, 125, 137, 140, 141, 142, 143, 144,
           148, 151, 158, 161, 162, 163, 164, 170, 186, 209, 211, 224, 229, 230, 231, 232, 233, 234, 236, 237, 238, 239,
           246, 249, 261, 262, 263]
    
    data_final_result_simplify_df = pd.DataFrame()
    data_final_result_simplify_df['same_color'] = [0] *52
    data_final_result_simplify_df['region_id'] = ids
elif components == 'complete':
    ids = [4,24,12,13,41,45,42,43,48,50,68,79,74,75,87,88,90,125,100,107, 113,114,116,120,127,128,151,140,137,141,142,152,143,144,148,158,161,162,163,164,170,166,186,209,211,224,229,230,231,239,232,233,234,236,237,238,263,243,244,246,249,261,262]

    data_final_result_simplify_df = pd.DataFrame()
    data_final_result_simplify_df['same_color'] = [0] *63
    data_final_result_simplify_df['region_id'] = ids
    print(data_final_result_simplify_df)

# theta_map_path = './visual_map/' + 'neighboring_regions_map' + '.html'
theta_map_path = './visual_map/' + 'attractions_map' + components +'.html'
theta_map_html_path = './model_dml/attractions_map' + components +'_html.html'

# Get center point at one region and geometry of current region
def get_center_point(input_region_geo_Manhattan_path, center_point_region_df_path, ids):
    region_polygon_gpd = gpd.read_file(input_region_geo_Manhattan_path)
    region_polygon_gpd["region_id"] = region_polygon_gpd["region_id"].apply(lambda x: int(x))
    region_polygon_gpd = region_polygon_gpd[region_polygon_gpd["region_id"].isin(ids)]
    region_polygon_gpd.set_index(["region_id"], inplace=True)
    region_polygon_gpd = region_polygon_gpd.sort_values(by=['region_id'])
    region_polygon_gpd = region_polygon_gpd.reset_index()
    
    center_point_region_df = gpd.GeoDataFrame()
    if os.path.exists(center_point_region_df_path):
        with open(center_point_region_df_path, 'rb') as f:
            center_point_region_df = pickle.load(f)
        return center_point_region_df, region_polygon_gpd
    
    center_point_region_df["region_id"] = region_polygon_gpd['region_id']
    center_point_region_df['center_point'] = region_polygon_gpd['geometry'].apply(lambda x: x.centroid)
    
    with open(center_point_region_df_path, 'wb') as f:
        pickle.dump(center_point_region_df, f)
    
    return center_point_region_df, region_polygon_gpd

def plot_map(data_final_result_simplify_df, region_polygon_gpd, center_point_region_df, plot_map_style):
    plot_map_style_list = plot_map_style.split('_')
    
    # plot map overview
    myscale = [i * 0.01 for i in range(-15, 1, 3)]
    m = folium.Map([40.730610, -73.935242], zoom_start=12, tiles=None)
    # m = folium.Map([40.730610, -73.935242], zoom_start=12, tiles=None)
    # tiles = ['stamenwatercolor', 'cartodbpositron', 'openstreetmap', 'stamenterrain']
    
    folium.raster_layers.TileLayer(
        tiles="https://tiles.stadiamaps.com/tiles/alidade_smooth/{z}/{x}/{y}{r}.png",
        attr='&copy; <a href="https://stadiamaps.com/">Stadia Maps</a>, &copy; <a href="https://openmaptiles.org/">OpenMapTiles</a> &copy; <a href="http://openstreetmap.org">OpenStreetMap</a> contributors'
    ).add_to(m)
    
    data_final_result_simplify_df = data_final_result_simplify_df.reset_index()
    region_polygon_gpd = region_polygon_gpd.reset_index()
    keys = ['same_color']
    for key in keys:
        region_polygon_gpd[key] = data_final_result_simplify_df[key]
    
    add_choropleth = folium.Choropleth(geo_data=region_polygon_gpd, data=data_final_result_simplify_df,
                                       columns=['region_id', 'same_color'],
                                       key_on='feature.properties.region_id', fill_color='Greys_r',
                                       fill_opacity=0.7, line_opacity=0.2,
                                       legend_name='Value of theta',
                                       highlight=True,
                                       threshold_scale=myscale,
                                       bins=10
                                       ).add_to(m)
    
    # show id
    if 'id' in plot_map_style_list:
        # Show region id
        center_point_layer = folium.FeatureGroup(name="Query Search")
        for i in range(0, len(center_point_region_df)):
            center_point_layer.add_child(folium.Marker(
                location=[center_point_region_df.iloc[i, 1].y, center_point_region_df.iloc[i, 1].x],
                popup=str(center_point_region_df.iloc[i, 0]),
                icon=folium.DivIcon(
                    html=f"""<div style="font-family: courier new; color: k">  {str(center_point_region_df.iloc[i, 0])}</div>"""),
                fill=True,  # Set fill to True
                color='red',
                fill_opacity=1.0
            )).add_to(m)
        m.add_child(center_point_layer)
    
    # show poi
    if 'poi' in plot_map_style_list:
        
        places_of_interests_region_df = [['One_World_Trade_Center', 40.7127, -74.0134],
                                         ['Brooklyn Bridge', 40.7061, -73.9969],
                                         ['Empire State Building', 40.7484, -73.9857],
                                         ['Times Square', 40.7580, -73.9855], ['Rockefeller Center', 40.7587, -73.9787],
                                         ['The Museum of Modern Art', 40.7614, -73.9776],
                                         ['Central Park', 40.7812, -73.9665],
                                         ['The Metropolitan Museum of Art', 40.7794, -73.9632],
                                         ['Wall Street', 40.706501206604756, -74.00943494897943]]
        places_of_interests_region_df = pd.DataFrame(places_of_interests_region_df)
        places_of_interests_region_df.columns = ['places_of_interests', 'x', 'y']
        
        # Show
        point_layer = folium.FeatureGroup(name="Query Search")
        
        for i in range(0, len(places_of_interests_region_df)):
            point_layer.add_child(folium.Marker(
                location=[places_of_interests_region_df.iloc[i, 1], places_of_interests_region_df.iloc[i, 2]],
                popup=places_of_interests_region_df.iloc[i, 0],
                # icon=folium.DivIcon(html=f"""<div><svg><circle cx="20" cy="20" r="20" fill="#ffcc00" opacity=".5"/></svg></div>"""),
                
                # fill=True,  # Set fill to True
                # color='red',
                # clustered_marker = True,
                fill_opacity=1.0
            
            )).add_to(m)
        m.add_child(point_layer)
    

    add_choropleth.geojson.add_child(
        folium.features.GeoJsonTooltip(['region_id', 'same_color']))
    m.add_child(folium.ClickForMarker(popup="Waypoint"))
    m.add_child(folium.map.LayerControl())
    m.add_child(folium.LatLngPopup())
    m.add_child(folium.LayerControl())
    
    return m


if __name__ == '__main__':
    center_point_region_df, region_polygon_gpd = get_center_point(input_region_geo_Manhattan_path,
                                                                  center_point_region_df_path, ids)
    
    m = plot_map(data_final_result_simplify_df, region_polygon_gpd, center_point_region_df, plot_map_style)
    
    m.save(theta_map_path)
    webbrowser.open(theta_map_html_path)