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

# model = 'DML'
# model = 'DSML'

# tag = 'GB'
# tag = 'RF'
# tag = 'Ada'
# tag = 'all'

model = 'LR'
tag = ''

plot_map_style = ' '
# plot_map_style = '_with_poi'
# plot_map_style = 'id_with_poi'
# plot_map_style = 'id'
# target_weekday = ['0', '6']
# Tues, Wed, Thurs
target_weekday = ['2', '3', '4']
identify_target_weekday = '_'.join(target_weekday)

target_hour = ['16', '17', '18', '19']
identify_target_hour = '_'.join(target_hour)

data_final_result_path = './model_dml/theta/' + 'theta_' + model + '_' + identify_target_weekday + '_' + identify_target_hour + tag+ '.pickle'
data_final_result_simplify_path = './model_dml/theta/' + 'theta_' + model + '_' + identify_target_weekday + '_' + identify_target_hour + tag+'_simplify.pickle'
theta_map_path = './model_dml/theta/' + 'theta_' + model + '_' + identify_target_weekday + '_' + identify_target_hour + tag+ '.html'
theta_map_html_path = './model_dml/theta/' + 'theta_' + model + '_' + identify_target_weekday + '_' + identify_target_hour + tag + '_html.html'

speed_fig_path = './Visualization/Pickup_Dropoff/'

input_region_geo_Manhattan_path = './data_set/NYC_geo/NYC_Taxi_Zones_Manhattan.geojson'
center_point_region_df_path = './Shortest_path/center_point_region_df.pickle'
ids = [4, 12, 13, 24, 43, 45, 48, 50, 68, 75, 79, 87, 88, 90, 100, 107, 113, 114, 125, 137, 140, 141, 142, 143, 144,148, 151, 158, 161, 162, 163, 164, 170, 186, 209, 211, 224, 229, 230, 231, 232, 233, 234, 236, 237, 238, 239,246, 249, 261, 262, 263]

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


# Rescale theta
def scale_value(x):
    if x >= 0:
        x = 0
    elif x <= -0.12:
        x = -0.12
    return x


def simplify_theta(data_final_result_path, ids, model):
    with open(data_final_result_path, 'rb') as f:
        data_final_result_dict = pickle.load(f)

    data_final_result_simplify_dict = {}
    for id in ids:
        data_final_result_simplify_dict[id] = {}
        
        if model == 'DSML':
            data_final_result_simplify_dict[id]['results_params'] = data_final_result_dict[id]['results_params'][1]
            data_final_result_simplify_dict[id]['p_value'] = data_final_result_dict[id]['p_value'][1]
        else:
            data_final_result_simplify_dict[id]['results_params'] = data_final_result_dict[id]['results_params']
            data_final_result_simplify_dict[id]['p_value'] = data_final_result_dict[id]['p_value']
        
    return data_final_result_simplify_dict


# Get theta for show
def rebuild_theta_show_df(data_final_result_simplify_dict, ids, data_final_result_simplify_path):
    
    data_final_result_simplify_df = pd.DataFrame()
    for id in ids:
        data_final_result_simplify_df = data_final_result_simplify_df.append(pd.json_normalize(data_final_result_simplify_dict[id]))

    data_final_result_simplify_df['results_params_rescale'] = data_final_result_simplify_df['results_params'].apply(lambda x: scale_value(x))
    
    data_final_result_simplify_df['region_id'] = pd.Series(ids, index=data_final_result_simplify_df.index)
    
    data_final_result_simplify_df = data_final_result_simplify_df[['region_id','results_params_rescale','results_params','p_value']]
    
    with open(data_final_result_simplify_path, 'wb') as f:
        pickle.dump(data_final_result_simplify_df, f)
        
    return data_final_result_simplify_df

# Get figs
def get_figs(center_point_region_df, speed_fig_path):
    
    filename = []
    fileid = []
    
    for photos in os.listdir(speed_fig_path):
        if photos.endswith(".png"):
            fileid.append(photos.split('_')[0])
            photos = speed_fig_path + photos
            filename.append(photos)
    
    fileid_df = pd.DataFrame(fileid)
    filename_df = pd.DataFrame(filename)
    file_df = pd.concat([fileid_df, filename_df], axis=1)
    file_df.reset_index()
    file_df.columns = ['region_id', 'file_path']
    
    show_fig_df = center_point_region_df.merge(file_df, how='left', on='region_id')
    return show_fig_df

def plot_map(data_final_result_simplify_df, region_polygon_gpd,center_point_region_df, plot_map_style):
    plot_map_style_list = plot_map_style.split('_')
    
    # plot map overview
    myscale = [i * 0.01 for i in range(-15, 1, 3)]
    m = folium.Map([40.730610, -73.935242], zoom_start=12, tiles=None)
    # m = folium.Map([40.730610, -73.935242], zoom_start=12, tiles=None)
    #tiles = ['stamenwatercolor', 'cartodbpositron', 'openstreetmap', 'stamenterrain']

    folium.raster_layers.TileLayer(
        tiles="https://tiles.stadiamaps.com/tiles/alidade_smooth/{z}/{x}/{y}{r}.png",
        attr='&copy; <a href="https://stadiamaps.com/">Stadia Maps</a>, &copy; <a href="https://openmaptiles.org/">OpenMapTiles</a> &copy; <a href="http://openstreetmap.org">OpenStreetMap</a> contributors'
    ).add_to(m)


    data_final_result_simplify_df = data_final_result_simplify_df.reset_index()
    region_polygon_gpd = region_polygon_gpd.reset_index()
    keys = ['results_params_rescale','results_params','p_value']
    for key in keys:
        region_polygon_gpd[key] = data_final_result_simplify_df[key]
    
    add_choropleth = folium.Choropleth(geo_data=region_polygon_gpd, data=data_final_result_simplify_df,
                                       columns=['region_id', 'results_params_rescale'],
                                       key_on='feature.properties.region_id', fill_color='Reds_r',
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
        
        places_of_interests_region_df = [['One_World_Trade_Center', 40.7127, -74.0134], ['Brooklyn Bridge', 40.7061, -73.9969], ['Empire State Building', 40.7484, -73.9857], ['Times Square',40.7580, -73.9855],['Rockefeller Center', 40.7587, -73.9787],['The Museum of Modern Art', 40.7614, -73.9776],['Central Park',40.7812, -73.9665],['The Metropolitan Museum of Art',40.7794, -73.9632],['Wall Street',40.706501206604756, -74.00943494897943]]
        places_of_interests_region_df = pd.DataFrame(places_of_interests_region_df)
        places_of_interests_region_df.columns = ['places_of_interests','x','y']
        
        # Show
        point_layer = folium.FeatureGroup(name="Query Search")

        
        for i in range(0,len(places_of_interests_region_df)):
            point_layer.add_child(folium.Marker(
                location = [places_of_interests_region_df.iloc[i,1],places_of_interests_region_df.iloc[i,2]],
                popup=places_of_interests_region_df.iloc[i,0],
                # icon=folium.DivIcon(html=f"""<div><svg><circle cx="20" cy="20" r="20" fill="#ffcc00" opacity=".5"/></svg></div>"""),
                
                # fill=True,  # Set fill to True
                # color='red',
                # clustered_marker = True,
                fill_opacity=1.0
                
           )).add_to(m)
        m.add_child(point_layer)
        
    # show fig
    if 'fig' in plot_map_style_list:
        show_fig_df = get_figs(center_point_region_df, speed_fig_path)
    
        show_fig_layer = folium.FeatureGroup(name="Query Search")
    
        for i in range(0, len(show_fig_df)):
            encoded = base64.b64encode(open(show_fig_df.iloc[i, 2], 'rb').read()).decode()
            html = '<img src="data:image/png;base64,{}">'.format
            resolution, width, height = 75, 50, 25
            iframe = IFrame(html(encoded), width=(width * resolution) + 20, height=(height * resolution) + 20)
            # popup = folium.Popup(iframe, max_width=1000)
            icon = folium.Icon()
            show_fig_layer.add_child(
                folium.Marker([show_fig_df.iloc[i, 1].y, show_fig_df.iloc[i, 1].x], icon=icon  # popup=popup,
                              )).add_to(m)
        m.add_child(show_fig_layer)
        
    add_choropleth.geojson.add_child(folium.features.GeoJsonTooltip(['region_id','results_params_rescale','results_params','p_value']))
    m.add_child(folium.ClickForMarker(popup="Waypoint"))
    m.add_child(folium.map.LayerControl())
    m.add_child(folium.LatLngPopup())
    m.add_child(folium.LayerControl())
    
    return m


if __name__ == '__main__':
    center_point_region_df, region_polygon_gpd = get_center_point(input_region_geo_Manhattan_path, center_point_region_df_path, ids)
    data_final_result_simplify_dict = simplify_theta(data_final_result_path, ids, model)
    data_final_result_simplify_df = rebuild_theta_show_df(data_final_result_simplify_dict, ids, data_final_result_simplify_path)
    m = plot_map(data_final_result_simplify_df, region_polygon_gpd,center_point_region_df, plot_map_style)
    
    m.save(theta_map_path)
    webbrowser.open(theta_map_html_path)