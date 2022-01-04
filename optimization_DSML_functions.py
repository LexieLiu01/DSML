# Imports
from shapely.geometry import Point, LineString
import geopandas as gpd
import json
import networkx as nx
import pandas as pd
from pyproj import Transformer
import numpy as np

import math
from datetime import datetime, timedelta
import pickle
import matplotlib
import os

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys

# Input Files
input_region_geo_Manhattan_path = './data_set/NYC_geo/NYC_Taxi_Zones_Manhattan.geojson'
input_trip_df_path= './data_set/NYC_triprecord/prepared_data/trip_all_for_optimization.csv'

# Cache Files
neighbor_region_path = './data_set/NYC_geo/neighbor_region_optimization.pickle'
trip_speed_precip_for_train_path = './data_set/Prepared_matrix_completion_data/trip_speed_precip_X_M_Nei_more_parameters.csv'
shortest_paths_path = './Shortest_path/shortest_paths.pickle'
region_speed_dict_path = './Shortest_path/region_speed_initialization.pickle'
trip_dict_path = './Shortest_path/trip_dict_initialization.pickle'
neighboring_nodes_subdistance_path = './Shortest_path/neighboring_nodes_distance.pickle'
center_point_region_path = './Shortest_path/center_point_region.pickle'
prepare_matrix_dict_path = './Shortest_path/prepare_matrix_just_finance_street.pickle'

# Output Files

# Initialize Variables
shortest_path_dict = {}
updated_pair_weight_dict = {}
updated_path_time_cost_dict = {}
updated_region_speed_value_dict = {}

ids = [4, 12, 13, 24, 43, 45, 48, 50, 68, 75, 79, 87, 88, 90, 100, 107, 113, 114, 125, 137, 140, 141, 142, 143, 144,148, 151, 158, 161, 162, 163, 164, 170, 186, 209, 211, 224, 229, 230, 231, 232, 233, 234, 236, 237, 238, 239,246, 249, 261, 262, 263]
id_o = ids

# run with arguments
# dest_idct = 'MidTown'/'WallStreet'/ Whole; sensitivity method = 'Add'/ 'Product';
dest_idct, Sens_idct = str(sys.argv[1]), str(sys.argv[2])
# Min_idct = 'withMin'/ 'withoutMin'; Q_hat_idct = 'withQhat'/ 'withoutQhat'
Min_idct, Q_hat_idct = str(sys.argv[3]), str(sys.argv[4])
# a: lower bound; b: upper bound; c: times
a,b,c = float(sys.argv[5]),float(sys.argv[6]),int(sys.argv[7])
# grad_method: "Nestgra"/"Adagra"/"Normal"; step_size: 0,1 the larger, the faster for updating. beta: 0.1/0.01/0.001
grad_method, beta, step_size= str(sys.argv[8]), float(sys.argv[9]), float(sys.argv[10])
# walking_speed; the theta_enlarge
walking_speed, theta_enlarge = float(sys.argv[11]), float(sys.argv[12]) * 1/12
# loop_times = 200
loop_times = int(sys.argv[13])
# Man_dist = 'withMandist'/ 'withoutMandist'
Man_dist = str(sys.argv[14])
# day ='weekdays'/ 'weekdays'
day = str(sys.argv[15])
# Setting of Destination
destination = 'MidTown'
if dest_idct == 'MidTown':
    id_d = [100, 186, 164]
elif dest_idct == 'WallStreet':
    id_d = [261, 87]
    # id_d = [125, 211]
elif dest_idct == 'CenterPark':
    id_d = [236,263]
else:
    dest_idct = 'Whole'
    id_d = ids

# # temporary value
# a = 99999
# b = 99999
# c = 10
# # grad_method = 'Normal'
# grad_method = 'Nestgra'
# beta = 0.1
# step_size = 0.05
# walking_speed = 5
# # theta_enlarge = 1/12
# theta_enlarge = 1

# step_size = 0.5

if day == 'weekends':
    weekdays = ['6', '0']
    theta_path = './model_dml/theta/theta_DSML_0_6_16_17_18_19all.pickle'
    # print('weekends')
elif day == 'weekdays':
    weekdays = ['2', '3', '4']
    theta_path = './model_dml/theta/theta_DSML_2_3_4_16_17_18_19all.pickle'
    # print('weekdays')

middle_filename = dest_idct + '_' + Sens_idct + '_' + Min_idct + '_' + Q_hat_idct + '_' + str(a) + '_' + str(b) + '_' + str(c) + '_' + grad_method + '_stepsize' + str(step_size) + '_walkingspeed' + str(walking_speed) +'_enlarge'+ str(theta_enlarge) +'_looptimes' +str(loop_times)+ '_'+str(Man_dist)+ '_' + str(day)
filename_result = 'optimization_result_' + middle_filename + '.pickle'
filename_rate = 'optimization_rate_' + middle_filename + '.pickle'
filename_fig = 'optimization_fig_' + middle_filename + '/'
result_final_path = './Shortest_path/'+ dest_idct + '/' + filename_result
result_rate_path = './Shortest_path/' + dest_idct + '/'+ filename_rate
fig_save_path = './Shortest_path/plot_optimization/' + dest_idct + '/'+ filename_fig

if not os.path.exists(fig_save_path):
    os.makedirs(fig_save_path)

# Initialize time
start = datetime(2019, 7, 1)
end = datetime(2019, 9, 30)
time_intervals = 60
hours = ['16', '17', '18', '19']

# Define Functions
# Get time sequence from start to the end

def datetime_range(start, end, delta):
    current = start
    if not isinstance(delta, timedelta):
        delta = timedelta(**delta)
    while current < end:
        yield current
        current += delta

def generate_time_sequence(time_intervals):
    time_seq = []
    for dt in datetime_range(start, end, {'minutes': time_intervals}):
        if str(dt)[11:13] in hours:
            if dt.strftime("%w") in weekdays:
                time_seq.append(str(dt)[0:19])
    return time_seq

# Get center point at one region
def get_center_point(input_region_geo_Manhattan_path, center_point_region_path, ids):
    region_polygon_gpd = gpd.read_file(input_region_geo_Manhattan_path)
    ids_str = [str(id) for id in ids]
    region_polygon_gpd = region_polygon_gpd[region_polygon_gpd["region_id"].isin(ids_str)]
    
    center_point_region_dict = {}
    if os.path.exists(center_point_region_path):
        with open(center_point_region_path, 'rb') as f:
            center_point_region_dict = pickle.load(f)
        return center_point_region_dict, region_polygon_gpd
    
    for index, row in region_polygon_gpd.iterrows():
        region_id = row['region_id']
        geometry = row['geometry']
        center = geometry.centroid
        
        # center_point_region[region_id] int
        center_point_region_dict[int(region_id)] = center
    
    with open(center_point_region_path, 'wb') as f:
        pickle.dump(center_point_region_dict, f)
    
    return center_point_region_dict, region_polygon_gpd

# Get neighboring region pairs
def get_neighbor_region(neighbor_region_path, region_polygon_gpd):
    if os.path.exists(neighbor_region_path):
        with open(neighbor_region_path, 'rb') as f:
            neighbor_region_dict = pickle.load(f)
        return neighbor_region_dict
    
    # sjoin region id and tmc code by geometry
    neighbor_region_dict = {}
    for id in ids:
        id_polygon = region_polygon_gpd[region_polygon_gpd['region_id'] == str(id)]['geometry'].item()
        
        for id_other in ids:
            if id != id_other:
                id_other_polygon = region_polygon_gpd[region_polygon_gpd.region_id == str(id_other)]['geometry'].item()
                if id_polygon.intersects(id_other_polygon):
                    
                    if id not in neighbor_region_dict.keys():
                        neighbor_region_dict[id] = []
                    neighbor_region_dict[id].append(id_other)
    
    with open(neighbor_region_path, 'wb') as f:
        pickle.dump(neighbor_region_dict, f)
        
    return neighbor_region_dict

# Get theta from DSML
def get_theta(theta_path, ids):
    region2theta_dict = {}
    
    with open(theta_path, 'rb') as fr:
        theta_dict = pickle.load(fr)
    for id in ids:
        region2theta_dict[id] = theta_dict[id]['results_params'][1]
    return region2theta_dict

# Get optimized_speed_value, Given delta_a, Related to time
def initial_region_speed(trip_speed_precip_for_train_path, ids, region_speed_dict_path):
    if os.path.exists(region_speed_dict_path):
        with open(region_speed_dict_path, 'rb') as f:
            region_speed_dict = pickle.load(f)
            
        return region_speed_dict
    
    region_speed_dict = {}
    speed_df = pd.read_csv(trip_speed_precip_for_train_path)
    speed_df['datetime_min_5'] = pd.to_datetime(speed_df['datetime_min_5'], format='%Y-%m-%d %H:%M', errors='ignore')
    speed_df['region_id'] = speed_df['region_id'].apply(lambda x: int(x))
    speed_df = speed_df.groupby(pd.Grouper(key='datetime_min_5', freq='60Min'))
    
    for group in speed_df:
        time = str(group[0])
        group_df = group[1]
        group_df = group_df.groupby(by=['region_id']).mean()
        
        group_df = group_df.reset_index()
        region_speed_dict[time] = {}
        
        for id in ids:
            region_group_df = group_df[group_df['region_id'] == int(id)]
            speed_value = region_group_df['relative_speed'].to_numpy()[0]
            
            if speed_value == 0:
                speed_value = 15
            
            # region_speed_dict[time][4] int
            region_speed_dict[time][id] = {}

            region_speed_dict[time][id]['speed_value'] = speed_value
    
    with open(region_speed_dict_path, 'wb') as f:
        pickle.dump(region_speed_dict, f)

    return region_speed_dict


# Update speed according to Delta_A
def update_speed(region_speed_dict, time, Delta_A_value, region2theta_dict, theta_enlarge):
    
    updated_region_speed_value_dict[time] = {}
    # print(time)
    
    # normal
    for id in ids:
    
        updated_region_speed_value_dict[time][id] = {}

        # normal
        index = ids.index(id)
    
        # region_speed_dynamic_dict[time][4] int
        # updated_region_speed_value_dict[time][id]['optimized_speed_value'] = region_speed_dict[time][id]['speed_value'] + Delta_A[index, 0] * region2theta_dict[id] / 6
        # just count value that is negative.
        
        if Min_idct == 'withMin':
            Delta_A_value = np.minimum(Delta_A_value, 0)
        
        updated_region_speed_value_dict[time][id]['optimized_speed_value'] = region_speed_dict[time][id]['speed_value'] + Delta_A_value[index, 0] * region2theta_dict[id] * theta_enlarge
        
        # print('id:', id,'the change of speed:',Delta_A_value[index, 0] * region2theta_dict[id] * theta_enlarge)
        # print('id:', id,'updated_speed:', updated_region_speed_value_dict[time][id]['optimized_speed_value'])
        if updated_region_speed_value_dict[time][id]['optimized_speed_value'] <= 1:
            updated_region_speed_value_dict[time][id]['optimized_speed_value'] = 1
    
    return updated_region_speed_value_dict


# Get distance between neighboring regions
def get_neighboring_nodes_subdistance(neighbor_region_dict, center_point_region_dict, region_polygon_gpd, neighboring_nodes_subdistance_path):
    
    if os.path.exists(neighboring_nodes_subdistance_path):
        with open(neighboring_nodes_subdistance_path, 'rb') as f:
            neighboring_nodes_subdistance = pickle.load(f)
        return neighboring_nodes_subdistance

    neighboring_nodes_subdistance = {}

    for id_r, id_ns in neighbor_region_dict.items():
        center_point_r = center_point_region_dict[id_r]
        
        for id_n in id_ns:
            neighboring_nodes_key = str(id_r) + '_' + str(id_n)
            neighboring_nodes_subdistance[neighboring_nodes_key] = {}
            center_point_n = center_point_region_dict[id_n]
            line = LineString([center_point_r, center_point_n])
            
            for index, row in region_polygon_gpd.iterrows():
                
                # get id_r's geometry data
                id = int(row['region_id'])
                geometry = row['geometry']
                
                if line.intersects(geometry):
                    neighboring_nodes_subdistance[neighboring_nodes_key][id] = {}
                    try:
                        sub_line = line.intersection(geometry)
                        sub_line_point_1 = Point(list(sub_line.coords)[0])
                        sub_line_point_2 = Point(list(sub_line.coords)[1])
                    
                    except:
                        sub_line = line.intersection(geometry)[0]
                        sub_line_point_1 = Point(list(sub_line.coords)[0])
                        sub_line_point_2 = Point(list(sub_line.coords)[1])
                    
                    # print(sub_line_point_1, sub_line_point_2)
                    
                    lon_1 = sub_line_point_1.y
                    lat_1 = sub_line_point_1.x
                    lon_2 = sub_line_point_2.y
                    lat_2 = sub_line_point_2.x
                    
                    transformer = Transformer.from_crs(4326, 2261)
                    sub_line_point_1 = transformer.transform(lon_1, lat_1)
                    sub_line_point_2 = transformer.transform(lon_2, lat_2)
                    
                    # for neighboring regions' distance
                    len_sub = math.sqrt(pow((sub_line_point_2[0] - sub_line_point_1[0]), 2) + pow((sub_line_point_2[1] - sub_line_point_1[1]), 2))
                    len_sub = len_sub / 5280.0
                    neighboring_nodes_subdistance[neighboring_nodes_key][id] = len_sub
            
    with open(neighboring_nodes_subdistance_path, 'wb') as f:
        pickle.dump(neighboring_nodes_subdistance, f)
        
    return neighboring_nodes_subdistance

# Get optimized_time_cost between neighboring regions update with for loop in one hour
def update_pair_weight(updated_region_speed_value_dict, time, neighboring_nodes_subdistance, walking_speed, Man_dist):
    updated_pair_weight_dict[time] = {}
    
    for neighboring_nodes_key, neighboring_nodes_id_subdistance in neighboring_nodes_subdistance.items():
        updated_pair_weight_dict[time][neighboring_nodes_key] = {}
        neighboring_nodes_time_cost = []
        neighboring_nodes_distance = []
        for id, subdistance in neighboring_nodes_id_subdistance.items():
            
            updated_speed_value = updated_region_speed_value_dict[time][id]['optimized_speed_value']
            if Man_dist == 'withMandist':
                subtime = subdistance * 1.25 / updated_speed_value
            elif Man_dist == 'withoutMandist':
                subtime = subdistance / updated_speed_value
            neighboring_nodes_time_cost.append(subtime)
            neighboring_nodes_distance.append(subdistance)
        
        neighboring_nodes_sum_time_cost = sum(neighboring_nodes_time_cost)
        neighboring_nodes_sum_distance = sum(neighboring_nodes_distance)
        
        updated_pair_weight_dict[time][neighboring_nodes_key]['updated_time_cost'] = neighboring_nodes_sum_time_cost
        updated_pair_weight_dict[time][neighboring_nodes_key]['total_distance'] = neighboring_nodes_sum_distance
        
        updated_pair_weight_dict[time][neighboring_nodes_key]['walking_time_cost'] = neighboring_nodes_sum_distance / walking_speed
        #updated_pair_weight_dict[time][neighboring_nodes_key]['walking_time_cost'] = 0
        
    return updated_pair_weight_dict

# Construct network by NetworkX static
def construct_network(shortest_paths_path, updated_pair_weight_dict):
    
    if os.path.exists(shortest_paths_path):
        with open(shortest_paths_path, 'rb') as f:
            shortest_paths_dict = pickle.load(f)
        return shortest_paths_dict
    
    G = nx.Graph()
    shortest_paths_dict = {}
    pair_weight_first_key = next(iter(updated_pair_weight_dict.items()))[0]
    pair_weight_first_value = next(iter(updated_pair_weight_dict.items()))[1]
    
    for neighboring_nodes, edges_weight in pair_weight_first_value.items():
        id_r, id_n = neighboring_nodes.split('_')
        weight = edges_weight['updated_time_cost']
        G.add_edge(id_r, id_n, weight=weight)
     
    shortest_paths_dict[pair_weight_first_key] = {}
    
    for id_r in ids:
        for id_s in ids:
            if id_r != id_s:
                edges = str(id_r) + '_' + str(id_s)
                shortest_paths_dict[pair_weight_first_key][edges] = {}
                # Update firstly at each hour
                path = nx.shortest_path(G, str(id_r), str(id_s), weight='weight')
                shortest_paths_dict[pair_weight_first_key][edges] = path
                
    pickle.dump(shortest_paths_dict, open(shortest_paths_path, "wb"))

    return shortest_paths_dict

# Update time cost in each epoch
def update_time_cost(updated_pair_weight_dict,shortest_paths_dict, time):
    
    updated_path_time_cost_dict[time] = {}
    shortest_paths_dict = next(iter(shortest_paths_dict.items()))[1]
    
    for edges, paths in shortest_paths_dict.items():

        # shortest_path_dict[time][str]
        updated_path_time_cost_dict[time][edges] = {}

        optimized_time_cost = 0.0
        shortest_path_distance = 0.0
        for _i in range(len(paths) - 1):
            neighboring_nodes_key = paths[_i] + "_" + paths[_i + 1]
            
            optimized_time_cost += updated_pair_weight_dict[time][neighboring_nodes_key]['updated_time_cost']
            shortest_path_distance += updated_pair_weight_dict[time][neighboring_nodes_key]['total_distance']
        
        updated_path_time_cost_dict[time][edges]['optimized_time_cost'] = optimized_time_cost
        updated_path_time_cost_dict[time][edges]['shortest_path_distance'] = shortest_path_distance
        # print('time',time,'path',edges,'optimized_time_cost',updated_path_time_cost_dict[time][edges]['optimized_time_cost'])
        
    return updated_path_time_cost_dict


# Get total number of pickup/dropoff in each region, Related to time
def initial_region_pickup_dropoff(input_trip_df_path,trip_dict_path):
    
    if os.path.exists(trip_dict_path):
        with open(trip_dict_path, 'rb') as f:
            trip_dict = pickle.load(f)
        return trip_dict
    
    trip_df = pd.read_csv(input_trip_df_path)

    trip_df['r'] = trip_df['r'].apply(lambda x: str(int(x)))
    trip_df['s'] = trip_df['s'].apply(lambda x: str(int(x)))
    
    # trip_df['total_number'] = trip_df['Pickup_number'] + trip_df['Dropoff_number']
    trip_df['datetime_hour_1'] = trip_df['datetime_min_5'].apply(lambda x: x[0:13] + ':00:00')
    
    # filter datetime
    trip_df = trip_df[trip_df.r != trip_df.s]
    trip_df['r_s'] = trip_df['r'].str.cat(trip_df['s'], sep='_')
    trip_df = trip_df.groupby(['r_s', 'datetime_hour_1']).sum()
    #trip_df['hat_Q_rs'] = trip_df['Dropoff_number'] * 9

    trip_df = trip_df.loc[:, ~trip_df.columns.str.contains('^Unnamed')]
    
    r_s_list = []
    for id_r in ids:
        for id_s in ids:
            if str(id_r) != str(id_s):
                r_s_list.append(str(id_r) + '_' + str(id_s))
    r_s_df = pd.DataFrame(r_s_list, columns=['r_s'])
    
    trip_df = trip_df.reset_index()
    trip_df = trip_df.groupby('datetime_hour_1')
    
    trip_dict = {}
    for group in trip_df:
        time = group[0]
        trip_in_hour_df = group[1]
        fill_value_trip_in_hour_df = r_s_df.merge(trip_in_hour_df, how='left', on='r_s')
        
        fill_value_trip_in_hour_df = fill_value_trip_in_hour_df.fillna(0)
        fill_value_trip_in_hour_df['datetime_hour_1'] = time
        
        fill_value_trip_in_hour_df = fill_value_trip_in_hour_df.set_index('r_s')
        input_trip_id_dict = fill_value_trip_in_hour_df.to_dict('index')
        
        trip_dict[time] = {}
        trip_dict[time] = input_trip_id_dict
    
    pickle.dump(trip_dict, open(trip_dict_path, "wb"))

    return trip_dict


# Generate matrix index: R_S & R_S_A
def prepare_matrix(id_o, id_d, neighbor_region_dict, prepare_matrix_dict_path):
    # test many settings
    # if os.path.exists(prepare_matrix_dict_path):
    #     with open(prepare_matrix_dict_path, 'rb') as f:
    #         prepare_matrix_dict = pickle.load(f)
    #         r_s = prepare_matrix_dict['r_s']
    #         r_s_a = prepare_matrix_dict['r_s_a']
    #         M0 = prepare_matrix_dict['M0']
    #         M1 = prepare_matrix_dict['M1']
    #         M2 = prepare_matrix_dict['M2']
    #     return r_s, r_s_a, M0, M1, M2
    
    prepare_matrix_dict = {}
    r_s = []
    r_s_a = []
    
    # just finance street
    for id_r in id_o:
        for id_s in id_d:
            if str(id_r) != str(id_s):
                r_s.append(str(id_r) + '_' + str(id_s))
                id_s_nbh = neighbor_region_dict[id_s]
                for id_a in id_s_nbh:
                    if str(id_r) != str(id_a):
                        r_s_a.append(str(id_r) + "_" + str(id_s) + '_' + str(id_a))
    
    # normal
    # for id_r in ids:
    #     for id_s in ids:
    #         if str(id_r) != str(id_s):
    #             r_s.append(str(id_r) + '_' + str(id_s))
    #             id_s_nbh = neighbor_region_dict[id_s]
    #             for id_a in id_s_nbh:
    #                 if str(id_r) != str(id_a):
    #                     r_s_a.append(str(id_r) + "_" + str(id_s) + '_' + str(id_a))
    
    prepare_matrix_dict['r_s'] = r_s
    prepare_matrix_dict['r_s_a'] = r_s_a
    # Generate 0-1 matrix: M0(len(r_s), len(r_s_a)) & M1(len(r),len(r_s)) & M2(len(r),len(r_s_a))
    M0_list = []
    for r_s_index in r_s:
        id_r1, id_s1 = r_s_index.split('_')
        for r_s_a_index in r_s_a:
            id_r2, id_s2, id_a2 = r_s_a_index.split('_')
            if id_r1 == id_r2 and id_s1 == id_s2:
                M0_list.append(1)
            else:
                M0_list.append(0)
    
    M0 = np.array(M0_list)
    # Shape of M: (len(r_s),len(r_s_a))
    M0 = np.reshape(M0, (len(r_s), len(r_s_a)))
    prepare_matrix_dict['M0'] = M0
    
    M1_list = []
    
    # normal
    for r in ids:
        for r_s_index in r_s:
            id_r1, id_s1 = r_s_index.split('_')
            if str(r) == id_s1:
                M1_list.append(1)
            else:
                M1_list.append(0)
    
    M1 = np.array(M1_list)
    # normal
    M1 = np.reshape(M1, (len(ids), len(r_s)))
    prepare_matrix_dict['M1'] = M1
    
    M2_list = []
    
    # normal
    for r in ids:
        for r_s_a_index in r_s_a:
            id_r2, id_s2, id_a2 = r_s_a_index.split('_')
            if str(r) == id_a2:
                M2_list.append(1)
            else:
                M2_list.append(0)
    
    M2 = np.array(M2_list)

    # normal
    M2 = np.reshape(M2, (len(ids), len(r_s_a)))
    prepare_matrix_dict['M2'] = M2
    
    with open(prepare_matrix_dict_path, 'wb') as f:
        pickle.dump(prepare_matrix_dict, f)
    
    return r_s, r_s_a, M0, M1, M2


def plot_figure(result_final_path, loop_times):
    result_optimization = pickle.load(open(result_final_path, "rb"))
    
    for time_i, time_value in result_optimization.items():
        plt.figure()
        ax = plt.gca()
        
        result_optimization_plot = pd.DataFrame([int(i) for i in range(0,loop_times)], columns=['iteration_epochs'])
        
        result_optimization_plot['objective_function'] = result_optimization_plot['iteration_epochs'].apply(lambda x: time_value[x]['objective_function'])
        result_optimization_plot['original_time_cost'] = [time_value[loop_times]['objective_function']] * (loop_times)
        result_optimization_plot.set_index('iteration_epochs')

        result_optimization_plot.plot(kind='line', x='iteration_epochs', y='objective_function', ax=ax)
        result_optimization_plot.plot(kind='line', x='iteration_epochs', y='original_time_cost', ax=ax)
        
        plt.ylabel('the value of objective function')
        plt.xlabel('iteration epochs')
        plt.title('objective function loss' + time_i)
        plt.savefig(fig_save_path + 'all_optimization_loss' + time_i + '.jpg')

def calculate_oprimized_rate(result_final_path,loop_times,result_rate_path):
    
    # if os.path.exists(result_rate_path):
    #     with open(result_rate_path, 'rb') as f:
    #         result_rate_df = pickle.load(f)
    #         print(result_rate_df)
    
    result_optimization = pickle.load(open(result_final_path, "rb"))
    
    result_rate_df = pd.DataFrame()
    
    i = 0
    for time_i, time_value in result_optimization.items():
        # print(time_i)
        result_rate_df.loc[i, ['time']] = time_i
        min_loop_i = min(time_value, key = lambda x: time_value[x]['objective_function'])

        min_value = time_value[min_loop_i]['objective_function']
        original_value = time_value[loop_times]['objective_function']
        optimized_rate = (original_value - min_value) / original_value

        result_rate_df.loc[i, ['min_value']] = min_value
        result_rate_df.loc[i, ['original_value']] = original_value
        result_rate_df.loc[i, ['optimized_rate']] = optimized_rate
        
        i += 1
    
    print(result_rate_df)
    with open(result_rate_path, 'wb') as f:
        pickle.dump(result_rate_df, f)
        
# Update frequency : each time by one hour
def generate_by_hour(trip_dict, time, ids, r_s, M1):

    # normal
    Delta_A_value = np.zeros((len(ids), 1))
    # Generate D
    # D_list = []
    
    # # normal
    # for id_s in ids:
    #     D_list_s = []
    #     for id_r in ids:
    #         if id_r != id_s:
    #             r_s_index = str(id_r) + '_' + str(id_s)
    #             D_list_s.append(trip_dict[time][r_s_index]['Dropoff_number'])
    #     D_list_r = sum(D_list_s)
    #     D_list.append(D_list_r)
    #
    # D = np.array(D_list)
    # D = np.reshape(D, (len(ids), 1))
    
    # Generate Q and Q_hat
    Q_list = []
    Q_hat_list = []
    for r_s_index in r_s:
        Q_list.append(trip_dict[time][r_s_index]['Dropoff_number'])
        # Q_hat_list.append(trip_dict[time][r_s_index]['hat_Q_rs'])
    
    Q = np.array(Q_list)
    
    # Shape of Q: (len(r_s),1))
    Q = np.reshape(Q, (len(r_s), 1))
    
    D = M1 @ Q
    
    # Test
    # np.random.seed(0)
    # Q = np.random.rand(len(r_s), 1) * 100
    # Q_hat = np.array(Q_hat_list)
    # Q_hat = Q * 20
    
    Q_hat = Q
    # Shape of Q_hat: (len(r_s),1)
    Q_hat = np.reshape(Q_hat, (len(r_s), 1))
    return Delta_A_value, D, Q, Q_hat


# Update parameters in each epoch
def generate_by_epoch(Delta_A_value, region_speed_dict, time, region2theta_dict, neighboring_nodes_subdistance,
                      shortest_paths_path, r_s, r_s_a, __i):
    # print("============begin update parameters" + str(__i) + "============")
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    # print("Current Time =", current_time)

    updated_region_speed_value_dict = update_speed(region_speed_dict, time, Delta_A_value, region2theta_dict, theta_enlarge)
    
    updated_pair_weight_dict = update_pair_weight(updated_region_speed_value_dict, time, neighboring_nodes_subdistance, walking_speed, Man_dist)
    
    # Get fixed shortest path
    shortest_paths_dict = construct_network(shortest_paths_path, updated_pair_weight_dict)
    
    updated_path_time_cost_dict = update_time_cost(updated_pair_weight_dict, shortest_paths_dict, time)
    
    # Generate C
    C_list = []
    T_list = []
    for r_s_a_index in r_s_a:
        r2, s2, a2 = r_s_a_index.split('_')
        
        t_index = r2 + '_' + a2
        w_index = a2 + '_' + s2
        
        C_list.append(
            updated_path_time_cost_dict[time][t_index]['optimized_time_cost'] + updated_pair_weight_dict[time][w_index][
                'walking_time_cost'])
        
        # print('Walking time cost:',updated_pair_weight_dict[time][w_index]['walking_time_cost'])
    
    for r_s_index in r_s:
        r1, s1 = r_s_index.split('_')
        t_index = r1 + '_' + s1
        T_list.append(updated_path_time_cost_dict[time][t_index]['optimized_time_cost'])
    
    C = np.array(C_list)
    # Shape of C: (len(r_s_a),1)
    C = np.reshape(C, (len(r_s_a), 1))
    
    T = np.array(T_list)
    # Shape of T: (len(r_s),1)
    T = np.reshape(T, (len(r_s), 1))
    
    # print("============begin probelm solver" + str(__i) + "============")
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    # print("Current Time =", current_time)
    return C, T