import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

ids = [4, 12, 13, 24, 43, 45, 48, 50, 68, 75, 79, 87, 88, 90, 100, 107, 113, 114, 125, 137, 140, 141, 142, 143, 144,148, 151, 158, 161, 162, 163, 164, 170, 186, 209, 211, 224, 229, 230, 231, 232, 233, 234, 236, 237, 238, 239,246, 249, 261, 262, 263]
region_speed_dict_path = './Shortest_path/region_speed_initialization.pickle'
theta_path = './model_dml/theta/theta_DSML_0_6_16_17_18_19.pickle'
trip_dict_path = './Shortest_path/trip_dict_initialization.pickle'

def get_average_speed(region_speed_dict_path,ids):
    
    with open(region_speed_dict_path, 'rb') as f:
        region_speed_dict = pickle.load(f)

    ave_region_speed_df = pd.DataFrame(ids, columns=['region_id'])
    ave_speed_list = []
    
    sum_speed = 0
    time_slot_cnt = 0
    for id in ids:
        for time, region_speed in region_speed_dict.items():
            sum_speed += region_speed[id]['speed_value']
            # unless you aren't sure that grade would be a int, in which case add exception
            time_slot_cnt += 1
        ave_speed = sum_speed / time_slot_cnt
        ave_speed_list.append(ave_speed)
        
    ave_region_speed_df['ave_speed'] = pd.DataFrame(ave_speed_list)
    
    return ave_region_speed_df

# Get theta from DSML
def get_theta(theta_path, ids):
    with open(theta_path, 'rb') as fr:
        theta_dict = pickle.load(fr)

    region2theta_df = pd.DataFrame(ids, columns=['region_id'])
    region2theta_list = []
    for id in ids:
        region2theta_list.append(-theta_dict[id]['results_params'][1])
    region2theta_df['theta'] = pd.DataFrame(region2theta_list)
    
    return region2theta_df

# Initialize time
start = datetime(2019, 2, 2)
end = datetime(2019, 2, 3)
time_intervals = 60
hours = ['16', '17', '18', '19']
#weekdays = ['2', '3', '4']
weekdays = ['6', '0']

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

def get_Demand(trip_dict_path, ids, time_seq):

    with open(trip_dict_path, 'rb') as f:
        trip_dict = pickle.load(f)

    Demand_df = pd.DataFrame(ids, columns=['region_id'])
    Demand_list = []
   
    for time in time_seq:
        D_list = []
        for id_s in ids:
            D_list_s = []
            for id_r in ids:
                if id_r != id_s:
                    r_s_index = str(id_r) + '_' + str(id_s)
                    D_list_s.append(trip_dict[time][r_s_index]['Dropoff_number'])
            D_list_r = sum(D_list_s)
            D_list.append(D_list_r)
        Demand_list.append(D_list)
    Demand_arr = np.array(Demand_list)
    Demand_arr = np.reshape(Demand_arr, (-1, len(ids)))
    Demand_arr_ave = np.mean(Demand_arr, axis = 0)
    Demand_df['ave_demand'] = pd.DataFrame(Demand_arr_ave)
    print('Demand_df',Demand_df)
    return Demand_df
    
def plot_fig(ave_region_speed_df, region2theta_df, Demand_df):
    fig, axes = plt.subplots(nrows=3, ncols=1)
    ave_region_speed_df.plot.bar(x="region_id", y="ave_speed", ax=axes[0])
    region2theta_df.plot.bar(x="region_id", y="theta", ax=axes[1])
    Demand_df.plot.bar( x="region_id", y="ave_demand", ax=axes[2])
    plt.show()
    
if __name__ == '__main__':
    ave_region_speed_df = get_average_speed(region_speed_dict_path,ids)
    region2theta_df = get_theta(theta_path, ids)
    
    # Generate time sequence
    time_seq = generate_time_sequence(time_intervals)

    Demand_df = get_Demand(trip_dict_path, ids, time_seq)
    # print(ave_region_speed_df)
    # print(region2theta_df)
    plot_fig(ave_region_speed_df, region2theta_df, Demand_df)