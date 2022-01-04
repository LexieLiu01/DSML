# encoding: utf-8
import numpy as np
import random
from sklearn.decomposition import NMF
import pickle
import pandas as pd

from sensing import *
from datetime import date, datetime, timedelta
import json

start = datetime(2019, 2, 1)
end = datetime(2020, 6, 30)

input_trip_and_speed = './data_set/NYC_speed_and_trip/trip_and_speed.csv'
input_precip_path = './data_set/NYC_precip/hourlyprecip.csv'
neighbor_region_path = './data_set/NYC_geo/neighbor_region.json'

trip_speed_precip_X_M_Nei_path = './data_set/Prepared_matrix_completion_data/trip_speed_precip_X_M_Nei_more_parameters.csv'

neighbor_region_file = open(neighbor_region_path, "r")
neighbor_region = json.load(neighbor_region_file)

# get time sequence from start to the end
def datetime_range(start, end, delta):
    current = start
    if not isinstance(delta, timedelta):
        delta = timedelta(**delta)
    while current < end:
        yield current
        current += delta


# fill up value of trip and speed.
def fill_value(input_trip_and_speed):
    input_trip_and_speed_df = pd.read_csv(input_trip_and_speed)
    # print(input_trip_and_speed_df)
    # just for testing in small dataset
    # input_trip_and_speed_df = input_trip_and_speed_df[0:100000]
    
    input_trip_and_speed_df = input_trip_and_speed_df.sort_values(by='datetime_min_5')
    input_trip_and_speed_df = input_trip_and_speed_df.groupby('region_id')
    
    # generate time sequence dataframe in each 5 minutes
    time_seq = []
    for dt in datetime_range(start, end, {'minutes': 5}):
        time_seq.append(str(dt)[0:16])
    
    time_seq_df = pd.DataFrame(time_seq)
    time_seq_df.columns = ['datetime_min_5']
    
    # with time sequence
    fill_value_region_list = []
    
    speed_region_list = []
    
    for group in input_trip_and_speed_df:
        region_id = group[0]
        group_df = group[1]
        
        # time_seq_df is complete time sequence; group_df is real data in life( has missing data).
        fill_value_region_df = time_seq_df.merge(group_df, how='left', on='datetime_min_5')
        
        fill_value_region_df['region_id'] = fill_value_region_df['region_id'].fillna(value=region_id)
        
        # fill_value_region_df['relative_speed_2'] = fill_value_region_df['relative_speed_2'].fillna(0)
        # fill_value_region_df['relative_speed_2'] = fill_value_region_df['relative_speed_2'].fillna(method='ffill')
        
        speed_region_list.append(fill_value_region_df['relative_speed_2'])
        
        fill_value_region_df['total_number'] = fill_value_region_df['total_number'].fillna(0)
        
        fill_value_region_list.append(fill_value_region_df)
    
    speed_region_join_df = pd.concat(speed_region_list, ignore_index=True, axis=1)
    
    fill_value_trip_and_speed_df = pd.concat(fill_value_region_list, ignore_index=True)
    fill_value_trip_and_speed_df = fill_value_trip_and_speed_df[['region_id', 'datetime_min_5', 'total_number', 'relative_speed_2']]
    #fill_value_trip_and_speed_df.to_csv(fill_value_trip_and_speed)
    print(fill_value_trip_and_speed_df)
    
    
    return fill_value_trip_and_speed_df, speed_region_join_df

def impute_na_with_SI(v):
    v_mat = v.copy()
    clf = vk_sensing("KNN")
    
    clf.CVfit(v_mat)
    new_mat = clf.fit_transform(v_mat)
    assert (np.sum(np.isnan(new_mat)) == 0)
    return new_mat


def fancyimpute(speed_region_join_df):
    v = speed_region_join_df.values
    
    #print(v[:10])
    #print(np.any(v == 0))
    #print(np.isnan(np.min(v)))
    
    v = v.reshape(-1, 288 * 63)
    
    new_v = impute_na_with_SI(v)
    new_v = new_v.reshape(-1, 63)
    
    #print(new_v[:10])
    #print(np.any(new_v == 0))
    #print(np.isnan(np.min(new_v)))
    
    return new_v


# input is new_v shape()
def get_axis_zero_speed(mat):
    speed_region_id_list = []
    for id in range(63):
        mat_region_id = mat[:, id]
        speed_region_id = pd.DataFrame(mat_region_id)
        speed_region_id_list.append(speed_region_id)
    
    speed_region_id_df = pd.concat(speed_region_id_list, axis=0, ignore_index=True)
    # print(speed_region_id_df)
    
    return speed_region_id_df


def rebuild_trip_speed(fill_value_trip_and_speed_df, speed_region_id_df):
    # print((fill_value_trip_and_speed_df == 0).astype(int).sum(axis=0))
    # print('=====================================')
    fill_value_trip_and_speed_df['relative_speed_2'] = speed_region_id_df
    # print((fill_value_trip_and_speed_df == 0).astype(int).sum(axis=0))
    # print('=====================================')
    # print(fill_value_trip_and_speed_df)
    return fill_value_trip_and_speed_df


def select_speed(fill_value_trip_and_speed_df):
    regionid2groups = {}
    
    fill_value_region_groups = fill_value_trip_and_speed_df.groupby('region_id')
    for group in fill_value_region_groups:
        region_id = group[0]
        group_df = group[1]
        regionid2groups[region_id] = group_df
    
    training_speed_data = []
    
    for region_id, group_df in regionid2groups.items():
        
        len_of_df = group_df.shape[0]
        for i in range(10, len_of_df):
            row = group_df.iloc[i]
            
            region_id = row['region_id']
            
            datetime_min_5 = row['datetime_min_5']
            datetime_hour = datetime_min_5[0:13]
            relative_speed = row['relative_speed_2']
            total_number = row['total_number']
            
            # select data at current weekday and hour
            training_sample = [region_id, datetime_min_5, relative_speed, total_number]
            X = []
            X_other = []
            M = []
            
            # get previous 10 units' speed data
            for _j in range(1, 11):
                X_datas = group_df.iloc[i - _j]
                X_relative_speed = X_datas['relative_speed_2']
                X.append(X_relative_speed)
                
                M_pickup_dropoff = X_datas['total_number']
                M.append(M_pickup_dropoff)
                
                other_average_speed = 0
                other_region_number = 0
                
                other_regions = neighbor_region[str(int(region_id))]
                
                for other_region_id in other_regions:
                    other_region_group_df = regionid2groups[float(other_region_id)]
                    X_other_datas = other_region_group_df.iloc[i - _j]
                    X_other_relative_speed = X_other_datas['relative_speed_2']
                    other_average_speed += X_other_relative_speed
                    other_region_number += 1
                
                other_average_speed = other_average_speed / other_region_number
                X_other.append(other_average_speed)
            
            X = X + X_other
            #M = M
            
            training_sample.append(X)
            training_sample.append(M)
            # print(training_sample)
            training_speed_data.append(training_sample)
    
    training_speed_data_df = pd.DataFrame(training_speed_data)
    training_speed_data_df.columns = ["region_id", "datetime_min_5", "relative_speed", "total_number", "X", "M"]
    # print(training_speed_data_df)
    
    # get w
    
    w_df = pd.read_csv(input_precip_path, low_memory=False)
    w_df = pd.DataFrame(w_df, columns=['valid', 'precip_in'])
    w_df['datetime_hour'] = w_df['valid'].apply(lambda x: x[0:13])
    
    training_speed_data_df['datetime_hour'] = training_speed_data_df['datetime_min_5'].apply(lambda x: x[0:13])
    training_data_df = training_speed_data_df.merge(w_df, how='left', on='datetime_hour')
    training_data_df['precip_in'] = training_data_df['precip_in'].fillna(0)
    training_data_df = training_data_df[['region_id', 'datetime_min_5', 'relative_speed', 'total_number', 'X','M', 'datetime_hour', 'precip_in']]
    
    # print(training_data_df)
    
    training_data_df.to_csv(trip_speed_precip_X_M_Nei_path, index=False)


if __name__ == '__main__':
    fill_value_trip_and_speed_df, speed_region_join_df = fill_value(input_trip_and_speed)
    new_v = fancyimpute(speed_region_join_df)
    speed_region_id_df = get_axis_zero_speed(new_v)
    
    fill_value_trip_and_speed_df = rebuild_trip_speed(fill_value_trip_and_speed_df, speed_region_id_df)
    select_speed(fill_value_trip_and_speed_df)