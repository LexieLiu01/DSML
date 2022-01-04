# caculate the total number of pick-up/ drop-offs in each 5 minute and in different regions

import pandas as pd
import geopandas as gpd
import os

raw_data_path_pre = './data_set/NYC_triprecord/'

input_region_geo_Manhanttan_path = './data_set/NYC_geo/NYC_Taxi_Zones_Manhattan.geojson'
# list file under path
input_path_foryellow = raw_data_path_pre + 'raw_data/01Yellow'
input_path_forgreen = raw_data_path_pre + 'raw_data/02Green'
input_path_forfhv = raw_data_path_pre + 'raw_data/03FHV'
input_path_forfhvhv = raw_data_path_pre + 'raw_data/04FHVHV'

output_path_foryellow = raw_data_path_pre + 'prepared_data/Yellow_for_optimization.csv'
output_path_forgreen = raw_data_path_pre + 'prepared_data/Green_for_optimization.csv'
output_path_forfhv = raw_data_path_pre + 'prepared_data/FHV_for_optimization.csv'
output_path_forfhvhv = raw_data_path_pre + 'prepared_data/FHVHV_for_optimization.csv'

output_path_forall = raw_data_path_pre + 'prepared_data/trip_all_for_optimization.csv'

# classify each 5 minutes
def prepare_for_5minutes(x):
    x_minute = x.minute
    x_minute = int(x_minute)
    a = x_minute // 5
    x_minute = 5 * a
    x = x.replace(minute=x_minute)
    return x


# select region_id of Manhanttan
def select_manhatten_regionid(input_region_geo_Manhanttan_path):
    region_Manhatten_gpd = gpd.read_file(input_region_geo_Manhanttan_path)
    Manhattan_region_id = []
    Manhattan_region_id = region_Manhatten_gpd['region_id'].values.tolist()
    Manhattan_region_id = [int(id) for id in Manhattan_region_id]
    
    return Manhattan_region_id


# for green taxi
def count_5minutes_forgreen(subfile_path, input_region_geo_Manhanttan_path):
    df_green = pd.read_csv(subfile_path, low_memory=False)
    # df_green = df_green[:1000]
    df_green['lpep_pickup_datetime'] = pd.to_datetime(df_green['lpep_pickup_datetime'])
    df_green['lpep_dropoff_datetime'] = pd.to_datetime(df_green['lpep_dropoff_datetime'])
    
    df_green['lpep_pickup_datetime_year'] = df_green['lpep_pickup_datetime'].apply(lambda x: x.strftime('%Y'))
    df_green['lpep_dropoff_datetime_year'] = df_green['lpep_dropoff_datetime'].apply(lambda x: x.strftime('%Y'))
    
    df_green['lpep_pickup_datetime_min_5'] = df_green['lpep_pickup_datetime'].apply(
        lambda x: prepare_for_5minutes(x).strftime('%Y-%m-%d %H:%M'))
    df_green['lpep_dropoff_datetime_min_5'] = df_green['lpep_dropoff_datetime'].apply(
        lambda x: prepare_for_5minutes(x).strftime('%Y-%m-%d %H:%M'))
    
    df_green = df_green[(df_green['PULocationID'] != '') & (df_green['DOLocationID'] != '')]
    df_green = df_green[
        (df_green['lpep_pickup_datetime_year'] == '2020') | (df_green['lpep_pickup_datetime_year'] == '2019') | (
                df_green['lpep_dropoff_datetime_year'] == '2020') | (
                df_green['lpep_dropoff_datetime_year'] == '2019')]
    
    # classify region_id in manhattan
    Manhattan_region_id = []
    Manhattan_region_id = select_manhatten_regionid(input_region_geo_Manhanttan_path)
    
    df_green = df_green[
        (df_green['PULocationID'].isin(Manhattan_region_id)) & (df_green['DOLocationID'].isin(Manhattan_region_id))]
    
    # get the number of dropoff
    df_green_dropoff = df_green.groupby(['lpep_pickup_datetime_min_5', 'PULocationID', 'DOLocationID']).count()[
        'lpep_pickup_datetime']
    df_green_dropoff = df_green_dropoff.reset_index()
    df_green_dropoff.columns = ['datetime_min_5', 'r', 's', 'Dropoff_number']
    Con_df_green = df_green_dropoff.fillna(0)
    return Con_df_green


# for yellow taxi
def count_5minutes_foryellow(subfile_path, input_region_geo_Manhanttan_path):
    df_yellow = pd.read_csv(subfile_path, low_memory=False)
    # df_yellow = df_yellow[:1000]
    df_yellow['lpep_pickup_datetime'] = pd.to_datetime(df_yellow['tpep_pickup_datetime'])
    df_yellow['lpep_dropoff_datetime'] = pd.to_datetime(df_yellow['tpep_dropoff_datetime'])
    
    df_yellow['lpep_pickup_datetime_year'] = df_yellow['lpep_pickup_datetime'].apply(lambda x: x.strftime('%Y'))
    df_yellow['lpep_dropoff_datetime_year'] = df_yellow['lpep_dropoff_datetime'].apply(lambda x: x.strftime('%Y'))
    
    df_yellow['lpep_pickup_datetime_min_5'] = df_yellow['lpep_pickup_datetime'].apply(
        lambda x: prepare_for_5minutes(x).strftime('%Y-%m-%d %H:%M'))
    df_yellow['lpep_dropoff_datetime_min_5'] = df_yellow['lpep_dropoff_datetime'].apply(
        lambda x: prepare_for_5minutes(x).strftime('%Y-%m-%d %H:%M'))
    
    df_yellow = df_yellow[(df_yellow['PULocationID'] != '') & (df_yellow['DOLocationID'] != '')]
    df_yellow = df_yellow[
        (df_yellow['lpep_pickup_datetime_year'] == '2020') | (df_yellow['lpep_pickup_datetime_year'] == '2019') | (
                df_yellow['lpep_dropoff_datetime_year'] == '2020') | (
                df_yellow['lpep_dropoff_datetime_year'] == '2019')]
    
    # classify region_id in manhattan
    # Manhattan_region_id = []
    Manhattan_region_id = select_manhatten_regionid(input_region_geo_Manhanttan_path)
    df_yellow = df_yellow.loc[
        (df_yellow['PULocationID'].isin(Manhattan_region_id)) & (df_yellow['DOLocationID'].isin(Manhattan_region_id))]
    
    df_yellow_dropoff = df_yellow.groupby(['lpep_pickup_datetime_min_5', 'PULocationID', 'DOLocationID']).count()[
        'lpep_pickup_datetime']
    df_yellow_dropoff = df_yellow_dropoff.reset_index()
    df_yellow_dropoff.columns = ['datetime_min_5', 'r', 's', 'Dropoff_number']
    Con_df_yellow = df_yellow_dropoff.fillna(0)
    
    return Con_df_yellow


# for fhv vehicles
def count_5minutes_forfhv(subfile_path, input_region_geo_Manhanttan_path):
    df_fhv = pd.read_csv(subfile_path, low_memory=False)
    # df_fhv = df_fhv[:1000]
    df_fhv['lpep_pickup_datetime_year'] = df_fhv['pickup_datetime'].apply(lambda x: x[0:4])
    df_fhv['lpep_dropoff_datetime_year'] = df_fhv['dropoff_datetime'].apply(lambda x: x[0:4])
    
    df_fhv['lpep_pickup_datetime'] = pd.to_datetime(df_fhv['pickup_datetime'])
    df_fhv['lpep_dropoff_datetime'] = pd.to_datetime(df_fhv['pickup_datetime'])
    
    df_fhv['lpep_pickup_datetime_min_5'] = df_fhv['lpep_pickup_datetime'].apply(
        lambda x: prepare_for_5minutes(x).strftime('%Y-%m-%d %H:%M'))
    df_fhv['lpep_dropoff_datetime_min_5'] = df_fhv['lpep_dropoff_datetime'].apply(
        lambda x: prepare_for_5minutes(x).strftime('%Y-%m-%d %H:%M'))
    
    df_fhv = df_fhv[(df_fhv['PULocationID'] != '') & (df_fhv['DOLocationID'] != '')]
    df_fhv = df_fhv[
        (df_fhv['lpep_pickup_datetime_year'] == '2020') | (df_fhv['lpep_pickup_datetime_year'] == '2019') | (
                df_fhv['lpep_dropoff_datetime_year'] == '2020') | (df_fhv['lpep_dropoff_datetime_year'] == '2019')]
    
    # classify region_id in manhattan
    Manhattan_region_id = []
    Manhattan_region_id = select_manhatten_regionid(input_region_geo_Manhanttan_path)
    df_fhv = df_fhv.loc[
        (df_fhv['PULocationID'].isin(Manhattan_region_id)) & (df_fhv['DOLocationID'].isin(Manhattan_region_id))]
    
    df_fhv_dropoff = df_fhv.groupby(['lpep_pickup_datetime_min_5', 'PULocationID', 'DOLocationID']).count()[
        'lpep_pickup_datetime']
    df_fhv_dropoff = df_fhv_dropoff.reset_index()
    df_fhv_dropoff.columns = ['datetime_min_5', 'r', 's', 'Dropoff_number']
    Con_df_fhv = df_fhv_dropoff.fillna(0)
    
    return Con_df_fhv


# for fhvhv vehicles
def count_5minutes_forfhvhv(subfile_path, input_region_geo_Manhanttan_path):
    df_fhvhv = pd.read_csv(subfile_path, low_memory=False)
    # df_fhvhv = df_fhvhv[:1000]
    df_fhvhv['lpep_pickup_datetime_year'] = df_fhvhv['pickup_datetime'].apply(lambda x: x[0:4])
    df_fhvhv['lpep_dropoff_datetime_year'] = df_fhvhv['dropoff_datetime'].apply(lambda x: x[0:4])
    
    df_fhvhv['lpep_pickup_datetime'] = pd.to_datetime(df_fhvhv['pickup_datetime'])
    df_fhvhv['lpep_dropoff_datetime'] = pd.to_datetime(df_fhvhv['pickup_datetime'])
    
    df_fhvhv['lpep_pickup_datetime_min_5'] = df_fhvhv['lpep_pickup_datetime'].apply(
        lambda x: prepare_for_5minutes(x).strftime('%Y-%m-%d %H:%M'))
    df_fhvhv['lpep_dropoff_datetime_min_5'] = df_fhvhv['lpep_dropoff_datetime'].apply(
        lambda x: prepare_for_5minutes(x).strftime('%Y-%m-%d %H:%M'))
    
    df_fhvhv = df_fhvhv[(df_fhvhv['PULocationID'] != '') & (df_fhvhv['DOLocationID'] != '')]
    df_fhvhv = df_fhvhv[
        (df_fhvhv['lpep_pickup_datetime_year'] == '2020') | (df_fhvhv['lpep_pickup_datetime_year'] == '2019') | (
                df_fhvhv['lpep_dropoff_datetime_year'] == '2020') | (
                df_fhvhv['lpep_dropoff_datetime_year'] == '2019')]
    
    # classify region_id in manhattan
    Manhattan_region_id = []
    Manhattan_region_id = select_manhatten_regionid(input_region_geo_Manhanttan_path)
    df_fhvhv = df_fhvhv.loc[
        (df_fhvhv['PULocationID'].isin(Manhattan_region_id)) & (df_fhvhv['DOLocationID'].isin(Manhattan_region_id))]
    
    df_fhvhv_dropoff = df_fhvhv.groupby(['lpep_pickup_datetime_min_5', 'PULocationID', 'DOLocationID']).count()[
        'lpep_pickup_datetime']
    df_fhvhv_dropoff = df_fhvhv_dropoff.reset_index()
    df_fhvhv_dropoff.columns = ['datetime_min_5', 'r', 's', 'Dropoff_number']
    Con_df_fhvhv = df_fhvhv_dropoff.fillna(0)
    
    return Con_df_fhvhv


def get_green(inputpath, output_path):
    filenames = os.listdir(inputpath)
    df_green = pd.DataFrame()
    for file in filenames:
        if file == '.DS_Store':
            continue
        subfile_path = inputpath + '/' + file
        Con_df_green = count_5minutes_forgreen(subfile_path, input_region_geo_Manhanttan_path)
        df_green = df_green.append(Con_df_green, ignore_index=True)
    
    df_green.to_csv(output_path)
    
    return df_green


def get_yellow(inputpath, output_path):
    filenames = os.listdir(inputpath)
    df_yellow = pd.DataFrame()
    for file in filenames:
        if file == '.DS_Store':
            continue
        subfile_path = inputpath + '/' + file
        Con_df_yellow = count_5minutes_foryellow(subfile_path, input_region_geo_Manhanttan_path)
        df_yellow = df_yellow.append(Con_df_yellow, ignore_index=True)
    
    df_yellow.to_csv(output_path)
    return df_yellow


def get_fhv(inputpath, output_path):
    filenames = os.listdir(inputpath)
    df_fhv = pd.DataFrame()
    for file in filenames:
        if file == '.DS_Store':
            continue
        subfile_path = inputpath + '/' + file
        Con_df_fhv = count_5minutes_forfhv(subfile_path, input_region_geo_Manhanttan_path)
        df_fhv = df_fhv.append(Con_df_fhv, ignore_index=True)
    
    df_fhv.to_csv(output_path)
    return df_fhv


def get_fhvhv(inputpath, output_path):
    filenames = os.listdir(inputpath)
    df_fhvhv = pd.DataFrame()
    for file in filenames:
        if file == '.DS_Store':
            continue
        subfile_path = inputpath + '/' + file
        Con_df_fhvhv = count_5minutes_forfhv(subfile_path, input_region_geo_Manhanttan_path)
        df_fhvhv = df_fhvhv.append(Con_df_fhvhv, ignore_index=True)
    
    df_fhvhv.to_csv(output_path)
    return df_fhvhv


# combine all pick-ups / dropoffs of yellow/green/fhv/fhfhv vehicles
def combine_all_trips(df_green, df_yellow, df_fhv, df_fhvhv):
    df_con = pd.concat([df_yellow, df_green, df_fhv, df_fhvhv], axis=0)
    df_con = df_con.loc[:, ~df_con.columns.str.contains('^Unnamed')]
    df_con.to_csv(output_path_forall)


if __name__ == '__main__':
    df_green = get_green(input_path_forgreen, output_path_forgreen)
    df_yellow = get_yellow(input_path_foryellow, output_path_foryellow)
    df_fhv = get_fhv(input_path_forfhv, output_path_forfhv)
    df_fhvhv = get_fhvhv(input_path_forfhvhv, output_path_forfhvhv)
    
    combine_all_trips(df_green, df_yellow, df_fhv, df_fhvhv)