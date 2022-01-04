import pandas as pd
import geopandas as gpd
import json

# cut whole data into several datasets to improve the running speed and save running memory

# raw speed data in 2019
raw_data_path_pre = './data_set/NYC_speed/raw_data/'
input_speed_path_2019 = raw_data_path_pre + 'nyc_2019_5min_0101_1231.csv'

month_split_2019_04 = ['2019-02', '2019-03', '2019-04']
month_split_2019_08 = ['2019-05', '2019-06', '2019-07', '2019-08']
month_split_2019_12 = ['2019-09', '2019-10', '2019-11', '2019-12']
input_speed_path_2019_04 = raw_data_path_pre + "nyc_2019_5min_0201_0430.csv"
input_speed_path_2019_08 = raw_data_path_pre +"nyc_2019_5min_0501_0831.csv"
input_speed_path_2019_12 = raw_data_path_pre +"nyc_2019_5min_0901_1231.csv"

# raw speed data of in 2020
input_speed_path_raw04_2020 = raw_data_path_pre +'nyc_2020_5min_0101_0430.csv'
input_speed_path_raw11_2020 = raw_data_path_pre +'nyc_2020_5min_0501_1130.csv'
input_speed_path_2020 = raw_data_path_pre + 'nyc_2020_5min_0101_1130.csv'

month_split_2020_02 = ['2020-01','2020-02']
month_split_2020_04 = ['2020-03','2020-04']
month_split_2020_06 = ['2020-05','2020-06']
input_speed_path_2020_02 = raw_data_path_pre +'nyc_2020_5min_0101_0229.csv'
input_speed_path_2020_04 = raw_data_path_pre +'nyc_2020_5min_0301_0430.csv'
input_speed_path_2020_06 = raw_data_path_pre +'nyc_2020_5min_0501_0630.csv'

# all the raw speed data
output_speed_path_2019_04 = raw_data_path_pre + 'nyc_2019_5min_0201_0430_prepared.csv'
output_speed_path_2019_08 = raw_data_path_pre + 'nyc_2019_5min_0501_0831_prepared.csv'
output_speed_path_2019_12 = raw_data_path_pre + 'nyc_2019_5min_0901_1231_prepared.csv'
output_speed_path_2020_02 = raw_data_path_pre + 'nyc_2020_5min_0101_0229_prepared.csv'
output_speed_path_2020_04 = raw_data_path_pre + 'nyc_2020_5min_0301_0430_prepared.csv'
output_speed_path_2020_06 = raw_data_path_pre + 'nyc_2020_5min_0501_0630_prepared.csv'

output_speed_path = './data_set/NYC_speed/prepared_data/nyc_5min_all_prepared.csv'

# geo data contains tmc code and linestring geometry given by Wei
input_tmc_path = './data_set/NYC_geo/ny.json'
input_tmc_geo_path = './data_set/NYC_geo/ny.geojson'

# geo_data includes region id and multiploygon geometry information
input_region_geo_path = './data_set/NYC_geo/NYC_Taxi_Zones.geojson'
input_region_geo_Manhanttan_path = './data_set/NYC_geo/NYC_Taxi_Zones_Manhattan.geojson'

region2tmc_path = './data_set/NYC_geo/region2tmc.json'

def combine_speed_2020(input_speed_path_raw04_2020,input_speed_path_raw11_2020,input_speed_path_2020):
    input_speed_path_2020_04_df = pd.read_csv(input_speed_path_raw04_2020)
    input_speed_path_2020_11_df = pd.read_csv(input_speed_path_raw11_2020)

    input_speed_path_2020_df = pd.concat([input_speed_path_2020_04_df, input_speed_path_2020_11_df], ignore_index=True)
    
    input_speed_path_2020_df.to_csv(input_speed_path_2020)

# get region2tmc
def correspond_tmccode_to_region(input_tmc_path, input_tmc_geo_path, input_region_geo_path, input_region_geo_Manhanttan_path, region2tmc_path):

    tmc_obj = json.load(open(input_tmc_path, encoding="utf-8"))
    tmc_geo = tmc_obj['geojson']
    json.dump(tmc_geo, open(input_tmc_geo_path, "w"))
    
    tmc_gpd = gpd.read_file(input_tmc_geo_path)
    tmc_gpd = tmc_gpd.rename(columns={'id': 'tmc_code'})
    
    # get region multiploygon
    region_gpd = gpd.read_file(input_region_geo_path)
    region_gpd = region_gpd[region_gpd["borough"] == "Manhattan"]
    region_gpd = region_gpd.loc[:, ['location_id', 'borough', 'geometry']]
    region_gpd = region_gpd.rename(columns={'location_id': 'region_id'})
    region_gpd = region_gpd.reset_index(drop=True)
    
    with open(input_region_geo_Manhanttan_path, 'w') as f:
        f.write(region_gpd.to_json())
    
    # sjoin region id and tmc code by geometry
    region_with_tmc = gpd.sjoin(region_gpd, tmc_gpd, how="inner", op='intersects')
    
    region_with_tmc['region_id'] = region_with_tmc['region_id'].apply(int)
    region_with_tmc = region_with_tmc.sort_values(by=['region_id'], ascending=True)
    region_with_tmc = region_with_tmc.loc[:, ['region_id', 'tmc_code']]
    
    region_with_tmc = pd.DataFrame(region_with_tmc)
    region_with_tmc.reset_index(inplace=True)
    
    with open(region2tmc_path, 'w') as f:
        f.write(region_with_tmc.to_json())

# cutting the whole data set into several files
def separate_data(input_path, month_split, output_path):
    speed_separate_df = pd.read_csv(input_path)
    
    #speed_separate_df = speed_separate_df[:100000]
    
    speed_separate_df['measurement_tstamp_month'] = speed_separate_df['measurement_tstamp'].apply(lambda x: x[0:7])
    speed_separate_df = speed_separate_df[speed_separate_df['measurement_tstamp_month'].isin(month_split)]
    speed_separate_df.to_csv(output_path)
    
    return speed_separate_df

# get flow according to different tmc_code's reference speed
def get_flow(reference_speed):
    if reference_speed <= 15:
        return 600
    elif reference_speed <= 25:
        return 800
    elif reference_speed <= 35:
        return 1200
    elif reference_speed <= 45:
        return 1600
    else:
        return 2000

# get different kinds of speed so that we can get a better result
def get_relative_speed_1(speed, reference_speed):
    return speed / reference_speed

def get_relative_speed_2(speed, reference_speed):
    flow = get_flow(reference_speed)
    return speed * flow

def get_relative_speed_3(speed, reference_speed):
    flow = get_flow(reference_speed)
    return speed * flow / reference_speed

# use chunk
def chunk_preprocessing(chunk, threshold):
    chunk = chunk[['tmc_code', 'measurement_tstamp', 'speed', 'reference_speed']]
    chunk['measurement_tstamp_month'] = chunk['measurement_tstamp'].apply(lambda x: x[0:7])
    chunk = chunk[chunk['reference_speed'] < threshold]
    # chunk['flow'] = get_flow(chunk['reference_speed'])
    return chunk

# combine all speed data and filter out highway speed
def combine2csv(input_speed_path, output_speed_path):
    df_chunk = pd.read_csv(input_speed_path, iterator=True, chunksize=10000)
    
    chunk_list = []
    
    for chunk in df_chunk:
        # filter out the data whose speed > 55
        chunk_filter = chunk_preprocessing(chunk, 55)
        chunk_list.append(chunk_filter)

    df_concat = pd.concat(chunk_list)
    
    speeds = []
    tmc2region = {}
    with open(region2tmc_path, 'r', encoding='UTF-8') as f:
        tmc2region_dict = json.load(f)
        #print(tmc2region_dict)
        
        for key in tmc2region_dict['region_id']:
            region_id = tmc2region_dict['region_id'][key]
            tmc_code = tmc2region_dict['tmc_code'][key]
            if tmc_code in tmc2region:
                tmc2region[tmc_code].append(region_id)
            else:
                tmc2region[tmc_code] = [region_id]
    
    for index, row in df_concat.iterrows():
        measurement_tstamp = row['measurement_tstamp']
        tmc_code = row['tmc_code']
        speed = row['speed']
        reference_speed = row['reference_speed']
        flow = get_flow(reference_speed)
        
        if reference_speed <= 0:
            continue
        
        relative_speed_1 = get_relative_speed_1(speed, reference_speed)
        relative_speed_2 = get_relative_speed_2(speed, reference_speed)
        relative_speed_3 = get_relative_speed_3(speed, reference_speed)
        
        try:
            region_ids = tmc2region[tmc_code]
        except:
            
            print(tmc_code)
            continue
        
        # get specialiesd id and its region, and three kinds of speed.
        for region_id in region_ids:
            speeds.append([measurement_tstamp, tmc_code, speed, reference_speed, region_id, flow, relative_speed_1,relative_speed_2, relative_speed_3])
    
    speed_df = pd.DataFrame(speeds)
    
    speed_df.columns = ["measurement_tstamp", "tmc_code", "speed", "reference_speed", "region_id", "flow", "relative_speed_1", "relative_speed_2", "relative_speed_3"]
    
    # speed_df = speed_df.groupby(['region_id', 'measurement_tstamp']).mean()[["speed", "reference_speed", "relative_speed_1", "relative_speed_2", "relative_speed_3"]]
    
    #get relative speed
    speed_df = speed_df.groupby(['region_id', 'measurement_tstamp']).agg({'speed': 'mean', "reference_speed": 'mean', "flow": 'sum', "relative_speed_1": 'mean', "relative_speed_2": 'sum', "relative_speed_3": 'sum'})
    speed_df['relative_speed_2'] = speed_df['relative_speed_2'] / speed_df['flow']
    speed_df['relative_speed_3'] = speed_df['relative_speed_3'] / speed_df['flow']
    
    speed_df = speed_df.reset_index()
    speed_df.to_csv(output_speed_path)
    return speed_df

def combine_all_csv(output_speed_path_2019_04_df, output_speed_path_2019_08_df, output_speed_path_2019_12_df, output_speed_path_2020_02_df,output_speed_path_2020_04_df ,output_speed_path_2020_06_df):
    
    df_output_speed_path = pd.concat([output_speed_path_2019_04_df, output_speed_path_2019_08_df, output_speed_path_2019_12_df,output_speed_path_2020_02_df,output_speed_path_2020_04_df ,output_speed_path_2020_06_df])
    df_output_speed_path = df_output_speed_path.loc[:, ~df_output_speed_path.columns.str.contains("^Unnamed")]
    df_output_speed_path.to_csv(output_speed_path)

if __name__ == '__main__':

    #correspond_tmccode_to_region(input_tmc_path, input_tmc_geo_path, input_region_geo_path, input_region_geo_Manhanttan_path, region2tmc_path)
    combine_speed_2020(input_speed_path_raw04_2020,input_speed_path_raw11_2020,input_speed_path_2020)
   
    separate_data(input_speed_path_2019, month_split_2019_04, input_speed_path_2019_04)
    separate_data(input_speed_path_2019, month_split_2019_08, input_speed_path_2019_08)
    separate_data(input_speed_path_2019, month_split_2019_12, input_speed_path_2019_12)
    separate_data(input_speed_path_2020, month_split_2020_02, input_speed_path_2020_02)
    separate_data(input_speed_path_2020, month_split_2020_04, input_speed_path_2020_04)
    separate_data(input_speed_path_2020, month_split_2020_06, input_speed_path_2020_06)

    output_speed_path_2019_04_df = combine2csv(input_speed_path_2019_04, output_speed_path_2019_04)
    output_speed_path_2019_08_df = combine2csv(input_speed_path_2019_08, output_speed_path_2019_08)
    output_speed_path_2019_12_df = combine2csv(input_speed_path_2019_12, output_speed_path_2019_12)
    output_speed_path_2020_02_df = combine2csv(input_speed_path_2020_02, output_speed_path_2020_02)
    output_speed_path_2020_04_df = combine2csv(input_speed_path_2020_04, output_speed_path_2020_04)
    output_speed_path_2020_06_df = combine2csv(input_speed_path_2020_06, output_speed_path_2020_06)
    
    combine_all_csv(output_speed_path_2019_04_df, output_speed_path_2019_08_df, output_speed_path_2019_12_df, output_speed_path_2020_02_df,output_speed_path_2020_04_df ,output_speed_path_2020_06_df)
    
