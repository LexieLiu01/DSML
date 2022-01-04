#merge data from speed data and trip record data into trip_and_speed.csv

import pandas as pd

input_trip='./data_set/NYC_triprecord/prepared_data/trip_all.csv'
input_speed='./data_set/NYC_speed/prepared_data/nyc_5min_all_prepared.csv'
input_trip_and_speed = './data_set/NYC_speed_and_trip/trip_and_speed.csv'

def groupby_trip_speed(input_trip,input_speed):
    #read trip record data
    input_trip_df = pd.read_csv(input_trip)
    input_trip_df['region_id'] = input_trip_df['region_id'].apply(lambda x:int(x))
    input_trip_df['total_number'] = input_trip_df['Pickup_number'] + input_trip_df['Dropoff_number']

    input_trip_df = input_trip_df.groupby(['region_id','datetime_min_5']).sum()

    input_trip_df = input_trip_df.reset_index()

    #read speed data
    input_speed_df = pd.read_csv(input_speed)
    input_speed_df['region_id'] = input_speed_df['region_id'].apply(lambda x: int(x))
    input_speed_df['measurement_tstamp'] = input_speed_df['measurement_tstamp'].apply(lambda x:x[0:16])
    input_speed_df.rename(columns={'measurement_tstamp': 'datetime_min_5'}, inplace=True)
    input_speed_df = input_speed_df.reset_index()
    
    #merge both speed data and triprecord data

    input_df = input_trip_df.merge(input_speed_df, on=['region_id','datetime_min_5'])
    input_df = input_df.reset_index()
    input_df.to_csv(input_trip_and_speed)

    return input_df

if __name__ == '__main__':
    groupby_trip_speed(input_trip, input_speed)
