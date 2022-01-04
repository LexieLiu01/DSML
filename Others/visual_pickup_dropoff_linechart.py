import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta

import matplotlib
import pickle

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

input_for_dml_train = './model_dml/training_data/trip_speed_precip_X_M_Nei_more_parameters.csv'
# ids = [4, 12, 13, 24,43, 45, 48, 50, 68,75, 79, 87, 88, 90, 100, 107, 113, 114, 125,137, 140, 141, 142, 143, 144, 148, 151,158, 161, 162, 163, 164,170, 186, 209, 211, 224, 229, 230,231, 232, 233, 234, 236, 237, 238, 239, 246, 249, 261, 262, 263]
ids = [232]

fig = plt.figure()

dayColor = 'darkgray'
aveColor = 'darkorange'
xlabel = "Time of Day"
ylabel = "Number of pick-up/drop-off in 5 Mins"

dayAlpha = 0.3
aveAlpha = 1

daylinewidth = 1
avelinewidth = 3

start = datetime(2019, 2, 1)
end = datetime(2020, 6, 30)
# end = datetime(2019, 2, 10)

# target_weekday = ['2', '3', '4']
target_weekday = ['6', '0']
target_hour = ['16', '17', '18', '19']
identify_target_weekday = '_'.join(target_weekday)
identify_target_hour = '_'.join(target_hour)

ave_speed_dict = {}

data_for_training_path = './model_dml/training_data/' + 'training_data_dmsl_XWXMW_more_parameters0_6_16_17_18_19.pickle'


# get time sequence from start to the end
def datetime_range(start, end, delta):
    current = start
    if not isinstance(delta, timedelta):
        delta = timedelta(**delta)
    while current < end:
        yield current
        current += delta


def generate_time_sequence(start, end, target_hour, target_weekday):
    time_seq = []
    for dt in datetime_range(start, end, {'minutes': 5}):
        if str(dt)[11:13] in target_hour:
            if dt.strftime("%w") in target_weekday:
                time_seq.append(str(dt)[0:16])
    
    time_seq_df = pd.DataFrame(time_seq)
    time_seq_df.columns = ['datetime_min_5']
    
    return time_seq_df


#
# def filter_different_days(input_for_dml_train):
#     training_data_df = pd.read_csv(input_for_dml_train)
#
#     training_data_df['datetime_min_5_strp'] = training_data_df['datetime_min_5'].apply(
#         lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M'))
#     training_data_df['weekday'] = training_data_df['datetime_min_5_strp'].apply(lambda x: x.strftime("%w"))
#     training_data_df['day'] = training_data_df['datetime_min_5'].apply(lambda x: x[0:10])
#     training_data_df['hour'] = training_data_df['datetime_min_5'].apply(lambda x: x[11:13])
#
#     holiday_list = ['2019-04-21', '2019-04-22', '2019-04-25', '2019-05-01', '2019-06-02', '2019-08-15', '2019-11-01',
#                     '2019-12-08', '2019-12-25', '2019-12-26', '2020-01-01', '2020-01-06', '2020-04-12', '2020-04-13',
#                     '2020-04-25', '2020-05-01', '2020-06-02']
#
#     training_data_df = training_data_df[~training_data_df['day'].isin(holiday_list)]
#     training_data_df = training_data_df[
#         (training_data_df['weekday'].isin(target_weekday)) & (training_data_df['hour'].isin(target_hour))]
#
#     # training_data_for_evening_df = time_seq_df.merge(training_data_df, how='left', on='datetime_min_5')
#     # training_data_for_evening_df = training_data_for_evening_df.fillna(0)
#     training_data_for_evening_df = training_data_df[
#         ['region_id', 'datetime_min_5', 'relative_speed', 'total_number', 'X', 'M', 'precip_in']]
#     # training_data_for_evening_df.to_csv()
#     return training_data_for_evening_df


def read_speed(data_for_training_path, id, time_seq_df):
    total_speed_ave_dict = {}
    
    with open(data_for_training_path, 'rb') as f:
        data_for_training_df = pickle.load(f)
    
    data_for_training_df = data_for_training_df[data_for_training_df['region_id'] == id]
    
    data_for_training_fill_value_df = time_seq_df.merge(data_for_training_df, how='left', on='datetime_min_5')
    data_for_training_fill_value_df = data_for_training_fill_value_df.fillna(0)
    
    data_for_training_fill_value_plot_df = data_for_training_fill_value_df[
        ['region_id', 'datetime_min_5', 'relative_speed', 'total_number']]
    
    date = []
    time = []
    for index, row in data_for_training_fill_value_plot_df.iterrows():
        # print(row['datetime_min_5'], row['relative_speed'])
        day = row['datetime_min_5'][0:10]
        ptime = row['datetime_min_5'][11:16]
        if ptime not in time:
            time.append(ptime)
        
        if day not in date:
            date.append(day)
    
    time.sort()
    
    total_speed = []
    total_pickdrop = []
    
    plt.figure()
    print('=====date=====')
    
    for day in date:
        tempdata_speed = []
        tempdata_pickdrop = []
        for index, row in data_for_training_fill_value_plot_df.iterrows():
            if day in row['datetime_min_5']:
                tempdata_speed.append(row['relative_speed'])
                tempdata_pickdrop.append(row['total_number'])
        
        # plt.plot(tempdata_speed, color=dayColor, alpha=dayAlpha)
        plt.plot(tempdata_pickdrop, color=dayColor, alpha=dayAlpha)
        
        total_speed.append(tempdata_speed)
        total_pickdrop.append(tempdata_pickdrop)
    
    total_speed = np.array(total_speed, dtype="float32")
    total_pickdrop = np.array(total_pickdrop, dtype="float32")
    
    total_speed_ave = np.true_divide(total_speed.sum(0), (total_speed != 0).sum(0))
    total_pickdrop_ave = np.true_divide(total_pickdrop.sum(0), (total_pickdrop != 0).sum(0))
    
    # total_speed_ave_dict[id] = total_speed_ave
    # total_pickdrop_ave_dict[id] = total_pickdrop_ave
    
    # plt.plot(total_speed_ave, color=aveColor, alpha=aveAlpha)
    plt.plot(total_pickdrop_ave, color=aveColor, alpha=aveAlpha)
    
    # spotlight = [i for i in range(0, 60, 6)]
    
    spotlight = [0, 6, 12, 18, 24, 30, 36, 42, 47]
    spotTime = []
    for idx, element in enumerate(time):
        if idx in spotlight:
            spotTime.append(element)
    
    plt.xticks(spotlight, spotTime)
    plt.xlim(0, 47)
    plt.gca().set_ylim((10, None))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    plt.savefig('./Visualization/Pickup_Dropoff/' + str(id) + '_pickdrop.png')
    
    plt.show()
    
    return None


if __name__ == '__main__':
    
    time_seq_df = generate_time_sequence(start, end, target_hour, target_weekday)
    for id in ids:
        ave_speed_dict = read_speed(data_for_training_path, id, time_seq_df)
