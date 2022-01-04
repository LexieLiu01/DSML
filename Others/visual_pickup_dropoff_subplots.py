import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date, datetime, timedelta

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

input_for_dml_train = './data_set/Prepared_matrix_completion_data/trip_speed_precip_X_M_Nei_more_parameters.csv'

ids = [4, 12, 13, 24,43, 45, 48, 50, 68,75, 79, 87, 88, 90, 100, 107, 113, 114, 125,137, 140, 141, 142, 143, 144, 148, 151,158, 161, 162, 163, 164,170, 186, 209, 211, 224, 229, 230,231, 232, 233, 234, 236, 237, 238, 239, 246, 249, 261, 262, 263]

#ids = [4]
dayColor = '#363A3E'
dayAlpha = 0.3
aveColor = '#e63946'
aveAlpha = 0.7
xlabel = "Time of Day"
ylabel = "The Total Number of Pickup and Dropoff in 5 Mins"

start = datetime(2019, 2, 1)
end = datetime(2020, 6, 30)

target_weekday = ['3', '4', '5']
target_hour = ['16', '17', '18', '19']

ave_dict = {}

# get time sequence from start to the end
def datetime_range(start, end, delta):
    current = start
    if not isinstance(delta, timedelta):
        delta = timedelta(**delta)
    while current < end:
        yield current
        current += delta

def generate_time_sequence(start, end):
	time_seq = []
	
	for dt in datetime_range(start, end, {'minutes': 5}):
		if str(dt)[11:13] in ['16', '17', '18', '19']:
			if dt.strftime("%w") in ['3', '4', '5']:
				time_seq.append(str(dt)[0:16])
	
	time_seq_df = pd.DataFrame(time_seq)

	time_seq_df.columns = ['datetime_min_5']
	print('=====time_seq_df======')
	print(time_seq_df)
	
	return time_seq_df

def filter_different_days(input_for_dml_train):
	training_data_df = pd.read_csv(input_for_dml_train)
	
	training_data_df['datetime_min_5_strp'] = training_data_df['datetime_min_5'].apply(
		lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M'))
	training_data_df['weekday'] = training_data_df['datetime_min_5_strp'].apply(lambda x: x.strftime("%w"))
	training_data_df['day'] = training_data_df['datetime_min_5'].apply(lambda x: x[0:10])
	training_data_df['hour'] = training_data_df['datetime_min_5'].apply(lambda x: x[11:13])
	
	holiday_list = ['2019-04-21', '2019-04-22', '2019-04-25', '2019-05-01', '2019-06-02', '2019-08-15', '2019-11-01',
					'2019-12-08', '2019-12-25', '2019-12-26', '2020-01-01', '2020-01-06', '2020-04-12', '2020-04-13',
					'2020-04-25', '2020-05-01', '2020-06-02']
	
	training_data_df = training_data_df[~training_data_df['day'].isin(holiday_list)]
	training_data_df = training_data_df[
		(training_data_df['weekday'].isin(target_weekday)) & (training_data_df['hour'].isin(target_hour))]
	
	#training_data_for_evening_df = time_seq_df.merge(training_data_df, how='left', on='datetime_min_5')
	#training_data_for_evening_df = training_data_for_evening_df.fillna(0)
	training_data_for_evening_df = training_data_df[
		['region_id', 'datetime_min_5', 'relative_speed', 'total_number', 'X', 'M', 'precip_in']]
	# training_data_for_evening_df.to_csv()
	return training_data_for_evening_df

def read_speed(training_data_for_evening_df, id,time_seq_df):
	fill_value_trip_and_speed_df  = training_data_for_evening_df
	
	fill_value_trip_and_speed_df = fill_value_trip_and_speed_df[fill_value_trip_and_speed_df['region_id'] == id]
	
	fill_value_trip_df = time_seq_df.merge(fill_value_trip_and_speed_df, how='left', on='datetime_min_5')
	fill_value_trip_df = fill_value_trip_df.fillna(0)
	
	plot_speed_df = fill_value_trip_df[['region_id','datetime_min_5', 'total_number','relative_speed']]
	date = []
	time = []
	for index, row in plot_speed_df.iterrows():
		#print(row['datetime_min_5'], row['relative_speed'])
		day = row['datetime_min_5'][0:10]
		ptime = row['datetime_min_5'][11:16]
		if ptime not in time:
			time.append(ptime)

		if day not in date:
			date.append(day)

	time.sort()
	
	total_pickup_dropoff = []
	total_speed = []
	
	fig, (ax1, ax2) = plt.subplots(2, 1)
	fig.suptitle('Sharing both axes')
	
	print('=====date=====')
	print(date)
	for day in date:

		tempdata_pickup_dropoff = []
		tempdata_speed = []

		for index, row in plot_speed_df.iterrows():
			if day in row['datetime_min_5']:
				#print(row['relative_speed'])
				tempdata_pickup_dropoff.append(row['total_number'])
				tempdata_speed.append(row['relative_speed'])
		
		ax1.plot(tempdata_pickup_dropoff, color=dayColor, alpha = dayAlpha)
		ax2.plot(tempdata_speed, color=dayColor, alpha = dayAlpha)
		total_pickup_dropoff.append(tempdata_pickup_dropoff)
		total_speed.append(tempdata_speed)

	total_pickup_dropoff = np.matrix(total_pickup_dropoff, dtype = "float32")
	total_speed = np.matrix(total_speed, dtype = "float32")
	
	ave_pickup_dropoff = np.mean(total_pickup_dropoff, axis=0)
	ave_speed = np.mean(total_speed, axis=0)
	print(ave_pickup_dropoff)
	print(ave_speed)
	exit()
	
	ax1.plot(ave_pickup_dropoff, color=aveColor, alpha=aveAlpha)
	ax2.plot(ave_speed, color=aveColor, alpha=aveAlpha)
	#ave_dict[id] = ave
	
	#spotlight = [i for i in range(0, 60, 6)]
	
	spotlight = [0, 6, 12, 18, 24, 30, 36, 42, 47]
	spotTime = []
	for idx, element in enumerate(time):
		if idx in spotlight:
			spotTime.append(element)
	plt.xticks(spotlight, spotTime)
	plt.xlim(0, 47)
	#plt.gca().set_ylim((10,None))
	#plt.xlabel(xlabel)
	#plt.ylabel(ylabel)
	plt.show()
	# plt.savefig('./Visualization/Subplot/'+ str(id) + '_pickup_dropoff_speed_subplot.png')
	#plt.show()
	
	return ave_dict

if __name__ == '__main__':
	
	time_seq_df = generate_time_sequence(start, end)
	training_data_for_evening_df = filter_different_days(input_for_dml_train)
	for id in ids:
		ave_dict = read_speed(training_data_for_evening_df,id,time_seq_df)
	
	#np.save('./Visualization/ave_pickup_dropoff.npy', ave_dict)