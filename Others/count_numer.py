
import csv


raw_data_path_pre = './data_set/NYC_speed/'
output_path_forall = raw_data_path_pre+ 'raw_data/nyc_2019_5min_0101_1231.csv'

file = open(output_path_forall)

numline = len(file.readlines())
print(file.readlines())
print (numline)


