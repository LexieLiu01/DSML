import pickle
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import seaborn as sns; sns.set()
plt.style.use('seaborn-poster')
np.set_printoptions(suppress=True)

# plt.style.use('seaborn-poster')

regions = ['MidTown', 'CenterPark']
# regions = ['MidTown']
days = ['weekdays', 'weekends']

# days = ['weekdays']

def generate_form(file_df):
    overall_list = []
    org_df = file_df[['time', 'original_value']]
    org_df = org_df.set_index('time')
    org_arr = org_df.to_numpy()
    org_arr = org_arr.reshape(-1, 4)
    
    org_arr_slot_mean = np.mean(org_arr, axis=0).reshape(-1, 1)
    org_arr_slot_std = np.std(org_arr, axis=0).reshape(-1, 1)
    
    org_arr_day_sum = np.sum(org_arr, axis=1)
    org_arr_day_sum_mean = np.mean(org_arr_day_sum)
    org_arr_day_sum_std = np.std(org_arr_day_sum)
    
    overall_list.append([org_arr_day_sum_mean, org_arr_day_sum_std])
    
    opt_df = file_df[['time', 'min_value']]
    opt_df = opt_df.set_index('time')
    
    opt_arr = opt_df.to_numpy()
    opt_arr = opt_arr.reshape(-1, 4)
    opt_arr_slot_mean = np.mean(opt_arr, axis=0).reshape(-1, 1)
    opt_arr_slot_std = np.std(opt_arr, axis=0).reshape(-1, 1)
    
    opt_arr_day_sum = np.sum(opt_arr, axis=1)
    opt_arr_day_sum_mean = np.mean(opt_arr_day_sum)
    opt_arr_day_sum_std = np.std(opt_arr_day_sum)
    
    overall_list.append([opt_arr_day_sum_mean, opt_arr_day_sum_std])
    
    rat_df = file_df[['time', 'optimized_rate']]
    rat_df = rat_df.set_index('time')
    
    rat_arr = rat_df.to_numpy()
    rat_arr = rat_arr.reshape(-1, 4)
    rat_arr_slot_mean = np.mean(rat_arr, axis=0).reshape(-1, 1)
    rat_arr_slot_std = np.std(rat_arr, axis=0).reshape(-1, 1)
    
    overall_rate = (org_arr_day_sum - opt_arr_day_sum) / org_arr_day_sum
    overall_rate_mean = np.mean(overall_rate)
    overall_rate_std = np.std(overall_rate)
    
    overall_list.append([overall_rate_mean, overall_rate_std])
    
    overall_array = np.array(overall_list).reshape(1, -1)
    
    detail_array = np.concatenate(
        (org_arr_slot_mean, org_arr_slot_std, opt_arr_slot_mean, opt_arr_slot_std, rat_arr_slot_mean, rat_arr_slot_std),
        axis=1)
    
    complete_array = np.concatenate((overall_array, detail_array), axis=0)
    
    complete_array_ard = np.round(complete_array, 4)
    return complete_array, complete_array_ard


def select_amount_df(place, days, form_df):
    # CenterPark
    # MidTown
    # place = 'MidTown'
    # days = 'weekends'
    
    plot_df = form_df[(form_df['region_name'] == place) & (form_df['lower_bound'] == '99999.0') & (
                form_df['upper_bound'] == '99999.0') & (form_df['days_type'] == days)]
    tilename = "Optimized rate curve in " + place + " on " + days + " controlling vehicle amount"
    return plot_df, tilename


def plot_amount_sub(place, days, form_df):
    
    fig, axs = plt.subplots(2, 3, figsize=(16, 8), sharey=True)
    
    plot_df, tilename = select_amount_df(place, days, form_df)
    # fig.suptitle(tilename)
    
    ax = axs[0, 0]
    
    groups = plot_df.groupby(['time_slot'])
    
    group1 = groups.get_group("16")
    ax.errorbar(group1['vehicle_amount'], group1['over_mean'], yerr=group1['over_dev'],
                fmt='.', markersize='5', color='firebrick', ls=':', label='16:00',
                ecolor='firebrick', elinewidth=1, capsize=3, capthick=1)
    
    
    # ax.grid()
    
    ax = axs[0, 1]
    group2 = groups.get_group("17")
    ax.errorbar(group2['vehicle_amount'], group2['over_mean'], yerr=group2['over_dev'],
                fmt='.', markersize='5', color='darkviolet', ls='--', dashes=(7, 10), label='17:00',
                ecolor='darkviolet', elinewidth=1, capsize=3, capthick=1)
    # ax.grid()
    
    ax = axs[0, 2]
    group3 = groups.get_group("18")
    ax.errorbar(group3['vehicle_amount'], group3['over_mean'], yerr=group3['over_dev'],
                fmt='.', markersize='5', color='royalblue', ls='-.', label='18:00',
                ecolor='royalblue', elinewidth=1, capsize=3, capthick=1)
    
    # ax.grid()
    
    ax = axs[1, 0]
    group4 = groups.get_group("19")
    ax.errorbar(group4['vehicle_amount'], group4['over_mean'], yerr=group4['over_dev'],
                fmt='.', markersize='5', color='tomato', ls='--', label='19:00',
                ecolor='tomato', elinewidth=1, capsize=3, capthick=1)
    
    # ax.grid()
    
    ax = axs[1, 1]
    group0 = groups.get_group("24")
    ax.errorbar(group0['vehicle_amount'], group0['over_mean'], yerr=group0['over_dev'],
                fmt='.', markersize='5', color='darkorange', ls='--', dashes=(4, 10), label='Sum',
                ecolor='darkorange', elinewidth=1, capsize=3, capthick=1)
    
    # ax.grid()
    
    #     axs[1, 2].axis('off')
    
    #     add the all lines into one figure
    ax = axs[1, 2]
    ax.plot(group1['vehicle_amount'], group1['over_mean'], markersize='5', color='firebrick', ls=':', label='16:00')
    ax.plot(group2['vehicle_amount'], group2['over_mean'], markersize='5', color='darkviolet', ls='--', dashes=(7, 10),
            label='17:00')
    ax.plot(group3['vehicle_amount'], group3['over_mean'], markersize='5', color='royalblue', ls='-.', label='18:00')
    ax.plot(group4['vehicle_amount'], group4['over_mean'], markersize='5', color='tomato', ls='--', label='19:00')
    ax.plot(group0['vehicle_amount'], group0['over_mean'], markersize='5', color='darkorange', ls='--', dashes=(4, 10),
            label='Sum')
    
    # ax.grid()
    
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
    
    fig.text(0.5,0, 'Times of vehicles controlled', ha='center', va='center', fontsize=20)
    fig.text(0.05, 0.5, 'Percentage improvement in total travel time by rerouting', ha='center', va='center', rotation='vertical', fontsize=20)
    labels = ['16:00', '17:00', '18:00', '19:00', 'average']
    fig.legend(labels, loc='lower right', bbox_to_anchor=(1.03, 0.37), ncol=1, bbox_transform=fig.transFigure)
    #     fig.legend(labels, loc='lower right', bbox_to_anchor=(0.9,0.2), ncol=len(labels), bbox_transform=fig.transFigure)
    
    plt.savefig('./visual_optimization_rate/' + tilename + '_line.pdf', bbox_inches="tight")
    plt.show()


# def plot_amount_lineplot(form_df):
#     plot_df, tilename = select_amount_df(form_df)
#
#     groups = plot_df.groupby(['time_slot'])
#
#     group1 = groups.get_group("16")
#     group1['min'] = group1['over_mean'] - group1['over_dev']
#     group1['max'] = group1['over_mean'] + group1['over_dev']
#
#     # Draw plot with error band and extra formatting to match seaborn style
#
#     fig, ax = plt.subplots(figsize=(15, 5))
#
#     ax.plot(group1['vehicle_amount'], group1['over_mean'], label='16:00', color='firebrick', ls=':')
#     ax.plot(group1['vehicle_amount'], group1['min'], color='firebrick', alpha=0.5)
#     ax.plot(group1['vehicle_amount'], group1['max'], color='firebrick', alpha=0.5)
#     ax.fill_between(group1['vehicle_amount'], group1['min'], group1['max'], alpha=0.2)
#     ax.set_xlabel('Times of vehicles controlled')
#     ax.set_ylabel('Optimized rate by rerouting')
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#
#     group2 = groups.get_group("17")
#     group2['min'] = group2['over_mean'] - group2['over_dev']
#     group2['max'] = group2['over_mean'] + group2['over_dev']
#
#     ax.plot(group2['vehicle_amount'], group2['over_mean'], label='17:00', color='darkviolet', ls='--', dashes=(7, 10))
#     ax.plot(group2['vehicle_amount'], group2['min'], color='darkviolet', alpha=0.5)
#     ax.plot(group2['vehicle_amount'], group2['max'], color='darkviolet', alpha=0.5)
#     ax.fill_between(group2['vehicle_amount'], group2['min'], group2['max'], alpha=0.2)
#     ax.set_xlabel('Times of vehicles controlled')
#     ax.set_ylabel('Optimized rate by rerouting')
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#
#     group3 = groups.get_group("18")
#     group3['min'] = group3['over_mean'] - group3['over_dev']
#     group3['max'] = group3['over_mean'] + group3['over_dev']
#
#     ax.plot(group3['vehicle_amount'], group3['over_mean'], label='18:00', color='royalblue', ls='-.')
#     ax.plot(group3['vehicle_amount'], group3['min'], color='royalblue', alpha=0.5)
#     ax.plot(group3['vehicle_amount'], group3['max'], color='royalblue', alpha=0.5)
#     ax.fill_between(group3['vehicle_amount'], group3['min'], group3['max'], alpha=0.2)
#     ax.set_xlabel('Times of vehicles controlled')
#     ax.set_ylabel('Optimized rate by rerouting')
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#
#     group4 = groups.get_group("19")
#     group4['min'] = group4['over_mean'] - group4['over_dev']
#     group4['max'] = group4['over_mean'] + group4['over_dev']
#
#     ax.plot(group4['vehicle_amount'], group4['over_mean'], label='19:00', color='tomato', ls='--')
#     ax.plot(group4['vehicle_amount'], group4['min'], color='tomato', alpha=0.5)
#     ax.plot(group4['vehicle_amount'], group4['max'], color='tomato', alpha=0.5)
#     ax.fill_between(group4['vehicle_amount'], group4['min'], group4['max'], alpha=0.2)
#     ax.set_xlabel('Times of vehicles controlled')
#     ax.set_ylabel('Optimized rate by rerouting')
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#
#     group0 = groups.get_group("24")
#     group0['min'] = group0['over_mean'] - group0['over_dev']
#     group0['max'] = group0['over_mean'] + group0['over_dev']
#
#     ax.plot(group0['vehicle_amount'], group0['over_mean'], label='sum', color='darkorange', ls='--', dashes=(4, 10))
#     ax.plot(group0['vehicle_amount'], group0['min'], color='darkorange', alpha=0.5)
#     ax.plot(group0['vehicle_amount'], group0['max'], color='darkorange', alpha=0.5)
#     ax.fill_between(group0['vehicle_amount'], group0['min'], group0['max'], alpha=0.2)
#     ax.set_xlabel('Times of vehicles controlled')
#     ax.set_ylabel('Optimized rate by rerouting')
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#
#     fig.legend(labels=['16:00', '17:00', '18:00', '19:00', 'sum'])
#
#     plt.savefig('./visual_optimization_rate/' + tilename + '.pdf')
#     plt.show()


def select_upperbound_df(place, days, form_df):
    # CenterPark
    # MidTown
    
    # place = 'MidTown'
    # days = 'weekdays'
    #
    plot_df = form_df[(form_df['region_name'] == place) & (form_df['lower_bound'] != '99999.0') & (
                form_df['upper_bound'] != '99999.0') & (form_df['days_type'] == days)]
    tilename = 'Optimized rate curve in ' + place + ' on ' + days + ' limiting maximum of vehicles controlled'
    return plot_df, tilename


def plot_upperbound(place, days, form_df):
    plt.subplots()
    plot_df, tilename = select_upperbound_df(place, days, form_df)
    
    groups = plot_df.groupby(['time_slot'])
    
    group1 = groups.get_group("16")
    plt.errorbar(group1['upper_bound'], group1['over_mean'], yerr=group1['over_dev'],
                 fmt='.', markersize='5', color='firebrick', ls=':', label='16:00',
                 ecolor='firebrick', elinewidth=1, capsize=3, capthick=1)
    group2 = groups.get_group("17")
    plt.errorbar(group2['upper_bound'], group2['over_mean'], yerr=group2['over_dev'],
                 fmt='.', markersize='5', color='darkviolet', ls='--', dashes=(7, 10), label='17:00',
                 ecolor='darkviolet', elinewidth=1, capsize=3, capthick=1)
    group3 = groups.get_group("18")
    plt.errorbar(group3['upper_bound'], group3['over_mean'], yerr=group3['over_dev'],
                 fmt='.', markersize='5', color='royalblue', ls='-.', label='18:00',
                 ecolor='royalblue', elinewidth=1, capsize=3, capthick=1)
    group4 = groups.get_group("19")
    plt.errorbar(group4['upper_bound'], group4['over_mean'], yerr=group4['over_dev'],
                 fmt='.', markersize='5', color='tomato', ls='--', label='19:00',
                 ecolor='tomato', elinewidth=1, capsize=3, capthick=1)
    group0 = groups.get_group("24")
    plt.errorbar(group0['upper_bound'], group0['over_mean'], yerr=group0['over_dev'],
                 fmt='.', markersize='5', color='darkorange', ls='--', dashes=(4, 10), label='Sum',
                 ecolor='darkorange', elinewidth=1, capsize=3, capthick=1)
    # plt.legend(bbox_to_anchor=(1.1, 0.9))
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
    plt.xlabel('Upperbound of vehicles controlled')
    plt.ylim(-0.01, 0.06)
    plt.ylabel('Optimized rate by rerouting')
    # plt.title(tilename)
    plt.savefig('./visual_optimization_rate/' + tilename + 'line.pdf')
    plt.show()


def plot_upperbound_sub(place, days, form_df):
    fig, axs = plt.subplots(2, 3, figsize=(16, 8), sharey=True)
    
    plot_df, tilename = select_upperbound_df(place, days, form_df)
    # fig.suptitle(tilename)
    
    groups = plot_df.groupby(['time_slot'])
    
    ax = axs[0, 0]
    group1 = groups.get_group("16")
    ax.errorbar(group1['upper_bound'], group1['over_mean'], yerr=group1['over_dev'],
                fmt='.', markersize='5', color='firebrick', ls=':', label='16:00',
                ecolor='firebrick', elinewidth=1, capsize=3, capthick=1)

    # ax.grid()
    
    ax = axs[0, 1]
    group2 = groups.get_group("17")
    ax.errorbar(group2['upper_bound'], group2['over_mean'], yerr=group2['over_dev'],
                fmt='.', markersize='5', color='darkviolet', ls='--', dashes=(7, 10), label='17:00',
                ecolor='darkviolet', elinewidth=1, capsize=3, capthick=1)

    # ax.grid()
    
    ax = axs[0, 2]
    group3 = groups.get_group("18")
    ax.errorbar(group3['upper_bound'], group3['over_mean'], yerr=group3['over_dev'],
                fmt='.', markersize='5', color='royalblue', ls='-.', label='18:00',
                ecolor='royalblue', elinewidth=1, capsize=3, capthick=1)

    # ax.grid()
    
    ax = axs[1, 0]
    group4 = groups.get_group("19")
    ax.errorbar(group4['upper_bound'], group4['over_mean'], yerr=group4['over_dev'],
                fmt='.', markersize='5', color='tomato', ls='--', label='19:00',
                ecolor='tomato', elinewidth=1, capsize=3, capthick=1)

    # ax.grid()
    
    ax = axs[1, 1]
    group0 = groups.get_group("24")
    ax.errorbar(group0['upper_bound'], group0['over_mean'], yerr=group0['over_dev'],
                fmt='.', markersize='5', color='darkorange', ls='--', dashes=(4, 10), label='Sum',
                ecolor='darkorange', elinewidth=1, capsize=3, capthick=1)
    # ax.grid()
    # axs[1, 2].axis('off')

    ax = axs[1, 2]
    ax.plot(group1['upper_bound'], group1['over_mean'], markersize='5', color='firebrick', ls=':', label='16:00')
    ax.plot(group2['upper_bound'], group2['over_mean'], markersize='5', color='darkviolet', ls='--', dashes=(7, 10),
            label='17:00')
    ax.plot(group3['upper_bound'], group3['over_mean'], markersize='5', color='royalblue', ls='-.', label='18:00')
    ax.plot(group4['upper_bound'], group4['over_mean'], markersize='5', color='tomato', ls='--', label='19:00')
    ax.plot(group0['upper_bound'], group0['over_mean'], markersize='5', color='darkorange', ls='--', dashes=(4, 10),
            label='Sum')

    # ax.grid()
    
    # plt.legend(bbox_to_anchor=(1.1, 0.9))
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
    
    fig.text(0.5, 0, 'Upperbound of vehicles controlled', ha='center', va='center', fontsize=20)
    fig.text(0.05, 0.5, 'Percentage improvement in total travel time by rerouting', ha='center', va='center', rotation='vertical', fontsize=20)
    labels = ['16:00', '17:00', '18:00', '19:00', 'average']
    fig.legend(labels, loc='lower right', bbox_to_anchor=(1.03, 0.37), ncol=1, bbox_transform=fig.transFigure)
    #     fig.legend(labels, loc='lower right', bbox_to_anchor=(0.9,0.2), ncol=len(labels), bbox_transform=fig.transFigure)
    
    plt.savefig('./visual_optimization_rate/' + tilename + '_line.pdf', bbox_inches="tight")
    plt.show()

# def plot_upperbound_lineplot(form_df):

#     plot_df, tilename = select_upperbound_df(form_df)
#
#     groups = plot_df.groupby(['time_slot'])
#
#     group1 = groups.get_group("16")
#     group1['min'] = group1['over_mean'] - group1['over_dev']
#     group1['max'] = group1['over_mean'] + group1['over_dev']
#
#     # Draw plot with error band and extra formatting to match seaborn style
#     fig, ax = plt.subplots(figsize=(9, 5))
#     ax.plot(group1['upper_bound'], group1['over_mean'], label='16:00', color='firebrick')
#     ax.plot(group1['upper_bound'], group1['min'], color='firebrick', alpha=0.1)
#     ax.plot(group1['upper_bound'], group1['max'], color='firebrick', alpha=0.1)
#     ax.fill_between(group1['upper_bound'], group1['min'], group1['max'], alpha=0.2)
#     ax.set_xlabel('Upperbound of vehicles controlled')
#     ax.set_ylabel('Optimized rate by rerouting')
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#
#     group2 = groups.get_group("17")
#     group2['min'] = group2['over_mean'] - group2['over_dev']
#     group2['max'] = group2['over_mean'] + group2['over_dev']
#
#     ax.plot(group2['upper_bound'], group2['over_mean'], label='17:00', color='darkviolet')
#     ax.plot(group2['upper_bound'], group2['min'], color='darkviolet', alpha=0.1)
#     ax.plot(group2['upper_bound'], group2['max'], color='darkviolet', alpha=0.1)
#     ax.fill_between(group2['upper_bound'], group2['min'], group2['max'], alpha=0.2)
#     ax.set_xlabel('Upperbound of vehicles controlled')
#     ax.set_ylabel('Optimized rate by rerouting')
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#
#     group3 = groups.get_group("18")
#     group3['min'] = group3['over_mean'] - group3['over_dev']
#     group3['max'] = group3['over_mean'] + group3['over_dev']
#
#     ax.plot(group3['upper_bound'], group3['over_mean'], label='18:00', color='royalblue')
#     ax.plot(group3['upper_bound'], group3['min'], color='royalblue', alpha=0.1)
#     ax.plot(group3['upper_bound'], group3['max'], color='royalblue', alpha=0.1)
#     ax.fill_between(group3['upper_bound'], group3['min'], group3['max'], alpha=0.2)
#     ax.set_xlabel('Upperbound of vehicles controlled')
#     ax.set_ylabel('Optimized rate by rerouting')
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#
#     group4 = groups.get_group("19")
#     group4['min'] = group4['over_mean'] - group4['over_dev']
#     group4['max'] = group4['over_mean'] + group4['over_dev']
#
#     ax.plot(group4['upper_bound'], group4['over_mean'], label='19:00', color='darkorange')
#     ax.plot(group4['upper_bound'], group4['min'], color='darkorange', alpha=0.1)
#     ax.plot(group4['upper_bound'], group4['max'], color='darkorange', alpha=0.1)
#     ax.fill_between(group4['upper_bound'], group4['min'], group4['max'], alpha=0.2)
#     ax.set_xlabel('Upperbound of vehicles controlled')
#     ax.set_ylabel('Optimized rate by rerouting')
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#
#     fig.legend(labels=['16:00', '17:00', '18:00', '19:00'])
#
#     #     plt.savefig('./visual_optimization_rate/' +tilename + '.pdf')
#     plt.show()


if __name__ == '__main__':
    i = 0
    
    # generate total form
    
    form_df = pd.DataFrame()
    
    for day in days:
        print('--------------------------------' + str(day)+ '--------------------------------')
        
        for region in regions:
            print('--------------------------------' + str(region) + '--------------------------------')
            
            file_paths_pre = '../Shortest_path/' + str(region) + '/'
            file_paths_sufs = [
                'optimization_rate_' + str(
                    region) + '_Product_withMin_withoutQhat_99999.0_99999.0_5_Nestgra_stepsize0.05_walkingspeed3.5_enlarge1.0_looptimes200_withMandist_' + str(
                    day) + '.pickle',
                'optimization_rate_' + str(
                    region) + '_Product_withMin_withoutQhat_99999.0_99999.0_10_Nestgra_stepsize0.05_walkingspeed3.5_enlarge1.0_looptimes200_withMandist_' + str(
                    day) + '.pickle',
                'optimization_rate_' + str(
                    region) + '_Product_withMin_withoutQhat_99999.0_99999.0_15_Nestgra_stepsize0.05_walkingspeed3.5_enlarge1.0_looptimes200_withMandist_' + str(
                    day) + '.pickle',
                'optimization_rate_' + str(
                    region) + '_Product_withMin_withoutQhat_99999.0_99999.0_20_Nestgra_stepsize0.05_walkingspeed3.5_enlarge1.0_looptimes200_withMandist_' + str(
                    day) + '.pickle',
                'optimization_rate_' + str(
                    region) + '_Product_withMin_withoutQhat_99999.0_99999.0_25_Nestgra_stepsize0.05_walkingspeed3.5_enlarge1.0_looptimes200_withMandist_' + str(
                    day) + '.pickle',
                'optimization_rate_' + str(
                    region) + '_Product_withMin_withoutQhat_99999.0_99999.0_30_Nestgra_stepsize0.05_walkingspeed3.5_enlarge1.0_looptimes200_withMandist_' + str(
                    day) + '.pickle'
            ]
            
            if region == 'CenterPark':
                file_paths_sufs = file_paths_sufs + [
                    'optimization_rate_' + str(
                        region) + '_Product_withMin_withoutQhat_0.0_1.4_15_Nestgra_stepsize0.05_walkingspeed3.5_enlarge1.0_looptimes200_withMandist_' + str(
                        day) + '.pickle',
                    'optimization_rate_' + str(
                        region) + '_Product_withMin_withoutQhat_0.0_1.5_15_Nestgra_stepsize0.05_walkingspeed3.5_enlarge1.0_looptimes200_withMandist_' + str(
                        day) + '.pickle',
                    'optimization_rate_' + str(
                        region) + '_Product_withMin_withoutQhat_0.0_1.6_15_Nestgra_stepsize0.05_walkingspeed3.5_enlarge1.0_looptimes200_withMandist_' + str(
                        day) + '.pickle',
                    'optimization_rate_' + str(
                        region) + '_Product_withMin_withoutQhat_0.0_1.7_15_Nestgra_stepsize0.05_walkingspeed3.5_enlarge1.0_looptimes200_withMandist_' + str(
                        day) + '.pickle',
                    'optimization_rate_' + str(
                        region) + '_Product_withMin_withoutQhat_0.0_1.8_15_Nestgra_stepsize0.05_walkingspeed3.5_enlarge1.0_looptimes200_withMandist_' + str(
                        day) + '.pickle']
            if region == 'MidTown':
                file_paths_sufs = file_paths_sufs + [
                    'optimization_rate_' + str(
                        region) + '_Product_withMin_withoutQhat_0.0_2.1_15_Nestgra_stepsize0.05_walkingspeed3.5_enlarge1.0_looptimes200_withMandist_' + str(
                        day) + '.pickle',
                    'optimization_rate_' + str(
                        region) + '_Product_withMin_withoutQhat_0.0_2.2_15_Nestgra_stepsize0.05_walkingspeed3.5_enlarge1.0_looptimes200_withMandist_' + str(
                        day) + '.pickle',
                    'optimization_rate_' + str(
                        region) + '_Product_withMin_withoutQhat_0.0_2.3_15_Nestgra_stepsize0.05_walkingspeed3.5_enlarge1.0_looptimes200_withMandist_' + str(
                        day) + '.pickle',
                    'optimization_rate_' + str(
                        region) + '_Product_withMin_withoutQhat_0.0_2.4_15_Nestgra_stepsize0.05_walkingspeed3.5_enlarge1.0_looptimes200_withMandist_' + str(
                        day) + '.pickle',
                    'optimization_rate_' + str(
                        region) + '_Product_withMin_withoutQhat_0.0_2.5_15_Nestgra_stepsize0.05_walkingspeed3.5_enlarge1.0_looptimes200_withMandist_' + str(
                        day) + '.pickle'
                ]
                
            if region == 'WallStreet':
                file_paths_sufs = file_paths_sufs + [
                    'optimization_rate_' + str(
                        region) + '_Product_withMin_withoutQhat_0.0_1.4_15_Nestgra_stepsize0.05_walkingspeed3.5_enlarge1.0_looptimes200_withMandist_' + str(
                        day) + '.pickle',
                    'optimization_rate_' + str(
                        region) + '_Product_withMin_withoutQhat_0.0_1.5_15_Nestgra_stepsize0.05_walkingspeed3.5_enlarge1.0_looptimes200_withMandist_' + str(
                        day) + '.pickle',
                    'optimization_rate_' + str(
                        region) + '_Product_withMin_withoutQhat_0.0_1.6_15_Nestgra_stepsize0.05_walkingspeed3.5_enlarge1.0_looptimes200_withMandist_' + str(
                        day) + '.pickle',
                    'optimization_rate_' + str(
                        region) + '_Product_withMin_withoutQhat_0.0_1.7_15_Nestgra_stepsize0.05_walkingspeed3.5_enlarge1.0_looptimes200_withMandist_' + str(
                        day) + '.pickle',
                    'optimization_rate_' + str(
                        region) + '_Product_withMin_withoutQhat_0.0_1.8_15_Nestgra_stepsize0.05_walkingspeed3.5_enlarge1.0_looptimes200_withMandist_' + str(
                        day) + '.pickle',
    
                    'optimization_rate_' + str(
                        region) + '_Product_withMin_withoutQhat_0.0_2.1_15_Nestgra_stepsize0.05_walkingspeed3.5_enlarge1.0_looptimes200_withMandist_' + str(
                        day) + '.pickle',
                    'optimization_rate_' + str(
                        region) + '_Product_withMin_withoutQhat_0.0_2.2_15_Nestgra_stepsize0.05_walkingspeed3.5_enlarge1.0_looptimes200_withMandist_' + str(
                        day) + '.pickle',
                    'optimization_rate_' + str(
                        region) + '_Product_withMin_withoutQhat_0.0_2.3_15_Nestgra_stepsize0.05_walkingspeed3.5_enlarge1.0_looptimes200_withMandist_' + str(
                        day) + '.pickle',
                    'optimization_rate_' + str(
                        region) + '_Product_withMin_withoutQhat_0.0_2.4_15_Nestgra_stepsize0.05_walkingspeed3.5_enlarge1.0_looptimes200_withMandist_' + str(
                        day) + '.pickle',
                    'optimization_rate_' + str(
                        region) + '_Product_withMin_withoutQhat_0.0_2.5_15_Nestgra_stepsize0.05_walkingspeed3.5_enlarge1.0_looptimes200_withMandist_' + str(
                        day) + '.pickle'
                
                ]
            
            for file_paths_suf in file_paths_sufs:
                print('=========' + str(i) + '=========')
                print(file_paths_suf)
                
                file_path = file_paths_pre + file_paths_suf
                
                if os.path.exists(file_path):
                    with open(file_path, 'rb') as f:
                        file_df = pickle.load(f)
                        complete_array, complete_array_ard = generate_form(file_df)
                        
                        print(complete_array_ard)
                        
                        complete_df = pd.DataFrame(complete_array_ard)
                        complete_df = complete_df.rename(
                            columns={0: 'ori_mean', 1: 'ori_dev', 2: 'opt_mean', 3: 'opt_dev', 4: 'over_mean',
                                     5: 'over_dev'})
                        #                         print('complete_df', complete_df)
                        file_path_list = file_path.split('_')
                        a, b, c, region_name, days_type = file_path_list[7], file_path_list[8], file_path_list[9], \
                                                          file_path_list[3], file_path_list[16].split('.')[0]
                        a_list, b_list, c_list, region_name_list, days_type_list = [a] * 5, [b] * 5, [c] * 5, [
                            region_name] * 5, [days_type] * 5
                        time_slot = ['24', '16', '17', '18', '19']
                        
                        complete_df['time_slot'] = pd.DataFrame(time_slot)
                        complete_df['lower_bound'] = pd.DataFrame(a_list)
                        complete_df['upper_bound'] = pd.DataFrame(b_list)
                        complete_df['vehicle_amount'] = pd.DataFrame(c_list)
                        complete_df['region_name'] = pd.DataFrame(region_name_list)
                        complete_df['days_type'] = pd.DataFrame(days_type_list)
                        
                        # print( complete_df)
                        form_df = form_df.append(complete_df, ignore_index=True)
                        
                        # print(form_df)
                        i += 1
                else:
                    print('file doesn\'t exit')

    
    # plot subfigures
    for day in days:
        #         print('--------------------------------' + str(day)+ '--------------------------------')
        for region in regions:
            plot_amount_sub(region, day, form_df)
            plot_upperbound_sub(region, day, form_df)