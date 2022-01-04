import pandas as pd
import json
import numpy as np

import datetime
from datetime import date, datetime, timedelta

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression, LogisticRegressionCV

from econml.sklearn_extensions.model_selection import GridSearchCVList
from econml.dml import LinearDML

from econml.cate_interpreter import SingleTreeCateInterpreter
import matplotlib
import os
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
import numpy as np
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
import statsmodels.api as sm

trip_speed_precip_X_label_DBSCAN_path = './data_set/Prepared_matrix_completion_data/trip_speed_precip_X_M_Nei_more_parameters.csv'

ids = [4, 12, 13, 24, 41, 42, 43, 45, 48, 50, 68, 74, 75, 79, 87, 88, 90, 100, 107, 113, 114, 116, 120, 125, 127, 128, 137, 140, 141, 142, 143, 144, 148, 151, 152, 158, 161, 162, 163, 164, 166, 170, 186, 209, 211, 224, 229, 230, 231, 232, 233, 234, 236, 237, 238, 239, 243, 244, 246, 249, 261, 262, 263]
start = datetime(2019, 2, 1)
end = datetime(2020, 6, 30)

# Saturday, Sunday
target_weekday = ['0','6']
# Tues, Wed, Thurs
# target_weekday = ['2', '3', '4']
identify_target_weekday = '_'.join(target_weekday)

target_hour = ['16', '17', '18', '19']
identify_target_hour = '_'.join(target_hour)

data_for_training_path = './model_dml/training_data/' + 'training_data_dsml_XWXMW_more_parameters'+identify_target_weekday+'_'+ identify_target_hour +'.pickle'

holiday_list = ['2019-04-21', '2019-04-22', '2019-04-25', '2019-05-01', '2019-06-02', '2019-08-15', '2019-11-01',
                '2019-12-08', '2019-12-25', '2019-12-26', '2020-01-01', '2020-01-06', '2020-04-12', '2020-04-13',
                '2020-04-25', '2020-05-01', '2020-06-02']

# you can select different models for training.
# model = 'DML'
model = 'DSML'
# model = 'LR'

# you can select the machine learning methods.
# tag = 'GB'
# tag = 'RF'
# tag = 'Ada'
tag = 'all'

data_final_result_path = './model_dml/theta/' + 'theta_' + model + '_' + identify_target_weekday + '_' + identify_target_hour + tag + '.pickle'

# Filter target hours and target days
def filter_different_days(trip_speed_precip_X_label_DBSCAN_path, holiday_list, data_for_training_path):
    
    if os.path.exists(data_for_training_path):
        with open(data_for_training_path, 'rb') as f:
            training_data_for_evening_df = pickle.load(f)
        return training_data_for_evening_df
    
    training_data_df = pd.read_csv(trip_speed_precip_X_label_DBSCAN_path)
    
    training_data_df['datetime_min_5_strp'] = training_data_df['datetime_min_5'].apply(
        lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M'))
    training_data_df['weekday'] = training_data_df['datetime_min_5_strp'].apply(lambda x: x.strftime("%w"))
    training_data_df['day'] = training_data_df['datetime_min_5'].apply(lambda x: x[0:10])
    training_data_df['hour'] = training_data_df['datetime_min_5'].apply(lambda x: x[11:13])
    
    training_data_df = training_data_df[~training_data_df['day'].isin(holiday_list)]
    training_data_df = training_data_df[(training_data_df['weekday'].isin(target_weekday)) & (training_data_df['hour'].isin(target_hour))]
    training_data_for_evening_df = training_data_df[['region_id', 'datetime_min_5', 'relative_speed', 'total_number', 'X', 'M', 'precip_in']]

    with open(data_for_training_path, 'wb') as f:
        pickle.dump(training_data_for_evening_df, f)
    
    return training_data_for_evening_df

def ge_tr_data(training_data_for_evening_df, id):
    training_data_df = training_data_for_evening_df[training_data_for_evening_df['region_id'] == id]
    training_data_df = training_data_df[training_data_df['total_number'] != 0]
    
    X = []
    X_str = training_data_df["X"].values.tolist()
    
    for x_str in X_str:
        x = json.loads(x_str)
        X.append(x)
    X = np.array(X)
    
    Y = training_data_df["relative_speed"].values.tolist()
    Y = np.array(Y)
    
    T = training_data_df["total_number"].values.tolist()
    
    # plt.hist(T, bins=10)
    # plt.savefig('./model_dml/T/' + str(id) + 'T_id.png')
 
    M = []
    M_str = training_data_df["M"].values.tolist()

    for m_str in M_str:
        m = json.loads(m_str)
        M.append(m)
    M = np.array(M)
    M = np.reshape(M, (-1, 10))

    W = training_data_df["precip_in"].values.tolist()
    W = np.array(W)
    # W = np.diff(W, axis=0)
    
    # T = T[10:, :]
    T = np.reshape(T, (-1, 1))
    Y = np.reshape(Y, (-1, 1))
    W = np.reshape(W, (-1, 1))

    X = X // 5

    # category
    X[X <= 3] = 0
    X[X == 4] = 1
    X[X == 5] = 2
    X[X == 6] = 3
    X[X >= 7] = 4
    
    M = M // 5
    
    return X,Y,T,M,W

# DMSL model
def DSML_model_train(X,Y,T,M,W):
    All_Array = np.hstack((Y,T,X,M,W))
    
    kf5 = KFold(n_splits=5, shuffle=False)
    
    Y_res_total = []
    T_res_total = []
    
    for train_index, test_index in kf5.split(All_Array):
        # print(train_index, test_index)
        
        train_data = All_Array[train_index, :]
        # print(train_data.shape)
        test_data = All_Array[test_index, :]
        # print(test_data.shape)
        
        Y_train = train_data[:, 0]
        Y_train = np.reshape(Y_train, (-1, 1))
        Y_val = test_data[:, 0]
        Y_val = np.reshape(Y_val, (-1, 1))
        # print(Y_train.shape)
        
        T_train = train_data[:, 1]
        T_train = np.reshape(T_train, (-1, 1))
        T_val = test_data[:, 1]
        T_val = np.reshape(T_val, (-1, 1))
        # print(T_train.shape)
        
        X_train = train_data[:, 2:22]
        X_val = test_data[:, 2:22]
        # print(X_train.shape)
        
        M_train = train_data[:, 22:-1]
        M_val = test_data[:, 22:-1]
        # print(M_train.shape)
        
        W_train = train_data[:, -1]
        W_train = np.reshape(W_train, (-1, 1))
        W_val = test_data[:, -1]
        W_val = np.reshape(W_val, (-1, 1))
        # print(W_train.shape)
        
        first_stage = lambda: GridSearchCVList(
            
            estimator_list =[GradientBoostingRegressor(), RandomForestRegressor(), AdaBoostRegressor()],
            param_grid_list=[
                [
                    {"alpha": [0.001, 0.01, 0.1]},
                    {"max_depth": [3, 5, None], "n_estimators": [10, 40, 70, 100, 130, 160, 190, 200]},
                    {"random_state": [0]}

                ],

                [

                    {"max_depth": [3, 5, None], "n_estimators": [10, 40, 70, 100, 130, 160, 190, 200]},
                    {"random_state": [0]}
                ],

                [

                    {"n_estimators": [10, 40, 70, 100, 130, 160, 190, 200]},
                    {"random_state": [0]}
                ]
            ],
            n_jobs=-1, scoring='neg_mean_squared_error'
        )
        
        XW_train = np.hstack((X_train, W_train))
        XW_val = np.hstack((X_val, W_val))
        
        XMW_train = np.hstack((X_train, M_train, W_train))
        XMW_val = np.hstack((X_val, M_val, W_val))
        
        #print(Y_train.shape, Y_val.shape, T_train.shape, T_val.shape, X_train.shape, X_val.shape, W_train.shape,W_val.shape, XW_train.shape, XW_val.shape, MW_train.shape, MW_val.shape)
        #print(XW_train.shape, Y_train.shape, XW_val.shape)
        model_y = first_stage().fit(XW_train, Y_train.flatten()).best_estimator_
        Y_hat = model_y.predict(XW_val)
        Y_res = Y_hat - Y_val.flatten()
        
        #print(MW_train.shape, T_train.shape, MW_val.shape)
        model_t = first_stage().fit(XMW_train, T_train.flatten()).best_estimator_
        T_hat = model_t.predict(XMW_val)
        T_res = T_hat - T_val.flatten()
        
        Y_res_total.append(Y_res)
        T_res_total.append(T_res)

    Y_res_total = np.concatenate(Y_res_total)
    T_res_total = np.concatenate(T_res_total)
    
    modely_score = np.sqrt(np.mean(Y_res_total ** 2))
    modelt_score = np.sqrt(np.mean(T_res_total ** 2))
    
    # store the value of theta
    T_res_total_OLS = sm.add_constant(T_res_total)
    regr = sm.OLS(endog=Y_res_total, exog=T_res_total_OLS)
    results = regr.fit()

    #print(results.pvalues)
    
    result_effect = {}
    result_effect['modely_score'] = modely_score
    result_effect['modelt_score'] = modelt_score
    result_effect['p_value'] = results.pvalues
    result_effect['results_params'] = results.params
    result_effect['results_fvalue'] = results.fvalue
    result_effect['results_f_pvalue'] = results.f_pvalue
    result_effect['results_tvalues'] = results.tvalues
    result_effect['results_summary'] = results.summary()
    result_effect['Y_res_total'] = Y_res_total
    result_effect['T_res_total'] = T_res_total
    
    return result_effect

# DML model
def DML_model_train(X,Y,T,M,W):

    X_M = np.hstack((X,M))
    X_M_train, X_M_val, Y_train, Y_val, T_train, T_val, W_train, W_val = train_test_split(X_M, Y, T, W, test_size=0.2)
    
    # adjust super parameter using GridsearchCV
    first_stage = lambda: GridSearchCVList(
        estimator_list=[GradientBoostingRegressor(), RandomForestRegressor(), AdaBoostRegressor()],
        param_grid_list=[
            [
                {"alpha": [0.001, 0.01, 0.1]},
                {"max_depth": [3, 5, None], "n_estimators": [10, 40, 70, 100, 130, 160, 190, 200]},
                {"random_state": [0]}
        
            ],
        
            [
            
                {"max_depth": [3, 5, None], "n_estimators": [10, 40, 70, 100, 130, 160, 190, 200]},
                {"random_state": [0]}
            ],
        
            [
            
                {"n_estimators": [10, 40, 70, 100, 130, 160, 190, 200]},
                {"random_state": [0]}
            ]
    
        ],
        cv=5, n_jobs=6, scoring='neg_mean_squared_error'
    )
    
    X_M_W_train = np.hstack((X_M_train, W_train))
    
    model_y = first_stage().fit(X_M_W_train, Y_train).best_estimator_
    model_t = first_stage().fit(X_M_W_train, T_train).best_estimator_
    est = LinearDML(
        model_y=model_y, model_t=model_t,
    )

    est.fit(Y_train, T_train, X=X_M_train, W=W_train, inference="statsmodels", cache_values=True)
    
    Y_res_total, T_res_total = est._cached_values.nuisances
    # store the value of theta
    result_effect = {}
    pnt_effect = est.const_marginal_ate(X_M_train)
    lb_effect, ub_effect = est.const_marginal_ate_interval(X_M_train, alpha=0.1)

    modely_score = np.sqrt(np.mean(Y_res_total ** 2))
    modelt_score = np.sqrt(np.mean(T_res_total ** 2))
    
    result_effect['modely_score'] = modely_score
    result_effect['modelt_score'] = modelt_score
    result_effect['results_summary'] = est.const_marginal_ate_inference(X_M_train)
    result_effect['p_value'] = str(result_effect['results_summary']).split()[16]
    result_effect['results_params'] = pnt_effect[0][0]
    result_effect['lb_effect'] = lb_effect[0][0]
    result_effect['ub_effect'] = ub_effect[0][0]
    result_effect['Y_res_total'] = Y_res_total
    result_effect['T_res_total'] = T_res_total
    
    return result_effect

# LR model
def LR_model_train(Y, T, W):

    X_train = np.hstack((T, W))
    X_train = sm.add_constant(X_train)
    regr = sm.OLS(endog=Y, exog=X_train)
    results = regr.fit()
    ypred = results.predict(X_train)
    
    modellr_score = np.sqrt(np.mean((ypred -Y) ** 2))

    result_effect = {}
    result_effect['modellr_score'] = modellr_score
    result_effect['p_value'] = results.pvalues[1]
    result_effect['results_params'] = results.params[1]
    result_effect['results_fvalue'] = results.fvalue
    result_effect['results_f_pvalue'] = results.f_pvalue
    result_effect['results_summary'] = results.summary()

    return result_effect

def plot_fig(data_final_result_path,ids,model):
    
    with open(data_final_result_path, 'rb') as f:
        result = pickle.load(f)
    
    for id in ids:
        
        Y_res_total = result[id]['Y_res_total']
        T_res_total = result[id]['T_res_total']

        # Figures Path
        Y_res_fig_path = './model_dml/fig/Y_res/' + str(id) + 'Y_res_' + model+ '_' + identify_target_weekday + '.png'
        T_res_fig_path = './model_dml/fig/T_res/' + str(id) + 'T_res' + model+ '_' + identify_target_weekday + '.png'
        Y_T_res_fig_path = './model_dml/fig/Y_T_res/' + str(
            id) + 'Y_T_res_' + model+ '_' + identify_target_weekday + '.png'
        seaborn_fig_path = './model_dml/fig/Y_T_res/' + str(
            id) + 'Y_T_res_' + model+ '_' + identify_target_weekday + '_seaborn.eps'
        
        # plot the distribution of Y_res
        plt.figure()
        plt.xlabel("Residual Y")
        plt.ylabel("Sequence")
        plt.hist(Y_res_total)
        plt.savefig(Y_res_fig_path)
        
        # plot the distribution of T_res
        plt.figure()
        plt.xlabel("Residual T")
        plt.ylabel("Sequence")
        plt.hist(T_res_total)
        plt.savefig(T_res_fig_path)

        # plot the residual of Y and T; model_final
        plt.figure()
        plt.xlabel("T_res")
        plt.ylabel("Y_res")
        plt.scatter(T_res_total, Y_res_total, s=0.3)
        # plt.show()
        plt.savefig(Y_T_res_fig_path)
        
        # array into dataframe
        # seaborn plot test
        plt.figure()
        sns.set_theme(style="darkgrid")
        # sns.set(style="white", color_codes=True)
        g = sns.jointplot(x=T_res_total, y=Y_res_total,
                          kind="reg", truncate=False,
                          color="m", height=7, scatter_kws = {'s': 2, 'linewidth': 1},
                          # xlim=[-20,None],
                          joint_kws = {'line_kws': {'linewidth': 0.5}})

        regline = g.ax_joint.get_lines()[0]
        regline.set_color('black')
        regline.set_zorder(5)
        g.set_axis_labels('Residual of pick-up/drop-offs', 'Residual of speed', fontsize=30)
        g.savefig(seaborn_fig_path)
        plt.show()
        

if __name__ == '__main__':

    training_data_for_evening_df = filter_different_days(trip_speed_precip_X_label_DBSCAN_path, holiday_list, data_for_training_path)
    result = {}

    for id in ids:

        X,Y,T,M,W = ge_tr_data(training_data_for_evening_df, id)
        print("=============================================================" + str(id) + "===============================================================")

        try:
            if model == 'DSML':
                result_effect = DSML_model_train(X,Y,T,M,W)
            if model == 'DML':
                result_effect = DML_model_train(X,Y,T,M,W)
            if model == 'LR':
                result_effect = LR_model_train(Y, T, W)

            result[id] = result_effect

        except Exception as e:
            print(e)


    with open(data_final_result_path, 'wb') as f:
        pickle.dump(result, f)
    
    # plot_fig(data_final_result_path, ids, model)
