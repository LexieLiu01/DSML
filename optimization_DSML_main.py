import cvxpy as cp
from optimization_DSML_functions import *

def optimization(region2theta_dict, region_speed_dict, trip_dict, time, ids, r_s, r_s_a, M0, M1, M2,
                 neighboring_nodes_subdistance,shortest_paths_path,a,b,c,grad_method, beta, step_size,loop_times,Sens_idct):
    
    if grad_method == 'Adagra':
        # step_size = 0.1
        F_sum_g_square = 1e-6
        H_sum_g_square = 1e-6
    
    Delta_A_value, D, Q, Q_hat = generate_by_hour(trip_dict,time,ids, r_s, M1)
    
    if Q_hat_idct == 'withQhat':
        Q = Q * 1
        Q_hat = Q * c
        D = D * 1
    elif Q_hat_idct == 'withoutQhat':
        Q = Q * c
        D = D * c
        # print('the demand:',D)
        
    # Q_hat = Q * 20
    # print('Path flow:',Q)
    
    result_optimization = {}
    
    # F_value = Q.copy()
    # H_value = np.zeros((len(r_s_a), 1))
    
    # H_value = np.random.randn(len(r_s_a), 1)
    # F_value = Q.copy() - M0 @ H_value

    C, T = generate_by_epoch(Delta_A_value, region_speed_dict, time, region2theta_dict, neighboring_nodes_subdistance,
                             shortest_paths_path, r_s, r_s_a, -1)
    
    result_optimization[loop_times] = {}
    if Q_hat_idct == 'withQhat':
        result_optimization[loop_times]['objective_function'] = ((Q_hat + Q).T @ T).flatten()[0]
    elif Q_hat_idct == 'withoutQhat':
        result_optimization[loop_times]['objective_function'] = (Q.T @ T).flatten()[0]
    
    # print("Orginal TT",result_optimization[loop_times]['objective_function'])
    seed_F = cp.Variable((len(r_s), 1))
    seed_H = cp.Variable((len(r_s_a), 1))
    F_value_seed = np.random.rand(len(r_s), 1)

    tmp_objective = cp.Minimize(cp.sum_squares(F_value_seed - seed_F))
    
    if a == 99999.0 and b == 99999.0:
        tmp_constraint = [Q == seed_F + M0 @ seed_H, seed_F >= 0, seed_H >= 0]
    elif Sens_idct == 'Product':
        tmp_constraint = [Q == seed_F + M0 @ seed_H, seed_F >= 0, seed_H >= 0, M1@seed_F + M2@seed_H >= a * D, M1@seed_F + M2@seed_H <= b * D]
    elif Sens_idct == 'Add':
        tmp_constraint = [Q == seed_F + M0 @ seed_H, seed_F >= 0, seed_H >= 0, M1 @ seed_F + M2 @ seed_H >= D - a, M1 @ seed_F + M2 @ seed_H <= D + b]
        
    prob = cp.Problem(tmp_objective, tmp_constraint)
    prob.solve(verbose = True)
    H_value = seed_H.value
    F_value = seed_F.value
    
    Tilde_D_value = np.matmul(M1, F_value) + np.matmul(M2, H_value)
    Delta_A_value = Tilde_D_value - D
    
    for __i in range(loop_times):
        
        C, T = generate_by_epoch(Delta_A_value, region_speed_dict, time, region2theta_dict, neighboring_nodes_subdistance, shortest_paths_path, r_s, r_s_a, __i)
        # print('Time:', time, 'Optimized time cost at epoch'+'str(__i)'+':',T,'Original time cost at each path flow:',Original_T,'Change of optimized time cost:', T - Original_T)
        
        result_optimization[__i] = {}

        # Generate Varibles:
        # Shape of H: (len(r_s_a), 1)
        H = cp.Variable((len(r_s_a), 1))
        # Shape of F: (len(r_s), 1)
        F = cp.Variable((len(r_s), 1))

        if Q_hat_idct == 'withQhat':
            result_optimization[__i]['objective_function'] = ((Q_hat + F_value).T @ T + H_value.T @ C).flatten()[0]
            objective = cp.Maximize(cp.sum((Q_hat + F).T @ T) + cp.sum(H.T @ C))
        elif Q_hat_idct == 'withoutQhat':
            result_optimization[__i]['objective_function'] = (F_value.T @ T + H_value.T @ C).flatten()[0]
            objective = cp.Minimize(cp.sum(F.T @ T) + cp.sum(H.T @ C))
        
        # print(result_optimization[__i]['objective_function'])
        # Generate Delta D_a
        # Tilde_D = M1@F + M2@H
        #print(Tilde_D)

        if a == 99999.0 and b == 99999.0:
            constraints = [Q == F + M0 @ H, F >= 0, H >= 0]
        elif Sens_idct == 'Product':
            constraints = [Q == F + M0 @ H, F >= 0, H >= 0, M1@F + M2@H >= a * D, M1@F + M2@H <= b * D]
        elif Sens_idct == 'Add':
            constraints = [Q == F + M0 @ H, F >= 0, H >= 0, M1 @ F + M2 @ H >= D - a, M1 @ F + M2 @ H <= D + b]
        
        prob = cp.Problem(objective, constraints)
        
        # result = prob.solve(solver='OSQP', verbose = True)
        result = prob.solve()

        # print(prob.value)
        # print('optimized F:', F.value, 'optimized H:', H.value)

        if grad_method == 'Adagra':
            F_grad =  F_value -F.value
            F_sum_g_square = F_sum_g_square + np.power(F_grad, 2)
            F_value -= step_size * F_grad / np.sqrt(F_sum_g_square)
            H_grad = H_value - H.value
            H_sum_g_square = H_sum_g_square + np.power(H_grad, 2)
            H_value -= step_size * H_grad / np.sqrt(H_sum_g_square)
            
        elif grad_method == 'Nestgra':
            
            F_grad = F_value - F.value
            F_value_upd = F.value - beta*F_grad
            F_value = (1 - step_size) * F_value + step_size * F_value_upd
            
            H_grad = H_value - H.value
            H_value_upd = H.value - beta*H_grad
            H_value = (1 - step_size) * H_value + step_size * H_value_upd
            
        elif grad_method == 'Normal':
            F_value = (1-step_size) * F_value + step_size * F.value
            H_value = (1-step_size) * H_value + step_size* H.value
            # print((Q_hat + F_value).T @ T + H_value.T @ C)

        Tilde_D_value = np.matmul(M1, F_value) + np.matmul(M2, H_value)
        Delta_A_value = Tilde_D_value - D
        # print("Tilde_D_value:",Tilde_D_value)
        # print("Delta_A_value:",Delta_A_value)
        # print('Sum of Delta_value',Delta_A_value.sum())

        result_optimization[__i]['Delta_A_value'] = Delta_A_value
        
        # if __i !=0 and np.abs(result_optimization[__i]['objective_function'] - result_optimization[__i - 1]['objective_function']) < 0.001:
        #     break
        
    return result_optimization

if __name__ == '__main__':

    # Generate time sequence
    time_seq = generate_time_sequence(time_intervals)

    # Get center point in region
    center_point_region_dict, region_polygon_gpd = get_center_point(input_region_geo_Manhattan_path, center_point_region_path, ids)

    # Get neighboring regions
    neighbor_region_dict = get_neighbor_region(neighbor_region_path, region_polygon_gpd)

    result_final = {}

    # Get theta
    region2theta_dict = get_theta(theta_path, ids)

    # Get neighboring nodes subdistance
    neighboring_nodes_subdistance = get_neighboring_nodes_subdistance(neighbor_region_dict, center_point_region_dict, region_polygon_gpd, neighboring_nodes_subdistance_path)

    # Get original speed before rerouting
    region_speed_dict = initial_region_speed(trip_speed_precip_for_train_path, ids, region_speed_dict_path)
    trip_dict = initial_region_pickup_dropoff(input_trip_df_path, trip_dict_path)

    # Prepare list and matrix
    # JUST FINANCE STREET
    r_s, r_s_a, M0, M1, M2 = prepare_matrix(id_o, id_d, neighbor_region_dict, prepare_matrix_dict_path)
    # NORMAL
    # r_s, r_s_a, M0, M1, M2 = prepare_matrix(ids, neighbor_region_dict, prepare_matrix_dict_path)

    for time in time_seq:
        result_optimization = optimization(region2theta_dict, region_speed_dict, trip_dict, time, ids, r_s, r_s_a, M0, M1, M2,
                 neighboring_nodes_subdistance,shortest_paths_path,a,b,c,grad_method, beta, step_size, loop_times, Sens_idct)

        result_final[time] = result_optimization
        pickle.dump(result_final, open(result_final_path, "wb"))

    plot_figure(result_final_path, loop_times)
    calculate_oprimized_rate(result_final_path, loop_times, result_rate_path)