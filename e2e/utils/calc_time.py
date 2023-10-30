import pandas as pd
import numpy as np
def arrayRankTransform(arr):
    return [rank+1 for rank, val in enumerate(sorted(set(arr)))] 


def get_time():
    # 读取Big_list
    Big_list = np.load('Big_list.npy', allow_pickle=True)
    new_task_assignment, order_1, order_2, demand, prepare_time, arrival_time, end_time, \
    free_power_list, work_power_list, Process_1_Time, Process_2_Time = Big_list

    # 将工件的工序1的完成时间和工序2的完成时间分别存入字典
    order = np.argsort(np.concatenate([order_1, order_2]))
    machine_job_list = [[] for _ in range(5)]

    for i in range(5):
        for j, order_id in enumerate(order):
            if order_id % 5 == i and new_task_assignment[order_id % 50] >= 1:
                machine_job_list[i].append(order_id)

    job_end_time_dict, job_start_time_dict, job_power_list, frees_power_list, job_last_time_list = {}, {}, [], [], []
    for i in range(5):
        total_job_power, total_free_power, job_last_time = 0, 0, -1
        job_id_list = [int(j % 50 / 5) for j in machine_job_list[i]]
        order_id_list = machine_job_list[i]
        machine_real_time = 0
        arrival_time_l = [[job_id, arrival_time[job_id], order_id] for job_id, order_id in zip(job_id_list, order_id_list)]
        arrival_time_l = sorted(arrival_time_l, key=lambda x: x[1])

        for ii in range(len(arrival_time_l)):
            min_arrival_time, min_arrival_time_index, order_id_now = arrival_time_l[ii][1], arrival_time_l[ii][0], arrival_time_l[ii][2]

            if order_id_now <= 50:
                per_unit = Process_1_Time[min_arrival_time_index][i]
                full_time = per_unit * new_task_assignment[order_id_now % 50]
                total_job_power += full_time * work_power_list[i]
            else:
                per_unit = Process_2_Time[min_arrival_time_index][i]
                full_time = per_unit * new_task_assignment[order_id_now % 50] + prepare_time[min_arrival_time_index]
                total_job_power += per_unit * new_task_assignment[order_id_now % 50] * work_power_list[i]
                total_free_power += prepare_time[min_arrival_time_index] * free_power_list[i]

            if machine_real_time < min_arrival_time:
                total_free_power += (min_arrival_time - machine_real_time) * free_power_list[i]
                machine_real_time = min_arrival_time

            job_start_time_dict[order_id_now] = machine_real_time
            job_end_time_dict[order_id_now] = machine_real_time + full_time
            machine_real_time = job_end_time_dict[order_id_now]
            job_last_time = max(job_last_time, machine_real_time)

        job_power_list.append(total_job_power)
        frees_power_list.append(total_free_power)
        job_last_time_list.append(job_last_time)

    power_cost = (sum(frees_power_list) + sum(job_power_list)) / 60
    tardiness = sum([job_last_time_list[i] - end_time[i] for i in range(len(job_last_time_list)) if job_last_time_list[i] > end_time[i]])
    alpha = 0.8
    final_obj = alpha * tardiness + (1 - alpha) * power_cost

    print('tardiness:', tardiness, 'power_cost:', power_cost)
    print('final_obj:', final_obj)

    return final_obj
                     
def format_data(data,prob):
    offset_ratio = np.random.uniform(prob-0.05, prob, size=len(data))
    offset_data = data * offset_ratio + data

    return offset_data

def Calc_Time(x, excel_name='./dataset/instance_data/data_10_5_1.xlsx'):
    data_dict = pd.read_excel(excel_name, sheet_name=None)
    job_info = data_dict["Job info"]
    demand = job_info['需求量'].values.tolist()
    prepare_time = job_info['准备时间'].values.tolist()
    arrival_time = job_info['到达时刻'].values.tolist()
    end_time = job_info['交货期'].values.tolist()

    machine_info = data_dict["Machine info"]
    free_power_list = machine_info["闲置功率"].tolist()
    work_power_list = machine_info["工作功率"].tolist()

    columns = ["Machine_0", "Machine_1", "Machine_2", "Machine_3", "Machine_4"]

    process_1_time = data_dict["Process 1 Time"][columns].values.tolist()
    process_2_time = data_dict["Process 2 Time"][columns].values.tolist()

    task_assignment = x[:50]
    new_task_assignment = [int(task_assignment[i] * demand[i // 5]) for i in range(50)]

    order = arrayRankTransform(x[50:150])
    order_1, order_2 = order[:50], order[50:100]

    big_list = [new_task_assignment, order_1, order_2, demand, prepare_time, arrival_time, end_time, free_power_list, work_power_list, process_1_time, process_2_time]

    np.save('./Big_list.npy', big_list)

    return get_time()



if __name__ == '__main__':
    # 固定随机数种子为1
    np.random.seed(1)
    excel_name = 'dataset/instance_data/data_10_5_2.xlsx'
    x = np.random.random(150)
    Calc_Time(x,excel_name)
    get_time()


