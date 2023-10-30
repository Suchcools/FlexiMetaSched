import numpy as np
import random
import pandas as pd
np.random.seed(42)
random.seed(42)

def read_env(path):
    # 定义参数
    Job_pd=pd.read_excel(path,sheet_name='Job info')
    Mac_pd=pd.read_excel(path,sheet_name='Machine info')
    Pr1_pd=pd.read_excel(path,sheet_name='Process 1 Time')
    Pr2_pd=pd.read_excel(path,sheet_name='Process 2 Time')
    # 定义问题的参数
    J = len(Job_pd)
    M = len(Mac_pd)
    W = Job_pd['准备时间'].mean()  # 订单的 Task0 和 Task1 之间的准备时间
    N = Job_pd['需求量'].values  # 订单的需求量
    A = Job_pd['到达时刻'].values  # 订单的到达时刻
    D = Job_pd['交货期'].values  # 订单的交货期
    pt= np.array([Pr1_pd[['Machine_0','Machine_1']].values,Pr2_pd[['Machine_0','Machine_1']].values])# 订单的 Task0 在生产线 i 上的单位处理时长 # 订单的 Task1 在生产线 i 上的单位处理时长
    p = np.array([Mac_pd[['闲置功率','工作功率']].values[:,0],Mac_pd[['闲置功率','工作功率']].values[:,1]])   # 生产线的闲置功率和工作功率
    pop = generate_population(20, J, M)
    return (pop,J,M,W,N,A,D,pt,p)

def parse_env(path):
    # 定义参数
    Job_pd=pd.read_excel(path,sheet_name='Job info')
    Mac_pd=pd.read_excel(path,sheet_name='Machine info')
    Pr1_pd=pd.read_excel(path,sheet_name='Process 1 Time')
    Pr2_pd=pd.read_excel(path,sheet_name='Process 2 Time')
    # 定义问题的参数
    J = len(Job_pd)
    M = len(Mac_pd)
    W = Job_pd['准备时间'].mean()  # 订单的 Task0 和 Task1 之间的准备时间
    N = Job_pd['需求量'].values  # 订单的需求量
    A = Job_pd['到达时刻'].values  # 订单的到达时刻
    D = Job_pd['交货期'].values  # 订单的交货期
    pt= np.array([Pr1_pd[['Machine_0','Machine_1']].values,Pr2_pd[['Machine_0','Machine_1']].values])# 订单的 Task0 在生产线 i 上的单位处理时长 # 订单的 Task1 在生产线 i 上的单位处理时长
    p = np.array([Mac_pd[['闲置功率','工作功率']].values[:,0],Mac_pd[['闲置功率','工作功率']].values[:,1]])   # 生产线的闲置功率和工作功率
    return J, M, A, D, N, pt, p, W


## 初始化种群
def generate_population(population_size, J, M):
    population = []
    for i in range(population_size):
        chromosome = random_chromosome(J, M)
        population.append(chromosome)
    return population


def random_chromosome(J, M):
    
    # 初始化染色体
    chromosome = np.zeros((2 * J, 3))

    # 随机生成Task0和Task1被分配到的生产线，使用二项分布
    result = np.random.binomial(M - 1, 0.5, 2 * J)
    proportion = np.sum(result) / len(result)

    # 控制Task0和Task1被分配到生产线的比例在0.4到0.6之间
    while proportion < 0.4 or proportion > 0.6:
        result = np.random.binomial(1, 0.5, 2 * J)
        proportion = np.sum(result) / len(result)

    # 将生成的Task0和Task1分配结果添加到染色体中
    chromosome[:, 0] = np.repeat(np.arange(J), 2)
    chromosome[:, 1] = result

    # 将染色体随机打乱，保证不同工件的分配顺序不同
    np.random.shuffle(chromosome)

    # 遍历第一列，记录已经出现过的订单编号
    appeared = set()
    for i in range(2 * J):
        current_value = chromosome[i, 0]
        if current_value in appeared:
            chromosome[i, 2] = 1
        else:
            appeared.add(current_value)
            chromosome[i, 2] = 0

    return chromosome.astype(int).tolist()

def calculate_cost(chromosome, J, M, A, D, N, pt, p, W):
    task0_start_time = {} # 创建空字典，存储Task0的开始时间
    task0_end_time = {} # 创建空字典，存储Task0的结束时间
    task1_start_time = {} # 创建空字典，存储Task1的开始时间
    delay_time = {} # 创建空字典，存储订单延迟时间
    production_line_status = {} # 创建空字典，存储生产线的占用情况
    production_line_power = {} # 创建空字典，存储生产线的功耗
    for i in range(M): # 初始化生产线的占用情况和功耗
        production_line_status[i] = 0
        production_line_power[i] = 0
    for j, m, t in chromosome: # 遍历个体 # 订单 产线 任务
        if t == 0: # 如果当前任务是Task0
            if m in production_line_status: # 如果生产线占用
                task0_start_time[j] = max(A[j], production_line_status[m]) # 取较大值作为任务开始时间
            else: # 如果生产线未被占用
                task0_start_time[j] = A[j] # 任务开始时间为订单到达时刻
            processing_time = N[j] * pt[t][j][m] # 计算任务加工时间
            production_line_power[m] += (task0_start_time[j] - production_line_status[m]) * p[m][0] + processing_time * p[m][1] # 更新生产线的功耗
            production_line_status[m] = task0_start_time[j] + processing_time # 更新生产线的占用情况
            task0_end_time[j] = task0_start_time[j] + processing_time # 更新生产线的占用情况
        elif t == 1: # 如果当前任务是Task1
            task1_start_time[j] = max(task0_end_time[j] + W, production_line_status[m]) # 取较大值作为任务开始时间
            processing_time = N[j] * pt[t][j][m] # 计算任务加工时间
            production_line_power[m] += (task1_start_time[j] - production_line_status[m]) * p[m][0] + processing_time * p[m][1] # 更新生产线的功耗
            production_line_status[m] = task1_start_time[j] + processing_time # 更新生产线的占用情况
            delay_time[j] = production_line_status[m] # 计算延期时间

    return delay_time, production_line_status, production_line_power 

def fitness(chromosome, J, M, A, D, N, pt, p, W):
    delay_time, production_line_status, production_line_power = calculate_cost(chromosome, J, M, A, D, N, pt, p, W)
    delay_time = sum(delay_time.values())
    total_time = max(production_line_status.values()) # 产线完成时间的最大值
    total_power = sum(production_line_power.values()) # 计算所有生产线的总功耗
    return -1.0 * (1* total_time + 0.12 * total_power + 0.007 * delay_time) # 返回适应度的倒数

def mutate(chromosome,prob=0.1): #选择订单序号，两两交换，保证Task位置相对不变
    for i in range(max(1,int(prob*len(chromosome)/2))): #突变订单长度的多少次
        idx1, idx2 = random.sample(range(len(chromosome)//2), 2)#选出两个订单
        job1, job2 = [i for i, tup in enumerate(chromosome) if tup[0] == idx1], [i for i, tup in enumerate(chromosome) if tup[0] == idx2] #找出订单的Task0和Task1的index：
        chromosome[job1[0]][1], chromosome[job1[1]][1]=random.randint(0,1),random.randint(0,1) # 随机更改产线
        swap=chromosome[job1[0]], chromosome[job1[1]] #临时变量
        chromosome[job2[0]][1], chromosome[job2[1]][1]=random.randint(0,1),random.randint(0,1)# 随机更改产线
        chromosome[job1[0]], chromosome[job1[1]] = chromosome[job2[0]], chromosome[job2[1]]#任务一定相同，则进行交换
        chromosome[job2[0]], chromosome[job2[1]] = swap#任务一定相同，则进行交换
    return chromosome

def sort_task(chromosome,J):
    # 遍历第一列，记录已经出现过的订单编号
    appeared = set()
    for i in range(2 * J):
        current_value = chromosome[i, 0]
        if current_value in appeared:
            chromosome[i, 2] = 1
        else:
            appeared.add(current_value)
            chromosome[i, 2] = 0
    return chromosome

def ruler(chromosome,J):
    arr = chromosome[:,0]
    rank = arr.argsort().argsort()
    seq = np.arange(arr.size)
    chromosome[:,0] = seq[rank]//2
    chromosome[:,1]= np.where(chromosome[:,1] <= chromosome[:,1].mean(), 0, 1)
    chromosome=sort_task(chromosome,J)
    return chromosome.astype(int).tolist()



# def formate(x,test,J):
#     return [ruler(np.array(eval(x)),J) for x in test.values]

# # 定义退火函数
# def temperature_plot():
#     # 参数设置
#     initial_temperature=1000
#     cooling_rate=0.001
#     iterations=3000
#     final_temperature=0.01

#     # 初始化温度
#     temperature = initial_temperature
#     temperature_list = [temperature]  # 记录温度变化

#     # 迭代退火过程
#     for i in range(iterations):
#         # 降温
#         temperature = initial_temperature * (cooling_rate ** (i / iterations))
#         temperature_list.append(temperature)  # 记录温度变化
#         if temperature < final_temperature:
#             break

#     # 可视化降温曲线
#     plt.plot(range(iterations+1), temperature_list)
#     plt.title('Temperature Annealing Curve')
#     plt.xlabel('Iteration')
#     plt.ylabel('Temperature')
#     plt.show()