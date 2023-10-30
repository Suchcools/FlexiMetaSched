import random
import math
from methods.common import random_chromosome, fitness, mutate
random.seed(42)


# 定义退火函数
def sa(env,record=False):
    # 参数设置
    pop,J,M,W,N,A,D,pt,p = env
    initial_temperature=1000
    cooling_rate=0.001
    iterations=12000
    final_temperature=1
    record_list=[]
    # Fast Annealing, FA 前面快后面慢 降温策略
    # 评估适应度
    fitness_values = [fitness(chromosome, J, M, A, D, N, pt, p, W) for chromosome in pop]
    best_fitness = max(fitness_values)
    current_state = pop[fitness_values.index(best_fitness)]
    current_energy = best_fitness
    best_state = current_state
    best_energy = current_energy
    global_energy = current_energy
    # 初始化温度
    temperature = initial_temperature

    # 迭代退火过程
    for i in range(iterations):
        current_state = best_state  # 将当前状态设置为上一轮迭代的最优状态
        normalized_prob = (  1 - final_temperature) / (initial_temperature - final_temperature)
        new_state = mutate(current_state,prob = max(normalized_prob/2,0.1))  # 根据温度生成一个新状态，并计算能量差
        new_energy = fitness(
            new_state, J, M, A, D, N, pt, p, W)  # 计算新的能量

        delta_energy = new_energy - current_energy  # 计算能量差

        if global_energy < new_energy:
            global_energy = new_energy

        # 判断是否接受新状态  #仔细看看接受函数
        if delta_energy > 0 or random.random() < math.exp(delta_energy / temperature):
            current_state = new_state
            current_energy = new_energy
            if record:
                print(-current_energy)
                record_list.append(-current_energy)


        # 更新最优状态 让退货和解空间相关(利用控制突变个数)
        if current_energy > best_energy:
            best_state = current_state
            best_energy = current_energy


        # 降温策略
        temperature = initial_temperature * (cooling_rate ** (i / iterations))

        if temperature < final_temperature:
            break
    if record:
        return record_list
    # 返回最优状态和最优能量
    return -global_energy
