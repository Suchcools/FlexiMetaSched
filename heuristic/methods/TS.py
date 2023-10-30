import copy
from methods.common import random_chromosome, fitness,mutate
def ts(env,record=False):
    pop,J,M,W,N,A,D,pt,p = env
    fitness_values = [fitness(chromosome, J, M, A, D, N, pt, p, W) for chromosome in pop]
    best_fitness = max(fitness_values)
    current_solution = pop[fitness_values.index(best_fitness)]
    tabu_size=2
    tabu_tenure=2
    max_iter=3
    # 定义禁忌表
    tabu_list = []
    record_list = []
    # 计算当前解的适应度函数
    current_fitness = -fitness(current_solution, J, M, A, D, N, pt, p, W)
    best_fitness = current_fitness
    # 进入主循环
    for i in range(max_iter):
        # 生成当前解的所有邻居解
        neighbors = []
        for j in range(100):
            neighbor = mutate(copy.deepcopy(current_solution))
            if neighbor not in tabu_list:
                neighbors.append(neighbor)
        if not neighbors: # 如果邻居解为空，则跳出循环
            break
        # 选择一个最优解，且不在禁忌表中
        best_neighbor = None
        best_neighbor_fitness = float('inf')
        for neighbor in neighbors:
            neighbor_fitness = -fitness(neighbor, J, M, A, D, N, pt, p, W)
            if neighbor_fitness < best_neighbor_fitness and neighbor not in tabu_list:
                best_neighbor = neighbor
                best_neighbor_fitness = neighbor_fitness
                if record:
                    print(best_neighbor_fitness)
                    record_list.append(best_neighbor_fitness)
        # 如果没有可行的邻居解，则跳出循环
        if best_neighbor is None:
            break
        # 将最优解加入禁忌表，并更新禁忌表中的所有解的期限
        tabu_list.append(best_neighbor + [tabu_tenure])
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)
        for k in range(len(tabu_list)):
            tabu_list[k][-1] -= 1
            if tabu_list[k][-1] <= 0:
                tabu_list.pop(k)
                break
        # 如果当前解的适应度函数优于历史最优解，则更新历史最优解
        if best_neighbor_fitness < best_fitness:
            best_solution = copy.deepcopy(best_neighbor)
            best_fitness = best_neighbor_fitness
        # 更新当前解和对应的适应度函数
        current_solution = copy.deepcopy(best_neighbor)
        current_fitness = best_neighbor_fitness
    if record:
        return record_list
    return  best_fitness