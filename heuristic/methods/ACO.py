import random
from methods.common import random_chromosome, fitness,mutate, calculate_cost, sort_task, ruler
random.seed(42)
def choice(elements, p):
    return random.choices(elements, weights=p, k=1)[0]
def aco(env,record=False,max_iterations=600):
    pop,J,M,W,N,A,D,pt,p = env
    num_ants = 20 # 蚂蚁数量
    alpha = 1 # 信息素的重要程度
    beta = 2 # 启发式规则的重要程度
    rho = 0.5 # 信息素的蒸发速率
    Q = 1 # 信息素的常数
    record_list=[]

    pheromone = [[[0.1 for k in range(M)] for j in range(2)] for i in range(J)]
    ants = []
    for i in range(num_ants):
        chromosome = pop[i]
        ants.append({'chromosome': chromosome, 'fitness': -fitness(chromosome, J, M, A, D, N, pt, p, W)})

    best_fitness = float('inf')
    for iteration in range(max_iterations):
        # 初始化每只蚂蚁的位置
        for ant in ants:
            ant['position'] = -1
            ant['task0_start_time'] = {}
            ant['task1_start_time'] = {}
            ant['production_line_power'] = {}
            for i in range(M):
                ant['production_line_power'][i] = 0
            # 为每个订单的Task 0和Task 1选择产线
            for j in range(J):
                for t in range(2):
                    allowed_lines = [m for m in range(M) if pt[t][j][m] > 0] # 可以处理该订单的生产线
                    pheromone_values = [pheromone[j][t][m] ** alpha * (1.0 / pt[t][j][m]) ** beta for m in allowed_lines]
                    total = sum(pheromone_values)
                    probabilities = [value / total for value in pheromone_values]
                    ant['position'] += 1
                    if probabilities:
                        ant['chromosome'][ant['position']][1] = choice(allowed_lines, p=probabilities)
                    else:
                        ant['chromosome'][ant['position']][1] = random.randint(0, M)
            # 检查任务约束
            for j in range(J):
                task0_idx = [i for i in range(len(ant['chromosome'])) if ant['chromosome'][i][0] == j and ant['chromosome'][i][2] == 0]
                task1_idx = [i for i in range(len(ant['chromosome'])) if ant['chromosome'][i][0] == j and ant['chromosome'][i][2] == 1]
                if not task0_idx or not task1_idx or task1_idx[0] < task0_idx[-1]:
                    ant['fitness'] = float('inf')
                    break
            # 计算适应度
            if ant['fitness'] != float('inf'):
                ant['task0_start_time'], ant['task1_start_time'], ant['production_line_power'] = calculate_cost(ant['chromosome'], J, M, A, D, N, pt, p, W)
                ant['fitness'] = -fitness(ant['chromosome'], J, M, A, D, N, pt, p, W)
            # 更新最佳解决方案
            if ant['fitness'] < best_fitness:
                best_solution = ant['chromosome']
                best_fitness = ant['fitness']
                if record:
                    print(best_fitness)
                    record_list.append(best_fitness)
                
        # 更新信息素
        for j in range(J):
            for t in range(2):
                for m in range(M):
                    delta_pheromone = 0
                    for ant in ants:
                        if (j, m, t) in ant['chromosome']:
                            delta_pheromone += Q / ant['fitness']
                    pheromone[j][t][m] = (1 - rho) * pheromone[j][t][m] + rho * delta_pheromone
    if record:
        return record_list
    # best_fitness = fitness(best_solution, J, M, A, D, N, pt, p, W)
    # print("Best Solution: ", best_solution)
    # print("Best Fitness: ", best_fitness)
    return best_fitness