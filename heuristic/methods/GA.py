import random
import numpy as np
import copy
from methods.common import random_chromosome, fitness,mutate, ruler, generate_population
random.seed(42)
np.random.seed(42)


def crossover(chromosome1, chromosome2, J):
    # 选取一个随机的交叉点
    cross_point = random.randint(0, len(chromosome1)-1)
    # 生成两个新的染色体
    new_chromosome1 = chromosome1[:cross_point] + chromosome2[cross_point:]
    new_chromosome2 = chromosome2[:cross_point] + chromosome1[cross_point:]

    # new_chromosome1=ruler(np.array(new_chromosome1), J)
    # new_chromosome2=ruler(np.array(new_chromosome2), J)
    # 检查任务约束
    for j in range(J):
        task0_idx = [i for i in range(len(new_chromosome1)) if new_chromosome1[i][0] == j and new_chromosome1[i][2] == 0]
        task1_idx = [i for i in range(len(new_chromosome1)) if new_chromosome1[i][0] == j and new_chromosome1[i][2] == 1]
        if not task0_idx or not task1_idx or task1_idx[0] < task0_idx[-1]:
            return crossover(chromosome1, chromosome2,J)
        task0_idx = [i for i in range(len(new_chromosome2)) if new_chromosome2[i][0] == j and new_chromosome2[i][2] == 0]
        task1_idx = [i for i in range(len(new_chromosome2)) if new_chromosome2[i][0] == j and new_chromosome2[i][2] == 1]
        if not task0_idx or not task1_idx or task1_idx[0] < task0_idx[-1]:
            return crossover(chromosome1, chromosome2,J)
    return new_chromosome1, new_chromosome2

def tournament_selection(population, tournament_size, J, M, A, D, N, pt, p, W):
    selected_population = []
    for i in range(len(population)):
        participants = random.sample(population, tournament_size)
        fitness_values = [fitness(chromosome, J, M, A, D, N, pt, p, W) for chromosome in participants]
        best_index = fitness_values.index(max(fitness_values))
        selected_population.append(participants[best_index])
    return selected_population


def ga(env,generations=100,record=False):
    pop,J,M,W,N,A,D,pt,p = env
    pop_size=20 
    crossover_prob=0.1
    mutation_prob=0.1
    tournament_size=5
    elite_size=5
    record_list =[]


    population = pop
    best_chromosome = None
    for generation in range(generations):
        
        # 评估适应度
        fitness_values = [fitness(chromosome, J, M, A, D, N, pt, p, W) for chromosome in population]
        best_fitness = max(fitness_values)
        best_chromosome_in_generation = population[fitness_values.index(best_fitness)]
        if record:
            print(-best_fitness)
            record_list.append((generation,-best_fitness))

        # 选择
        elites = []
        if elite_size > 0:
            elite_indices = np.argpartition(fitness_values, -elite_size)[-elite_size:] #种群选出最好的elite_size个基因
            current_best=[population[i] for i in elite_indices]
            if best_fitness <= max([fitness(chromosome, J, M, A, D, N, pt, p, W) for chromosome in current_best]):
                elites = copy.deepcopy(current_best) #这里精英也被改了

        selected_population =copy.deepcopy(elites)
        while len(selected_population) < pop_size:
            # 锦标赛选择
            tournament = random.sample(population, tournament_size)
            fitness_values = [fitness(chromosome, J, M, A, D, N, pt, p, W) for chromosome in tournament]
            winner_index = fitness_values.index(max(fitness_values))
            winner = tournament[winner_index]
            selected_population.append(winner)
        # 交叉
        for i in range(0, pop_size - elite_size, 2):
            if random.random() < crossover_prob:
                chromosome1, chromosome2 = crossover(selected_population[i], selected_population[i+1], J)
                selected_population[i] = chromosome1
                selected_population[i+1] = chromosome2

        # 变异
        for i in range(0, pop_size):
            if random.random() < mutation_prob:
                selected_population[i] = mutate(selected_population[i])

        # 更新种群
        population = elites + selected_population[elite_size:]
        # 更新最佳染色体
        if best_chromosome is None or fitness(best_chromosome_in_generation, J, M, A, D, N, pt, p, W) > fitness(best_chromosome, J, M, A, D, N, pt, p, W):
            best_chromosome = best_chromosome_in_generation
    if record :
        return record_list
    return -fitness(best_chromosome_in_generation, J, M, A, D, N, pt, p, W)