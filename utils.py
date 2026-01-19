from problem import Problem
from population import Individual
from copy import deepcopy
from decode_time_greedy import *
import random
def random_permutation(n):
    numbers = list(range(1, n+1))
    random.shuffle(numbers)
    return numbers

def random_binary_vector(n, p):
    vec = []
    for _ in range(n):
        if random.random() < p:   # random() ~ U[0,1)
            vec.append(0)
        else:
            vec.append(1)
    return vec

def cal_fitness(problem: Problem, indi: Individual):
    temp_chromosome = deepcopy(indi.chromosome)
    temp_chromosome = repair_distance(temp_chromosome, problem)
    temp_chromosome = repair_capacity(temp_chromosome, problem)
    if temp_chromosome == False:
        return indi.chromosome, np.inf, np.inf, np.inf
    assigned_truck_customers, assigned_drone_customers = extract_routes(temp_chromosome, problem)
    truck_solutions, drone_solutions = find_solution(assigned_truck_customers, assigned_drone_customers, problem)
    i = 0
    while truck_solutions == False and i <= problem.number_customer:
        idx = temp_chromosome[0].index(drone_solutions)
        temp_chromosome[1][idx] = 0

        temp_chromosome = repair_distance(temp_chromosome, problem)
        temp_chromosome = repair_capacity(temp_chromosome, problem)
        if temp_chromosome == False:
            return indi.chromosome, np.inf, np.inf, np.inf
        assigned_truck_customers, assigned_drone_customers = extract_routes(temp_chromosome, problem)
        truck_solutions, drone_solutions = find_solution(assigned_truck_customers, assigned_drone_customers, problem)
        i = i + 1
    if truck_solutions == False:
        return indi.chromosome, np.inf, np.inf, np.inf
    total_cost = problem.cal_total_cost(truck_solutions, drone_solutions)
    wait_time = problem.customer_wait_max(truck_solutions, drone_solutions)
    fainess = problem.cal_truck_fairness(truck_solutions)
    return temp_chromosome, total_cost, wait_time, fainess


def init_random(problem: Problem, pro_drone):
    dimension = problem.number_customer + problem.number_of_trucks+ problem.number_of_drones-2
    per_customers = random_permutation(dimension)
    drone_truck_assign = random_binary_vector(dimension, pro_drone)
    chromosome = [per_customers, drone_truck_assign]
    indi = Individual(chromosome)
    return indi


import random

def crossover_PMX(problem: Problem, parent1: Individual, parent2: Individual):
    size = len(parent1.chromosome[0])
    p1_perm, p1_bin = parent1.chromosome
    p2_perm, p2_bin = parent2.chromosome

    # chọn 2 điểm cắt
    cx_point1 = random.randint(0, size - 2)
    cx_point2 = random.randint(cx_point1 + 1, size - 1)

    off1_perm = [None] * size
    off2_perm = [None] * size

    # copy đoạn cắt từ cha
    off1_perm[cx_point1:cx_point2 + 1] = p1_perm[cx_point1:cx_point2 + 1]
    off2_perm[cx_point1:cx_point2 + 1] = p2_perm[cx_point1:cx_point2 + 1]

    # PMX mapping
    for i in range(cx_point1, cx_point2 + 1):
        if p2_perm[i] not in off1_perm:
            pos = i
            while off1_perm[pos] is not None:
                pos = p2_perm.index(p1_perm[pos])
            off1_perm[pos] = p2_perm[i]

        if p1_perm[i] not in off2_perm:
            pos = i
            while off2_perm[pos] is not None:
                pos = p1_perm.index(p2_perm[pos])
            off2_perm[pos] = p1_perm[i]

    # điền chỗ trống
    for i in range(size):
        if off1_perm[i] is None:
            off1_perm[i] = p2_perm[i]
        if off2_perm[i] is None:
            off2_perm[i] = p1_perm[i]

    # ánh xạ lại binary theo vị trí
    idx_map1 = {val: i for i, val in enumerate(off1_perm)}
    idx_map2 = {val: i for i, val in enumerate(off2_perm)}
    off1_bin = [0] * size
    off2_bin = [0] * size

    for i, val in enumerate(p1_perm):
        off1_bin[idx_map1[val]] = p1_bin[i]
    for i, val in enumerate(p2_perm):
        off2_bin[idx_map2[val]] = p2_bin[i]

    # trả về offspring
    off1 = Individual([off1_perm, off1_bin])
    off2 = Individual([off2_perm, off2_bin])
    return off1, off2


def mutation_flip(problem: Problem, indi: Individual, num_flips=None):
    perm, bin_vec = indi.chromosome
    size = len(perm)

    if num_flips is None:
        num_flips = max(1, size // 10)  # mặc định 10% số gene

    # copy để không sửa trực tiếp parent
    new_bin = bin_vec.copy()

    flip_positions = random.sample(range(size), num_flips)
    for pos in flip_positions:
        new_bin[pos] = 1 - new_bin[pos]  # lật bit

    # tạo offspring mới
    offspring = Individual([perm.copy(), new_bin])
    return offspring