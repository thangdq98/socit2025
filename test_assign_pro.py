from utils import *
from data import load_data
from moo_algorithm.nsga_ii import run_nsga_ii
from moo_algorithm.pfg_moea import run_pfgmoea
from moo_algorithm.nsga_iii import run_nsga_iii
from moo_algorithm.moead import run_moead, init_weight_vectors_3d
import json, os, time


#Store indi_list to json file
def restore_initial_population(file_path, indi_list):
    indi_json = []
    for indi in indi_list:
        indi_data = {
            "chromosome_customer": indi.chromosome[0],
            "chromosome_assign": indi.chromosome[1]
        }
        indi_json.append(indi_data)

    # 🔹 Tạo thư mục cha nếu chưa có
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    result = {"population": indi_json}

    # 🔹 Ghi dữ liệu JSON
    with open(file_path, 'w') as f:
        json.dump(result, f)



def build_data_paths(num_customers, types=["C", "R", "RC"], K_list=[1,2], i=4, j_list=[1,2,3,4,5]):
    paths = []
    for t in types:
        for K in K_list:
            for j in j_list:
                if i == 0:
                    filename = f"{t}{K}{i}{j}.txt"
                    path = os.path.join("data", f"{num_customers}customers", filename)
                else:
                    filename = f"{t}{K}_{i}_{j}.TXT"
                    path = os.path.join("data", f"{num_customers}customers", filename)
                paths.append(path)
    return paths


### 100customers
if __name__ == "__main__":
    number_customer = 100 
    number_truck = 3
    number_drone = 4
    processing_number = 12
    pro_drone = 0.5
    pop_size = 100
    max_gen = 100
    crossover_rate = 0.9
    mutation_rate = 0.1
    G = 5
    sigma = 0.1

    # Sinh danh sách file data
    data_files = build_data_paths(number_customer, types=["c", "r", "rc"], K_list=[1,2], i=0, j_list=[1,2,3])

    for data_file in data_files:
        print(f"Đang chạy file: {data_file}")
        problem = load_data(data_file, number_customer, number_truck, number_drone, 2000)


        # Tạo thư mục kết quả gốc
        base_path = os.path.join("result_probability", f"{number_customer}customers")
        os.makedirs(base_path, exist_ok=True)

        file_name = os.path.splitext(os.path.basename(data_file))[0] + ".json"


        ##### PFGMOEA 0.5 ############
        pfg_start = time.time()
                # Khởi tạo quần thể ban đầu
        indi_list = []
        for i in range(pop_size):
            indi = init_random(problem, 0.5)
            indi_list.append(indi)
        pfg_history = run_pfgmoea(processing_number, problem, indi_list, pop_size, max_gen,
                                  G, sigma, crossover_PMX, mutation_flip, crossover_rate, mutation_rate, cal_fitness)
        pfg_end = time.time()
        pfg_result = {'time': pfg_end - pfg_start, 'history': pfg_history}
        pfg_path = os.path.join(base_path, "50", file_name)
        os.makedirs(os.path.dirname(pfg_path), exist_ok=True)
        with open(pfg_path, 'w') as f:
            json.dump(pfg_result, f)

        ##### PFGMOEA 0.6 ############
        pfg_start = time.time()
                # Khởi tạo quần thể ban đầu
        indi_list = []
        for i in range(pop_size):
            indi = init_random(problem, 0.6)
            indi_list.append(indi)
        pfg_history = run_pfgmoea(processing_number, problem, indi_list, pop_size, max_gen,
                                  G, sigma, crossover_PMX, mutation_flip, crossover_rate, mutation_rate, cal_fitness)
        pfg_end = time.time()
        pfg_result = {'time': pfg_end - pfg_start, 'history': pfg_history}
        pfg_path = os.path.join(base_path, "60", file_name)
        os.makedirs(os.path.dirname(pfg_path), exist_ok=True)
        with open(pfg_path, 'w') as f:
            json.dump(pfg_result, f)


        ##### PFGMOEA 0.8 ############
        pfg_start = time.time()
                # Khởi tạo quần thể ban đầu
        indi_list = []
        for i in range(pop_size):
            indi = init_random(problem, 0.8)
            indi_list.append(indi)  
        pfg_history = run_pfgmoea(processing_number, problem, indi_list, pop_size, max_gen,
                                  G, sigma, crossover_PMX, mutation_flip, crossover_rate, mutation_rate, cal_fitness)
        pfg_end = time.time()
        pfg_result = {'time': pfg_end - pfg_start, 'history': pfg_history}
        pfg_path = os.path.join(base_path, "80", file_name)
        os.makedirs(os.path.dirname(pfg_path), exist_ok=True)
        with open(pfg_path, 'w') as f:
            json.dump(pfg_result, f)
        ##### PFGMOEA 0.9 ############
        pfg_start = time.time()
                # Khởi tạo quần thể ban đầu
        indi_list = []
        for i in range(pop_size):
            indi = init_random(problem, 0.9)
            indi_list.append(indi)
        pfg_history = run_pfgmoea(processing_number, problem, indi_list, pop_size, max_gen,
                                  G, sigma, crossover_PMX, mutation_flip, crossover_rate, mutation_rate, cal_fitness)
        pfg_end = time.time()
        pfg_result = {'time': pfg_end - pfg_start, 'history': pfg_history}
        pfg_path = os.path.join(base_path, "90", file_name) 
        os.makedirs(os.path.dirname(pfg_path), exist_ok=True)
        with open(pfg_path, 'w') as f:
            json.dump(pfg_result, f)    

        ################ PFGMOEA 1.0 ############
        pfg_start = time.time()
                # Khởi tạo quần thể ban đầu
        indi_list = []
        for i in range(pop_size):
            indi = init_random(problem, 1.0)
            indi_list.append(indi)
        pfg_history = run_pfgmoea(processing_number, problem, indi_list, pop_size, max_gen,
                                  G, sigma, crossover_PMX, mutation_flip, crossover_rate, mutation_rate, cal_fitness)
        pfg_end = time.time()
        pfg_result = {'time': pfg_end - pfg_start, 'history': pfg_history}
        pfg_path = os.path.join(base_path, "100", file_name)
        os.makedirs(os.path.dirname(pfg_path), exist_ok=True)
        with open(pfg_path, 'w') as f:
            json.dump(pfg_result, f)
    print("Đã hoàn thành tất cả các file.")
