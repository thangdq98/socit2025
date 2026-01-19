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

def read_initial_population(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    indi_list = []
    for indi_data in data["population"]:
        chromosome = [indi_data["chromosome_customer"], indi_data["chromosome_assign"]]
        indi = Individual(chromosome)
        indi_list.append(indi)
    
    return indi_list



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


### 400customers
if __name__ == "__main__":
    number_customer = 400 
    number_truck = 12
    number_drone = 16
    processing_number = 12
    pro_drone = 0.7
    pop_size = 100
    max_gen = 100

    crossover_rate = 0.9
    mutation_rate = 0.1
    G = 5
    sigma = 0.1

    # Sinh danh sách file data
    data_files = build_data_paths(number_customer, types=["C", "R", "RC"], K_list=[1,2], i=4, j_list=[1,2,3])

    for data_file in data_files:
        print(f"Đang chạy file: {data_file}")
        problem = load_data(data_file, number_customer, number_truck, number_drone, 2000)

        # Khởi tạo quần thể ban đầu
        # indi_list = []
        # for i in range(pop_size):
        #     indi = init_random(problem, pro_drone)
        #     indi_list.append(indi)

        #     init_pop_path = os.path.join("init_population", data_file)

        #     restore_initial_population(init_pop_path, indi_list)
        indi_list = read_initial_population(os.path.join("init_population", data_file))

        # Tạo thư mục kết quả gốc
        base_path = os.path.join("result", f"{number_customer}customers")
        os.makedirs(base_path, exist_ok=True)

        file_name = os.path.splitext(os.path.basename(data_file))[0] + ".json"

        ##### NSGA-III ############
        nsgaiii_start = time.time()
        nsgaiii_history = run_nsga_iii(processing_number, problem, indi_list, pop_size, max_gen,
                                       crossover_PMX, mutation_flip, crossover_rate, mutation_rate, cal_fitness)
        nsgaiii_end = time.time()
        nsgaiii_result = {'time': nsgaiii_end - nsgaiii_start, 'history': nsgaiii_history}
        nsgaiii_path = os.path.join(base_path, "NSGAIII", file_name)
        os.makedirs(os.path.dirname(nsgaiii_path), exist_ok=True)
        with open(nsgaiii_path, 'w') as f:
            json.dump(nsgaiii_result, f)

     

# ### 200customers
# if __name__ == "__main__":
#     number_customer = 200 
#     number_truck = 6
#     number_drone = 8
#     processing_number = 8
#     pro_drone = 0.7
#     pop_size = 100
#     max_gen = 100
#     crossover_rate = 0.9
#     mutation_rate = 0.1
#     G = 5
#     sigma = 0.1

#     # Sinh danh sách file data
#     data_files = build_data_paths(number_customer, K_list=[1,2], i=2, j_list=[1,2,3])

#     for data_file in data_files:
#         print(f"Đang chạy file: {data_file}")
#         problem = load_data(data_file, number_customer, number_truck, number_drone, 2000)

#         # # Khởi tạo quần thể ban đầu
#         # indi_list = []
#         # for i in range(pop_size):
#         #     indi = init_random(problem, pro_drone)
#         #     indi_list.append(indi)
#         indi_list = read_initial_population(os.path.join("init_population", data_file))

#         # Tạo thư mục kết quả gốc
#         base_path = os.path.join("result", f"{number_customer}customers")
#         os.makedirs(base_path, exist_ok=True)

#         file_name = os.path.splitext(os.path.basename(data_file))[0] + ".json"


#         ##### NSGA-III ############
#         nsgaiii_start = time.time()
#         nsgaiii_history = run_nsga_iii(processing_number, problem, indi_list, pop_size, max_gen,
#                                        crossover_PMX, mutation_flip, crossover_rate, mutation_rate, cal_fitness)
#         nsgaiii_end = time.time()
#         nsgaiii_result = {'time': nsgaiii_end - nsgaiii_start, 'history': nsgaiii_history}
#         nsgaiii_path = os.path.join(base_path, "NSGAIII", file_name)
#         os.makedirs(os.path.dirname(nsgaiii_path), exist_ok=True)
#         with open(nsgaiii_path, 'w') as f:
#             json.dump(nsgaiii_result, f)


#         print(f"Hoàn thành {data_file}, kết quả đã lưu.")


# ### 100customers
# if __name__ == "__main__":
#     number_customer = 100 
#     number_truck = 3
#     number_drone = 4
#     processing_number = 12
#     pro_drone = 0.5
#     pop_size = 100
#     max_gen = 100
#     crossover_rate = 0.9
#     mutation_rate = 0.1
#     G = 5
#     sigma = 0.1

#     # Sinh danh sách file data
#     data_files = build_data_paths(number_customer, types=["c", "r", "rc"], K_list=[1,2], i=0, j_list=[1,2,3])

#     for data_file in data_files:
#         print(f"Đang chạy file: {data_file}")
#         problem = load_data(data_file, number_customer, number_truck, number_drone, 2000)

#         # Khởi tạo quần thể ban đầu
#         # indi_list = []
#         # for i in range(pop_size):
#         #     indi = init_random(problem, pro_drone)
#         #     indi_list.append(indi)

#         # init_pop_path = os.path.join("init_population", data_file)
#         # restore_initial_population(init_pop_path, indi_list)
#         indi_list = read_initial_population(os.path.join("init_population", data_file))

#         # Tạo thư mục kết quả gốc
#         base_path = os.path.join("result", f"{number_customer}customers")
#         os.makedirs(base_path, exist_ok=True)

#         file_name = os.path.splitext(os.path.basename(data_file))[0] + ".json"

#         ##### NSGA-III ############
#         nsgaiii_start = time.time()
#         nsgaiii_history = run_nsga_iii(processing_number, problem, indi_list, pop_size, max_gen,
#                                        crossover_PMX, mutation_flip, crossover_rate, mutation_rate, cal_fitness)
#         nsgaiii_end = time.time()
#         nsgaiii_result = {'time': nsgaiii_end - nsgaiii_start, 'history': nsgaiii_history}
#         nsgaiii_path = os.path.join(base_path, "NSGAIII", file_name)
#         os.makedirs(os.path.dirname(nsgaiii_path), exist_ok=True)
#         with open(nsgaiii_path, 'w') as f:
#             json.dump(nsgaiii_result, f)

        # print(f"Hoàn thành {data_file}, kết quả đã lưu.")