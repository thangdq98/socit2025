from problem import Customer, Problem
import math
def read_data_file(filename):
    customers = []
    capacity = None
    reading_customers = False

    with open(filename, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line.startswith('VEHICLE'):
            continue
        if 'CAPACITY' in line:
            continue
        if line and line[0].isdigit() and capacity is None:
            parts = line.split()
            if len(parts) == 2:
                capacity = int(parts[1])
            continue

        if line.startswith('CUSTOMER'):
            reading_customers = True
            continue

        if reading_customers and line and line[0].isdigit():
            parts = line.split()
            if len(parts) >= 7:
                cust_no = int(parts[0])
                x = int(parts[1])
                y = int(parts[2])
                demand = int(parts[3])
                ready_time = int(parts[4])
                due_date = int(parts[5])
                service_time = int(parts[6])
                customer = Customer(arrive_time=ready_time, quantity=demand, service_time=service_time, x=x, y=y)
                customers.append(customer)

    return customers, capacity

def manhattan_distance(c1, c2):
    return abs(c1.x - c2.x) + abs(c1.y - c2.y)
def euclidean_distance(c1, c2):
    return math.hypot(c1.x - c2.x, c1.y - c2.y)

def create_distance_matrices(customers):
    n = len(customers)
    distance_matrix_truck = [[0] * n for _ in range(n)]
    distance_matrix_drone = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix_truck[i][j] = manhattan_distance(customers[i], customers[j])
                distance_matrix_drone[i][j] = euclidean_distance(customers[i], customers[j])
            else:
                distance_matrix_truck[i][j] = 0
                distance_matrix_drone[i][j] = 0.0

    return distance_matrix_truck, distance_matrix_drone


def load_data(file_path, number_customer, number_of_trucks, number_of_drones, capacity_of_truck_mofi = None):
    customer_list, truck_capacity = read_data_file(file_path)
    if capacity_of_truck_mofi != None:
        truck_capacity = capacity_of_truck_mofi
    customer_list = customer_list[:number_customer+1]
    drone_capacity = 30
    weight_of_drone = 10
    drone_energy = 180 #Wh
    speed_of_truck = 40 # km/h
    speed_of_drone = 60 # km/h
    launch_time = 1/60
    land_time = 1/60
    energy_consumption_rate = 9
    cost_truck_unit = 0.78 # $/km
    cost_drone_unit = 0.0104 # $/km
    fix_truck_cost = 15
    fix_drone_cost = 5

    for customer in customer_list:
        customer.arrive_time = customer.arrive_time/speed_of_truck/2
        customer.service_time = customer.service_time/1200
    
    distance_matrix_truck, distance_matrix_drone = create_distance_matrices(customer_list)
    problem = Problem(number_customer, customer_list, number_of_trucks, number_of_drones, distance_matrix_truck,
                distance_matrix_drone, truck_capacity, drone_capacity, drone_energy,
                speed_of_truck, speed_of_drone, launch_time, land_time,
                energy_consumption_rate, weight_of_drone,
                cost_truck_unit, cost_drone_unit, fix_truck_cost, fix_drone_cost)
    return problem
