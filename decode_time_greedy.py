from problem import Problem, Truck_Solution, Drone_Solution, Drone_Trip
from copy import deepcopy
import numpy as np
from typing import List, Tuple, Dict


def extract_routes(chromosome, problem: Problem):
    truck_routes = []
    drone_routes = []
    tmp_truck_route = []
    tmp_drone_route = []

    k = problem.number_customer + 1
    d = problem.number_customer + problem.number_of_trucks

    for i in range(len(chromosome[0])):
        if chromosome[0][i] >= k and chromosome[0][i] < d:
            truck_routes.append(tmp_truck_route)
            tmp_truck_route = []
            continue
        elif (
            chromosome[0][i] >= d
            and chromosome[0][i] < d + problem.number_of_drones - 1
        ):
            drone_routes.append(tmp_drone_route)
            tmp_drone_route = []
            continue

        if chromosome[1][i] == 0:
            tmp_truck_route.append(chromosome[0][i])
        elif chromosome[1][i] == 1:
            tmp_drone_route.append(chromosome[0][i])

    truck_routes.append(tmp_truck_route)
    drone_routes.append(tmp_drone_route)
    return truck_routes, drone_routes


def update_chromosome(truck_routes, drone_routes, problem: Problem):
    list_nodes = []
    list_mode = []

    k = problem.number_customer + 1
    d = problem.number_customer + problem.number_of_trucks

    for idx, route in enumerate(truck_routes):
        for cust in route:
            list_nodes.append(cust)
            list_mode.append(0)  # truck
        if idx < len(truck_routes) - 1:
            list_nodes.append(k + idx)   # marker truck
            list_mode.append(0)

    for idx, route in enumerate(drone_routes):
        for cust in route:
            list_nodes.append(cust)
            list_mode.append(1)  # drone
        if idx < len(drone_routes) - 1:
            list_nodes.append(d + idx)   # marker drone
            list_mode.append(1)

    return [list_nodes, list_mode]

# def repair_capacity(chromosome, problem: Problem):
#     for i in range(len(chromosome[0])):
#         if chromosome[1][i] == 1 and chromosome[0][i] <= problem.number_customer:
#             if problem.customer_list[chromosome[0][i]].quantity > problem.drone_capacity:
#                 chromosome[1][i] = 0

#     truck_routes, drone_routes = extract_routes(chromosome, problem)
#     def route_load(route):
#         return sum(problem.customer_list[cust].quantity for cust in route)
    
#     while True:
#         loads = [route_load(route) for route in truck_routes]
#         max_load = max(loads)
#         min_load = min(loads)
#         max_idx = loads.index(max_load)
#         min_idx = loads.index(min_load)

#         if all(load <= problem.truck_capacity for load in loads):
#             break

#         if max_load > problem.truck_capacity:
#             max_cust = max(
#                 truck_routes[max_idx],
#                 key=lambda c: problem.customer_list[c].quantity
#             )
#             truck_routes[max_idx].remove(max_cust)
#             truck_routes[min_idx].append(max_cust)
#         else:
#             break
#     loads = [route_load(route) for route in truck_routes]
#     if all(load <= problem.truck_capacity for load in loads):
#         return False

#     # B4: Encode lại chromosome
#     chromosome = update_chromosome(truck_routes, drone_routes, problem)
#     truck_routes, drone_routes = extract_routes(chromosome, problem)
#     return chromosome

def repair_capacity(chromosome, problem: Problem):
    for i in range(len(chromosome[0])):
        if chromosome[1][i] == 1 and chromosome[0][i] <= problem.number_customer:
            if problem.customer_list[chromosome[0][i]].quantity > problem.drone_capacity:
                chromosome[1][i] = 0

    truck_routes, drone_routes = extract_routes(chromosome, problem)
    def route_load(route):
        return sum(problem.customer_list[cust].quantity for cust in route)
    i = 0
    while True:
        if i > 1000:
            return False
        i = i+1
        loads = [route_load(route) for route in truck_routes]
        max_load = max(loads)
        min_load = min(loads)
        max_idx = loads.index(max_load)
        min_idx = loads.index(min_load)

        if all(load <= problem.truck_capacity for load in loads):
            break

        if max_load > problem.truck_capacity:
            max_cust = max(
                truck_routes[max_idx],
                key=lambda c: problem.customer_list[c].quantity
            )
            truck_routes[max_idx].remove(max_cust)
            truck_routes[min_idx].append(max_cust)
        else:
            break

    # B4: Encode lại chromosome
    chromosome = update_chromosome(truck_routes, drone_routes, problem)
    truck_routes, drone_routes = extract_routes(chromosome, problem)
    return chromosome

def repair_distance(chromosome, problem: Problem):
    max_distance = (
        (problem.drone_energy / (problem.energy_consumption_rate * problem.weight_of_drone))
        - problem.land_time
        - problem.launch_time
    ) * problem.speed_of_drone

    for i in range(len(chromosome[0])):
        customer = chromosome[0][i]
        if chromosome[1][i] == 1 and customer <= problem.number_customer:
            nearest_dist = float("inf")
            for j in range(len(chromosome[0])):
                if i == j:
                    continue
                other = chromosome[0][j]
                if other <= problem.number_customer:
                    dist = problem.distance_matrix_drone[customer][other]
                    if dist < nearest_dist:
                        nearest_dist = dist

            if nearest_dist == float("inf") or nearest_dist > max_distance:
                chromosome[1][i] = 0

    return chromosome


def find_nearest_customer(customer_drone_list, customer_truck_list, distance_drone_matrix):
    nearest_dict = {}

    for customer in customer_drone_list:
        distances = [
            (other, distance_drone_matrix[customer][other])
            for other in customer_truck_list if other != customer
        ]
        distances.append((0, distance_drone_matrix[0][customer]))
        distances.sort(key=lambda x: x[1])
        nearest_dict[customer] = [other for other, _ in distances]

    return nearest_dict


def update_nearest_dict(nearest_dict, customer_remove, distance_drone_matrix):
    if customer_remove in nearest_dict:
        del nearest_dict[customer_remove]

    for key, value in nearest_dict.items():
        dist_remove = distance_drone_matrix[key][customer_remove]

        inserted = False
        new_list = []
        for other in value:
            dist_other = distance_drone_matrix[key][other]
            if not inserted and dist_remove < dist_other:
                new_list.append(customer_remove)
                inserted = True
            new_list.append(other)
        if not inserted:
            new_list.append(customer_remove)

        nearest_dict[key] = new_list

    return nearest_dict

def init_truck_solution(route, problem: Problem):
    num_point = len(route)
    if num_point == 0:
        return Truck_Solution([], [], [], [], [])
    assigned_customer = deepcopy(route)
    recived_truck = np.zeros(num_point)
    recived_drone = np.zeros(num_point)
    arrive_time = np.zeros(num_point)
    leave_time = np.zeros(num_point)
    arrive_time[0] = problem.distance_matrix_truck[0][route[0]]/problem.speed_of_truck
    recived_truck[0] = problem.customer_list[route[0]].quantity
    recived_drone[0] = 0
    leave_time[0] = max(arrive_time[0], problem.customer_list[route[0]].arrive_time) + problem.customer_list[route[0]].service_time
    for i in range(1, num_point):
        arrive_time[i] = leave_time[i-1] + problem.distance_matrix_truck[route[i-1]][route[i]]/problem.speed_of_truck
        recived_truck[i] = problem.customer_list[route[i]].quantity
        recived_drone[i] = 0
        leave_time[i] = max(arrive_time[i], problem.customer_list[route[i]].arrive_time) + problem.customer_list[route[i]].service_time
    truck_solution = Truck_Solution(assigned_customer, recived_truck, recived_drone, arrive_time, leave_time)
    return truck_solution


def update_truck_solution(truck_solution: Truck_Solution, problem:Problem, customer, add_quantity, leave_arrive, index_insert = None):
    if customer in  truck_solution.assigned_customers:
        idx = truck_solution.assigned_customers.index(customer)
        truck_solution.recived_drone[idx] += add_quantity
        truck_solution.leave_time[idx] = max(truck_solution.leave_time[idx], leave_arrive)
        if truck_solution.leave_time[idx] == leave_arrive:
            route = truck_solution.assigned_customers
            for i in range(idx+1, len(truck_solution.assigned_customers)):
                truck_solution.arrive_time[i] = truck_solution.leave_time[i-1] + problem.distance_matrix_truck[route[i-1]][route[i]]/problem.speed_of_truck
                truck_solution.leave_time[i] = max(truck_solution.arrive_time[i], problem.customer_list[route[i]].arrive_time) + problem.customer_list[route[i]].service_time
        return truck_solution


    truck_solution.assigned_customers.insert(index_insert, customer)
    truck_solution.recived_truck = np.insert(truck_solution.recived_truck, index_insert, 0)
    truck_solution.recived_drone = np.insert(truck_solution.recived_drone, index_insert, add_quantity)
    truck_solution.arrive_time = np.insert(truck_solution.arrive_time, index_insert, 0)
    truck_solution.leave_time = np.insert(truck_solution.leave_time, index_insert, 0)

    route = truck_solution.assigned_customers
    n = len(route)

    if index_insert == 0:
        truck_solution.arrive_time[0] = problem.distance_matrix_truck[0][route[0]] / problem.speed_of_truck
    else:
        truck_solution.arrive_time[index_insert] = truck_solution.leave_time[index_insert-1] + problem.distance_matrix_truck[route[index_insert-1]][route[index_insert]] / problem.speed_of_truck

    truck_solution.leave_time[index_insert] = max(max(truck_solution.arrive_time[index_insert],
                                                 problem.customer_list[route[index_insert]].arrive_time) + problem.customer_list[route[index_insert]].service_time, leave_arrive)

    for i in range(index_insert+1, n):
        truck_solution.arrive_time[i] = truck_solution.leave_time[i-1] + problem.distance_matrix_truck[route[i-1]][route[i]] / problem.speed_of_truck
        truck_solution.leave_time[i] = max(truck_solution.arrive_time[i],
                                           problem.customer_list[route[i]].arrive_time) + \
                                       problem.customer_list[route[i]].service_time

    return truck_solution




def find_solution(truck_routes: List[List[int]], drone_trips: List[List[int]], problem: Problem):
    truck_solutions = [init_truck_solution(route, problem) for route in truck_routes]
    customer_to_truck: Dict[int, Tuple[int, int]] = {}
    for t_idx, tsol in enumerate(truck_solutions):
        for pos, c in enumerate(tsol.assigned_customers):
            customer_to_truck[c] = (t_idx, pos)
    
    drone_solutions = []
    # print("********************Drone trip solution******************")
    
    for trip in drone_trips:
        trip_list = []
        if not trip:
            drone_solutions.append(Drone_Solution(0, []))
            continue
        

        customer_on_trip = [(customer, problem.customer_list[customer].arrive_time) for customer in trip]
        customer_on_trip.sort(key= lambda x: x[1])
        # print("Customer on trip:", customer_on_trip)
        unserved = [x[0] for x in customer_on_trip]  
        
        current_system_time = 0
        first_trip = True
        current_customer = 0
        while unserved:
            flag = False
            first_unserved_customer = unserved[0]
            # print("first_unserved_customer", first_unserved_customer)
            
            arrive_time_request = problem.customer_list[first_unserved_customer].arrive_time
            ucv_launch = []

            if current_customer == 0 :
                drone_fly_duration = problem.launch_time + problem.distance_matrix_drone[0][first_unserved_customer]/problem.speed_of_drone + problem.land_time
                drone_launch_time = max(current_system_time, arrive_time_request - drone_fly_duration)
                # print(drone_launch_time)
                depot_consumption_energy = problem.energy_consumption_rate*(problem.weight_of_drone)*drone_fly_duration + problem.energy_consumption_rate*(problem.weight_of_drone + problem.customer_list[first_unserved_customer].quantity)*problem.customer_list[first_unserved_customer].service_time
                # print(depot_consumption_energy) 
                drone_arrive_to_customer_time = drone_launch_time + drone_fly_duration 


                customer_wait_time = drone_arrive_to_customer_time - problem.customer_list[first_unserved_customer].arrive_time
                # print("customer_wait_time", customer_wait_time)
                drone_leave_time = drone_launch_time + drone_fly_duration+ problem.customer_list[first_unserved_customer].service_time
                ucv_launch.append((0,customer_wait_time, current_system_time, drone_launch_time, drone_leave_time - problem.customer_list[first_unserved_customer].service_time, drone_leave_time, depot_consumption_energy, first_unserved_customer))
            # if first_trip == True:  
                for truck_sol in truck_solutions:
                    for pos, cus in enumerate(truck_sol.assigned_customers):
                        drone_fly_duration = problem.launch_time + problem.distance_matrix_drone[cus][first_unserved_customer]/problem.speed_of_drone + problem.land_time
                        drone_launch_time = max(problem.customer_list[first_unserved_customer].arrive_time - drone_fly_duration, truck_sol.arrive_time[pos])
                        if drone_launch_time < current_system_time:
                            continue
                        if drone_launch_time <= truck_sol.leave_time[pos]:
                            consumption_energy = problem.energy_consumption_rate*(problem.weight_of_drone)*drone_fly_duration + problem.energy_consumption_rate*(problem.weight_of_drone + problem.customer_list[first_unserved_customer].quantity)*problem.customer_list[first_unserved_customer].service_time
                            drone_arrive_to_customer_time = drone_launch_time + drone_fly_duration 
                            customer_wait_time = drone_arrive_to_customer_time - problem.customer_list[first_unserved_customer].arrive_time
                            drone_leave_time = drone_launch_time  + drone_fly_duration + problem.customer_list[first_unserved_customer].service_time
                            ucv_launch.append((cus, customer_wait_time,  drone_launch_time,  drone_launch_time, drone_leave_time - problem.customer_list[first_unserved_customer].service_time, drone_leave_time, consumption_energy, first_unserved_customer))
            if current_customer !=0:    
                for truck_sol in truck_solutions:
                    if current_customer not in truck_sol.assigned_customers:
                            continue
                    for pos, cus in enumerate(truck_sol.assigned_customers):
                        drone_fly_duration = problem.launch_time + problem.distance_matrix_drone[cus][first_unserved_customer]/problem.speed_of_drone + problem.land_time
                        drone_launch_time = max(problem.customer_list[first_unserved_customer].arrive_time - drone_fly_duration, truck_sol.arrive_time[pos])
                        if drone_launch_time < current_system_time:
                            continue
                        if drone_launch_time <= truck_sol.leave_time[pos]:
                            consumption_energy = problem.energy_consumption_rate*(problem.weight_of_drone)*drone_fly_duration + problem.energy_consumption_rate*(problem.weight_of_drone + problem.customer_list[first_unserved_customer].quantity)*problem.customer_list[first_unserved_customer].service_time
                            drone_arrive_to_customer_time = drone_launch_time + drone_fly_duration 
                            customer_wait_time = drone_arrive_to_customer_time - problem.customer_list[first_unserved_customer].arrive_time
                            drone_leave_time = drone_launch_time  + drone_fly_duration + problem.customer_list[first_unserved_customer].service_time
                            ucv_launch.append((cus, customer_wait_time,  drone_launch_time,  drone_launch_time, drone_leave_time - problem.customer_list[first_unserved_customer].service_time, drone_leave_time, consumption_energy, first_unserved_customer))

            ucv_launch.sort(key= lambda x: (x[1], x[6]))
            # print(ucv_launch)
            # print("*********************************")

            best_land_tuple = None
            best_customer_sequence = None
            best_good_add = None
            best_leave_time = None
            best_assigned_customer = None
            best_arrive_time = None
            best_leave_time = None
            best_recive_done = None
            for launch_node_candidate_tup in ucv_launch:
                best_assigned_customer = [launch_node_candidate_tup[0], launch_node_candidate_tup[-1]]
                best_arrive_time = [launch_node_candidate_tup[2], launch_node_candidate_tup[4]]
                best_leave_time = [launch_node_candidate_tup[3], launch_node_candidate_tup[5]]
                best_recive_done = [0, problem.customer_list[launch_node_candidate_tup[-1]].quantity]
                current_cus = launch_node_candidate_tup[-1]
                current_drone_leave_time = launch_node_candidate_tup[3]
                current_consumption_energey = launch_node_candidate_tup[6]
                current_quantity = problem.customer_list[current_cus].quantity
                current_system_time = launch_node_candidate_tup[3]
                ucv_land = []

                drone_fly_duration = problem.launch_time + problem.land_time + problem.distance_matrix_drone[current_cus][0]/problem.speed_of_drone
                energy_fly_to_depot = problem.energy_consumption_rate*(problem.weight_of_drone + current_quantity)*drone_fly_duration
                if energy_fly_to_depot + current_consumption_energey <= problem.drone_energy:
                    if len(unserved) >  1:
                        next_distance = problem.distance_matrix_drone[0][unserved[1]]
                    else:
                        next_distance = 0
                    drone_leave_time = current_drone_leave_time + drone_fly_duration
                    ucv_land.append((0, next_distance, drone_leave_time))

                for truck_sol in truck_solutions:
                    if np.sum(truck_sol.recived_truck) + np.sum(truck_sol.recived_drone) + sum(best_recive_done) > problem.truck_capacity:
                            continue
                    for pos, cus in enumerate(truck_sol.assigned_customers):
        
                        drone_fly_duration = problem.launch_time + problem.land_time + problem.distance_matrix_drone[current_cus][cus]/problem.speed_of_drone
                        energy_fly_to_land_point = problem.energy_consumption_rate*(problem.weight_of_drone + current_quantity)*drone_fly_duration
                        drone_land_time = current_drone_leave_time + drone_fly_duration
                        if energy_fly_to_land_point + current_consumption_energey <= problem.drone_energy and drone_land_time >= truck_sol.arrive_time[pos] and drone_land_time <= truck_sol.leave_time[pos]:    
                            if len(unserved) >  1:
                                next_distance = problem.distance_matrix_drone[cus][unserved[1]]
                            else:
                                next_distance = 0
                            ucv_land.append((cus, next_distance, drone_land_time))
                ucv_land.sort(key= lambda x: x[1])
                if len(ucv_land) != 0:
                    best_assigned_customer.append(ucv_land[0][0])
                    best_recive_done.append(0)
                    best_arrive_time.append(ucv_land[0][2])
                    best_leave_time.append(ucv_land[0][2])
   
                    # print("best_assigned_customer", best_assigned_customer)
                    # print("best_recive_drone", best_recive_done)
                    # print("best_arrive_time", best_arrive_time)
                    # print("best_leave_time", best_leave_time)
                temp_best_assigned = None
                temp_best_recive = None
                temp_best_arrive = None
                temp_best_leave = None
                for i in range(2, len(unserved) + 1):
                    customers_to_serve = unserved[:i]
                    temp_arrive_time = []
                    temp_leave_time = []
                    temp_recive_drone = []
                    total_capacity_drone = problem.customer_list[customers_to_serve[0]].quantity
                    consumption_energy_new = current_consumption_energey
                    ready_time = current_drone_leave_time
                    temp_arrive_time.append(launch_node_candidate_tup[4])
                    temp_leave_time.append(ready_time)
                    temp_recive_drone.append(total_capacity_drone)

                    for iii in range(1, len(customers_to_serve)):
                        # print("customer_to_serve", customers_to_serve)
                        cus = customers_to_serve[iii]
                        drone_arrive_time = ready_time + problem.launch_time + problem.distance_matrix_drone[customers_to_serve[iii -1]][cus]/problem.speed_of_drone + problem.land_time
                        drone_wait_duration = max(0, problem.customer_list[cus].arrive_time - drone_arrive_time)
                        consumption_energy_new = consumption_energy_new + problem.energy_consumption_rate*(problem.weight_of_drone + total_capacity_drone)*(problem.launch_time + problem.distance_matrix_drone[customers_to_serve[iii -1]][cus]/problem.speed_of_drone + problem.land_time + drone_wait_duration) + problem.energy_consumption_rate*(problem.weight_of_drone + total_capacity_drone + problem.customer_list[cus].quantity)*(problem.customer_list[cus].service_time)
                        total_capacity_drone = total_capacity_drone + problem.customer_list[cus].quantity
                        ready_time = ready_time + problem.launch_time + problem.distance_matrix_drone[customers_to_serve[iii -1]][cus]/problem.speed_of_drone + problem.land_time + problem.customer_list[cus].service_time + drone_wait_duration
                        temp_arrive_time.append(drone_arrive_time)
                        temp_leave_time.append(ready_time)
                        temp_recive_drone.append(problem.customer_list[cus].quantity)

                        # print("Log nang luong:", )
                    if total_capacity_drone > problem.drone_capacity or consumption_energy_new + current_consumption_energey > problem.drone_energy:
                        break
                    else:
                        ucv_land = []
                        energy_fly_to_depot = problem.energy_consumption_rate*(problem.weight_of_drone + total_capacity_drone)*(problem.launch_time + problem.land_time + problem.distance_matrix_drone[customers_to_serve[-1]][0]/problem.speed_of_drone)
                        if consumption_energy_new + current_consumption_energey + energy_fly_to_depot <= problem.drone_energy:
                            if i == len(unserved):
                                next_distance = 0
                            else:
                                next_distance = problem.distance_matrix_drone[0][unserved[i]]
                            land_drone = ready_time + problem.launch_time + problem.land_time + problem.distance_matrix_drone[customers_to_serve[-1]][0]/problem.speed_of_drone
                            ucv_land.append((0, next_distance, land_drone))

                        for truck_sol in truck_solutions:
                            for pos, cus in enumerate(truck_sol.assigned_customers):
                                if np.sum(truck_sol.recived_truck) + np.sum(truck_sol.recived_drone) + total_capacity_drone > problem.truck_capacity:
                                    continue
                                energy_fly_to_land_point = problem.energy_consumption_rate*(problem.weight_of_drone + total_capacity_drone)*(problem.launch_time + problem.land_time + problem.distance_matrix_drone[customers_to_serve[-1]][cus]/problem.speed_of_drone)
                                drone_land_time = ready_time + problem.launch_time + problem.land_time + problem.distance_matrix_drone[customers_to_serve[-1]][cus]/problem.speed_of_drone
                                if energy_fly_to_land_point + current_consumption_energey + consumption_energy_new <= problem.drone_energy and drone_land_time >= truck_sol.arrive_time[pos] and drone_land_time <= truck_sol.leave_time[pos]:
                                    if i == len(unserved):
                                        next_distance = 0
                                    else:
                                        next_distance = problem.distance_matrix_drone[cus][unserved[i]]
                                    ucv_land.append((cus, next_distance, drone_land_time))
                        ucv_land.sort(key= lambda x: x[1])
                        if len(ucv_land) != 0:
                            temp_arrive_time.append(ucv_land[0][-1])
                            temp_leave_time.append(ucv_land[0][-1])
                            temp_recive_drone.append(0)
                            temp_best_assigned = deepcopy(customers_to_serve)
                            temp_best_assigned.append(ucv_land[0][0])
                            temp_best_arrive = deepcopy(temp_arrive_time)
                            temp_best_leave = deepcopy(temp_leave_time)
                            temp_best_recive = deepcopy(temp_recive_drone)

                if temp_best_assigned != None:
                    if best_recive_done[-1] != 0:
                        best_assigned_customer = best_assigned_customer + temp_best_assigned[1:]
                        best_recive_done = best_recive_done + temp_best_recive[1:]
                        best_arrive_time = best_arrive_time + temp_best_arrive[1:]
                        best_leave_time = best_leave_time + temp_best_leave[1:]
                    else:
                        best_assigned_customer = best_assigned_customer[:-1] + temp_best_assigned[1:]
                        best_recive_done = best_recive_done[:-1] + temp_best_recive[1:]
                        best_arrive_time = best_arrive_time[:-1] + temp_best_arrive[1:]
                        best_leave_time = best_leave_time[:-1] + temp_best_leave[1:]
                
                if best_recive_done[-1] == 0:
                    drone_trip_ob = Drone_Trip(best_assigned_customer, best_recive_done, best_arrive_time, best_leave_time)
                    trip_list.append(drone_trip_ob)

                    add_good = sum(best_recive_done)
                    leave_time_update = best_leave_time[-1]
                    cus_update = best_assigned_customer[-1]
                    current_system_time = best_arrive_time[-1]
                    
                    if cus_update != 0:
                        idx_truck = None
                        
                        for idx, truck_sol in enumerate(truck_solutions):
                            if cus_update in truck_sol.assigned_customers:
                                idx_truck = idx
                                break
                        truck_solutions[idx_truck] = update_truck_solution(truck_solutions[idx_truck], problem, cus_update, add_good, leave_time_update)
                    flag = True
                    unserved = unserved[len(best_assigned_customer) - 2:]
                    current_customer = best_assigned_customer[-1]
                    first_trip = False
                    break
            
            if flag == False:
                return False, first_unserved_customer          

        trip_sol = Drone_Solution(len(trip_list), trip_list)
        drone_solutions.append(trip_sol)
    return truck_solutions, drone_solutions

