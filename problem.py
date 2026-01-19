import numpy as np


class Customer:
    def __init__(self, arrive_time, quantity, service_time, x, y):
        self.arrive_time = arrive_time
        self.quantity = quantity
        self.service_time = service_time
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Customer(x={self.x}, y={self.y}, demand={self.quantity}, time={self.arrive_time}, service={self.service_time})"


class Truck_Solution:
    def __init__(
        self, assigned_customers, recived_truck, recived_drone, arrive_time, leave_time
    ):
        self.assigned_customers = assigned_customers
        self.recived_truck = recived_truck
        self.recived_drone = recived_drone
        self.arrive_time = arrive_time
        self.leave_time = leave_time


class Drone_Trip:
    def __init__(self, assigned_customers, recived_drone, arrive_time, leave_time):
        self.assigned_customers = assigned_customers
        self.recived_drone = recived_drone
        self.arrive_time = arrive_time
        self.leave_time = leave_time


class Drone_Solution:
    def __init__(self, num_of_trips, trip_list):
        self.num_of_trips = num_of_trips
        self.trip_list = trip_list


class Problem:
    def __init__(
        self,
        number_customer,
        customer_list,
        number_of_trucks,
        number_of_drones,
        distance_matrix_truck,
        distance_matrix_drone,
        truck_capacity,
        drone_capacity,
        drone_energy,
        speed_of_truck,
        speed_of_drone,
        launch_time,
        land_time,
        energy_consumption_rate,
        weight_of_drone,
        cost_truck_unit,
        cost_drone_unit,
        fix_truck_cost,
        fix_drone_cost,
    ):
        self.number_customer = number_customer
        self.customer_list = customer_list
        self.number_of_trucks = number_of_trucks
        self.number_of_drones = number_of_drones
        self.distance_matrix_truck = distance_matrix_truck
        self.distance_matrix_drone = distance_matrix_drone
        self.truck_capacity = truck_capacity
        self.drone_capacity = drone_capacity
        self.drone_energy = drone_energy
        self.speed_of_truck = speed_of_truck
        self.speed_of_drone = speed_of_drone
        self.launch_time = launch_time
        self.land_time = land_time
        self.energy_consumption_rate = energy_consumption_rate
        self.weight_of_drone = weight_of_drone
        self.cost_truck_unit = cost_truck_unit
        self.cost_drone_unit = cost_drone_unit
        self.fix_truck_cost = fix_truck_cost
        self.fix_drone_cost = fix_drone_cost

    def check_capacity_truck_constraint(self, route):
        total = 0
        for cust in route:
            total = total + self.customer_list[cust].quantity
            if total > self.truck_capacity:
                return False

        return True

    def check_capacity_truck_contraint(self, truck_solution: Truck_Solution):
        customer_length = len(truck_solution.assigned_customers)
        if customer_length == 0:
            return True
        total_goods = 0
        for i in range(customer_length):
            total_goods = (
                total_goods
                + truck_solution.recived_truck[i]
                + truck_solution.recived_drone[i]
            )
        if total_goods > self.truck_capacity:
            return False
        return True

    def check_capacity_drone_trip_constraint(self, drone_trip: Drone_Trip):
        if len(drone_trip.assigned_customers) == 0:
            return True
        total_goods = 0
        for i in range(len(drone_trip.assigned_customers)):
            total_goods = total_goods + drone_trip.recived_drone[i]
        if total_goods > self.drone_capacity:
            return False
        return True

    def check_capacity_drone_constraint(self, drone_solution: Drone_Solution):
        for i in range(drone_solution.num_of_trips):
            if not self.check_capacity_drone_trip_constraint(
                drone_solution.trip_list[i]
            ):
                return False
        return True

    def check_energy_drone(self, drone_trip: Drone_Trip):
        number_point = len(drone_trip.assigned_customers)
        if number_point == 0:
            return True
        total_energy = 0
        total_goods = 0
        for i in range(number_point - 1):
            u, u_next = drone_trip.assigned_customers[i], drone_trip.assigned_customers[i+1]
            total_energy = total_energy + self.energy_consumption_rate*(self.weight_of_drone + total_goods)*(self.distance_matrix_drone[u][u_next]/self.speed_of_drone + self.launch_time)
            # print("i = ", i)
            if drone_trip.recived_drone[i+1] != 0:
                wait_time = max(self.customer_list[u_next].arrive_time - drone_trip.arrive_time[i+1], 0)
                # print("wait_time: ", wait_time)
                total_energy = total_energy + self.energy_consumption_rate*(self.weight_of_drone + total_goods)*(self.land_time + wait_time)
                total_goods = total_goods + drone_trip.recived_drone[i]
                total_energy = (
                    total_energy
                    + self.energy_consumption_rate
                    * (self.weight_of_drone + total_goods)
                    * self.customer_list[u_next].service_time
                )
            else:
                wait_time = max(0, drone_trip.leave_time[i+1] - drone_trip.arrive_time[i+1])
                # print("wait_time 2:", wait_time)
                total_energy = total_energy + self.energy_consumption_rate*(self.weight_of_drone + total_goods)*wait_time
        # print("total_energy", total_energy, self.drone_energy)
        if total_energy > self.drone_energy:
            return False
        else:
            return True

    def cal_truck_cost(self, truck_solution: Truck_Solution):
        if len(truck_solution.assigned_customers) == 0:
            return 0
        else:
            total_cost = (
                self.fix_truck_cost
                + self.distance_matrix_truck[0][truck_solution.assigned_customers[0]]
                * self.cost_truck_unit
            )
            for i in range(len(truck_solution.assigned_customers) - 1):
                total_cost = (
                    total_cost
                    + self.distance_matrix_truck[truck_solution.assigned_customers[i]][
                        truck_solution.assigned_customers[i + 1]
                    ]
                    * self.cost_truck_unit
                )
            return total_cost

    def cal_drone_trip_cost(self, drone_trip: Drone_Trip):
        num_point = len(drone_trip.assigned_customers)
        if num_point == 0:
            return 0
        else:
            total_cost = 0
            for i in range(num_point - 1):
                total_cost = (
                    total_cost
                    + self.distance_matrix_drone[drone_trip.assigned_customers[i]][
                        drone_trip.assigned_customers[i + 1]
                    ]
                    * self.cost_drone_unit
                )
            return total_cost

    def cal_drone_cost(self, drone_solution: Drone_Solution):
        if drone_solution.num_of_trips == 0:
            return 0
        total_cost = self.fix_drone_cost
        for trip in drone_solution.trip_list:
            total_cost = total_cost + self.cal_drone_trip_cost(trip)
        return total_cost

    def cal_total_cost(self, truck_solution_list, drone_solution_list):
        total_cost = 0
        for truck_solution in truck_solution_list:
            total_cost = total_cost + self.cal_truck_cost(truck_solution)
        for drone_solution in drone_solution_list:
            total_cost = total_cost + self.cal_drone_cost(drone_solution)
        return total_cost

    def customer_wait_max(self, truck_solution_list, drone_solution_list):
        customer_wait_time = np.zeros(len(self.customer_list))
        for truck_solution in truck_solution_list:
            num_point = len(truck_solution.assigned_customers)
            if num_point == 0:
                continue
            else:
                for i in range(num_point):
                    if truck_solution.recived_truck[i] == 0:
                        continue
                    else:
                        customer = truck_solution.assigned_customers[i]
                        customer_wait_time[customer] = max(
                            truck_solution.arrive_time[i]
                            - self.customer_list[customer].arrive_time,
                            0,
                        )

        for drone_solution in drone_solution_list:
            for drone_trip in drone_solution.trip_list:
                num_point = len(drone_trip.assigned_customers)
                if num_point == 0:
                    continue
                else:
                    for i in range(num_point):
                        if drone_trip.recived_drone[i] == 0:
                            continue
                        else:
                            customer = drone_trip.assigned_customers[i]
                            customer_wait_time[customer] = max(
                                drone_trip.arrive_time[i]
                                - self.customer_list[customer].arrive_time,
                                0,
                            )
        return np.max(customer_wait_time)

    def cal_distance_truck(self, truck_solution: Truck_Solution):
        total_distance = 0
        num_point = len(truck_solution.assigned_customers)
        if num_point == 0:
            return total_distance
        else:
            total_distance = (
                total_distance
                + self.distance_matrix_truck[0][truck_solution.assigned_customers[0]]
                + self.distance_matrix_truck[truck_solution.assigned_customers[-1]][0]
            )
            for i in range(num_point -1):
                u, u_next = (
                    truck_solution.assigned_customers[i],
                    truck_solution.assigned_customers[i + 1],
                )
                total_distance = total_distance + self.distance_matrix_truck[u][u_next]
            return total_distance

    def cal_truck_fairness(self, truck_soltuon_list):
        num_truck = len(truck_soltuon_list)
        distance_list = np.zeros(num_truck)
        for i in range(num_truck):
            distance_list[i] = self.cal_distance_truck(truck_soltuon_list[i])
        return np.std(distance_list)
