# -*- coding: utf-8 -*-
"""
@author: Afei

"""

# %% Package
import os, time, signal, sys, math, random, copy, threading 
from PIL import Image, ImageDraw
from heapq import heappush, heapify, nlargest
import matplotlib.pyplot as plt
from bisect import bisect_right
from itertools import accumulate
import numpy as np
#from mandatoryclass import ga
# %% Class definition


#---------------------------- Mandatory class -----------------------------
class routeclass(object):
    def __init__(self, route=[], cost=0, is_valid=False, demand=0):
        self.is_valid = is_valid
        self.route = route
        self.cost = cost
        self.demand = demand

    def InsertRoute(self, index, route):
        self.is_valid = False
        self.route = self.route[:index + 1] + route + self.route[index + 1:]

    def AppendNode(self, node):
        self.is_valid = False
        self.route = self.route[:-1] + [node] + [1]

    def RemoveNode(self, x):
        self.is_valid = False
        del self.route[self.route.index(x)]
        
    #print
    def __repr__(self):
        debug_str = ", cost = " + str(self.cost) + ", demand = " + str(self.demand)
        ret_str = "->".join(str(n) for n in self.route)
        return ret_str + (debug_str if False else "")
    
    
class solutionclass(object):
    def __init__(self, routes=[], cost=0, is_valid=False, demand=0):
        self.is_valid = is_valid
        self.routes = routes
        self.cost = cost
        self.demand = demand
        self.penalty = 0

    def shuffle(self):
        random.shuffle(self.routes)

    def RemoveNode(self, x):
        for route in self.routes:
            if x in route.route:
                route.RemoveNode(x)
        self.is_valid = False

    def InsertRoute(self, route_id, route_index, route):
        self.routes[route_id].InsertRoute(route_index, route)
        self.is_valid = False

    def RandomSubroute(self):
        r_i = random.randrange(0, len(self.routes))
        while len(self.routes[r_i].route) == 2:
            r_i = random.randrange(0, len(self.routes))
        c_s = random.randrange(1, len(self.routes[r_i].route))
        c_e = c_s
        while c_e == c_s:
            c_e = random.randrange(1, len(self.routes[r_i].route))
        if c_s > c_e:
            c_s, c_e = c_e, c_s
        return self.routes[r_i].route[c_s:c_e]

    def __repr__(self):
        return "\n".join([str(route) for route in self.routes])
    
    def __lt__(self, other):
        return self.cost < other.cost


#---------------------------- STEP 1 --------------------------------------
#attributes: data, dimension, capacity, demand, coords
class cvrpinitial(object):
    
    def __init__(self, data_file):
        self.ReadData(data_file)
        self.CalcDistance()
        self.start_node = 1
        self.max_route_len = math.ceil(self.dimension/self.instance_num)+2
        
    # get customer number, capacity, demand, coordinate and trucks number
    def ReadData(self, data_file):
        with open(data_file) as f:
            result = [line.rstrip("\n") for line in f.readlines()] #remove the '\n' 
        
        self.dimension = int(result[0].split()[-1]) # spilt by space and select the number
        self.capacity = int(result[1].split()[-1])
        self.demand = [-1 for _ in range(self.dimension + 1)]
        self.coords = [(-1, -1) for _ in range(self.dimension + 1)]
        for i in range(3, self.dimension + 3): #get the index and coordinate of every place
            index, xc, yc = [float(x) for x in result[i].split()]
            self.coords[int(index)] = (xc, yc)
        for i in range(self.dimension + 4, 2 * (self.dimension + 2)):
            index, demand = [int(x) for x in result[i].split()]
            self.demand[index] = demand
        self.instance_num = math.ceil(sum(self.demand)/self.capacity)
            
    # calculate distance between all costumers
    def CalcDistance(self):
        self.dist = [list([0 for _ in range(self.dimension + 1)]) \
                        for _ in range(self.dimension + 1)]
        for xi in range(self.dimension + 1):
            for yi in range(self.dimension + 1):
                self.dist[xi][yi] = math.sqrt((self.coords[xi][0] - self.coords[yi][0])**2 + (self.coords[xi][1] - self.coords[yi][1])**2)
                
                
                
#---------------------------- STEP 2 --------------------------------------

#get initial population(route)
class InitPopulation(object):
    
    def __init__(self, init, greedy, popsize_init):
        self.pop_size = popsize_init # set population size
        self.initial = init # 
        self.chromosomes = [] # 1 choromosome = 1 solution
        self.greedy = greedy
        
        for x in [self.InitializeRandomPopulation(self.greedy) for _ in range(self.pop_size)]:
            heappush(self.chromosomes, (x.cost, x))
        self.best_solution = self.chromosomes[0][1]
        random.seed()
    
    
        
    def InitializeRandomPopulation(self, greedy=False):
        customers_index = [i for i in range(2, self.initial.dimension + 1)] # i= 2->dimension
        random.shuffle(customers_index) # change the sequence 
        routes = []
        cur_route = [self.initial.start_node] # set depot as first node
        route_demand = 0
        route_length = 0
        #creaing 1 solution(chromosome) every loop

        while customers_index:
            route_num = len(routes)
            if route_num == self.initial.instance_num:
                customers_index = [i for i in range(2, self.initial.dimension + 1)]
                random.shuffle(customers_index) # change the sequence 
                routes = []
                cur_route = [self.initial.start_node]
                route_demand = 0
                route_length = 0
            
            i = 0
            if greedy:
                i = min([i for i in range(len(customers_index))], \
                        key=lambda x: self.initial.dist[cur_route[-1] if random.uniform(0, 1) < 0.9 else 1][customers_index[x]])
            else:
                i = random.randint(0,len(customers_index)-1)
            node = customers_index[i]
    
            if route_length <= self.initial.max_route_len and route_demand + self.initial.demand[node] <= self.initial.capacity:
                cur_route += [node]
                route_length += 1
                route_demand += self.initial.demand[node]
                #print(cur_route)
                del customers_index[i]
                continue # if meet the condition, the program will continue the while loop at start
            cur_route += [1]  # creating a circle by adding depot as the last node
            routes += [self.CreateRoute(cur_route)] # declare routes(list) using class Route by CreateRoute()
            
            # initialize parameter after CreateRoute()
            cur_route = [self.initial.start_node]
            route_demand = 0
            route_length = 0
            
        routes += [self.CreateRoute(cur_route + [1])] # add the last solution
        
        return self.CreateSolution(routes)
    
    # ========================= GENERAL FUNCTIONS ===========================
    
    def CreateRoute(self, route_list):
        if route_list[0] != self.initial.start_node:# make sure to create a valid route with the same start node
            return None
        cost = 0
        demand = 0
        is_valid = True
        for i in range(1, len(route_list)):
            n1, n2 = route_list[i - 1], route_list[i]
            cost += self.initial.dist[n1][n2]
            demand += self.initial.demand[n2]
        if demand > self.initial.capacity:
            is_valid = False
            
        #using a new class Route to save the route information in route
        route = routeclass(route_list, cost, is_valid, demand)
        return route
        
    def CreateSolution(self, routes):
        cost = 0
        demand = 0
        is_valid = True
        visited = set()
        for route in routes:
            if not route.is_valid:
                is_valid = False
            for x in route.route:
                visited.add(x)
            cost += route.cost
            demand += route.demand
        if len(visited) != self.initial.dimension:
            print("NOT ALL VISITED")
            print(visited)
        sol = solutionclass(routes, cost, is_valid, demand)
        
        #input(junk)
        return sol
    
    def BoundingBoxes(self, route):
        x_min = min(self.initial.coords[node][0] for node in route)
        x_max = max(self.initial.coords[node][0] for node in route)
        y_min = min(self.initial.coords[node][1] for node in route)
        y_max = max(self.initial.coords[node][1] for node in route)
        return x_min, x_max, y_min, y_max
    
    
    def SteepestAscentRoute(self, route):
        savings = 1
        iters = 0
        while savings > 0:
            savings = 0
            if iters > 1000:
                return route
            for t1_i in range(len(route) - 2):
                for t4_i in range(len(route) - 2):
                    if t4_i != t1_i and t4_i != t1_i + 1 and t4_i + 1 != t1_i:
                        t1 = route[t1_i]
                        t2 = route[t1_i + 1]
                        t3 = route[t4_i + 1]
                        t4 = route[t4_i]
                        diff = self.initial.dist[t1][t2] + self.initial.dist[t4][t3] - self.initial.dist[t2][t3] - self.initial.dist[t1][t4]
                        if diff > savings:
                            savings = diff
                            t1best = t1_i
                            t4best = t4_i
            if savings > 0:
                route[t1best+1], route[t4best] = route[t4best], route[t1best+1]
            iters += 1
        return route
    
    #AGAPopulation
    # input: solution(class Solution) with 4 attributes: cost(int), demand(int), is_valid(bool), routes(list class Route)
    # output: solution with different routes sequence
    def SteepestAscentSolution(self, solution):
        new_routes = []
        for route in solution.routes:
            route = self.SteepestAscentRoute(route.route)
            new_routes += [self.CreateRoute(route)]
        return self.CreateSolution(new_routes)

    
#---------------------------- STEP3 --------------------------------------

class ga(object):
    
    def __init__(self, data, iteration_time, replace_rate_ga, time_ga, mutate_rate_ga):
        self.pop = data
        self.num_iter = iteration_time
        self.iter = 0
        self.iters = 0
        self.print_cycle = 20
        self.best_solution = self.pop.chromosomes[0][1]
        self.num_populations = 1
        self.min_nodes = 4
        self.pop_bests = [0]        
        self.replace_rate = replace_rate_ga
        self.running_time = time_ga
        self.mutate_rate = mutate_rate_ga
        
    def RunGA(self):
        global iteration, best_cost, average_cost
        self.start_time = time.time()
        
        while self.iter < self.num_iter:
            # CVRPAdvanceGA.step()
            best = self.RunStep()
            self.best = best
            if self.iter % self.print_cycle == 0:
                #self.timings_file.write("{0} at {1}s\n".format(best.cost, time.time() - self.start_time))

                cost = [self.pop.chromosomes[i][0] for i in range(len(self.pop.chromosomes))]
                average_cost_iter = np.mean(cost)
                
                iteration += [self.iter]
                best_cost += [self.best.cost]
                average_cost += [average_cost_iter]
                print("iter: {0} best:{1} average:{2}".format(self.iter, self.best.cost, average_cost_iter), int(time.time() - self.start_time))
            self.iter += 1
            # 1800 secs or 10000 iterations
            if time.time() - self.start_time > self.running_time:
#                self.write_to_file("best-solution-marking.txt")
                break
#            if self.iter % self.num_iter:
#                 self.im = self.visualise(self.best_solution)
#                 self.im.save("image/"+str(self.best_solution.cost) + ".png")
        print("Best solution: " + str(best))
        print("Cost: " + str(best.cost))
        print(time.time() - self.start_time)
        
    # if (num_population != 1), it will work for parallization. 
    def RunStep(self):
        for i in range(self.num_populations):
            self.pop_bests[i] = self.MainStep()
        self.best_solution = min(self.pop_bests, key = lambda x: x.cost)
        return self.best_solution    
    
    def MainStep(self):
        global child, token
        replace = 1
        for select_time in range(math.ceil(len(self.pop.chromosomes)*self.replace_rate)):
            #normalize
            fit = [self.pop.chromosomes[i][0] for i in range(len(self.pop.chromosomes))]
            fit = [(i - max(fit)) for i in fit]
            # Create roulette wheel.
            sum_fit = sum(fit)
            wheel = list(accumulate([i/sum_fit for i in fit]))
            
            i = bisect_right(wheel, random.random())
            j = (i + 1) % len(wheel)
            ic, jc = self.pop.chromosomes[i][1], self.pop.chromosomes[j][1] # select 2 chromosomes
                
            child = self.RandomCrossover(ic, jc)   
            if random.uniform(0, 1) < 0.95:
                for _ in range(3):
                    c = self.BiggestOverlapCrossover(ic, child)
                    self.refresh(c)
                    if c.cost < child.cost:
                        child = c
            else:
                for _ in range(3):
                    c = self.RandomCrossover(ic, child)
                    self.refresh(c)
                    if c.cost < child.cost:
                        child = c
                     
            # control the length of routes 
            token = 0
            for i in range(self.pop.initial.instance_num):
                if len(child.routes[i].route) < self.min_nodes:
                    child.routes[i].is_valid = False
                    token += 1
            #print(token) 
            if token > 0:
                token = 0
                continue
            
            self.refresh(child)
#            self.RandomSwapMutation(child)
#            self.refresh(child)
            self.RepairOperator(child)
            self.refresh(child)
            self.pop.SteepestAscentSolution(child)
            self.refresh(child)
            self.pop.chromosomes[-replace] = (self.fitness(child), child) # replace the worst chromosome by child
            replace += 1
#        
        for mutate_index in range(math.ceil(len(self.pop.chromosomes))):
            if random.uniform(0,1) < self.mutate_rate:
                self.RandomSwapMutation(self.pop.chromosomes[mutate_index][1])
                self.refresh(self.pop.chromosomes[mutate_index][1])
                self.RepairOperator(self.pop.chromosomes[mutate_index][1])
                self.refresh(self.pop.chromosomes[mutate_index][1])

        # heap sort algorithm
        heapify(self.pop.chromosomes)
        self.iters += 1
        if (self.pop.chromosomes[0][1].cost<self.best_solution.cost):
            self.best_solution = self.pop.chromosomes[0][1]
        return self.best_solution

    def fitness(self, chromosome):
        penalty = self.penalty(chromosome)
        return chromosome.cost + penalty
    
    def penalty(self, chromosome):
        penalty_sum = 0
        for route in chromosome.routes:
            penalty_sum += max(0, route.demand - self.pop.initial.capacity)**2
        mnv = sum(self.pop.initial.demand[i] for i in range(self.pop.initial.dimension)) / self.pop.initial.capacity
        alpha = self.best_solution.cost / ((1 / (self.iters + 1)) * (self.pop.initial.capacity * mnv / 2)**2 + 0.00001)
        penalty = alpha * penalty_sum * self.iters / self.num_iter
        chromosome.penalty = penalty
        return penalty
    
    def RandomCrossover(self, chrom1, chrom2):
        child = copy.deepcopy(chrom1)
        sub_route = chrom2.RandomSubroute()
        for x in sub_route:
            child.RemoveNode(x)
        r_id, n_id = self.BestInsertionIndex(child, sub_route)
        child.InsertRoute(r_id, n_id, sub_route)
        return child

    def RandomSwapMutation(self, chromosome):
        global r_i
        r_i = random.randrange(0, len(chromosome.routes))
        c_i = random.randrange(1, len(chromosome.routes[r_i].route) - 1)
        node = chromosome.routes[r_i].route[c_i]
        chromosome.RemoveNode(node)
        if random.uniform(0, 1) < 0.25: # pSame
            _, best = self.BestInsertionRoute([node], chromosome.routes[r_i].route)
            best_i = (r_i, best)
        else:
            r_r_i = random.randrange(0, len(chromosome.routes))
            while r_r_i == r_i:
                r_r_i = random.randrange(0, len(chromosome.routes))
            _, best = self.BestInsertionRoute([node], chromosome.routes[r_r_i].route)
            best_i = (r_r_i, best)
        chromosome.InsertRoute(best_i[0], best_i[1], [node])

    def BiggestOverlapCrossover(self, c1, c2):
        child = copy.deepcopy(c1)
        sub_route = c2.RandomSubroute()
        routes = []
        for x in sub_route:
            child.RemoveNode(x)
        for i, route in enumerate(child.routes):
            x_min, x_max, y_min, y_max = self.BoundingBoxes(route.route)
            sx_min, sx_max, sy_min, sy_max = self.BoundingBoxes(sub_route)
            x_overlap = max(0, min(x_max, sx_max) - max(x_min, sx_min))
            y_overlap = max(0, min(y_max, sy_max) - max(y_min, sy_min))
            heappush(routes, (x_overlap * y_overlap, i))
        top3 = nlargest(6, routes)
        min_i = min((i[1] for i in top3), key = lambda x: child.routes[x].demand)
        _, best = self.BestInsertionRoute(sub_route, child.routes[min_i].route)
        child.InsertRoute(min_i, best, sub_route)
        return child


    #--------------------------------------------------------------------------------------------
    #----------------------------------General Functions-----------------------------------------
    #--------------------------------------------------------------------------------------------    
    def BoundingBoxes(self, route):
        x_min = min(self.pop.initial.coords[node][0] for node in route)
        x_max = max(self.pop.initial.coords[node][0] for node in route)
        y_min = min(self.pop.initial.coords[node][1] for node in route)
        y_max = max(self.pop.initial.coords[node][1] for node in route)
        return x_min, x_max, y_min, y_max

        
    def BestInsertionIndex(self, child, sub_route):
        best_payoff, best_rid, best_nid = -1, 0, 0
        for r_id, route in enumerate(child.routes):
            route = route.route
            subopt_best, n_id = self.BestInsertionRoute(sub_route, route)
            if subopt_best > best_payoff:
                best_payoff, best_rid, best_nid = subopt_best, r_id, n_id
        return best_rid, best_nid
    
    def BestInsertionRoute(self, sub_route, route):
        start = sub_route[0]
        end = sub_route[-1]
        best_payoff, best_i = 0, 0
        dist = self.pop.initial.dist
        i = 0
        for i in range(0, len(route) - 1):
            init_cost = dist[route[i]][route[i + 1]]
            payoff = init_cost - dist[route[i]][start] - dist[end][route[i + 1]]
            if payoff > best_payoff:
                best_payoff, best_i = payoff, i
        return best_payoff, best_i


    def refresh(self, solution):
        solution.cost, solution.demand = 0, 0
        for route_obj in solution.routes:
            route = route_obj.route
            route_obj.demand, route_obj.cost = 0, 0
            for i in range(0, len(route) - 1):
                route_obj.demand += self.pop.initial.demand[route[i]]
                route_obj.cost += self.pop.initial.dist[route[i]][route[i + 1]]
            solution.cost += route_obj.cost
            solution.demand += route_obj.demand
            if route_obj.demand > self.pop.initial.capacity:
                route_obj.is_valid = False
                solution.is_valid = False

    def RepairOperator(self, chromosome):
        routes = chromosome.routes
        r_max_i = max((i for i in range(len(routes))), key = lambda i: routes[i].demand)
        r_min_i = min((i for i in range(len(routes))), key = lambda i: routes[i].demand)
        if routes[r_max_i].demand > self.pop.initial.capacity:
            rint = random.randrange(1, len(routes[r_max_i].route) - 1)
            routes[r_min_i].AppendNode(routes[r_max_i].route[rint])
            routes[r_max_i].RemoveNode(routes[r_max_i].route[rint])
            return True
        return False
    

    def visualise(self, solution):
        im = Image.new( 'RGB', (1980,1980), "white") # create a new black image
        draw = ImageDraw.Draw(im)
        for i, route in enumerate(solution.routes):
            r_c = (i*i)%255
            g_c = (i*r_c)%255
            b_c = (i*g_c)%255
            nodes = route.route
            norm = lambda x, y: ((10*x + 250), (10*y + 250))
            draw.line([norm(*self.pop.initial.coords[n]) for n in nodes], fill=(r_c, g_c, b_c), width=5)
        return im
    
        
# %% Main 

if __name__ == "__main__": 
    # step 0: parameters setting
    
    data_file = ["test1_E-n51-k5_521.vrp","test2_B-n34-k5_788.vrp","test3_B-n78-k10_1221.vrp"]
    running_time = 1800
    running_iteration=15000
    replace_rate = [0.10, 0.15]
    mutation_prob = [0.01, 0.02, 0.03]
    popsize = [100, 200]
    greedy=False
    

    loop_time = 0
    loop_length = 12
    data_iteration = [[] for i in range(loop_length)]
    data_best_cost = [[] for i in range(loop_length)]
    data_average_cost = [[] for i in range(loop_length)]

    for popsize_loop in popsize:
        for replace_rate_loop in replace_rate:
            for mutate_rate_loop in mutation_prob:
                for data_file_loop in data_file:
                    iteration = []
                    best_cost = []
                    average_cost = []    
                    # step 1: read data and declare mandatory parameters
                    cvrp_data = cvrpinitial(data_file_loop)
                    # step 2: create initial population
                    initpop = InitPopulation(cvrp_data, greedy, popsize_loop)
                    # setp 3: evolve
                    best_sol = ga(initpop, running_iteration, replace_rate_loop, running_time, mutate_rate_loop)
                    best_sol.RunGA()
                    # step 4: output and visualization                    
                    data_iteration[loop_time].append(iteration)
                    data_best_cost[loop_time].append(best_cost)
                    data_average_cost[loop_time].append(average_cost)
                    
                    loop_time += 1
                    print(loop_time)
    
    test_best = [] 
    test_average = []
    test_iter = []

    for i in range(len(data_iteration)):
        test_best += data_best_cost[i]
        test_average += data_average_cost[i]
        test_iter += data_iteration[i]
