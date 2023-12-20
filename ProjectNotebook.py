#!/usr/bin/env python
# coding: utf-8

# # Part 1

# In[257]:


from gurobipy import *

import numpy as np


# In[258]:


node_no = {
    'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7
}


# In[259]:


with open('distances.txt', 'r') as file:
    distances = [list(map(float, line.split())) for line in file]


# In[260]:


paths = []
with open('paths.txt', 'r') as file:
    for line in file:
        at_node = {}
        path = line.strip().split(': ')[1].split('-')
        actual_path = []
        start = node_no[path[0]]
        end = node_no[path[-1]]
        total_dist = 0
        at_node[0] = path[0]
        for el in range(len(path)-1):
            dest = path[el + 1]
            source = path[el] 
            
            total_dist += distances[node_no[source]][node_no[dest]]
            at_node[total_dist] = path[el+1]
        paths.append({'total_dist':total_dist, 'start':start, 'end': end, 'actual_path':path, 'arrival_times':at_node})


# In[261]:


starts_with = {}
for i in range(15):
    for j in range(8):
        starts_with[i, j] = 1 if paths[i]['start'] == j else 0


# In[262]:


distances_to_depots = []
with open('depot_node_distances.txt', 'r') as file:
    for line in file:
        distances_to_depots.append([int(x) for x in line.strip().split(': ')[1].split('-')])


# In[263]:


depot_no = {
    'X':0, 'Y':1
}


# In[264]:


odd_even = {}

for i in range(len(paths)):
        for j in range(2):
            time = 20
            #for X depot
            time -= distances_to_depots[j][paths[i]['start']]
            tour = 0
            while(time > paths[i]['total_dist']):
                time -= paths[i]['total_dist']
                tour += 1
            if tour % 2 == 0: # even number of tours
                time -= distances_to_depots[j][paths[i]['start']]
                if time >= 0:
                    # print("success! begin and end nodes are same for route")
                    odd_even[i, j] = 1
                else:
                    time += paths[i]['total_dist']
                    tour -= 1
                    odd_even[i, j] = 0

                    #print("you need to take one less tour, sorry, ODDIZED")
            else: # odd number of tours
                time -= distances_to_depots[j][paths[i]['end']]
                if time >= 0:
                    # print("success! begin and end nodes are not same for route")
                    odd_even[i, j] = 0
                else:
                    time = time + paths[i]['total_dist']
                    tour -= 1
                    odd_even[i, j] = 1
                    # print("you need to take one less tour, sorry , EVENIZED")
            paths[i][f'tours_taken_{"X" if j == 0 else "Y"}'] = tour
            


# In[265]:


X = {}
model = Model('Part1_model')

# 1 if train i assigned to depot j
for i in range(1,16):
    for j in ['X','Y']:
            X[i,j] = model.addVar(vtype = GRB.BINARY,name=f'x_{i}_{j}')

# Constraints
# Trains can be assigned to one depot only
for i in range(1,16):
    model.addConstr(quicksum(X[i,j] for j in ['X','Y']) == 1)
     
# X -> A - max 3        
# A route cannot be used more than 3 trains
for j in ['X','Y']:
    for k in range(0,8):
        model.addConstr(quicksum(starts_with[i-1,k]*X[i,j] for i in range(1,16)) <= 3)
        
# Each depot should have at least 5 trains assigned to it
for j in ['X','Y']:
    model.addConstr(quicksum(X[i,j] for i in range(1,16)) >= 5)
        

objective_func = quicksum((1 + odd_even[i - 1, depot_no[j]]) * X[i,j] * distances_to_depots[depot_no[j]][paths[i-1]['start']] + 
                          (1 - odd_even[i - 1, depot_no[j]]) * X[i,j] * distances_to_depots[depot_no[j]][paths[i-1]['end']] 
                          for i in range(1,16) for j in ['X', 'Y'])
model.setObjective(objective_func, GRB.MINIMIZE)

model.optimize()


# In[266]:


print("Optimal total distance is", model.objVal)
print()
assigned = {}
for j in ['X', 'Y']:
    assigned_trains = [i for i in range(1, 16) if X[i, j].X == 1]
    assigned[j] = assigned_trains
    print(f"Trains assigned to depot {j}: {assigned_trains}")


# # Part 2

# In[273]:


for i in assigned['X']:    
    paths[i-1].pop('tours_taken_Y', None)
    paths[i-1]['X'] = 1
    paths[i-1]['Y'] = 0
    
for i in assigned['Y']:    
    paths[i-1].pop('tours_taken_X', None)
    paths[i-1]['X'] = 0
    paths[i-1]['Y'] = 1


# In[268]:


for i in range(len(paths)):
    arr_times = paths[i]['arrival_times']
    start_depot = 0 if 'tours_taken_X' in paths[i] else 1
    first_dis = distances_to_depots[start_depot][paths[i]['start']]
    
    tours = paths[i]['tours_taken_X'] if 'tours_taken_X' in paths[i] else paths[i]['tours_taken_Y']
    
    actual_path = paths[i]['actual_path']
    
    new_path = actual_path
    for j in range(tours-1):
        if j % 2 == 0:
            new_path = new_path + actual_path[::-1][1:]
        else:
            new_path = new_path + actual_path[1:]
    
    arr_times_new = {}
    arr_times_new[0] = 'X' if start_depot == 0 else 'Y'
    
    first_dis = distances_to_depots[start_depot][paths[i]['start']]
    
    total_dist = 0
    arr_times_new[first_dis] = new_path[0]
    
    for el in range(len(new_path)-1):
            dest = new_path[el + 1]
            source = new_path[el] 
            
            total_dist += distances[node_no[source]][node_no[dest]]
            arr_times_new[total_dist + first_dis] = new_path[el+1]
    if(odd_even[i, start_depot] == 1):
        arr_times_new[total_dist + 2 * first_dis] = 'X' if start_depot == 0 else 'Y'
    else:
        arr_times_new[total_dist + first_dis + distances_to_depots[start_depot][paths[i]['end']]] = 'X' if start_depot == 0 else 'Y'
    

    
    total_dur = list(arr_times_new)[-1]
    for k in range(25):
        if k not in arr_times_new:
            arr_times_new[k] = 'OW' # On way
        if k > total_dur:
            arr_times_new[k] = 'MA' # Maintenance 
      
    node_no['X'] = -1
    node_no['Y'] = -1
    node_no['MA'] = -1
    node_no['OW'] = -1
    
    where_is_train = []
    for k in range(25):
        node = node_no[arr_times_new[k]]
        row = []
        for l in range(0,8):
            if(l == node):
                row.append(1)
            else:
                row.append(0)
        where_is_train.append(row)
            
    # 21: 'MA' means maintenance from 20 to 21        
    paths[i]['train_locs'] = dict(sorted(arr_times_new.items()))
    paths[i]['path_after_tours'] = new_path
    paths[i]['total_operation_time'] = total_dur
    paths[i]['where_is_train'] = where_is_train


# In[269]:


# read parameters.txt
other_params = {}

with open('parameters.txt', 'r') as file:
    for line in file:
        other_params[line.strip().split(': ')[0]] = int(line.strip().split(': ')[1].split("$")[0].replace(",", ""))


# In[270]:


other_params


# In[274]:


paths[0] # information we keep for train 1 as an instance.


# In[275]:


Y = {}
I = {}
O = {}
F = {}


D = {}

ET = {}
E = {}

model2 = Model('Part2_model')

# Y_i_j Is train i type j, 'E'lectric or 'D'iesel
for i in range(1,16):
    for j in ['E','D']:
            Y[i,j] = model2.addVar(vtype = GRB.BINARY,name=f'y_{i}_{j}')

# I_i: number of in depot charging station built at depot i
for i in ['X', 'Y']:
    I[i] = model2.addVar(vtype = GRB.INTEGER, name=f'I_{i}')
    
# F_i: number of in depot fueling station built at depot i
for i in ['X', 'Y']:
    F[i] = model2.addVar(vtype = GRB.INTEGER, name=f'F_{i}')

# O_i: number of on-route charging station built at node i
for i in range(1,9):
    O[i] = model2.addVar(vtype = GRB.INTEGER, name=f'O_{i}')
    
# D_it: number of trains being charged at node i at time t
for i in range(1,9):
    for t in range(1,21):
        D[i, t] = model2.addVar(vtype = GRB.INTEGER, name=f'D_{i}_{t}')
        
#which train is charged at which node calculation
# ET_it Remaining time without charging for electric train i in time t
for i in range(1,16):
    for t in range(0, 21):
        ET[i,t] = model2.addVar(ub = 8, vtype = GRB.INTEGER, name=f'ET_{i}_{t}')
    
#If the train i chose to be charged at node j at time t
for i in range(1,16):
    for j in range(1,9):
        for t in range(1,21):
            E[i,j,t] = model2.addVar(vtype = GRB.BINARY, name=f'E_{i}_{j}_{t}')

for i in range(1,16):
    model2.addConstr(ET[i, 0] == 8)
    
# capacity constraints -> in depot - 3 electric or 2 diesels, on route - 1 electric only
# O[i]*1 - on route charging capacity
for i in range(1,9):
    for t in range(1,21):
        model2.addConstr(D[i, t] <= O[i] * 1, name=f'ORC_{i}_{t}')
        
# I[i]*3 - in depot charging capacity  
for i in ['X', 'Y']:
        model2.addConstr((quicksum(Y[j, 'E'] * paths[j-1][i] for j in range(1,16))) / 5 <= I[i] * 3, name=f'IDCC_{i}')
         
# F[i]*2 - in depot fueling capacity
for i in ['X', 'Y']:
        model2.addConstr((quicksum(Y[j, 'D'] * paths[j-1][i] for j in range(1,16))) / 5 <= F[i] * 2, name=f'IDFC_{i}')

# Number of trains charged at node i at time t equals to the sum of trains that chose to be charged there.
for j in range(1,9):
    for t in range(1,21):
        model2.addConstr(quicksum(E[i,j,t] for i in range(1,16)) == D[j,t], name=f'NOC_{j}_{t}')

# each train can be charged at atmost 1 node, for every instance
for i in range(1,16):
    for t in range(1,21):
        model2.addConstr(quicksum(E[i,j,t] for j in range(1,9)) <= 1, name=f'MC_{i}_{t}')

# if Eijt is true for i,t pair for any j, ETit should be 8, ow, it should be ETi,t-1 -1
for i in range(1,16):
        for t in range(1,21):
            # If quicksum(E[i, j, t] for j in range(1, 9)) is 0, then ET[i, t] = ET[i, t-1] - 1
            model2.addConstr(ET[i, t] == (ET[i, t-1] - 1) * (1 - (quicksum(E[i, j, t] for j in range(1, 9)))) + 8 * (quicksum(E[i, j, t] for j in range(1, 9))), name=f'ELC_{i}_{t}')
            
# a train can choose to be charged at a node if and only if it is at a node at that time instance, i.e, ch_av=1
for i in range(1,16):
    for j in range(1,9):
        for t in range(1,21):
            model2.addConstr(E[i,j,t] <= paths[i-1]['where_is_train'][t][j-1], name=f'IODC_{i}_{j}_{t}')
            
# energy levels should be nonnegative
for i in range(1,16):
        for t in range(1,21):
            model2.addConstr(ET[i,t] >= 0, name=f'ELNN_{i}_{j}_{t}')
            
# Ei,j,t can be 1 if and only if Yi_E is 1, that is a train can choose be charged at 
# an instance if it is an electric train
for i in range(1,16):
    for j in range(1,9):
        for t in range(1,21):
            model2.addConstr(E[i,j,t] <= Y[i, 'E'], name=f'OCET_{i}_{j}_{t}')

# total train number should be 15
model2.addConstr(quicksum(Y[i,j] for i in range(1,16) for j in ['E','D']) == 15, name=f'TN')

# a train can be either E or D
for i in range(1,16):
    model2.addConstr((Y[i, 'E'] + Y[i, 'D']) == 1, name=f'TT_{i}')

# number of charging & fueling stations must be non-negative integers
for i in ['X', 'Y']:
    model2.addConstr(F[i] >= 0, name=f'FNN_{i}')
    
for i in ['X', 'Y']:
    model2.addConstr(I[i] >= 0, name=f'ICNN_{i}')

for i in range(1,9):
    model2.addConstr(O[i] >= 0, name=f'OCNN_{i}')
    
        
# minimize cost: total hours of operation based on train type + no of diesel & electric trains bought + number of fuel&charge stations, objective func
fueling_station_cost = quicksum(other_params["Cost of In-Depot Fuel Station"]*F[i] for i in ['X', 'Y'])
in_depot_ch_cost = quicksum(other_params["Cost of In-Depot Charging Station"]*I[i] for i in ['X', 'Y'])
on_route_ch_cost = quicksum(other_params["Cost of On-Route Charging Station"]*O[i] for i in range(1,9))

train_cost = quicksum(other_params["Cost of Purchasing an Electric Train"]*Y[i,'E'] 
                      + other_params["Cost of Purchasing a Diesel Train"]*Y[i,'D'] 
                     for i in range(1,16))

# energy spent by each train
energy_cost = quicksum(other_params["Cost of Energy Spend by Diesel Train (by working hour)"]*paths[i-1]['total_operation_time']*Y[i,'D'] 
                + other_params["Cost of Energy Spend by Electric Train (by working hour)"]*paths[i-1]['total_operation_time']*Y[i,'E'] for i in range(1,16))

objective_func = fueling_station_cost + in_depot_ch_cost + on_route_ch_cost + train_cost + energy_cost

model2.setObjective(objective_func, GRB.MINIMIZE)

model2.write("model2.lp")

model2.optimize()


# In[276]:


print("Optimal cost is {:,}$".format(model2.objVal))
print()
diesel = []
electric = []

for i in range(1,16):
    for j in ['E','D']:
        if j == 'E' and Y[i,j].X == 1:
            electric.append(i)
        if j == 'D' and Y[i,j].X == 1:
            diesel.append(i)

print("Electric trains: ", electric)
print("Diesel trains: ", diesel)
print()
print("Number of in-depot charging stations at depot X: ", int(I['X'].X))
print("Number of in-depot charging stations at depot Y: ", int(I['Y'].X))
print()
print("Number of in-depot fueling stations at depot X: ", int(F['X'].X))
print("Number of in-depot fueling stations at depot Y: ", int(F['Y'].X))
print()

nodes = "ABCDEFGH"
for i in range(1,9):
    print(f"Number of on-route charging stations at node {nodes[i-1]}: ", int(O[i].X))

