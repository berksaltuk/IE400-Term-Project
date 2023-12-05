from gurobipy import *

# Read depot node distances
depot_distances = {'X': [1, 1, 1, 2, 3, 2, 1, 1], 'Y': [3, 2, 1, 1, 1, 1, 1, 2]}

# Read distances between nodes
with open('distances.txt', 'r') as file:
    distances = [list(map(float, line.split())) for line in file]

# Read paths
with open('paths.txt', 'r') as file:
    paths = [line.strip().split(': ')[1].split('-') for line in file]

# Create the Gurobi model
model = Model("Depot_Train_Assignment")

# Define decision variables
x = {}
for i in ['X', 'Y']:
    for j in range(1, 11):
        for k in range(1, 16):
            x[i, j, k] = model.addVar(vtype=GRB.BINARY, name=f'x_{i}_{j}_{k}')

# Set up objective function
objective = quicksum(x[i, j, k] * distances[ord(paths[k-1][0]) - ord('A')][j-1] +
                     x[i, j, k] * distances[ord(paths[k-1][-1]) - ord('A')][0 if i == 'X' else 1]
                     for i in ['X', 'Y'] for j in range(1, 11) for k in range(1, 16))

model.setObjective(objective, GRB.MINIMIZE)

# Add constraints
for i in ['X', 'Y']:
    model.addConstr(quicksum(x[i, j, k] for j in range(1, 11) for k in range(1, 16)) >= 5)

for i in ['X', 'Y']:
    for j in range(1, 11):
        model.addConstr(quicksum(x[i, j, k] for k in range(1, 16)) <= 3)

# Optimize the model
model.optimize()

# Print results
for i in ['X', 'Y']:
    assigned_trains = [k for j in range(1, 11) for k in range(1, 16) if x[i, j, k].x > 0.5]
    print(f"Depot {i} assigned trains: {assigned_trains}")
