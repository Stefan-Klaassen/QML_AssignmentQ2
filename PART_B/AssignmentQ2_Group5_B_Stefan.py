# -*- coding: utf-8 -*-

"""
Title: Vehicle routing problem
Course: ME44206 Quantitative Methods for Logistics
Part: B
Authors:
    Stefan Klaassen - 6076947
    Peter Nederveen - 
    Britt van de Geer - 
    Bart Verzijl - 
Last updated: 2025-12-04
Version: 1.0

Usage:
    ../
     ├── AssignmentQ2_Group5_B.py
     └── data_small.txt

Dependencies:
    Python 3.13.7+
    gurobipy

"""

# IMPORTS
#==================================================================================================

import math, sys
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Generator
from gurobipy import Model, GRB, quicksum


# MODEL DATA
#==================================================================================================

# CONSTANTS
INF = 1e5
MAX_TIME = 2000
MAX_BATTERY = 200
FLEET_SIZE = 4
VEHICLE_PACE = 2
VEHICLE_CAPACITY = 120
VEHICLE_RANGE = 110
VEHICLE_CHARGE_RATE = 1.0
VEHICLE_DISCHARGE_RATE = 1.0

@dataclass
class Node:
    LOC_ID: int
    XCOORD: int
    YCOORD: int
    DEMAND: int
    READYTIME: int
    DUETIME: int
    SERVICETIME: int
    CHARGING: int

def get_node_data(filename: str) -> list[Node]:
    try:
        file = Path(__file__).parent / filename
        assert file.is_file(), f"Kan '{filename}' niet vinden in '{file.parent}'."
    except Exception as e: 
        print(e)
        sys.exit(1)
    data = []
    if not file.exists():
        raise FileNotFoundError(f"Data file not found: {file}")
    with file.open('r') as f:
        for line in f:
            row = [int(v) for v in line.strip().split()]
            instance = Node(*row)
            data.append(instance)
    return data

def build_distance_mat(data: list[Node]) -> list[list[float]]:

    def _euclidean_distance(node1: Node, node2: Node) -> float:
        return math.sqrt( (node2.XCOORD - node1.XCOORD)**2 + (node2.YCOORD - node1.YCOORD)**2 )

    mat = []
    for i, start in enumerate(data):
        row = []
        for j, dest in enumerate(data):
            if i == j:
                row.append(INF)
                continue
            row.append(_euclidean_distance(start, dest))
        mat.append(row)
    return mat

node_data = get_node_data('data_small.txt')
# node_data = node_data[:5] # Slice data

# SETS
#==================================================================================================

N = range(len(node_data))
V = range(FLEET_SIZE)

# PARAMETERS
#==================================================================================================

c  = VEHICLE_CAPACITY                       # Vehicle capacity per vehicle v
d  = build_distance_mat(node_data)          # Distance between node i and j
s  = VEHICLE_PACE                           # Pace of vehicle v
q  = [n.DEMAND for n in node_data]          # Demand at node i
bm = VEHICLE_RANGE                          # Maximum travel time of vehicle v
bc = VEHICLE_CHARGE_RATE                    # travel time per charge time
bd = VEHICLE_DISCHARGE_RATE                 # battery time per travel time
bs = [n.CHARGING for n in node_data]        # Charger station at node i
ts = [n.SERVICETIME for n in node_data]     # Minimal service time at node i
tr = [n.READYTIME for n in node_data]       # Ready time at node i
td = [n.DUETIME for n in node_data]         # Due time at node i
K  = FLEET_SIZE                             # number of vehicles to be used


# MODEL DEFINITION
#==================================================================================================

# MODEL
model = Model('Vehicle Routing Problem')

# DESISION VARIABLES
x      = model.addVars(N, N, V, vtype=GRB.BINARY, name='x')                          # If 1, indicates if vehicle v travels from node i to j
z      = model.addVars(N, V, vtype=GRB.BINARY, name='z')                             # If 1, node is visited by vehicle v
u      = model.addVars(N, V, lb=0, ub=len(N)-1, vtype=GRB.CONTINUOUS, name='u')      # order of node i in the tour
k      = model.addVar(lb=0, ub=len(V), vtype=GRB.INTEGER, name='k')                  # outgoing vehicles
beta_c = model.addVars(N, V, lb=0, ub=MAX_BATTERY, vtype=GRB.CONTINUOUS, name='β^c') # Amount charged at node i
beta_a = model.addVars(N, V, lb=0, ub=bm, vtype=GRB.CONTINUOUS, name='β^a')          # Battery level at arrival of vehicle v at node i
beta_d = model.addVars(N, V, lb=0, ub=bm, vtype=GRB.CONTINUOUS, name='β^d')          # Battery level at departure of vehicle v at node i
tau_a  = model.addVars(N, V, lb=0, ub=MAX_TIME, vtype=GRB.CONTINUOUS, name='τ^a')    # Time of arrival at node i        
tau_d  = model.addVars(N, V, lb=0, ub=MAX_TIME, vtype=GRB.CONTINUOUS, name='τ^d')    # Time of depature at node i

# OBJECTIVE
obj = quicksum(d[i][j] * x[i, j, v] for i in N for j in N for v in V)
model.setObjective(obj, GRB.MINIMIZE)

# CONSTRAINTS
constraints = {

    # ROUTE
    'visit_nodes_once':
    (quicksum(z[i, v] for v in V) == 1 for i in N if i != 0),

    'outgoing_vehicles':
    quicksum(z[0, v] for v in V) == k,

    'capacity':
    (quicksum(q[i] * z[i, v] for i in N) <= c for v in V),

    'route_incoming':
    (quicksum(x[j, i, v] for j in N) == z[i, v]
     for i in N for v in V),

    'route_outgoing':
    (quicksum(x[i, j, v] for j in N) == z[i, v]
     for i in N for v in V),

    # Subtour elimination: MTZ (Miller–Tucker–Zemlin)
    'subtour_elimination(1)':
    (u[0, v] == 1 for v in V),

    'subtour_elimination(2)':
    (u[i, v] >= 2 for i in N if i != 0 for v in V),

    'subtour_elimination(3)':
    (u[i, v] <= len(N) for i in N if i != 0 for v in V),

    'subtour_elimination(4)':
    (u[i, v] - u[j, v] + len(N) * x[i, j, v] <= len(N) - 1
     for i in N for j in N if j != 0 for v in V),

    # BATTERY:
    'initial_charge': # set initial charge to max battery cap
    (beta_a[0, v] == bm for v in V),

    'battery_dynamics_traveling': # departure charge - discharge == arrival charge
    (beta_d[j, v] - d[i][j] * s * bd + MAX_BATTERY * (1 - x[j, i, v]) >= beta_a[i, v]
     for i in N[1:] for j in N if i != j for v in V),

    'battery_dynamics_charging': # arrival charge + charge == departure charge
    (beta_a[i, v] + beta_c[i, v] == beta_d[i, v]
     for i in N[1:] for v in V),

    'charging': # charged -> 0 if no charger
    (beta_c[i, v] <= MAX_BATTERY * bs[i]
     for i in N for v in V),

    'final_charge': # departure charge last node - discharge ( +inf if not last node) >= 0
    (beta_d[i, v] - d[i][0] * s * bd + MAX_BATTERY * (1 - x[i, 0, v]) >= 0
     for i in N[1:] for v in V),

    # TIME:
    'start_time': # sets start time to ready time depot
    (tau_a[0, v] == tr[0] for v in V),

    'arrival_time_dynamics': # departure time outgoing node + travel time ( -inf if not prev arc) <= arrival time
    (tau_d[j, v] + d[i][j] * s - MAX_TIME * (1 - x[j, i, v]) <= tau_a[i, v]
     for i in N[1:] for j in N if i != j for v in V),

    'departure_time_dynamics_charging': # departure time >= arrival time + charging time
    (tau_d[i, v] >= tau_a[i, v] + bc * beta_c[i, v]
     for i in N for v in V),

    'departure_time_dynamics_service': # departure time >= arrival time + service time
    (tau_d[i, v] >= tau_a[i, v] + ts[i]
     for i in N for v in V),

    'ready_time_service': # service window start <= arrival time
    (tr[i] <= tau_a[i, v] for i in N if i != 0 for v in V),

    'due_time_service': # service window end >= arrival time
    (td[i] >= tau_a[i, v] for i in N if i != 0 for v in V),

    'finish_time': # departure time + travel time ( -inf if not prev arc) <= due time depot
    (tau_d[i, v] + d[i][0] * s - MAX_TIME * (1 - x[i, 0, v]) <= td[0]
     for i in N[1:] for v in V),
}

for name, con in constraints.items():
    if isinstance(con, Generator): model.addConstrs(con, name=name) 
    else: model.addConstr(con, name=name)


# SOLVE
#==================================================================================================

model.update()
# model.write('TSPmodel.lp')
setattr(model.Params, 'timeLimit', 30)
model.optimize()
# model.write('TSPmodel.sol')

try: sol = model.ObjVal
except: sol = None

if __name__ == "__main__":
    print('\n')
    if not sol:
        print('Failed')
        sys.exit(1)
    print('Success, obj:', sol)




