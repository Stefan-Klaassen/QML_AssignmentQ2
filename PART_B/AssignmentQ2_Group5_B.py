# -*- coding: utf-8 -*-

"""
Title: Vehicle routing problem
Course: ME44206 Quantitative Methods for Logistics
Authors:
    Stefan Klaassen - 6076947
    Peter Nederveen - 
    Britt van de Geer - 
    Bart Verzijl - 
Last updated: 2025-12-02
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
MAX_TIME = 10000
MAX_BATTERY = 2000
VEHICLES = 4
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
                row.append(float('inf'))
                continue
            row.append(_euclidean_distance(start, dest))
        mat.append(row)
    return mat

node_data = get_node_data('data_small.txt')

# SETS
#==================================================================================================

N = range(len(node_data))
V = range(VEHICLES)

# PARAMETERS
#==================================================================================================

c  = VEHICLE_CAPACITY                       # Vehicle capacity per vehicle v
d  = build_distance_mat(node_data)          # Distance between node i and j
s  = VEHICLE_PACE                           # Pace of vehicle v
q  = [n.DEMAND for n in node_data]          # Demand at node i
bm = VEHICLE_RANGE                          # Maximum battery capacity of vehicle v
bc = VEHICLE_CHARGE_RATE                    # Energy charging rate
bd = VEHICLE_DISCHARGE_RATE                 # Energy discharging rate
bs = [n.CHARGING for n in node_data]        # Charger station at node i
ts = [n.SERVICETIME for n in node_data]     # Minimal service time at node i
tr = [n.READYTIME for n in node_data]       # Ready time at node i
td = [n.DUETIME for n in node_data]         # Due time at node i


# MODEL DEFINITION
#==================================================================================================

# MODEL
model = Model('Vehicle Routing Problem')


# DESISION VARIABLES
tau_a_i = model.addVars(N, lb=0, vtype = GRB.CONTINUOUS, name='τ^a')        # Time of arrival at node i
tau_c_i = model.addVars(N, lb=0, vtype = GRB.CONTINUOUS, name='τ^c')        # Time at node i with charging
tau_w_i = model.addVars(N, lb=0, vtype = GRB.CONTINUOUS, name='τ^w')        # Time at node i without charging
z_iv    = model.addVars(N, V, vtype=GRB.BINARY, name='z')                   # If 1, node is visited by vehicle v 
beta_iv = model.addVars(N, V, lb=0, name='β')                               # Battery level of vehicle v at node i
x_ijv   = model.addVars(N, N, V, vtype=GRB.BINARY, name='x')                # If 1, indicates if vehicle v travels from node i to j


# OBJECTIVE
obj = quicksum(d[i][j] * x_ijv[i, j, v] for i in N for j in N if i != j for v in V)
model.setObjective(obj, GRB.MINIMIZE)


# CONSTRAINTS
constraints = {
    'visit_constraint':
    (quicksum(z_iv[i, v] for v in V) == 1 for i in N[1:]),

    'depot_constraint':
    quicksum(z_iv[0, v] for v in V) == VEHICLES,

    'vehicle_capacity_constraint': 
    (quicksum(q[i] * z_iv[i, v] for i in N) <= c for v in V),

    'departure_constraint':  
    (quicksum(x_ijv[i, j, v] for j in N) == quicksum(x_ijv[j, i, v]
        for j in N) for i in N for v in V),

    'departure_constraint(2)':  
    (quicksum(x_ijv[i, j, v] for j in N) == z_iv[i, v] for i in N for v in V),

    'no self loops':
    (x_ijv[i, i, v] == 0 for i in N for v in V),

    'time_constraint(1)':
    (tau_a_i[i] + tau_c_i[i] + tau_w_i[i] + (d[i][j] * s) - MAX_TIME * (1 - x_ijv[i, j, v]) <= tau_a_i[j]
        for i in N for j in N[1:] if i != j for v in V),
    
    'time_constraint(2)':
    (tau_a_i[i] + tau_c_i[i] + tau_w_i[i] + (d[i][j] * s) + MAX_TIME * (1 - x_ijv[i, j, v]) >= tau_a_i[j]
        for i in N for j in N[1:] if i != j for v in V),

    'end_time_constraint(1)':
    (tau_a_i[i] + tau_c_i[i] + tau_w_i[i] + (d[i][0] * s) - MAX_TIME * (1 - x_ijv[i, 0, v]) <= td[0]
        for i in N[1:] for v in V),
    
    'end_time_constraint(2)':
    (tau_a_i[i] + tau_c_i[i] + tau_w_i[i] + (d[i][0] * s) + MAX_TIME * (1 - x_ijv[i, 0, v]) >= td[0]
        for i in N[1:] for v in V),

    'time_window_constraint_start':
    (tr[i] <= tau_a_i[i] for i in N),

    'time_window_constraint_end':
    (tau_a_i[i] <= td[i] for i in N),

    'service_time_constraint':
    (tau_c_i[i] + tau_w_i[i] >= ts[i] for i in N),

    'battery_capacity_constraint_bottom':
    ((beta_iv[i, v] - (d[i][j] * s * bd) + tau_c_i[i] * bc * bs[i]) + (1 - x_ijv[i, j, v]) * MAX_BATTERY >= 0 
        for i in N for j in N if i != j for v in V),

    'battery_capacity_constraint_top':
    ((beta_iv[i, v] + tau_c_i[i] * bc * bs[i]) - (1 - x_ijv[i, j, v]) * MAX_BATTERY <= bm 
        for i in N for j in N for v in V),
    
    'battery_update_constraint':
    (beta_iv[i, v] - (d[i][j] * s * bd) + tau_c_i[i] * bc * bs[i] + (1 - x_ijv[i, j, v]) * MAX_BATTERY >= beta_iv[j, v]
        for i in N for j in N[1:] if i != j for v in V),

    'battery_update_constraint(2)':
    (beta_iv[i, v] - (d[i][j] * s * bd) + tau_c_i[i] * bc * bs[i] - (1 - x_ijv[i, j, v]) * MAX_BATTERY <= beta_iv[j, v]
        for i in N for j in N[1:] if i != j for v in V),

    'initial_battery_constraint':
    (beta_iv[0, v] == bm for v in V),

    'charging_time_constraint':
    (tau_c_i[i] == tau_c_i[i] * bs[i] for i in N),
}

for name, con in constraints.items():
    if isinstance(con, Generator): model.addConstrs(con, name=name) 
    else: model.addConstr(con, name=name)


# SOLVE
#==================================================================================================

# model.computeIIS()
# print('IIS written to model.iis')
# # optional: list IIS constraints
# print([c.ConstrName for c in model.getConstrs() if c.IISConstr == 1])

model.update()
model.write('TSPmodel.lp')
setattr(model.Params, 'timeLimit', 3600)
model.optimize()
model.write('TSPmodel.sol')

solved = model.Status == GRB.OPTIMAL

if __name__ == "__main__":
    print('\n')
    print('Success, obj:', model.ObjVal) if solved else print('Failed')



