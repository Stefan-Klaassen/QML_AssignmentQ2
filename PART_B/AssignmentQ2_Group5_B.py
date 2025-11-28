# -*- coding: utf-8 -*-

"""
Title: Vehicle routing problem
Course: ME44206 Quantitative Methods for Logistics
Authors:
    Stefan Klaassen - 6076947
    Peter Nederveen - 
    Britt van de Geer - 
    Bart Verzijl - 
Last updated: 2025-11-27
Version: 1.0

Usage:
    ../
     ├── AssignmentQ2_Group5_B.py
     └── data_small.txt

Dependencies:
    Python 3.13.7+
    gurobipy

"""

# CONSTANTS
#==================================================================================================

MAX_FLOAT = 1e6
VEHICLES = 4
VEHICLE_PACE = 2
VEHICLE_CAPACITY = 120
VEHICLE_RANGE = 110
VEHICLE_CHARGE_RATE = 1.0
VEHICLE_DISCHARGE_RATE = 1.0


# IMPORTS
#==================================================================================================

from pathlib import Path
from dataclasses import dataclass
from collections.abc import Generator
import math
from gurobipy import Model, GRB, quicksum


# MODEL DATA
#==================================================================================================

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

def get_node_data() -> list[Node]:
    file = Path(__file__).parent / 'data_small.txt'
    data = []
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
                row.append(MAX_FLOAT)
                continue
            row.append(_euclidean_distance(start, dest))
        mat.append(row)
    return mat

node_data = get_node_data()

# SETS
N = range(len(node_data))
V = range(VEHICLES)

# PARAMETERS
c  = [VEHICLE_CAPACITY for _ in V]          # Vehicle capacity per vehicle v
d  = build_distance_mat(node_data)          # Distance between node i and j
s  = [VEHICLE_PACE for _ in V]              # Velocity of vehicle v
q  = [n.DEMAND for n in node_data]          # Demand at node i
bm = [VEHICLE_RANGE for _ in V]             # Maximum battery capacity of vehicle v
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
x     = model.addVars(N, N, V, vtype=GRB.BINARY, name='x')                # If 1, indicates if vehicle v travels from node i to j
z     = model.addVars(N, V, vtype=GRB.BINARY, name='z')                   # If 1, node is visited by vehicle v 
tau_a = model.addVars(N, lb=0, vtype = GRB.CONTINUOUS, name='τ^a')        # Time of arrival at node i
tau_c = model.addVars(N, lb=0, vtype = GRB.CONTINUOUS, name='τ^c')        # Time at node i with charging
tau_l = model.addVars(N, lb=0, vtype = GRB.CONTINUOUS, name='τ^l')        # Time at node i without charging
beta  = model.addVars(N, V, lb=0, name='β')                               # Battery level of vehicle v at node i
l     = model.addVars(N, V, lb=0, name='l')     
u     = model.addVars(N, V, lb=0, name='u')  


# OBJECTIVE
obj = quicksum(d[i][j] * x[i, j, v] for i in N for j in N for v in V)
model.setObjective(obj, GRB.MINIMIZE)

# CONSTRAINTS
constraints = { #TODO: allemaal
    'visit_constraint':
    (quicksum(z_iv[i, v] for v in V) == 1 for i in N[1:]),

    'depot_constraint':
    quicksum(z_iv[0, v] for v in V) == VEHICLES,

    'vehicle_capacity_constraint': 
    (quicksum(q[i] * z_iv[i, v] for i in N) <= c[v] for v in V),

    'departure_constraint':  
    (quicksum(x_ijv[i, j, v] for j in N) == quicksum(x_ijv[j, i, v]
        for j in N) for i in N for v in V),

    'departure_constraint(2)':  
    (quicksum(x_ijv[j, i, v] for j in N) == z_iv[i, v] for i in N for v in V),

    'time_constraint':
    (tau_a_i[i] + tau_c_i[i] + tau_w_i[i] + (d[i][j] * s[v]) - MAX_FLOAT * (1 - x_ijv[i, j, v]) <= tau_a_i[j]
        for i in N for j in N[1:] for v in V),

    'time_window_constraint':
    (tr[i] <= tau_a_i[i] for i in N),

    'time_window_constraint(2)':
    (tau_a_i[i] <= td[i] for i in N),

    'service_time_constraint':
    (tau_c_i[i] + tau_w_i[i] >= ts[i] for i in N),

    'battery_capacity_constraint':
    (beta_iv[i, v] - x_ijv[i, j, v] * (d[i][j] * s[v] * bd) >= 0 
        for i in N for j in N for v in V),

    'battery_capacity_constraint(2)':
    (beta_iv[i, v] - x_ijv[i, j, v] * (d[i][j] * s[v] * bd) + tau_c_i[j] * bc >= bm[v]
        for i in N for j in N[1:] for v in V),

    'battery_update_constraint':
    (beta_iv[i, v] - x_ijv[i, j, v] * (d[i][j] * s[v] * bd) + tau_c_i[j] * bc == beta_iv[j, v]
        for i in N for j in N[1:] for v in V),

    'initial_battery_constraint':
    (beta_iv[0, v] == bm[v] for v in V)
}

for name, con in constraints.items():
    model.addConstrs(con, name=name) if isinstance(con, Generator) else model.addConstr(con, name=name)


# SOLVE
#==================================================================================================

model.update()
model.write('TSPmodel.lp')
setattr(model.Params, 'timeLimit', 3600)
model.optimize()
# model.write('TSPmodel.sol')

res = model.ObjVal if model.Status == GRB.OPTIMAL else "😢"
print('\n'*2, 'Result: ', res, sep='')