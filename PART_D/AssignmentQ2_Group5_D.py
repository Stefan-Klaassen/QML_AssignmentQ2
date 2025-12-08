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

# IMPORTS
#==================================================================================================

from pathlib import Path
from dataclasses import dataclass
from collections.abc import Generator
import math
import sys
from gurobipy import Model, GRB, quicksum
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from typing import Any




# MODEL DATA
#==================================================================================================

# CONSTANTS
INF = 1e5
MAX_TIME = 2000
MAX_BATTERY = 500
VEHICLE_PACE = 2

# CLASSES
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

@dataclass
class ChargePeriod:
    PER_ID: int
    STARTTIME: int
    ENDTIME: int
    COST: int

@dataclass
class Case:
    fleet_size: int
    capacity: int
    battery_range: int
    charge_rate: float
    discharge_rate: float
    charge_periods: list[ChargePeriod]

# FUNCTIONS
def get_data(filename: str, Cls: Any) -> list[Any]:
    try:
        file = Path(__file__).parent / filename
        assert file.is_file(), f"Kan '{filename}' niet vinden in '{file.parent}'."
    except Exception as e: 
        print(e)
        sys.exit(1)
    data = []
    with file.open('r') as f:
        for line in f:
            row = [int(v) for v in line.strip().split()]
            instance = Cls(*row)
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

# GET EXTERNAL DATA
node_data: list[Node] = get_data('data_small.txt', Node)
charge_periods_from_file = get_data('data_periodsCharge.txt', ChargePeriod)

# CASES
cases = {
    1: Case(4, 120, 110, 1.0, 1.0, [ChargePeriod(0, 0, MAX_TIME, 2)]),
    2: Case(4, 120, 110, 1.0, 1.0, charge_periods_from_file),
    3: Case(4, 120, 110, 1.1, 0.7, charge_periods_from_file),
    4: Case(4, 120,  90, 1.1, 0.7, charge_periods_from_file),
    5: Case(4, 200, 140, 1.1, 0.7, charge_periods_from_file),
}

case = None
while not case:
    try:
        inp = input("Enter case number: ")
        case = cases[int(inp)]
    except:
        print("Enter a number from 1 - 5. (press ctrl+c to exit)")


# SETS
#==================================================================================================

N = range(len(node_data))
V = range(case.fleet_size)
G = range(len(case.charge_periods))


# PARAMETERS
#==================================================================================================

c  = case.capacity                                      # Vehicle capacity per vehicle v
d = build_distance_mat(node_data)                       # Distance between node i and j
s  = VEHICLE_PACE                                       # Pace of vehicle v
q  = [n.DEMAND for n in node_data]                      # Demand at node i
bm = case.battery_range                                 # Maximum travel time of vehicle v
bc = case.charge_rate                                   # travel time per charge time
bd = case.discharge_rate                                # battery time per travel time
bs = [n.CHARGING for n in node_data]                    # Charger station at node i
ts = [n.SERVICETIME for n in node_data]                 # Minimal service time at node i
tr = [n.READYTIME for n in node_data]                   # Ready time at node i
td = [n.DUETIME for n in node_data]                     # Due time at node i
wd = 1.0                                                # weight of distance in objective
wc = 0.5                                                # weight of charging costs in objective
p_g = [p.COST for p in case.charge_periods]              # Charging cost in period p
ta_g = [p.STARTTIME for p in case.charge_periods]         # Open time for charge period p
tb_g = [p.ENDTIME for p in case.charge_periods]           # Close time for charge period p
VEHICLES = case.fleet_size                                 # Number of vehicles available

# MODEL DEFINITION
#==================================================================================================

# MODEL
model = Model('Vehicle Routing Problem')

# DESISION VARIABLES
tau_a_i = model.addVars(N, lb=0, vtype = GRB.CONTINUOUS, name='τ^a')        # Time of arrival at node i
tau_d_i = model.addVars(N, lb=0, vtype = GRB.CONTINUOUS, name='τ^d')        # Time of departure at node i
tau_s_i = model.addVars(N, lb=0, vtype = GRB.CONTINUOUS, name='τ^s')        # Time of Service start at node i
tau_c_i = model.addVars(N, lb=0, vtype = GRB.CONTINUOUS, name='τ^c')        # Time at node i with charging
z_iv    = model.addVars(N, V, vtype=GRB.BINARY, name='z')                   # If 1, node is visited by vehicle v 
beta_iv = model.addVars(N, V, lb=0, name='β')                               # Battery level of vehicle v at node i
x_ijv   = model.addVars(N, N, V, vtype=GRB.BINARY, name='x')                # If 1, indicates if vehicle v travels from node i to j

k_gi  = model.addVars(G, N, vtype = GRB.BINARY, name='k')        # Load of vehicle at node i
tau_gs_gi = model.addVars(G, N, lb=0, vtype = GRB.CONTINUOUS, name='τ^g_s')        # Start charging time of period g at node i
tau_ge_gi = model.addVars(G, N, lb=0, vtype = GRB.CONTINUOUS, name='τ^g_e')        # End charging time of period g at node i



# OBJECTIVE
obj = wd*quicksum(d[i][j] * x_ijv[i, j, v] for i in N for j in N for v in V) +     wc*quicksum(k_gi[g, i] * p_g[g] * (tau_ge_gi[g, i] - tau_gs_gi[g, i])  for g in G for i in N)
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

    'no_self_loops':
    (x_ijv[i, i, v] == 0 for i in N for v in V),

    'time_constraint(1)':
    (tau_d_i[i] + (d[i][j] * s) - MAX_TIME * (1 - x_ijv[i, j, v]) <= tau_a_i[j]
        for i in N for j in N[1:] if i != j for v in V),

    'end_time_constraint(1)':
    (tau_d_i[i] + (d[i][0] * s) - MAX_TIME * (1 - x_ijv[i, 0, v]) <= td[0]
        for i in N[1:] for v in V),
    

    'time_window_constraint_start':
    (tr[i] <= tau_s_i[i] for i in N),

    'time_window_constraint_end':
    (tau_s_i[i] <= td[i] for i in N),

    'time_service_constraint_start':
    (tau_s_i[i] >= tau_a_i[i] for i in N),
     
    'time_service_constraint_end':
    (tau_s_i[i] <= tau_d_i[i] for i in N),

    'service_time_constraint':
    (tau_d_i[i] - tau_s_i[i] >= ts[i] for i in N),

    'service_time_constraint_battery':
    (tau_d_i[i] - tau_s_i[i] >= tau_c_i[i] for i in N),


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
    (tau_c_i[i] * (1 - bs[i]) == 0 for i in N),



    'charging_time_distribution_constraint':
    (tau_c_i[i] <= quicksum((tau_ge_gi[g, i] - tau_gs_gi[g, i]) * k_gi[g, i] for g in G) for i in N),   

    'period_selection_constraint_start':
    (tau_gs_gi[g, i] >= ta_g[g]         - MAX_TIME * (1 - k_gi[g, i])   for g in G for i in N),

    'period_selection_constraint_end':
    (tau_ge_gi[g, i] <= tb_g[g]         + MAX_TIME * (1 - k_gi[g, i])   for g in G for i in N),

    'period_time_constraint_start':
    (tau_gs_gi[g, i] >= tau_a_i[ i]     - MAX_TIME * (1 - k_gi[g, i])   for g in G for i in N),

    'period_time_constraint_end':
    (tau_ge_gi[g, i] <= tau_d_i[ i]     + MAX_TIME * (1 - k_gi[g, i])   for g in G for i in N),

    'positive_charging_time_constraint':
    (tau_ge_gi[g, i] - tau_gs_gi[g, i] >= 0 for g in G for i in N),


}

for name, con in constraints.items():
    model.addConstrs(con, name=name) if isinstance(con, Generator) else model.addConstr(con, name=name)

# SOLVE
#==================================================================================================
# model.computeIIS()

# print('IIS written to model.iis')
# # optional: list IIS constraints
# print([c.ConstrName for c in model.getConstrs() if c.IISConstr == 1])



# SOLVE
#==================================================================================================

model.update()
# model.write('TSPmodel.lp')
setattr(model.Params, 'timeLimit', 3600)
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



