# -*- coding: utf-8 -*-

"""
Title: Vehicle routing problem
Course: ME44206 Quantitative Methods for Logistics
Authors:
    Stefan Klaassen - 6076947
    Peter Nederveen - 
    Britt van de Geer - 
    Bart Verzijl - 
Last updated: 2025-11-24
Version: 1.0

Usage:
    ../
     ├── AssignmentQ2_Group5_B.py
     └── data_small.txt

Dependencies:
    gurobipy

"""

# CONSTANTS
#==================================================================================================

MAX_FLOAT = 1e6
VEHICLES = 4
VEHICLE_VELOCITY = 2
VEHICLE_CAPACITY = 120
VEHICLE_RANGE = 110
VEHICLE_CHARGE_RATE = 1.0
VEHICLE_DISCHARGE_RATE = 1.0


# IMPORTS
#==================================================================================================

from pathlib import Path
from dataclasses import dataclass
import math
from gurobipy import *


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
bm = VEHICLE_CAPACITY                       # Maximimum battery capcity
bc = VEHICLE_CHARGE_RATE                    # Energy charging rate
bd = VEHICLE_DISCHARGE_RATE                 # Energy discharging rate [battery_unit/time_unit]
bs = [n.CHARGING for n in node_data]        # Charger station at node j [battery_unit]
c  = VEHICLE_CAPACITY                       # Vehicle capacity [demand_unit]
d  = build_distance_mat(node_data)          # Distance between node i and j [distance_unit]
s  = VEHICLE_VELOCITY                       # Velocity [time_unit/distance_unit]
q  = [n.DEMAND for n in node_data]          # Demand at node i [demand_unit]
ts = [n.SERVICETIME for n in node_data]     # Minimal service time at node j [time_unit]
tr = [n.READYTIME for n in node_data]       # Ready time [time_unit]
td = [n.DUETIME for n in node_data]         # Due time [time_unit]


# MODEL DEFINITION
#==================================================================================================

# MODEL
model = Model('Vehicle Routing Problem')

# DESISION VARIABLES
x = model.addVars(N, N, V, vtype=GRB.BINARY, name='x')
z = model.addVars(N, V, vtype=GRB.BINARY, name='z')

# OBJECTIVE
obj = quicksum(d[i][j] * x[i, j, v] for i in N for j in N for v in V)
model.setObjective(obj, GRB.MINIMIZE)

# CONSTRAINTS
con_visit = (quicksum(z[i, v] for v in V) == 1 for i in N)
model.addConstrs(con_visit, name="visit constraint")

#TODO: Constraints

# SOLVE
#==================================================================================================

model.update()
model.write('TSPmodel.lp')
setattr(model.Params, 'timeLimit', 3600)
model.optimize()
model.write('TSPmodel.sol')