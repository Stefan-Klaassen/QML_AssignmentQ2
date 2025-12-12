# -*- coding: utf-8 -*-

"""
Title: Vehicle routing problem
Course: ME44206 Quantitative Methods for Logistics
Part: G  (Heterogeneous fleet implementation 2)
Authors:
    Stefan Klaassen - 6076947
    Peter Nederveen - 5607175
    Britt van de Geer - 6088023
    Bart Verzijl - 6343772

Usage:
    ./ 
     ├── AssignmentQ2_Group5_G.py
     ├── data_periodsCharge.txt
     └── data_large.txt

Dependencies:
    Python 3.13.7+
    gurobipy
"""

# IMPORTS
# ================================================================================================

import math, sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any
from collections.abc import Generator
from gurobipy import Model, GRB, quicksum


# MODEL DATA
# ================================================================================================

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

    def __repr__(self) -> str:
        return f"ChargePeriod(cost={self.COST})"

@dataclass
class Case:
    fleet_size: int
    capacity: int
    battery_range: int
    charge_rate: float
    discharge_rate: float
    charge_periods: list[ChargePeriod]

    def __str__(self) -> str:
        return '\n'.join(f"{k}: {v}" for k, v in self.__dict__.items())


# FUNCTIONS
# ================================================================================================

def get_data(filename: str, Cls: Any) -> list[Any]:
    """Read txt data file and return list of dataclass instances."""
    try:
        file = Path(__file__).parent / filename
        assert file.is_file(), f"Kan '{filename}' niet vinden in '{file.parent}'."
    except Exception as e:
        print(e)
        sys.exit(1)

    print(f"Using file: {file}\n")

    data: list[Any] = []
    with file.open('r') as f:
        for line in f:
            row = [int(v) for v in line.strip().split()]
            instance = Cls(*row)
            data.append(instance)
    return data


def build_distance_mat(data: list[Node]) -> list[list[float]]:
    """Return matrix with Euclidean distances between all node pairs."""

    def _euclidean_distance(node1: Node, node2: Node) -> float:
        return math.sqrt(
            (node2.XCOORD - node1.XCOORD) ** 2 +
            (node2.YCOORD - node1.YCOORD) ** 2
        )

    mat: list[list[float]] = []
    for i, start in enumerate(data):
        row: list[float] = []
        for j, dest in enumerate(data):
            if i == j:
                row.append(INF)  # disallow self-loops
                continue
            row.append(_euclidean_distance(start, dest))
        mat.append(row)
    return mat


# GET EXTERNAL DATA
# ================================================================================================

print('\n')
node_data: list[Node] = get_data('data_large.txt', Node)
charge_periods_from_file = get_data('data_periodsCharge.txt', ChargePeriod)

# =============================== ASSIGNMENT G SETTINGS ==========================================

NUM_EV = 3
NUM_DV = 3

case = Case(
    fleet_size    = NUM_EV + NUM_DV,
    capacity      = 300,      # capacity of each vehicle (EV and DV)
    battery_range = 120,       # EV battery range (time units)
    charge_rate   = 1.1,      # EV charge rate
    discharge_rate= 0.7,      # EV discharge rate
    charge_periods= charge_periods_from_file,
)

print("Case G settings:\n", case, "\n")

# Cost parameters (per vehicle type)
FIXED_COST_DV = 100.0
FIXED_COST_EV = 120.0
VAR_COST_DV   = 2.00
VAR_COST_EV   = 1.25


# SETS
# ================================================================================================

N   = range(len(node_data))             # all nodes (0 = depot)
V   = range(case.fleet_size)            # all vehicles
V_EV = range(NUM_EV)                    # 0,1,2  -> EVs
V_DV = range(NUM_EV, NUM_EV + NUM_DV)   # 3,4,5  -> DVs
P   = range(len(case.charge_periods))   # charging periods


# PARAMETERS
# ================================================================================================

c  = case.capacity                                   # vehicle capacity
d  = build_distance_mat(node_data)                   # distance matrix
s  = VEHICLE_PACE                                    # travel pace
q  = [n.DEMAND for n in node_data]                   # demand at node i
bm = case.battery_range                              # max battery for EVs
bc = case.charge_rate                                # charge rate
bd = case.discharge_rate                             # discharge rate
bs = [n.CHARGING for n in node_data]                 # charger availability at node i (0/1)
ts = [n.SERVICETIME for n in node_data]              # service time
tr = [n.READYTIME for n in node_data]                # ready time
td = [n.DUETIME for n in node_data]                  # due time
cc = [p.COST for p in case.charge_periods]           # charging cost per period
to = [p.STARTTIME for p in case.charge_periods]      # period open time
tc = [p.ENDTIME for p in case.charge_periods]        # period close time

# Per-vehicle fixed and variable distance costs
fixed_cost = {v: (FIXED_COST_EV if v in V_EV else FIXED_COST_DV) for v in V}
var_cost   = {v: (VAR_COST_EV   if v in V_EV else VAR_COST_DV)   for v in V}


# MODEL DEFINITION
# ================================================================================================

model = Model('Vehicle Routing Problem - Assignment G')

# DECISION VARIABLES
x      = model.addVars(N, N, V, vtype=GRB.BINARY,   name='x')     # travel from i to j by v
z      = model.addVars(N, V, vtype=GRB.BINARY,      name='z')     # node i visited by v
k      = model.addVar(lb=0, ub=len(V), vtype=GRB.INTEGER, name='k')  # number of used vehicles

# Battery / charging variables (for all v, maar constraints alleen voor EV's)
beta_q = model.addVars(N, P, V, lb=0, ub=MAX_BATTERY,
                       vtype=GRB.CONTINUOUS, name='β^q')   # charged amount
beta_c = model.addVars(N, P, V, vtype=GRB.BINARY,          name='β^c')   # charging decision
beta_a = model.addVars(N, V, lb=0, ub=bm,
                       vtype=GRB.CONTINUOUS, name='β^a')   # battery on arrival
beta_d = model.addVars(N, V, lb=0, ub=bm,
                       vtype=GRB.CONTINUOUS, name='β^d')   # battery on departure

tau_cs = model.addVars(N, P, V, lb=0, ub=MAX_TIME,
                       vtype=GRB.CONTINUOUS, name='τ^cs')  # charge start time
tau_ce = model.addVars(N, P, V, lb=0, ub=MAX_TIME,
                       vtype=GRB.CONTINUOUS, name='τ^ce')  # charge end time
tau_ss = model.addVars(N, V, lb=0, ub=MAX_TIME,
                       vtype=GRB.CONTINUOUS, name='τ^ss')  # service start
tau_a  = model.addVars(N, V, lb=0, ub=MAX_TIME,
                       vtype=GRB.CONTINUOUS, name='τ^a')   # arrival time
tau_d  = model.addVars(N, V, lb=0, ub=MAX_TIME,
                       vtype=GRB.CONTINUOUS, name='τ^d')   # departure time


# OBJECTIVE FUNCTION (cost-based)
# ================================================================================================
# Minimize: fixed cost + variable distance cost + charging cost for EVs

obj = (
    # Fixed cost per used vehicle (if it leaves the depot)
    quicksum(fixed_cost[v] * z[0, v] for v in V)
    +
    # Distance-based cost per vehicle
    quicksum(var_cost[v] * d[i][j] * x[i, j, v]
             for i in N for j in N for v in V)
    +
    # Time-dependent charging cost (EVs only, i != depot)
    quicksum(cc[p] * beta_q[i, p, v]
             for i in N if i != 0 for p in P for v in V_EV)
)

model.setObjective(obj, GRB.MINIMIZE)


# CONSTRAINTS
# ================================================================================================

constraints = {

    # ---------------- ROUTING CONSTRAINTS ----------------
    'visit_nodes_once':
        (quicksum(z[i, v] for v in V) == 1
         for i in N if i != 0),

    'outgoing_vehicles':
        quicksum(z[0, v] for v in V) == k,

    'capacity':
        (quicksum(q[i] * z[i, v] for i in N) <= c
         for v in V),

    'route_incoming':
        (quicksum(x[j, i, v] for j in N) == z[i, v]
         for i in N for v in V),

    'route_outgoing':
        (quicksum(x[i, j, v] for j in N) == z[i, v]
         for i in N for v in V),

    # ---------------- BATTERY CONSTRAINTS (EV ONLY) ----------------
    'initial_charge':
        (beta_a[0, v] == bm
         for v in V_EV),

    'battery_dynamics_traveling': (
        # upper bound
        (beta_d[i, v] - d[i][j] * s * bd
         <= beta_a[j, v] + MAX_BATTERY * (1 - x[i, j, v])
         for i in N for j in N if j != 0 and i != j for v in V_EV),

        # lower bound
        (beta_d[i, v] - d[i][j] * s * bd
         >= beta_a[j, v] - MAX_BATTERY * (1 - x[i, j, v])
         for i in N for j in N if j != 0 and i != j for v in V_EV),
    ),

    'battery_dynamics_charging':
        (beta_a[i, v] + quicksum(beta_q[i, p, v] for p in P) == beta_d[i, v]
         for i in N for v in V_EV),

    # optional “no charging if no charger / no charging decision”
    'amount_charged':
        (beta_q[i, p, v] ==
         (tau_ce[i, p, v] - tau_cs[i, p, v]) * bc * bs[i] * beta_c[i, p, v]
         for i in N for p in P for v in V_EV),

    'final_charge':
        (beta_d[i, v] - d[i][0] * s * bd
         >= -MAX_BATTERY * (1 - x[i, 0, v])
         for i in N if i != 0 for v in V_EV),

    # ---------------- TIME & SERVICE CONSTRAINTS ----------------
    'arrival_time_dynamics': (
        # upper
        (tau_d[i, v] + d[i][j] * s
         <= tau_a[j, v] + MAX_TIME * (1 - x[i, j, v])
         for i in N for j in N if j != 0 and i != j for v in V),

        # lower
        (tau_d[i, v] + d[i][j] * s
         >= tau_a[j, v] - MAX_TIME * (1 - x[i, j, v])
         for i in N for j in N if j != 0 and i != j for v in V),
    ),

    'departure_time_dynamics_service':
        (tau_d[i, v] >= tau_ss[i, v] + ts[i] - MAX_TIME * (1 - z[i, v])
         for i in N for v in V),

    # Charging-related time constraints (EV only)
    'departure_time_dynamics_charging':
        (tau_d[i, v] >= tau_ce[i, p, v] - MAX_TIME * (1 - beta_c[i, p, v])
         for i in N for p in P for v in V_EV),

    'non_neg_chargetime':
        (tau_cs[i, p, v] <= tau_ce[i, p, v] + MAX_TIME * (1 - beta_c[i, p, v])
         for i in N for p in P for v in V_EV),

    'charge_period_open':
        (tau_cs[i, p, v] >= to[p] - MAX_TIME * (1 - beta_c[i, p, v])
         for i in N for p in P for v in V_EV),

    'charge_period_close':
        (tau_ce[i, p, v] <= tc[p] + MAX_TIME * (1 - beta_c[i, p, v])
         for i in N for p in P for v in V_EV),

    'charge_after_arrival':
        (tau_cs[i, p, v] >= tau_a[i, v] - MAX_TIME * (1 - beta_c[i, p, v])
         for i in N for p in P for v in V_EV),

    'start_service':
        (tau_ss[i, v] >= tau_a[i, v] - MAX_TIME * (1 - z[i, v])
         for i in N for v in V),

    'ready_time_service':
        (tr[i] <= tau_ss[i, v] + MAX_TIME * (1 - z[i, v])
         for i in N for v in V),

    'due_time_service':
        (td[i] >= tau_ss[i, v] - MAX_TIME * (1 - z[i, v])
         for i in N if i != 0 for v in V),

    'finish_time':
        (tau_d[i, v] + d[i][0] * s
         <= td[0] + MAX_TIME * (1 - x[i, 0, v])
         for i in N if i != 0 for v in V),
}

# Add all constraints to the model
for name, con in constraints.items():
    if isinstance(con, tuple):
        for c in con:
            model.addConstrs(c, name=name)
    elif isinstance(con, Generator):
        model.addConstrs(con, name=name)
    else:
        model.addConstr(con, name=name)   # type: ignore


# SOLVE
# ================================================================================================

sol = None

def solve():
    try:
        global sol
        model.update()
        model.Params.timeLimit = 3600
        model.optimize()

        try:
            sol = model.ObjVal
        except Exception:
            raise Exception("Solution not found")

    except Exception as e:
        print(f"Error: {e}")


if input("\nSolve (y/n)? ") == 'y':
    solve()
else:
    sys.exit()

if __name__ == "__main__":
    if sol is None:
        sys.exit(1)
    print('\nSuccess, obj:', sol)
