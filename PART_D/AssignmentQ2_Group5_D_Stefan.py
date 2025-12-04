# -*- coding: utf-8 -*-

"""
Title: Vehicle routing problem
Course: ME44206 Quantitative Methods for Logistics
Part: D
Authors:
    Stefan Klaassen - 6076947
    Peter Nederveen - 
    Britt van de Geer - 
    Bart Verzijl - 
Last updated: 2025-12-04
Version: 1.0

Usage:
    ../
     ├── AssignmentQ2_Group5_D.py
     ├── data_periodsCharge.txt
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
from typing import Any
from collections.abc import Generator
from gurobipy import Model, GRB, quicksum


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
charge_periods_from_file: list[ChargePeriod] = get_data('data_periodsCharge.txt', ChargePeriod)

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
        print("Enter an number from 1 - 5. (press ctrl+c to exit)")


# SETS
#==================================================================================================

N = range(len(node_data))
V = range(case.fleet_size)
P = range(len(case.charge_periods))


# PARAMETERS
#==================================================================================================

c  = case.capacity                              # Vehicle capacity per vehicle v
cd  = build_distance_mat(node_data)             # Distance between node i and j
s  = VEHICLE_PACE                               # Pace of vehicle v
q  = [n.DEMAND for n in node_data]              # Demand at node i
bm = case.battery_range                         # Maximum travel time of vehicle v
bc = case.charge_rate                           # travel time per charge time
bd = case.discharge_rate                        # battery time per travel time
bs = [n.CHARGING for n in node_data]            # Charger station at node i
ts = [n.SERVICETIME for n in node_data]         # Minimal service time at node i
tr = [n.READYTIME for n in node_data]           # Ready time at node i
td = [n.DUETIME for n in node_data]             # Due time at node i
wd = 1.0                                        # weight of distance in objective
wc = 0.5                                        # weight of charging costs in objective
cc = [p.COST for p in case.charge_periods]      # Charging cost in period p
to = [p.STARTTIME for p in case.charge_periods] # Open time for charge period p
tc = [p.ENDTIME for p in case.charge_periods]   # Close time for charge period p


# MODEL DEFINITION
#==================================================================================================

# MODEL
model = Model('Vehicle Routing Problem')

# DESISION VARIABLES
x      = model.addVars(N, N, V, vtype=GRB.BINARY, name='x')                             # If 1, indicates if vehicle v travels from node i to j
z      = model.addVars(N, V, vtype=GRB.BINARY, name='z')                                # If 1, node is visited by vehicle v
u      = model.addVars(N, V, lb=0, ub=len(N)-1, vtype=GRB.CONTINUOUS, name='u')         # order of node i in the tour
k      = model.addVar(lb=0, ub=len(V), vtype=GRB.INTEGER, name='k')                     # outgoing vehicles
tau_s  = model.addVars(N, P, V, lb=0, ub=MAX_TIME, vtype=GRB.CONTINUOUS, name='τ^c')    # charging start time at node i inperiod p for vehicle v
tau_e  = model.addVars(N, P, V, lb=0, ub=MAX_TIME, vtype=GRB.CONTINUOUS, name='τ^c')    # charging end time at node i inperiod p for vehicle v
beta_q = model.addVars(N, P, V, lb=0, ub=MAX_BATTERY, vtype=GRB.CONTINUOUS, name='β^c') # Amount charged at node i in period p
beta_c = model.addVars(N, P, V, vtype=GRB.BINARY, name='β^b')                           # if 1, vehicle v has charged in period p at node i
beta_a = model.addVars(N, V, lb=0, ub=bm, vtype=GRB.CONTINUOUS, name='β^a')             # Battery level at arrival of vehicle v at node i
beta_d = model.addVars(N, V, lb=0, ub=bm, vtype=GRB.CONTINUOUS, name='β^d')             # Battery level at departure of vehicle v at node i
tau_a  = model.addVars(N, V, lb=0, ub=MAX_TIME, vtype=GRB.CONTINUOUS, name='τ^a')       # Time of arrival at node i        
tau_d  = model.addVars(N, V, lb=0, ub=MAX_TIME, vtype=GRB.CONTINUOUS, name='τ^d')       # Time of depature at node i

# OBJECTIVE
obj = (
    wd * quicksum(cd[i][j] * x[i, j, v] for i in N for j in N for v in V) + 
    wc * quicksum(cc[p] * beta_q[i, p, v] for i in N[1:] for p in P for v in V)
)
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
    (beta_d[0, v] == bm for v in V),

    'battery_dynamics_traveling': # departure charge - discharge == arrival charge
    (beta_d[j, v] - cd[i][j] * s * bd + MAX_BATTERY * (1 - x[j, i, v]) >= beta_a[i, v]
     for i in N[1:] for j in N if i != j for v in V),

    'battery_dynamics_charging': # arrival charge + total charge == departure charge
    (beta_a[i, v] + quicksum(beta_q[i, p, v] for p in P) == beta_d[i, v]
     for i in N[1:] for v in V),

    'has_charger': # charged -> 0 if no charger
    (beta_q[i, p, v] <= MAX_BATTERY * bs[i]
     for i in N for p in P for v in V),

    'amount_charged': # == charge time * charge rate * has charged
    (beta_q[i, p, v] == (tau_e[i, p, v] - tau_s[i, p, v]) * bc * beta_c[i, p, v]
     for i in N[1:] for p in P for v in V),

    'final_charge': # departure charge last node - discharge ( +inf if not last node) >= 0
    (beta_d[i, v] - cd[i][0] * s * bd + MAX_BATTERY * (1 - x[i, 0, v]) >= 0
     for i in N[1:] for v in V),

    # TIME:
    'start_time': # sets start time to ready time depot
    (tau_a[0, v] == tr[0] for v in V),

    'arrival_time_dynamics': # departure time outgoing node + travel time ( -inf if not prev arc) <= arrival time
    (tau_d[j, v] + cd[i][j] * s - MAX_TIME * (1 - x[j, i, v]) <= tau_a[i, v]
     for i in N[1:] for j in N if i != j for v in V),

    'departure_time_dynamics_service': # departure time >= arrival time + service time
    (tau_d[i, v] >= tau_a[i, v] + ts[i]
     for i in N for v in V),

    'departure_time_dynamics_charging': # departure time >= end charging time ( -inf if no charging)
    (tau_d[i, v] >= tau_e[i, p, v] - MAX_TIME * (1 - beta_c[i, p, v])
     for i in N for p in P for v in V),

    # 'non_neg_chargetime': # start time charging is before or eq to end time charging
    # (tau_s[i, p, v] <= tau_e[i, p, v]
    #  for i in N for p in P for v in V),

    'charge_period_open': # start time charging after period start ( -inf if no charging)
    (tau_s[i, p, v] >= to[p] - MAX_TIME * (1 - beta_c[i, p, v])
     for i in N for p in P for v in V),

    'charge_period_close': # end time charging before period end ( +inf if no charging)
    (tau_e[i, p, v] <= tc[p] + MAX_TIME * (1 - beta_c[i, p, v])
     for i in N for p in P for v in V),

    'charge_after_arrival': # start time charging after arrival ( -inf if no charging)
    (tau_s[i, p, v] >= tau_a[i, v] - MAX_TIME * (1 - beta_c[i, p, v])
     for i in N[1:] for p in P for v in V),

    'ready_time_service': # service window start <= arrival time
    (tr[i] <= tau_a[i, v] for i in N if i != 0 for v in V),

    'due_time_service': # service window end >= arrival time
    (td[i] >= tau_a[i, v] for i in N if i != 0 for v in V),

    'finish_time': # departure time + travel time ( -inf if not prev arc) <= due time depot
    (tau_d[i, v] + cd[i][0] * s - MAX_TIME * (1 - x[i, 0, v]) <= td[0]
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