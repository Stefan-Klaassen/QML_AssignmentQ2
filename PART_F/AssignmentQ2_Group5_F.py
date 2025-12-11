# -*- coding: utf-8 -*-

"""
Title: Vehicle routing problem
Course: ME44206 Quantitative Methods for Logistics
Part: F
Authors:
    Stefan Klaassen - 6076947
    Peter Nederveen - 5607175
    Britt van de Geer - 6088023
    Bart Verzijl - 6343772
Last updated: 2025-12-10
Version: 1.0

Usage:
    ../
     ├── AssignmentQ2_Group5_F.py
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
from typing import Any, Literal
from collections.abc import Generator
from gurobipy import Model, GRB, quicksum


# MODEL DATA
#==================================================================================================

# CONSTANTS
INF = 100_000
MAX_TIME = 2000
MAX_BATTERY = 500
VEHICLE_PACE = 2
ELECTRIC_VEHICLES = 3
DIESEL_VEHICLES = 3

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
class Vehicle:
    type: Literal['EV'] | Literal['DV']
    capacity: int
    range: int | None
    charge_rate: float | None
    discharge_rate: float|  None
    fixed_cost: int
    variable_cost: float

    def __str__(self) -> str:
        return '\n'.join(f"{k}: {v}" for k, v in self.__dict__.items())
    
    def __repr__(self) -> str:
        return f"Vehicle({self.type})"

# FUNCTIONS
def get_data(filename: str, Cls: Any) -> list[Any]:
    try:
        file = Path(__file__).parent / filename
        assert file.is_file(), f"Kan '{filename}' niet vinden in '{file.parent}'."
    except Exception as e: 
        print(e)
        sys.exit(1)
    print(f"Using file: {file}\n")
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
print('\n')
node_data: list[Node] = get_data('data_small.txt', Node)
charge_periods_data: list[ChargePeriod] = get_data('data_periodsCharge.txt', ChargePeriod)

# VEHICLES
electric_vehicle = Vehicle('EV', 100, 90, 1.1, 0.7, 120, 1.25)
print(f"\n{electric_vehicle}")
diesel_vehicle = Vehicle('DV', 100, None, None, None, 100, 2.0)
print(f"\n{diesel_vehicle}")
fleet = [electric_vehicle] * ELECTRIC_VEHICLES + [diesel_vehicle] * DIESEL_VEHICLES
print(f"\nFleet: {fleet}\n")


# SETS
#==================================================================================================

N = range(len(node_data))
V = range(len(fleet))
V_EV = [v for v, ins in zip(V, fleet) if ins.type == 'EV']
V_DV = [v for v, ins in zip(V, fleet) if ins.type == 'DV']
P = range(len(charge_periods_data))


# PARAMETERS
#==================================================================================================

c  = [v.capacity for v in fleet]                        # Vehicle capacity per vehicle v
d = build_distance_mat(node_data)                       # Distance between node i and j
s  = VEHICLE_PACE                                       # Pace of vehicle v
q  = [n.DEMAND for n in node_data]                      # Demand at node i
bm = [v.range for v in fleet]                           # Maximum travel time of vehicle v
bc = [v.charge_rate for v in fleet]                     # travel time per charge time
bd = [v.discharge_rate for v in fleet]                  # battery time per travel time
bs = [n.CHARGING for n in node_data]                    # Charger station at node i
ts = [n.SERVICETIME for n in node_data]                 # Minimal service time at node i
tr = [n.READYTIME for n in node_data]                   # Ready time at node i
td = [n.DUETIME for n in node_data]                     # Due time at node i
cc = [p.COST for p in charge_periods_data]              # Charging cost in period p
to = [p.STARTTIME for p in charge_periods_data]         # Open time for charge period p
tc = [p.ENDTIME for p in charge_periods_data]           # Close time for charge period p
cf = [v.fixed_cost for v in fleet]                      # Fixed cost for deploying vehicle v
cv = [v.variable_cost for v in fleet]                   # Variable distance cost of vehicle v 

# MODEL DEFINITION
#==================================================================================================

# MODEL
model = Model('Vehicle Routing Problem')

# DESISION VARIABLES
x      = model.addVars(N, N, V, vtype=GRB.BINARY, name='x')                                 # If 1, indicates if vehicle v travels from node i to j
z      = model.addVars(N, V, vtype=GRB.BINARY, name='z')                                    # If 1, node is visited by vehicle v
# u      = model.addVars(N, V, lb=0, ub=len(N)-1, vtype=GRB.CONTINUOUS, name='u')             # order of node i in the tour
k      = model.addVar(lb=0, ub=len(V), vtype=GRB.INTEGER, name='k')                         # outgoing vehicles
beta_q = model.addVars(N, P, V, lb=0, ub=MAX_BATTERY, vtype=GRB.CONTINUOUS, name='β^q')     # Amount charged at node i in period p
beta_c = model.addVars(N, P, V, vtype=GRB.BINARY, name='β^c')                               # if 1, vehicle v has charged in period p at node i
beta_a = model.addVars(N, V, lb=0, ub=MAX_BATTERY, vtype=GRB.CONTINUOUS, name='β^a')        # Battery level at arrival of vehicle v at node i
beta_d = model.addVars(N, V, lb=0, ub=MAX_BATTERY, vtype=GRB.CONTINUOUS, name='β^d')        # Battery level at departure of vehicle v at node i
tau_cs = model.addVars(N, P, V, lb=0, ub=MAX_TIME, vtype=GRB.CONTINUOUS, name='τ^cs')       # charging start time at node i in period p for vehicle v
tau_ce = model.addVars(N, P, V, lb=0, ub=MAX_TIME, vtype=GRB.CONTINUOUS, name='τ^ce')       # charging end time at node i in period p for vehicle v
tau_ss = model.addVars(N, V, lb=0, ub=MAX_TIME, vtype=GRB.CONTINUOUS, name='τ^ss')          # Time of arrival at node i
tau_a  = model.addVars(N, V, lb=0, ub=MAX_TIME, vtype=GRB.CONTINUOUS, name='τ^a')           # Time of arrival at node i        
tau_d  = model.addVars(N, V, lb=0, ub=MAX_TIME, vtype=GRB.CONTINUOUS, name='τ^d')           # Time of depature at node i

# OBJECTIVE
obj = (
    quicksum(cv[v] * d[i][j] * x[i, j, v] for i in N for j in N for v in V) +               # Variable distance cost
    quicksum(cc[p] * beta_q[i, p, v] for i in N[1:] for p in P for v in V) +                # Charging costs
    quicksum(cf[v] * z[0, v] for v in V)                                                    # Fixed outgoing cost
)
model.setObjective(obj, GRB.MINIMIZE)

# CONSTRAINTS
constraints = {

    # ROUTE
    'visit_nodes_once':
        (quicksum(z[i, v] for v in V) == 1 for i in N[1:]),

    'outgoing_vehicles':
        quicksum(z[0, v] for v in V) == k,

    'capacity':
        (quicksum(q[i] * z[i, v] for i in N) <= c[v] for v in V),

    'route_incoming':
        (quicksum(x[j, i, v] for j in N) == z[i, v]
        for i in N for v in V),

    'route_outgoing':
        (quicksum(x[i, j, v] for j in N) == z[i, v]
        for i in N for v in V),

    # #Subtour elimination: MTZ (Miller–Tucker–Zemlin) - Not needed due to time dynamics
    # 'subtour_elimination(1)':
    #     (u[0, v] == 1 for v in V),

    # 'subtour_elimination(2)':
    #     (u[i, v] >= 2 for i in N[1:] for v in V),

    # 'subtour_elimination(3)':
    #     (u[i, v] <= len(N) for i in N[1:] for v in V),

    # 'subtour_elimination(4)':
    #     (u[i, v] - u[j, v] + len(N) * x[i, j, v] <= len(N) - 1
    #     for i in N for j in N[1:] for v in V),

    # BATTERY:
    'initial_charge': # set initial charge to max battery cap
        (beta_a[0, v] == bm[v] for v in V_EV),

    'battery_dynamics_traveling': # departure charge - discharge == arrival charge, if (i, j) is arc
    (
        (beta_d[i, v] - d[i][j] * s *  bd[v] <= beta_a[j, v] + MAX_BATTERY * (1 - x[i, j, v])           # type: ignore
        for i in N for j in N[1:] if i != j for v in V_EV),

        (beta_d[i, v] - d[i][j] * s *  bd[v] >= beta_a[j, v] - MAX_BATTERY * (1 - x[i, j, v])           # type: ignore
        for i in N for j in N[1:] if i != j for v in V_EV),
    ),

    'battery_dynamics_charging': # arrival charge + total charge == departure charge
        (beta_a[i, v] + quicksum(beta_q[i, p, v] for p in P) == beta_d[i, v]
        for i in N for v in V_EV),

    'amount_charged': # == charge time * charge rate * has charger * has charged
        (beta_q[i, p, v] == (tau_ce[i, p, v] - tau_cs[i, p, v]) * bc[v] * bs[i] * beta_c[i, p, v]       # type: ignore
        for i in N for p in P for v in V_EV),

    'battery_capacity': # departure charge <= battery capacity, if i is visited (Arrival capacity should always be lower)
        (beta_d[i, v] <= bm[v] + MAX_BATTERY * (1 - z[i, v]) for i in N for v in V_EV),                 # type: ignore

    'final_charge': # departure charge last node - discharge >= 0, if (i, 0) is arc
        (beta_d[i, v] - d[i][0] * s * bd[v] >= 0 - MAX_BATTERY * (1 - x[i, 0, v])                       # type: ignore
        for i in N[1:] for v in V_EV),

    # TIME:
    'arrival_time_dynamics': # departure time outgoing node + travel time == arrival time, if (i, j) is arc
    (
        (tau_d[i, v] + d[i][j] * s <= tau_a[j, v] + MAX_TIME * (1 - x[i, j, v])
        for i in N for j in N[1:] if i != j for v in V),

        (tau_d[i, v] + d[i][j] * s >= tau_a[j, v] - MAX_TIME * (1 - x[i, j, v])
        for i in N for j in N[1:] if i != j for v in V),
    ),

    'departure_time_dynamics_service': # departure time >= start service time + service time, if i is visited
        (tau_d[i, v] >= tau_ss[i, v] + ts[i] - MAX_TIME * (1 - z[i, v])
        for i in N for v in V),

    'departure_time_dynamics_charging': # departure time >= end charging time, if has charged
        (tau_d[i, v] >= tau_ce[i, p, v] - MAX_TIME * (1 - beta_c[i, p, v])
        for i in N for p in P for v in V_EV),

    'non_neg_chargetime': # start time charging >= end time charging, if has charged (to not sell more exp energy)
        (tau_cs[i, p, v] <= tau_ce[i, p, v] + MAX_TIME * (1 - beta_c[i, p, v])
        for i in N for p in P for v in V_EV),

    'charge_period_open': # start time charging >= period start, if has charged
        (tau_cs[i, p, v] >= to[p] - MAX_TIME * (1 - beta_c[i, p, v])
        for i in N for p in P for v in V_EV),

    'charge_period_close': # end time charging <= period end, if has charged
        (tau_ce[i, p, v] <= tc[p] + MAX_TIME * (1 - beta_c[i, p, v])
        for i in N for p in P for v in V_EV),

    'charge_after_arrival': # start time charging after arrival, if has charged
        (tau_cs[i, p, v] >= tau_a[i, v] - MAX_TIME * (1 - beta_c[i, p, v])
        for i in N for p in P for v in V_EV),

    'start_service': # Start time serive after arrival time, if i is visited
        (tau_ss[i, v] >= tau_a[i, v] - MAX_TIME * (1 - z[i, v])
        for i in N for v in V),

    'ready_time_service': # service window start <= start time service, if i is visited
        (tr[i] <= tau_ss[i, v] + MAX_TIME * (1 - z[i, v])
        for i in N for v in V),

    'due_time_service': # service window end >= start time service, if i is visited
        (td[i] >= tau_ss[i, v] - MAX_TIME * (1 - z[i, v])
        for i in N[1:] for v in V),

    'finish_time': # departure time + travel time <= due time depot, if (i, 0) is arc
        (tau_d[i, v] + d[i][0] * s <= td[0] + MAX_TIME * (1 - x[i, 0, v])
        for i in N[1:] for v in V),
}

for name, con in constraints.items():
    try:
        # Tuple of generators
        if isinstance(con, tuple):
            for c in con: model.addConstrs(c, name=name)
        # Generator
        elif isinstance(con, Generator):
            model.addConstrs(con, name=name)
        # Single
        else:
            model.addConstr(con, name=name) # type: ignore
    except:
        print(f"Fout in constraint: {name}")
        sys.exit(1)


# SOLVE
#==================================================================================================

sol = None
def solve():
    try:
        global sol
        model.update()
        # model.write('TSPmodel.lp')
        setattr(model.Params, 'timeLimit', 3600)
        model.optimize()
        # model.write('TSPmodel.sol')

        try: sol = model.ObjVal
        except: raise Exception("Solution not found")

    except Exception as e:
        print(f"Error: {e}")


if input("\nSolve (y/n)? ") == 'y':
    solve()
else: 
    sys.exit()

if __name__ == "__main__":
    if not sol:
        sys.exit(1)
    print('\n')
    print('Success, obj:', sol)