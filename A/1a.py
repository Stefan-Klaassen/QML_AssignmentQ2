from gurobipy import *
from data import data_small as data

VEHICLES = 4 

model = Model ('Vehicle Routing Problem')

# ---- Parameters ----
N = range(len(data))
V = range(len(VEHICLES))


