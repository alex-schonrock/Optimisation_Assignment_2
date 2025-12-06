from pathlib import Path
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from Data_ops.data_loader_processor import *

class Expando(object):
    pass


class OptModel1:
    def __init__(self, input_data: InputData, model_type: str):
        self.data = input_data
        self.model_type = model_type
        self.results = Expando()
        self.model = gp.Model()
        self.vars = Expando()
        self.cons = Expando() 
        self.cons.balance = {}
        self.cons.allowance = {}
        self.cons.unmet_demand_max = {}
        self.cons.plant_min = {}
        n = len(self.data.demand)
        self.T = list(range(0, n))
        # self.T = list(range(len(self.data.demand)))

    def _set_objective(self):
        if self.model_type == "Model_1":
            # obj_fn = gp.quicksum(
            #     self.data.price[t] * self.vars.bought[t] + 
            #     self.data.unmet_demand_cost * self.vars.unmet_demand[t] for t in self.T)
            obj_fn = gp.quicksum(self.vars.unmet_demand[t] for t in self.T)
        self.model.setObjective(obj_fn, GRB.MINIMIZE)

    def _build(self):
        v_bought = self.model.addVars(self.T, name="bought", lb=0)
        v_unmet_demand = self.model.addVars(self.T, name="unmet_demand", lb=0)

        # store variable handles in self.vars
        self.vars.bought = v_bought
        self.vars.unmet_demand = v_unmet_demand

        for i in self.T:
            self.cons.balance[i] = self.model.addConstr(self.vars.bought[i] ==
                                                        self.data.demand[i] - self.vars.unmet_demand[i], name=f"balance_{i}")
            self.cons.unmet_demand_max[i] = self.model.addConstr(self.vars.unmet_demand[i] <= self.data.demand[i], name=f"unmet_demand_max_{i}")
            self.cons.plant_min[i] = self.model.addConstr(self.vars.bought[i] >= self.data.demand_min, name=f"bought_min_{i}")
            
            self.cons.allowance[i] = self.model.addConstr(self.vars.bought[i]*self.data.price[i] <= 
                                                   self.data.exp_allowance, name="max ependiture")
        self._set_objective()
    
    def solve(self, verbose: bool = False):
        if not verbose:
            self.model.Params.OutputFlag = 0
        self.model.optimize()
        if self.model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            raise RuntimeError(f"Gurobi status: {self.model.Status}")

        v = self.vars
        self.results.v_bought = np.array([v.bought[t].X for t in self.T])
        self.results.v_unmet_demand = np.array([v.unmet_demand[t].X for t in self.T])
        self.results.prices = np.asarray(self.data.price, dtype=float).reshape(-1)
        self.results.demand = np.asarray(self.data.demand, dtype=float).reshape(-1)
        self.results.allowance = self.data.exp_allowance

        self.results.obj = self.model.ObjVal

        duals = Expando()

        duals.balance = np.array([self.cons.balance[t].Pi for t in self.T], dtype=float)
        duals.unmet_demand_max = np.array([self.cons.unmet_demand_max[t].Pi for t in self.T], dtype=float)
        duals.plant_min = np.array([self.cons.plant_min[t].Pi for t in self.T], dtype=float)
        duals.allowance = np.array([-self.cons.allowance[t].Pi for t in self.T], dtype=float)
        # duals.allowance = float(self.cons.allowance.Pi)
        
        self.results.duals = duals
        return self.results
        # build constraints and store handles in self.cons



