from Runner.opt_model_1 import InputData
from Runner.opt_model_1 import DataProcessor
from pathlib import Path
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

class Expando(object):
    pass

class OptModel2:
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
        self.cons.bought_min = {}
        self.cons.ramp_up = {}
        self.cons.ramp_down = {}
        self.cons.depreciation = {}
        self.cons.budget = {}
        self.cons.initial_budget = {}
        n = len(self.data.demand)
        self.T = list(range(0, n))
        # self.T = list(range(len(self.data.demand)))

    def _set_objective(self):

        obj_fn = gp.quicksum(
                self.data.price[t] * self.vars.bought[t] + 
                self.data.unmet_demand_cost * self.vars.unmet_demand[t] +
                self.data.storage_cost * self.vars.stored[t]
                for t in self.T)
        self.model.setObjective(obj_fn, GRB.MINIMIZE)

    def _build(self):
        v_bought = self.model.addVars(self.T, name="bought", lb=0)
        v_unmet_demand = self.model.addVars(self.T, name="unmet_demand", lb=0)
        v_stored = self.model.addVars(self.T, name="stored", lb=0)
        v_budget = self.model.addVars(self.T, name="budget", lb=0)

        # store variable handles in self.vars
        self.vars.bought = v_bought
        self.vars.unmet_demand = v_unmet_demand
        self.vars.stored = v_stored
        self.vars.budget = v_budget

        for i in self.T:
            #### fix this one for s0 

            

            self.cons.unmet_demand_max[i] = self.model.addConstr(self.vars.unmet_demand[i] <= self.data.demand[i], name=f"unmet_demand_max_{i}")
            self.cons.bought_min[i] = self.model.addConstr(self.vars.bought[i] >= self.data.demand_min, name=f"bought_min_{i}")
            self.cons.budget[i] = self.model.addConstr(self.vars.bought[i]*self.data.price[i] <= self.vars.budget[i], name=f"budget_{i}")


        # constraints that dont apply in the first day due to intertemporal nature
        for i in list(range(1, len(self.data.demand))):
            self.cons.ramp_up[i] = self.model.addConstr(self.vars.bought[i] - self.vars.bought[i-1] <= self.data.ramp_rate)
            self.cons.ramp_down[i] = self.model.addConstr(self.vars.bought[i-1] - self.vars.bought[i] <= self.data.ramp_rate)
            self.cons.depreciation[i] = self.model.addConstr(self.vars.budget[i] == self.data.exp_allowance + 
                                                             self.data.depreciation*(self.vars.budget[i-1] - self.vars.bought[i-1]*self.data.price[i-1]))
            self.cons.balance[i] = self.model.addConstr(self.vars.bought[i] + self.vars.stored[i-1] ==
                                                        self.data.demand[i] - self.vars.unmet_demand[i] + self.vars.stored[i], name=f"balance_{i}")
        self.cons.balance[0] = self.model.addConstr(self.vars.bought[0] ==
                                                    self.data.demand[0] - self.vars.unmet_demand[0] + self.vars.stored[0], name=f"balance_0")
        self.cons.inital_budget = self.model.addConstr(self.vars.budget[0] == self.data.exp_allowance)

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
        self.results.v_stored = np.array([v.stored[t].X for t in self.T])
        self.results.v_budget = np.array([v.budget[t].X for t in self.T])

        self.results.obj = self.model.ObjVal

        duals = Expando()

        duals.balance = np.array([self.cons.balance[t].Pi for t in self.T], dtype=float)
        duals.unmet_demand_max = np.array([self.cons.unmet_demand_max[t].Pi for t in self.T], dtype=float)
        duals.bought_min = np.array([self.cons.bought_min[t].Pi for t in self.T], dtype=float)
        #duals.allowance = float(self.cons.allowance.Pi)
        
        self.results.duals = duals
        return self.results
        # build constraints and store handles in self.cons

path = r"C:\Users\alex\OneDrive\Desktop\DTU\Optimistation\Optimisation_Assignment_2\Data"
data_processor = DataProcessor(path, "Model_2")
data_processor.get_coefficients()