from pathlib import Path
from Data_ops.data_loader_processor import *
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

class Expando(object):
    pass

import gurobipy as gp
from gurobipy import GRB
import numpy as np


    
class OptModelCVaR_IndependentStorageBudget:
    def __init__(self, input_data, model_type: str, alpha: float = 0.8, beta: float = 0):
        # Input data for the model
        self.data = input_data
        self.model_type = model_type
        self.alpha = alpha  # Confidence level for CVaR
        self.beta = beta  # Weight for expected unmet demand
        self.results = Expando()
        self.model = gp.Model()
        self.vars = Expando()
        self.cons = Expando()
        self.cons.balance = {}
        self.cons.unmet_demand_max = {}
        self.cons.plant_min = {}
        self.cons.ramp_up = {}
        self.cons.ramp_down = {}
        self.cons.depreciation = {}
        self.cons.budget = {}
        self.cons.initial_budget = {}
        self.cons.eta_up = {}
        self.T = list(range(len(self.data.demand_scenarios[0])))  # Time periods
        self.S = list(range(len(self.data.demand_scenarios)))     # Scenarios
        self.probs = self.data.probabilities  # Probabilities of each scenario
        self.price_scenarios = self.data.price_scenarios

    def _set_objective_with_cvar(self):
        # CVaR penalty based on `eta`
        cvar_penalty = self.vars.z + (1 / (1 - self.alpha)) * gp.quicksum(self.probs[s] * self.vars.eta[s] for s in self.S)
        
        unmet_demand = gp.quicksum(self.probs[s] * gp.quicksum(self.vars.unmet_demand[t, s] for t in self.T) for s in self.S)

        # Objective function: Minimize expected unmet demand + CVaR penalty
        obj_fn = (1-self.beta)*unmet_demand + self.beta * cvar_penalty
        self.model.setObjective(obj_fn, GRB.MINIMIZE)

    def _build(self):
        # Add variables - bought and unmet_demand vary by scenario
        v_bought = self.model.addVars(self.T, self.S, name="bought", lb=0)
        v_unmet_demand = self.model.addVars(self.T, self.S, name="unmet_demand", lb=0)
        
        # CHANGED: Storage only varies with time, NOT scenario
        v_stored = self.model.addVars(self.T, name="stored", lb=0, ub=self.data.max_storage_capacity)
        
        # CHANGED: Budget only varies with time, NOT scenario
        v_budget = self.model.addVars(self.T, name="budget", lb=0)
        
        z = self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name='z')
        eta = self.model.addVars(self.S, name="eta", lb=0)
        
        # Store variable handles in self.vars
        self.vars.bought = v_bought
        self.vars.unmet_demand = v_unmet_demand
        self.vars.stored = v_stored
        self.vars.budget = v_budget
        self.vars.z = z
        self.vars.eta = eta
        
        # Define constraints for each time period and scenario
        for t in self.T:
            for s in self.S:
                # Constraints for unmet demand (max demand that can be unmet)
                self.cons.unmet_demand_max[t, s] = self.model.addConstr(
                    self.vars.unmet_demand[t, s] <= self.data.demand_scenarios[s, t], 
                    name=f"unmet_demand_max_{t}_{s}")
                
                # Constraints for plant minimum
                self.cons.plant_min[t, s] = self.model.addConstr(
                    self.data.demand_scenarios[s, t] - self.vars.unmet_demand[t, s] >= self.data.demand_min, 
                    name=f"bought_min_{t}_{s}")
                
                # CHANGED: Budget constraint uses v_budget[t] and v_stored[t]
                self.cons.budget[t, s] = self.model.addConstr(
                    self.vars.bought[t, s] * self.data.price_scenarios[s, t] + 
                    self.vars.stored[t] * self.data.storage_cost <= self.vars.budget[t], 
                    name=f"budget_{t}_{s}")

        # Constraints that don't apply in the first day
        for t in list(range(1, len(self.data.demand_scenarios[0]))):
            for s in self.S:
                # Ramp-up and ramp-down constraints
                self.cons.ramp_up[t, s] = self.model.addConstr(
                    self.data.demand_scenarios[s, t] - self.vars.unmet_demand[t, s] - 
                    (self.data.demand_scenarios[s, t-1] - self.vars.unmet_demand[t-1, s]) <= self.data.ramp_rate)
                
                self.cons.ramp_down[t, s] = self.model.addConstr(
                    self.data.demand_scenarios[s, t-1] - self.vars.unmet_demand[t-1, s] - 
                    (self.data.demand_scenarios[s, t] - self.vars.unmet_demand[t, s]) <= self.data.ramp_rate)
                
                # CHANGED: Balance constraint uses v_stored[t] and v_stored[t-1]
                self.cons.balance[t, s] = self.model.addConstr(
                    self.vars.bought[t, s] + self.vars.stored[t-1] == 
                    self.data.demand_scenarios[s, t] - self.vars.unmet_demand[t, s] + self.vars.stored[t], 
                    name=f"balance_{t}_{s}")
            
            # CHANGED: Depreciation constraint - one per time period
            # Use EXPECTED spending across scenarios
            expected_spending = gp.quicksum(
                self.probs[s] * self.vars.bought[t-1, s] * self.data.price_scenarios[s, t-1]
                for s in self.S
            ) + self.vars.stored[t-1] * self.data.storage_cost  # Storage is deterministic
            
            self.cons.depreciation[t] = self.model.addConstr(
                self.vars.budget[t] == self.data.exp_allowance + 
                self.data.depreciation * (self.vars.budget[t-1] - expected_spending),
                name=f"depreciation_{t}")
        
        # Balance and initial constraints for first time period
        for s in self.S:
            # CHANGED: Balance constraint for t=0 uses v_stored[0]
            self.cons.balance[0, s] = self.model.addConstr(
                self.vars.bought[0, s] == 
                self.data.demand_scenarios[s, 0] - self.vars.unmet_demand[0, s] + self.vars.stored[0], 
                name=f"balance_0_{s}")
            
            self.cons.ramp_up[0, s] = self.model.addConstr(
                self.data.demand_scenarios[s, 0] - self.vars.unmet_demand[0, s] <= self.data.ramp_rate, 
                name=f"ramp_up_0_{s}")
            
            self.cons.ramp_down[0, s] = self.model.addConstr(
                self.data.demand_scenarios[s, 0] - self.vars.unmet_demand[0, s] <= self.data.ramp_rate, 
                name=f"ramp_down_0_{s}")
            
            self.cons.eta_up[s] = self.model.addConstr(
                eta[s] >= gp.quicksum(self.vars.unmet_demand[t, s] for t in self.T) - self.vars.z, 
                name=f"eta_abs_upper_{s}")
        
        # CHANGED: Initial budget constraint
        self.cons.initial_budget = self.model.addConstr(
            self.vars.budget[0] == self.data.exp_allowance, 
            name=f"initial_budget")
        
        self._set_objective_with_cvar()

    def solve(self, verbose: bool = False):
        if not verbose:
            self.model.Params.OutputFlag = 0
        
        # Optimize the model
        self.model.optimize()

        v = self.vars
        self.results.v_bought = np.array([[v.bought[t, s].X for s in self.S] for t in self.T])
        self.results.v_unmet_demand = np.array([[v.unmet_demand[t, s].X for s in self.S] for t in self.T])
        self.results.price_scenarios = np.asarray(self.data.price_scenarios, dtype=float)
        self.results.demand_scenarios = np.asarray(self.data.demand_scenarios, dtype=float)
        
        # CHANGED: Storage is now (time,) not (time, scenarios)
        # Replicate it across scenarios for compatibility
        stored_1d = np.array([v.stored[t].X for t in self.T])
        self.results.v_stored = np.tile(stored_1d.reshape(-1, 1), (1, len(self.S)))
        
        # CHANGED: Budget is now (time,) not (time, scenarios)
        # Replicate it across scenarios for compatibility
        budget_1d = np.array([v.budget[t].X for t in self.T])
        self.results.v_budget = np.tile(budget_1d.reshape(-1, 1), (1, len(self.S)))
        
        self.results.allowance = np.asarray(self.data.exp_allowance, dtype=float).reshape(-1)
        self.results.z = v.z.X
        self.results.eta = np.array([v.eta[s].X for s in self.S])
        self.results.obj = self.model.ObjVal

        duals = Expando()
        duals.balance = np.array([[self.cons.balance[t, s].Pi for s in self.S] for t in self.T], dtype=float)
        duals.unmet_demand_max = np.array([[self.cons.unmet_demand_max[t, s].Pi for s in self.S] for t in self.T], dtype=float)
        duals.plant_min = np.array([[self.cons.plant_min[t, s].Pi for s in self.S] for t in self.T], dtype=float)
        duals.ramp_up = np.array([[self.cons.ramp_up[t, s].Pi for s in self.S] for t in self.T], dtype=float)
        duals.ramp_down = np.array([[self.cons.ramp_down[t, s].Pi for s in self.S] for t in self.T], dtype=float)
        
        # CHANGED: Depreciation duals are now 1D
        duals.depreciation = np.array([self.cons.depreciation[t].Pi for t in self.T[1:]], dtype=float)
        
        duals.budget = np.array([[self.cons.budget[t, s].Pi for s in self.S] for t in self.T], dtype=float)

        self.results.duals = duals
        return self.results


