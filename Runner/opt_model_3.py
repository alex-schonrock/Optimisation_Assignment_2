# from Runner.opt_model_1 import InputData
# from Runner.opt_model_1 import DataProcessor
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


class OptModelCVaR:
    def __init__(self, input_data, model_type: str, alpha: float = 0.95):
        # Input data for the model (e.g., demand, prices, costs, etc.)
        self.data = input_data
        self.model_type = model_type
        self.alpha = alpha  # Confidence level for CVaR
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
        self.cons.eta_constraint = {}
        self.T = list(range(len(self.data.demand_scenarios[0])))  # Time periods
        self.S = list(range(len(self.data.demand_scenarios)))     # Scenarios
        self.probs = self.data.probabilities  # Probabilities of each scenario
        self.expected_unmet_demand = self.data.expected_unmet_demand

    # def _set_objective_with_cvar(self):
    #     # Calculate the average of the worst losses (CVaR)
    #     cvar = gp.quicksum(self.probs[s] * self.vars.z[s] for s in self.S)  # Weighted by scenario probabilities
        
    #     # Objective function: Minimize expected unmet demand + CVaR penalty
    #     obj_fn = gp.quicksum(self.probs[s] * gp.quicksum(self.vars.unmet_demand[t, s] for t in self.T) for s in self.S) + self.alpha * cvar
    #     self.model.setObjective(obj_fn, GRB.MINIMIZE)

    def _set_objective_with_cvar(self):
        # Calculate the average of the worst losses (CVaR)
        cvar = gp.quicksum(self.probs[s] * self.vars.z[s] for s in self.S)  # Weighted by scenario probabilities
        
        # CVaR penalty based on `eta`
        cvar_penalty = self.alpha * gp.quicksum(self.probs[s] * self.vars.eta[s] for s in self.S)
        
        # Objective function: Minimize expected unmet demand + CVaR penalty
        obj_fn = gp.quicksum(self.probs[s] * gp.quicksum(self.vars.unmet_demand[t, s] for t in self.T) for s in self.S) + cvar_penalty
        self.model.setObjective(obj_fn, GRB.MINIMIZE)

    def _build(self):
        # Add variables for each time period and scenario
        v_bought = self.model.addVars(self.T, self.S, name="bought", lb=0)
        v_unmet_demand = self.model.addVars(self.T, self.S, name="unmet_demand", lb=0)
        v_stored = self.model.addVars(self.T, self.S, name="stored", lb=0, ub=self.data.max_storage_capacity)
        v_budget = self.model.addVars(self.T, self.S, name="budget", lb=0)
        z = self.model.addVars(self.S, name="z", lb=0)
        eta = self.model.addVars(self.S, name="eta", lb=0)
        
        # Store variable handles in self.vars
        self.vars.bought = v_bought
        self.vars.unmet_demand = v_unmet_demand
        self.vars.stored = v_stored
        self.vars.budget = v_budget
        self.vars.zeta = z
        self.vars.eta = eta

        
        # Define constraints for each time period and scenario
        for t in self.T:
            for s in self.S:
                # Constraints for unmet demand (max demand that can be unmet)
                self.cons.unmet_demand_max[t, s] = self.model.addConstr(self.vars.unmet_demand[t, s] <= self.data.demand_scenarios[s, t], name=f"unmet_demand_max_{t}_{s}")
                # Constraints for plant minimum
                self.cons.plant_min[t, s] = self.model.addConstr(self.data.demand_scenarios[s, t] - self.vars.unmet_demand[t, s] >= self.data.demand_min, name=f"bought_min_{t}_{s}")
                # Constraints for the budget (bought energy + storage cost)
                self.cons.budget[t, s] = self.model.addConstr(self.vars.bought[t, s] * self.data.price[t] + self.vars.stored[t, s] * self.data.storage_cost <= self.vars.budget[t, s], name=f"budget_{t}_{s}")
                # define eta as the deviation from expected unmet demand
                self.cons.eta_constraint[t, s] = self.model.addConstr(self.vars.eta[s] >= abs(self.vars.unmet_demand[t, s] - self.data.expected_unmet_demand[t]), name=f"eta_constraint_{t}_{s}")
        
        # Constraints that don't apply in the first day due to intertemporal nature
        for t in list(range(1, len(self.data.demand_scenarios[0]))):  # Loop over time periods
            for s in self.S:  # Loop over scenarios
                # Ramp-up and ramp-down constraints (capacity change between time periods)
                self.cons.ramp_up[t, s] = self.model.addConstr(self.data.demand_scenarios[s, t] - self.vars.unmet_demand[t, s] - (self.data.demand_scenarios[s, t-1] - self.vars.unmet_demand[t-1, s]) <= self.data.ramp_rate)
                self.cons.ramp_down[t, s] = self.model.addConstr(self.data.demand_scenarios[s, t-1] - self.vars.unmet_demand[t-1, s] - (self.data.demand_scenarios[s, t] - self.vars.unmet_demand[t, s]) <= self.data.ramp_rate)
                # Depreciation constraint
                self.cons.depreciation[t, s] = self.model.addConstr(self.vars.budget[t, s] == self.data.exp_allowance + self.data.depreciation * (self.vars.budget[t-1, s] - self.vars.bought[t-1, s] * self.data.price[t-1] - self.vars.stored[t-1, s] * self.data.storage_cost))
                # Balance constraint (ensuring demand is met or stored)
                self.cons.balance[t, s] = self.model.addConstr(self.vars.bought[t, s] + self.vars.stored[t-1, s] == self.data.demand_scenarios[s, t] - self.vars.unmet_demand[t, s] + self.vars.stored[t, s], name=f"balance_{t}_{s}")
        
        # Balance and depreciation for the first time period (index 0)
        for s in self.S:
            self.cons.balance[0, s] = self.model.addConstr(self.vars.bought[0, s] == self.data.demand_scenarios[s, 0] - self.vars.unmet_demand[0, s] + self.vars.stored[0, s], name=f"balance_0_{s}")
            self.cons.depreciation[0, s] = self.model.addConstr(self.vars.budget[0, s] == self.data.exp_allowance, name=f"initial_budget_{s}")
            self.cons.ramp_up[0, s] = self.model.addConstr(self.data.demand_scenarios[s, 0] - self.vars.unmet_demand[0, s] <= self.data.ramp_rate, name=f"ramp_up_0_{s}")
            self.cons.ramp_down[0, s] = self.model.addConstr(self.data.demand_scenarios[s, 0] - self.vars.unmet_demand[0, s] <= self.data.ramp_rate, name=f"ramp_down_0_{s}")
        
        self._set_objective_with_cvar()

    def solve(self, verbose: bool = False):
        if not verbose:
            self.model.Params.OutputFlag = 0
        
        # Optimize the model
        self.model.optimize()

        # Check if the model is infeasible
        if self.model.Status != GRB.OPTIMAL:
            print(f"Gurobi status: {self.model.Status}")
            
            if self.model.Status == GRB.INFEASIBLE:
                print("Model is infeasible. Running Infeasibility Analyzer...")
                self.model.computeIIS()  # Compute Infeasible Irreducible Set
                self.model.write("model.ilp")  # Write the infeasible model to a file
                
                # Optional: Print out the infeasible constraints
                print("Infeasible Constraints:")
                for c in self.model.getConstrs():
                    if c.IISConstr:  # If the constraint is part of the IIS (Infeasible Irreducible Set)
                        print(f"Infeasible Constraint: {c.constrName}")
            else:
                print("Optimization did not return optimal or suboptimal solution.")

            # Raise an error to stop further execution
            raise RuntimeError(f"Gurobi status: {self.model.Status}")
        
        # Store the results after optimization
        v = self.vars
        self.results.v_bought = np.array([[v.bought[t, s].X for s in self.S] for t in self.T])
        self.results.v_unmet_demand = np.array([[v.unmet_demand[t, s].X for s in self.S] for t in self.T])
        self.results.prices = np.asarray(self.data.price, dtype=float).reshape(-1)
        self.results.demand_scenarios = np.asarray(self.data.demand_scenarios, dtype=float)
        self.results.v_stored = np.array([[v.stored[t, s].X for s in self.S] for t in self.T])
        self.results.v_budget = np.array([[v.budget[t, s].X for s in self.S] for t in self.T])
        self.results.allowance = np.asarray(self.data.exp_allowance, dtype=float).reshape(-1)

        self.results.obj = self.model.ObjVal

        self.results.zeta = self.vars.zeta.X  # CVaR worst-case value
        self.results.eta = np.array([self.vars.eta[s].X for s in self.S])  # Deviation for each scenario

        duals = Expando()
        duals.balance = np.array([[self.cons.balance[t, s].Pi for s in self.S] for t in self.T], dtype=float)
        duals.unmet_demand_max = np.array([[self.cons.unmet_demand_max[t, s].Pi for s in self.S] for t in self.T], dtype=float)
        duals.plant_min = np.array([[self.cons.plant_min[t, s].Pi for s in self.S] for t in self.T], dtype=float)
        duals.ramp_up = np.array([[self.cons.ramp_up[t, s].Pi for s in self.S] for t in self.T], dtype=float)
        duals.ramp_down = np.array([[self.cons.ramp_down[t, s].Pi for s in self.S] for t in self.T], dtype=float)
        duals.depreciation = np.array([[self.cons.depreciation[t, s].Pi for s in self.S] for t in self.T], dtype=float)
        duals.budget = np.array([[self.cons.budget[t, s].Pi for s in self.S] for t in self.T], dtype=float)

        self.results.duals = duals
        return self.results
    
    
path = r"C:\Users\alex\OneDrive\Desktop\DTU\Optimistation\Optimisation_Assignment_2\Data"
data_processor = DataProcessor(path, "Model_3")
input_data = data_processor.get_coefficients()
print(input_data.demand_scenarios)
# Initialize the OptModelCVaR model
model = OptModelCVaR(input_data, model_type="Model_1")
model._build()
results = model.solve()

# Access the results
v_bought = results.v_bought  # Procurement (bought) values for each time period and scenario
v_unmet_demand = results.v_unmet_demand  # Unmet demand for each time period and scenario
prices = results.prices  # Prices for each time period
demand_scenarios = results.demand_scenarios  # Demand scenarios
v_stored = results.v_stored  # Stored energy values for each time period and scenario
v_budget = results.v_budget  # Budget values for each time period and scenario
allowance = results.allowance  # Allowance values for each time period

# Print the total objective value
print(f"Objective Value: {results.obj}")

# Print CVaR-related results
if model.alpha > 0:  # If CVaR is included
    print(f"CVaR (zeta): {results.zeta}")  # Print the CVaR value (worst-case value)
    print("Eta values (deviation for each scenario):")
    for s in model.S:
        print(f"  Scenario {s}: {results.eta[s]}")

# Now, display the results for each scenario
for s in model.S:
    print(f"Results for Scenario {s}:")
    for t in model.T:
        # Display the results for each time period (t) in each scenario (s)
        print(f"  Time Period {t}:")
        print(f"    Bought: {v_bought[t, s]}")  # Procurement for this time period and scenario
        print(f"    Unmet Demand: {v_unmet_demand[t, s]}")  # Unmet demand for this time period and scenario
        print(f"    Demand: {demand_scenarios[s, t]}")  # Actual demand for this time period and scenario
        print(f"    Stored: {v_stored[t, s]}")  # Stored energy for this time period and scenario
        print(f"    Budget: {v_budget[t, s]}")  # Budget for this time period and scenario

# You can also access other results (like duals) if needed, for example:
print(f"Duals for balance constraints:")
for t in model.T:
    for s in model.S:
        print(f"  Time Period {t}, Scenario {s}: {results.duals['balance'][t, s]}")



class OptModel:
    def __init__(self, input_data, model_type: str):
        # Input data for the model (e.g., demand, prices, costs, etc.)
        self.data = input_data
        self.model_type = model_type
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
        self.T = list(range(len(self.data.demand[0])))  # Time periods
        self.S = list(range(len(self.data.demand)))     # Scenarios
        self.probs = self.data.probabilities  # Probabilities of each scenario

    def _set_objective_without_cvar(self):
        # Objective function: Minimize the expected total unmet demand across all time periods and scenarios
        obj_fn = gp.quicksum(self.probs[s] * gp.quicksum(self.vars.unmet_demand[t] for t in self.T) for s in self.S)
        self.model.setObjective(obj_fn, GRB.MINIMIZE)

    def _build(self):
        # Add variables for each time period (single variable for each time period across all scenarios)
        v_bought = self.model.addVars(self.T, name="bought", lb=0)
        v_unmet_demand = self.model.addVars(self.T, self.S, name="unmet_demand", lb=0)
        v_stored = self.model.addVars(self.T, self.S, name="stored", lb=0, ub=self.data.max_storage_capacity)
        v_budget = self.model.addVars(self.T, self.S, name="budget", lb=0)

        # Store variable handles in self.vars
        self.vars.bought = v_bought
        self.vars.unmet_demand = v_unmet_demand
        self.vars.stored = v_stored
        self.vars.budget = v_budget

        # Define constraints for each time period and scenario
        for t in self.T:
            for s in self.S:
                # Constraints for unmet demand (max demand that can be unmet)
                self.cons.unmet_demand_max[t, s] = self.model.addConstr(self.vars.unmet_demand[t, s] <= self.data.demand[s][t], name=f"unmet_demand_max_{t}_{s}")
                # Constraints for plant minimum
                self.cons.plant_min[t, s] = self.model.addConstr(self.data.demand[s][t] - self.vars.unmet_demand[t, s] >= self.data.demand_min, name=f"bought_min_{t}_{s}")
                # Constraints for the budget (bought energy + storage cost)
                self.cons.budget[t, s] = self.model.addConstr(self.vars.bought[t] * self.data.price[t] + self.vars.stored[t, s] * self.data.storage_cost <= self.vars.budget[t, s], name=f"budget_{t}_{s}")
        
        # Constraints that don't apply in the first day due to intertemporal nature
        for t in list(range(1, len(self.data.demand[0]))):  # Loop over time periods
            for s in self.S:  # Loop over scenarios
                # Ramp-up and ramp-down constraints (capacity change between time periods)
                self.cons.ramp_up[t, s] = self.model.addConstr(self.data.demand[s][t] - self.vars.unmet_demand[t, s] - (self.data.demand[s][t-1] - self.vars.unmet_demand[t-1, s]) <= self.data.ramp_rate)
                self.cons.ramp_down[t, s] = self.model.addConstr(self.data.demand[s][t-1] - self.vars.unmet_demand[t-1, s] - (self.data.demand[s][t] - self.vars.unmet_demand[t, s]) <= self.data.ramp_rate)
                # Depreciation constraint
                self.cons.depreciation[t, s] = self.model.addConstr(self.vars.budget[t, s] == self.data.exp_allowance + self.data.depreciation * (self.vars.budget[t-1, s] - self.vars.bought[t-1] * self.data.price[t-1] - self.vars.stored[t-1, s] * self.data.storage_cost))
                # Balance constraint (ensuring demand is met or stored)
                self.cons.balance[t, s] = self.model.addConstr(self.vars.bought[t] + self.vars.stored[t-1, s] == self.data.demand[s][t] - self.vars.unmet_demand[t, s] + self.vars.stored[t, s], name=f"balance_{t}_{s}")
        
        # Balance and depreciation for the first time period (index 0)
        for s in self.S:
            self.cons.balance[0, s] = self.model.addConstr(self.vars.bought[0] == self.data.demand[s][0] - self.vars.unmet_demand[0, s] + self.vars.stored[0, s], name=f"balance_0_{s}")
            self.cons.depreciation[0, s] = self.model.addConstr(self.vars.budget[0, s] == self.data.exp_allowance, name=f"initial_budget_{s}")
            self.cons.ramp_up[0, s] = self.model.addConstr(self.data.demand[s][0] - self.vars.unmet_demand[0, s] <= self.data.ramp_rate, name=f"ramp_up_0_{s}")
            self.cons.ramp_down[0, s] = self.model.addConstr(self.data.demand[s][0] - self.vars.unmet_demand[0, s] <= self.data.ramp_rate, name=f"ramp_down_0_{s}")
        
        self._set_objective_without_cvar()

    def solve(self, verbose: bool = False):
        if not verbose:
            self.model.Params.OutputFlag = 0
        self.model.optimize()
        if self.model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            raise RuntimeError(f"Gurobi status: {self.model.Status}")
        
        # Store the results after optimization
        v = self.vars
        self.results.v_bought = np.array([v.bought[t].X for t in self.T])  # One variable for each time period
        self.results.v_unmet_demand = np.array([[v.unmet_demand[t, s].X for s in self.S] for t in self.T])
        self.results.prices = np.asarray(self.data.price, dtype=float).reshape(-1)
        self.results.demand = np.asarray(self.data.demand, dtype=float)
        self.results.v_stored = np.array([[v.stored[t, s].X for s in self.S] for t in self.T])
        self.results.v_budget = np.array([[v.budget[t, s].X for s in self.S] for t in self.T])
        self.results.allowance = np.asarray(self.data.exp_allowance, dtype=float).reshape(-1)

        self.results.obj = self.model.ObjVal

        duals = Expando()
        duals.balance = np.array([[self.cons.balance[t, s].Pi for s in self.S] for t in self.T], dtype=float)
        duals.unmet_demand_max = np.array([[self.cons.unmet_demand_max[t, s].Pi for s in self.S] for t in self.T], dtype=float)
        duals.plant_min = np.array([[self.cons.plant_min[t, s].Pi for s in self.S] for t in self.T], dtype=float)
        duals.ramp_up = np.array([[self.cons.ramp_up[t, s].Pi for s in self.S] for t in self.T], dtype=float)
        duals.ramp_down = np.array([[self.cons.ramp_down[t, s].Pi for s in self.S] for t in self.T], dtype=float)
        duals.depreciation = np.array([[self.cons.depreciation[t, s].Pi for s in self.S] for t in self.T], dtype=float)
        duals.budget = np.array([[self.cons.budget[t, s].Pi for s in self.S] for t in self.T], dtype=float)

        self.results.duals = duals
        return self.results




# data_processor = DataProcessor(path, "Model_3")
# input_data = data_processor.get_coefficients()
# print(input_data.demand_scenarios)
# model = OptModelCVaR(input_data, model_type="Model_1")
# model._build()

# # Retrieve results
# results = model.solve()

# # Access the results
# v_bought = results.v_bought
# v_unmet_demand = results.v_unmet_demand
# prices = results.prices
# demand_scenarios = results.demand_scenarios
# v_stored = results.v_stored
# v_budget = results.v_budget
# allowance = results.allowance

# # Now, to display the results for each scenario:
# for s in model.S:
#     print(f"Results for Scenario {s}:")
#     for t in model.T:
#         # Display the results for each time period (t) in each scenario (s)
#         print(f"  Time Period {t}:")
#         print(f"    Bought: {v_bought[t, s]}")
#         print(f"    Unmet Demand: {v_unmet_demand[t, s]}")
#         print(f"    Demand: {demand_scenarios[s][t]}")
#         print(f"    Stored: {v_stored[t, s]}")
#         print(f"    Budget: {v_budget[t, s]}")