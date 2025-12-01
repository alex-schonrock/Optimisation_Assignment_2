from pathlib import Path
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from Data_ops.data_loader import DataLoader
### class inputdata
class InputData:
    """Class to hold input data for the optimization model."""
    def __init__(
        self,
        price: list[float],
        demand: list[float],
        demand_min: float,
        exp_allowance: float,
        unmet_demand_cost: float,

        # optional for model 2 and 3
        storage_cost: float | None = None,  
        ramp_rate: float | None = None,
        depreciation: float | None = None
        ):

        # defining model 1 data inputs
        self.price = price
        self.demand = demand
        self.demand_min = demand_min
        self.exp_allowance = exp_allowance
        self.unmet_demand_cost = unmet_demand_cost

        # defining model 2 and 3 data inputs
        self.storage_cost = storage_cost
        self.ramp_rate = ramp_rate
        self.depreciation = depreciation
        

class DataProcessor():

    def __init__(self, input_path: str, model_type: str):
        self.input_path = input_path
        self.model_type = model_type
        data = DataLoader(input_path, model_type)
        self.data_loader = data

    def get_coefficients (self):
        """
        Extract and process data from the data files to create InputData instance
        """
        price_df = self.data_loader.load_data_file("price_data.csv")

        # 1) Convert Date to datetime and set as index
        price_df["Date"] = pd.to_datetime(price_df["Date"], dayfirst=True)
        price_df = price_df.set_index("Date").sort_index()

        # 2) Filter by your desired window
        start_date = "2016-02-01"
        end_date = "2016-03-31"
        price_df = price_df.loc[start_date:end_date]

        # 3) Create full daily date range and reindex
        full_idx = pd.date_range(start=start_date, end=end_date, freq="D")
        price_df_full = price_df.reindex(full_idx)

        # 4) Fill missing prices with interpolation
        col = "Gas Spot Price ($/mil Btu)"   # adjust if your column name is different
        price_df_full[col] = price_df_full[col].interpolate(method="time")

        price = price_df_full[col].astype(float).to_numpy()
        price = price * 6.44 / 10**6 # convert $/mil btu to dkk/btu

        demand_df = self.data_loader.load_data_file("demand_data.csv")
        demand_df["Date"] = pd.to_datetime(demand_df["Date"], dayfirst=True)
        demand_df = demand_df.set_index("Date").sort_index()
        demand_df = demand_df.loc[start_date:end_date]
        demand_MW = demand_df['load MW'].astype(float).to_numpy()
        ### Need to scale demand data to be relative to our power plant size 
        demand_MW_max = demand_MW.max()
        plant_capacity_MW = 205
        demand_MW_plant = demand_MW * (plant_capacity_MW / demand_MW_max)

        heat_rate = 6421 # this is in btu/kwh
        demand_plant_Btu = demand_MW_plant * heat_rate * 1000 * 24 # this is demand in btu/day
        demand_min_Btu = 1.4216 * 10**10 # in btu/day
        exp_allowance = 3 * 10**5 # in dkk/day
        unmet_demand_cost = 3 * 10**(-5) # in dkk/btu
        
        if self.model_type == "Model_1":
            return InputData(
             price,
             demand_plant_Btu,
             demand_min_Btu,
             exp_allowance,
             unmet_demand_cost,
            )
        
        if self.model_type == "Model_2":
            storage_cost = 1 # dkk/btu stored per day
            ramp_rate = 1.11 * 10**13 # Btu per day
            depreciation = 0.95 # 5% depreciation per day


            return InputData(
            price,
             demand_plant_Btu,
             demand_min_Btu,
             exp_allowance,
             unmet_demand_cost,
             storage_cost,
             ramp_rate,
             depreciation
            )
        pass


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
        self.cons.bought_min = {}
        n = len(self.data.demand)
        self.T = list(range(0, n))
        # self.T = list(range(len(self.data.demand)))

    def _set_objective(self):
        if self.model_type == "Model_1":
            obj_fn = gp.quicksum(
                self.data.price[t] * self.vars.bought[t] + 
                self.data.unmet_demand_cost * self.vars.unmet_demand[t] for t in self.T)
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
            self.cons.bought_min[i] = self.model.addConstr(self.vars.bought[i] >= self.data.demand_min, name=f"bought_min_{i}")
            
        self.cons.allowance = self.model.addConstr(gp.quicksum(self.vars.bought[t]*self.data.price[t] for t in self.T) <= 
                                                   self.data.exp_allowance*max(self.T), name="max ependiture")
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

        self.results.obj = self.model.ObjVal

        duals = Expando()

        duals.balance = np.array([self.cons.balance[t].Pi for t in self.T], dtype=float)
        duals.unmet_demand_max = np.array([self.cons.unmet_demand_max[t].Pi for t in self.T], dtype=float)
        duals.bought_min = np.array([self.cons.bought_min[t].Pi for t in self.T], dtype=float)
        duals.allowance = float(self.cons.allowance.Pi)
        
        self.results.duals = duals
        return self.results
        # build constraints and store handles in self.cons



path = r"C:\Users\alex\OneDrive\Desktop\DTU\Optimistation\Optimisation_Assignment_2\Data"
data_processor = DataProcessor(path, "Model_1")
data_processor.get_coefficients()
