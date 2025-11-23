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
        storage_cost: float):

        # defining model 1 data inputs
        self.price = price
        self.demand = demand
        self.demand_min = demand_min
        self.exp_allowance = exp_allowance
        self.unmet_demand_cost = unmet_demand_cost
        

class DataProcessor():

    def __init__(self, input_path: str, model_type: str):
        self.input_path = input_path
        self.model = model_type
        data = DataLoader(input_path, model_type)
        self.data_loader = data

    def get_coefficients (self):
        """ 
        Extract and process data from the data files to create InputData instance
        """
        # start_date = "2016-02-01"
        # end_date = "2016-03-31"
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

        # 5) If you want Date back as a column:
        #price_df_full = price_df_full.rename_axis("Date").reset_index()
        #demand = self.data_loader.load_data_file("demand_data.csv")['demand'].values.tolist()
        print("Interpolated values is: ", price_df_full)
        print("Original data is: ", price_df)


        demand_df = self.data_loader.load_data_file("demand_data.csv")
        demand_df["Date"] = pd.to_datetime(demand_df["Date"], dayfirst=True)
        demand_df = demand_df.set_index("Date").sort_index()
        demand_df = demand_df.loc[start_date:end_date]
        ### Need to scale demand data to be relative to our power plant size 
        print("Demand data is: ", demand_df)
       
        # demand_min =
        # exp_allowance =
        # unmet_demand_cost =
        # storage_cost =
        # return InputData(
        #     price,
        #     demand,
        #     demand_min,
        #     exp_allowance,
        #     unmet_demand_cost,
        #     storage_cost)
        pass


class Expando(object):
    pass


class OptModel:
    def __init__(self, input_data: InputData, model_type: str):
        self.data = input_data
        self.model_type = model_type
        self.results = Expando()
        self.model = gp.Model()
        self.vars = Expando()
        self.cons = Expando() 
        self.cons.balance = {}
        self.cons.allowance = {}
        self.cons.unmet_demand = {}
        self.T = len(self.data.demand)

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

        #for i in self.T:


        # build constraints and store handles in self.cons



path = r"C:\Users\alex\OneDrive\Desktop\DTU\Optimistation\Optimisation_Assignment_2\Data"
data_processor = DataProcessor(path, "Model_1")
data_processor.get_coefficients()
