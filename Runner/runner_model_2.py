import pandas as pd
from pathlib import Path
import numpy as np
from Runner.opt_model_2 import OptModel2
from Runner.opt_model_1 import DataProcessor
from Utils.utils_and_plotting import *
from Data_ops.data_loader_processor import *
import csv


class RunnerModel2:
    """Class to run optimization model 2."""
    def __init__(self, input_path: str):
        self.input_path = input_path
        self.model_type = "Model_2"

    def run_model(self):
        # Load and process data
        data_processor = DataProcessor(self.input_path, model_type="Model_2")
        input_data = data_processor.get_coefficients()

        # Initialize and solve optimization model
        model = OptModel2(input_data, model_type="Model_2")
        model._build()
    
        # Retrieve results
        results = model.solve()
        flows_results_df = pd.DataFrame({
            "Bought (MWh)": results.v_bought,
            "Unmet_Demand (MWh)": results.v_unmet_demand,
            "Demand (MWh)": results.demand,
            "Stored (MWh)": results.v_stored,
            # "Price (dkk/MWh)": results.v_stored,
        }, index =pd.Index(range(len(results.v_bought)), name = "day"))

        # cost_of_unmet_demand = results.v_unmet_demand * input_data.unmet_demand_cost
        bought_dkk = results.v_bought * input_data.price
        cost_of_storage = results.v_stored * input_data.storage_cost

        allowance = results.allowance
        carry_over_budget = results.v_budget - allowance

        spending_results_df = pd.DataFrame({
            "Bought fuel": np.around(bought_dkk,1),
            # "Unmet demand (1000 DKK)": np.around(cost_of_unmet_demand,1)/1000,
            "Storage cost": cost_of_storage,
            "Carry over budget": carry_over_budget,
            "Allowance": allowance,
            "Price (dkk/MWh)": results.prices
        }, index =pd.Index(range(len(results.v_bought)), name = "day"))

        dual_results_df = pd.DataFrame({
            "Power Balance dual": results.duals.balance,
            "Max unmet demand dual": results.duals.unmet_demand_max,
            "Plant min dual": results.duals.plant_min,
            # "Ramp up dual": results.duals.ramp_up,
            # "Ramp down dual": results.duals.ramp_down,
            "Depreciation dual": results.duals.depreciation,
            "Budget": results.duals.budget
        }, index =pd.Index(range(len(results.v_bought)), name = "day"))

        budget = results.v_budget
        unmet_demand = results.obj
        return flows_results_df, unmet_demand, spending_results_df, budget, dual_results_df

