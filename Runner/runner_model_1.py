### class runner model 1
import pandas as pd
from pathlib import Path
import numpy as np
from Runner.opt_model_1 import OptModel1
from Runner.opt_model_1 import DataProcessor
from Utils.utils import *
from Data_ops.data_loader_processor import *
from Runner.runner_model_1_v2 import *

class RunnerModel1:
    """Class to run optimization model 1."""
    def __init__(self, input_path: str):
        self.input_path = input_path
        self.model_type = "Model_1"

    def run_model(self):
        # Load and process data
        data_processor = DataProcessor(self.input_path, model_type="Model_1")
        input_data = data_processor.get_coefficients()

        # Initialize and solve optimization model
        model = OptModel1(input_data, model_type="Model_1")
        model._build()
    
        # Retrieve results
        results = model.solve()
        primal_results_df = pd.DataFrame({
            "Bought (MWh)": results.v_bought,
            "Unmet_Demand (MWh)": results.v_unmet_demand,
            "Price (DKK/MWh)": results.prices,
            "Demand (MWh)": results.demand
        }, index =pd.Index(range(len(results.v_bought)), name = "day"))

        bought_dkk = results.v_bought * input_data.price
        # cost_of_storage = results.v_stored * input_data.storage_cost

        allowance = results.allowance
        # carry_over_budget = results.v_budget - allowance

        spending_results_df = pd.DataFrame({
            "Bought fuel (1000 DKK)": np.around(bought_dkk,1),
            # "Unmet demand (1000 DKK)": np.around(cost_of_unmet_demand,1)/1000,
            # "Storage(1000 dkk)": cost_of_storage,
            # "Carry over budget (1000 DKK)": carry_over_budget,
            "Allowance (1000 DKK)": allowance,
            "Price (dkk/MWh)": results.prices
        }, index =pd.Index(range(len(results.v_bought)), name = "day"))

        dual_results_df = pd.DataFrame({
            "Power Balance dual": results.duals.balance,
            "Max unmet demand dual": results.duals.unmet_demand_max,
            "Plant min dual": results.duals.plant_min,
            "Allowance dual": results.duals.allowance
        }, index =pd.Index(range(len(results.v_bought)), name = "day"))

        unmet_demand = results.obj
        return primal_results_df, unmet_demand, spending_results_df, dual_results_df

path = r"C:\Users\alex\OneDrive\Desktop\DTU\Optimistation\Optimisation_Assignment_2\Data"
primal_results, unmet_demand, spending_results, dual_results = RunnerModel1(path).run_model()

print_model_1_metrics(primal_results, spending_results, unmet_demand)

plot_model_1_energy_balance(
    primal_results=primal_results, unmet_demand=unmet_demand,
    save_path=Path(path) / "figures" / "Model_1_energy_balance"
)

plot_model_1_economics(
    primal_results=primal_results,
    spending_results=spending_results, unmet_demand=unmet_demand,
    save_path=Path(path) / "figures" / "Model_1_economics"
)



# plot_primal_results(primal_results, save_path=Path(path)/"figures"/"Model_1_primal_results", show=True, show_price_line=True, line_label="Price (DKK/MWh)",
# title=f"Primal Results for Model 1, Total Unmet Demand = {unmet_demand:.2f} MWh")
# plot_primal_results(spending_results, save_path=Path(path)/"figures"/"Model_1_spending_results", show=True, show_price_line=True, line_label="Price (dkk/MWh)",
# title=f"Spending Results for Model 2, Total Unmet Demand = {unmet_demand:.2f} MWh")
# plot_all_duals(dual_results, save_path=Path(path)/"figures"/"Model_1_dual_results", show=True)