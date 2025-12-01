import pandas as pd
from pathlib import Path
import numpy as np
from Runner.opt_model_2 import OptModel2
from Runner.opt_model_1 import DataProcessor
from Utils.utils import plot_primal_results

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
        primal_results_df = pd.DataFrame({
            "Bought (Btu)": results.v_bought,
            "Unmet_Demand (Btu)": results.v_unmet_demand,
            "Price (DKK/Btu)": results.prices,
            "Demand (Btu)": results.demand,
            "Stored (Btu)": results.v_stored,
            "Budget (DKK)": results.v_budget
        }, index =pd.Index(range(len(results.v_bought)), name = "day"))

        expenditure = results.obj
        return primal_results_df, expenditure

path = r"C:\Users\alex\OneDrive\Desktop\DTU\Optimistation\Optimisation_Assignment_2\Data"
primal_results, expenditure = RunnerModel2(path).run_model()
print("Primal Results:\n", primal_results)
print("Total Expenditure:", expenditure)