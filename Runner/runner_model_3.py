from Data_ops.data_loader_processor import *
from Runner.opt_model_3 import OptModelCVaR
from Utils.plotting_model_3 import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# RUNNER FOR MODEL 3
# ============================================================================

class RunnerModel3nocvar:
    """Class to run optimization model 3 (stochastic with scenarios)."""
    def __init__(self, input_path: str, alpha: float = 0.8, beta: float = 0.0):
        self.input_path = input_path
        self.model_type = "Model_3"
        self.alpha = alpha
        self.beta = beta
    
    def run_model(self):
        # Load and process data
        data_processor = DataProcessor(self.input_path, model_type="Model_3")
        input_data = data_processor.get_coefficients()
        
        # Initialize and solve optimization model
        model = OptModelCVaR(input_data, model_type="Model_3", alpha=self.alpha, beta=self.beta)
        model._build()
        
        # Retrieve results
        results = model.solve()
        
        # Process results into dataframes
        # For Model 3, we have scenarios, so we'll compute statistics
        n_time = len(model.T)
        n_scenarios = len(model.S)
        
        # Calculate medians and percentiles across scenarios for each time period
        bought_median = np.median(results.v_bought, axis=1)
        bought_p5 = np.percentile(results.v_bought, 5, axis=1)
        bought_p95 = np.percentile(results.v_bought, 95, axis=1)
        
        unmet_median = np.median(results.v_unmet_demand, axis=1)
        unmet_p5 = np.percentile(results.v_unmet_demand, 5, axis=1)
        unmet_p95 = np.percentile(results.v_unmet_demand, 95, axis=1)
        
        demand_median = np.median(results.demand_scenarios, axis=0)
        demand_p5 = np.percentile(results.demand_scenarios, 5, axis=0)
        demand_p95 = np.percentile(results.demand_scenarios, 95, axis=0)
        
        stored_median = np.median(results.v_stored, axis=1)
        stored_p5 = np.percentile(results.v_stored, 5, axis=1)
        stored_p95 = np.percentile(results.v_stored, 95, axis=1)
        
        budget_median = np.median(results.v_budget, axis=1)
        
        # Flow results (medians and percentiles)
        flow_results_df = pd.DataFrame({
            "Bought_median (MWh)": bought_median,
            "Bought_p5 (MWh)": bought_p5,
            "Bought_p95 (MWh)": bought_p95,
            "Unmet_Demand_median (MWh)": unmet_median,
            "Unmet_Demand_p5 (MWh)": unmet_p5,
            "Unmet_Demand_p95 (MWh)": unmet_p95,
            "Demand_median (MWh)": demand_median,
            "Demand_p5 (MWh)": demand_p5,
            "Demand_p95 (MWh)": demand_p95,
            "Stored_median (MWh)": stored_median,
            "Stored_p5 (MWh)": stored_p5,
            "Stored_p95 (MWh)": stored_p95,
        }, index=pd.Index(range(n_time), name="day"))
        
        # Calculate spending for each scenario
        bought_fuel_all = results.v_bought * input_data.price.reshape(-1, 1)
        storage_cost_all = results.v_stored * input_data.storage_cost
        
        bought_fuel_median = np.median(bought_fuel_all, axis=1)
        bought_fuel_p5 = np.percentile(bought_fuel_all, 5, axis=1)
        bought_fuel_p95 = np.percentile(bought_fuel_all, 95, axis=1)
        
        storage_cost_median = np.median(storage_cost_all, axis=1)
        storage_cost_p5 = np.percentile(storage_cost_all, 5, axis=1)
        storage_cost_p95 = np.percentile(storage_cost_all, 95, axis=1)
        
        carry_over_median = budget_median - input_data.exp_allowance
        allowance = input_data.exp_allowance
        
        spending_results_df = pd.DataFrame({
            "Bought_fuel_median": bought_fuel_median,
            "Bought_fuel_p5": bought_fuel_p5,
            "Bought_fuel_p95": bought_fuel_p95,
            "Storage_cost_median": storage_cost_median,
            "Storage_cost_p5": storage_cost_p5,
            "Storage_cost_p95": storage_cost_p95,
            "Carry_over_median": carry_over_median,
            "Allowance": allowance,
            "Price (dkk/MWh)": results.prices
        }, index=pd.Index(range(n_time), name="day"))
        
        # Calculate total unmet demand for each scenario (for risk distribution)
        total_unmet_per_scenario = np.sum(results.v_unmet_demand, axis=0)
        
        # Store statistics
        unmet_demand_mean = np.mean(total_unmet_per_scenario)
        unmet_demand_median = np.median(total_unmet_per_scenario)
        unmet_demand_std = np.std(total_unmet_per_scenario)
        
        risk_stats = {
            'total_unmet_per_scenario': total_unmet_per_scenario,
            'mean': unmet_demand_mean,
            'median': unmet_demand_median,
            'std': unmet_demand_std,
            'min': np.min(total_unmet_per_scenario),
            'max': np.max(total_unmet_per_scenario),
            'p5': np.percentile(total_unmet_per_scenario, 5),
            'p25': np.percentile(total_unmet_per_scenario, 25),
            'p75': np.percentile(total_unmet_per_scenario, 75),
            'p95': np.percentile(total_unmet_per_scenario, 95)
        }
        
        return flow_results_df, spending_results_df, risk_stats, results
    
path = r"C:\Users\alex\OneDrive\Desktop\DTU\Optimistation\Optimisation_Assignment_2\Data"

# Run model (beta=0 means no CVaR, just expected value)
flow_results, spending_results, risk_stats, full_results = RunnerModel3nocvar(path, alpha=0.95, beta=0.0).run_model()

# Print metrics
print_model_3_metrics(flow_results, spending_results, risk_stats)

# Create plots
plot_model_3_energy_balance(
    flow_results=flow_results,
    save_path=Path(path) / "figures" / "Model_3_energy_balance"
)

plot_model_3_economics(
    spending_results=spending_results,
    risk_stats=risk_stats,
    save_path=Path(path) / "figures" / "Model_3_economics"
)

plot_model_3_risk_distribution(
    risk_stats=risk_stats,
    save_path=Path(path) / "figures" / "Model_3_risk_distribution"
)