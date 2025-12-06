from Data_ops.data_loader_processor import *
from Runner.opt_model_3 import *
# from Utils.plotting_model_3 import *
from Utils.model_3_plotting import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from Utils.utils_and_plotting import *
from Runner.runner_model_2 import RunnerModel2
# ============================================================================
# RUNNER
# ============================================================================

class RunnerModel3_IndependentStorageBudget:
    """Class to run optimization model 3 with scenario-independent storage and budget."""
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
        model = OptModelCVaR_IndependentStorageBudget(input_data, model_type="Model_3", 
                                                      alpha=self.alpha, beta=self.beta)
        model._build()
        
        # Retrieve results
        results = model.solve()
        
        # Process results
        n_time = len(model.T)
        n_scenarios = len(model.S)
        
        # Calculate medians and percentiles across scenarios
        bought_median = np.median(results.v_bought, axis=1)
        bought_p5 = np.percentile(results.v_bought, 5, axis=1)
        bought_p95 = np.percentile(results.v_bought, 95, axis=1)
        
        unmet_median = np.median(results.v_unmet_demand, axis=1)
        unmet_p5 = np.percentile(results.v_unmet_demand, 5, axis=1)
        unmet_p95 = np.percentile(results.v_unmet_demand, 95, axis=1)
        
        demand_median = np.median(results.demand_scenarios, axis=0)
        demand_p5 = np.percentile(results.demand_scenarios, 5, axis=0)
        demand_p95 = np.percentile(results.demand_scenarios, 95, axis=0)
        
        # Storage and budget are deterministic (same across scenarios)
        stored_value = results.v_stored[:, 0]  # Take from any scenario (all identical)
        budget_value = results.v_budget[:, 0]  # Take from any scenario (all identical)
        
        # Price statistics
        price_mean = np.mean(results.price_scenarios, axis=0)
        price_median = np.median(results.price_scenarios, axis=0)
        price_p5 = np.percentile(results.price_scenarios, 5, axis=0)
        price_p95 = np.percentile(results.price_scenarios, 95, axis=0)
        
        # Flow results
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
            "Stored (MWh)": stored_value,  # Deterministic
        }, index=pd.Index(range(n_time), name="day"))
        
        # Calculate spending with scenario-dependent prices
        bought_fuel_all = results.v_bought * results.price_scenarios.T
        storage_cost_all = results.v_stored * input_data.storage_cost
        
        bought_fuel_median = np.median(bought_fuel_all, axis=1)
        bought_fuel_p5 = np.percentile(bought_fuel_all, 5, axis=1)
        bought_fuel_p95 = np.percentile(bought_fuel_all, 95, axis=1)
        
        storage_cost_median = np.median(storage_cost_all, axis=1)
        storage_cost_p5 = np.percentile(storage_cost_all, 5, axis=1)
        storage_cost_p95 = np.percentile(storage_cost_all, 95, axis=1)
        
        carry_over = budget_value - input_data.exp_allowance
        allowance = input_data.exp_allowance
        
        # Spending results
        spending_results_df = pd.DataFrame({
            "Bought_fuel_median": bought_fuel_median,
            "Bought_fuel_p5": bought_fuel_p5,
            "Bought_fuel_p95": bought_fuel_p95,
            "Storage_cost_median": storage_cost_median,
            "Storage_cost_p5": storage_cost_p5,
            "Storage_cost_p95": storage_cost_p95,
            "Carry_over": carry_over,  # Deterministic
            "Allowance": allowance,
            "Budget": budget_value,  # Deterministic
            "Price_mean (dkk/MWh)": price_mean,
            "Price_median (dkk/MWh)": price_median,
            "Price_p5 (dkk/MWh)": price_p5,
            "Price_p95 (dkk/MWh)": price_p95,
        }, index=pd.Index(range(n_time), name="day"))
        
        # Risk statistics
        total_unmet_per_scenario = np.sum(results.v_unmet_demand, axis=0)
        total_cost_per_scenario = np.sum(bought_fuel_all, axis=0) + np.sum(storage_cost_all, axis=0)
        
        risk_stats = {
            'total_unmet_per_scenario': total_unmet_per_scenario,
            'total_cost_per_scenario': total_cost_per_scenario,
            'mean': np.mean(total_unmet_per_scenario),
            'median': np.median(total_unmet_per_scenario),
            'std': np.std(total_unmet_per_scenario),
            'min': np.min(total_unmet_per_scenario),
            'max': np.max(total_unmet_per_scenario),
            'p5': np.percentile(total_unmet_per_scenario, 5),
            'p25': np.percentile(total_unmet_per_scenario, 25),
            'p75': np.percentile(total_unmet_per_scenario, 75),
            'p95': np.percentile(total_unmet_per_scenario, 95),
            'cost_mean': np.mean(total_cost_per_scenario),
            'cost_std': np.std(total_cost_per_scenario),
            'cost_max': np.max(total_cost_per_scenario),
            'cost_p95': np.percentile(total_cost_per_scenario, 95)
        }
        
        return flow_results_df, spending_results_df, risk_stats, results



