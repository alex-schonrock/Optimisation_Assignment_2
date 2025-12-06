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
        
        # NEW: Calculate price statistics across scenarios
        price_median = np.median(results.price_scenarios, axis=0)
        price_p5 = np.percentile(results.price_scenarios, 5, axis=0)
        price_p95 = np.percentile(results.price_scenarios, 95, axis=0)
        
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
        
        # UPDATED: Calculate spending with scenario-dependent prices
        # bought_fuel_all is now (time × scenarios) using price_scenarios
        bought_fuel_all = results.v_bought * results.price_scenarios.T  # Transpose to match dimensions
        storage_cost_all = results.v_stored * input_data.storage_cost
        
        bought_fuel_median = np.median(bought_fuel_all, axis=1)
        bought_fuel_p5 = np.percentile(bought_fuel_all, 5, axis=1)
        bought_fuel_p95 = np.percentile(bought_fuel_all, 95, axis=1)
        
        storage_cost_median = np.median(storage_cost_all, axis=1)
        storage_cost_p5 = np.percentile(storage_cost_all, 5, axis=1)
        storage_cost_p95 = np.percentile(storage_cost_all, 95, axis=1)
        
        carry_over_median = budget_median - input_data.exp_allowance
        allowance = input_data.exp_allowance
        
        # UPDATED: Include price statistics in spending results
        spending_results_df = pd.DataFrame({
            "Bought_fuel_median": bought_fuel_median,
            "Bought_fuel_p5": bought_fuel_p5,
            "Bought_fuel_p95": bought_fuel_p95,
            "Storage_cost_median": storage_cost_median,
            "Storage_cost_p5": storage_cost_p5,
            "Storage_cost_p95": storage_cost_p95,
            "Carry_over_median": carry_over_median,
            "Allowance": allowance,
            "Price_median (dkk/MWh)": price_median,
            "Price_p5 (dkk/MWh)": price_p5,
            "Price_p95 (dkk/MWh)": price_p95,
        }, index=pd.Index(range(n_time), name="day"))
        
        # Calculate total unmet demand for each scenario (for risk distribution)
        total_unmet_per_scenario = np.sum(results.v_unmet_demand, axis=0)
        
        # NEW: Calculate total cost per scenario (with scenario-dependent prices)
        total_cost_per_scenario = np.sum(bought_fuel_all, axis=0) + \
                                  np.sum(storage_cost_all, axis=0)
        
        # Store statistics
        unmet_demand_mean = np.mean(total_unmet_per_scenario)
        unmet_demand_median = np.median(total_unmet_per_scenario)
        unmet_demand_std = np.std(total_unmet_per_scenario)
        
        risk_stats = {
            'total_unmet_per_scenario': total_unmet_per_scenario,
            'total_cost_per_scenario': total_cost_per_scenario,  # NEW
            'mean': unmet_demand_mean,
            'median': unmet_demand_median,
            'std': unmet_demand_std,
            'min': np.min(total_unmet_per_scenario),
            'max': np.max(total_unmet_per_scenario),
            'p5': np.percentile(total_unmet_per_scenario, 5),
            'p25': np.percentile(total_unmet_per_scenario, 25),
            'p75': np.percentile(total_unmet_per_scenario, 75),
            'p95': np.percentile(total_unmet_per_scenario, 95),
            'cost_mean': np.mean(total_cost_per_scenario),  # NEW
            'cost_std': np.std(total_cost_per_scenario),    # NEW
            'cost_max': np.max(total_cost_per_scenario),    # NEW
            'cost_p95': np.percentile(total_cost_per_scenario, 95)  # NEW
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

# flow_results_beta07, spending_results_beta07, risk_beta07, results_beta07 = RunnerModel3nocvar(path, alpha=0.9, beta=0.7).run_model()

# plot_model_3_risk_distribution(
#     risk_stats=results_beta07,
#     save_path=Path(path) / "figures" / "Model_3_risk_distribution_beta05"
# )

# # Run model with β=0.7
# path = r"C:\Users\alex\OneDrive\Desktop\DTU\Optimistation\Optimisation_Assignment_2\Data"

# print("Running Model 3 with β=0.7 (CVaR-focused)...")
# flow_beta07, spending_beta07, risk_beta07, results_beta07 = \
#     RunnerModel3_CostBased_Original(path, alpha=0.85, beta=0.7).run_model()

# Create CVaR-specific plot
# ============================================================================
# COMPREHENSIVE CVAR PARAMETER SENSITIVITY ANALYSIS
# ============================================================================

import numpy as np
import pandas as pd
from pathlib import Path

def comprehensive_cvar_experiment(input_path: str):
    """
    Test multiple combinations of alpha and beta to see if CVaR produces
    different operational decisions.
    """
    
    # Parameter ranges to test
    beta_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    alpha_values = [0.75, 0.85, 0.90, 0.95]  # Different confidence levels
    
    print("="*80)
    print("COMPREHENSIVE CVAR PARAMETER SENSITIVITY ANALYSIS")
    print("="*80)
    print(f"\nTesting {len(beta_values)} beta values × {len(alpha_values)} alpha values")
    print(f"Total runs: {len(beta_values) * len(alpha_values)}")
    print(f"\nBeta values: {beta_values}")
    print(f"Alpha values: {alpha_values}")
    print("\n" + "="*80)
    
    # Store all results
    all_results = []
    
    # Run all combinations
    for alpha in alpha_values:
        for beta in beta_values:
            print(f"\n{'='*80}")
            print(f"Running: α={alpha}, β={beta}")
            print(f"{'='*80}")
            
            # Run model
            runner = RunnerModel3nocvar(input_path, alpha=alpha, beta=beta)
            flow_df, spending_df, risk_stats, results = runner.run_model()
            
            # Extract key metrics
            total_unmet = risk_stats['total_unmet_per_scenario']
            
            # Calculate CVaR manually
            tail_threshold = np.percentile(total_unmet, alpha * 100)
            tail_scenarios = total_unmet[total_unmet >= tail_threshold]
            cvar_value = np.mean(tail_scenarios)
            
            # Store comprehensive results
            result_dict = {
                'alpha': alpha,
                'beta': beta,
                'objective': results.obj,
                'var_z': results.z,
                'cvar_calculated': cvar_value,
                'mean_unmet': risk_stats['mean'],
                'median_unmet': risk_stats['median'],
                'std_unmet': risk_stats['std'],
                'min_unmet': risk_stats['min'],
                'max_unmet': risk_stats['max'],
                'p5_unmet': risk_stats['p5'],
                'p25_unmet': risk_stats['p25'],
                'p75_unmet': risk_stats['p75'],
                'p95_unmet': risk_stats['p95'],
                'total_bought': np.sum(results.v_bought),
                'mean_bought_per_scenario': np.mean(np.sum(results.v_bought, axis=0)),
                'std_bought_per_scenario': np.std(np.sum(results.v_bought, axis=0)),
                'total_stored': np.sum(results.v_stored),
                'mean_stored_per_scenario': np.mean(np.sum(results.v_stored, axis=0)),
                'unmet_array_hash': hash(total_unmet.tobytes()),  # Hash to detect identical arrays
                'bought_array_hash': hash(results.v_bought.tobytes()),
                'stored_array_hash': hash(results.v_stored.tobytes())
            }
            
            all_results.append(result_dict)
            
            # Quick print
            print(f"  Objective: {results.obj:.4f}")
            print(f"  VaR: {results.z:.4f}")
            print(f"  CVaR: {cvar_value:.4f}")
            print(f"  Mean unmet: {risk_stats['mean']:.4f}")
            print(f"  Max unmet: {risk_stats['max']:.4f}")
    
    # Create comprehensive results dataframe
    results_df = pd.DataFrame(all_results)
    
    print("\n" + "="*80)
    print("ANALYSIS: SEARCHING FOR DIFFERENCES")
    print("="*80)
    
    # ========================================================================
    # ANALYSIS 1: Check if any decision variables differ
    # ========================================================================
    print("\n1. DECISION VARIABLE UNIQUENESS CHECK")
    print("-" * 80)
    
    unique_unmet_hashes = results_df['unmet_array_hash'].nunique()
    unique_bought_hashes = results_df['bought_array_hash'].nunique()
    unique_stored_hashes = results_df['stored_array_hash'].nunique()
    
    total_runs = len(results_df)
    
    print(f"\nOut of {total_runs} runs:")
    print(f"  Unique unmet demand solutions: {unique_unmet_hashes}")
    print(f"  Unique bought solutions: {unique_bought_hashes}")
    print(f"  Unique stored solutions: {unique_stored_hashes}")
    
    if unique_unmet_hashes == 1:
        print("\n  ⚠️  ALL unmet demand solutions are IDENTICAL")
    else:
        print(f"\n  ✅ Found {unique_unmet_hashes} different unmet demand solutions!")
    
    if unique_bought_hashes == 1:
        print("  ⚠️  ALL bought solutions are IDENTICAL")
    else:
        print(f"  ✅ Found {unique_bought_hashes} different bought solutions!")
    
    if unique_stored_hashes == 1:
        print("  ⚠️  ALL stored solutions are IDENTICAL")
    else:
        print(f"  ✅ Found {unique_stored_hashes} different stored solutions!")
    
    # ========================================================================
    # ANALYSIS 2: Check metrics variation within each alpha
    # ========================================================================
    print("\n\n2. METRICS VARIATION WITHIN EACH ALPHA")
    print("-" * 80)
    
    for alpha in alpha_values:
        alpha_results = results_df[results_df['alpha'] == alpha]
        
        print(f"\nα = {alpha}:")
        print(f"  {'Metric':<25} {'Min':>12} {'Max':>12} {'Range':>12} {'Varies?':>10}")
        print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")
        
        metrics = ['objective', 'var_z', 'cvar_calculated', 'mean_unmet', 
                   'median_unmet', 'std_unmet', 'max_unmet', 'p95_unmet']
        
        for metric in metrics:
            min_val = alpha_results[metric].min()
            max_val = alpha_results[metric].max()
            range_val = max_val - min_val
            varies = "YES" if range_val > 1e-6 else "NO"
            
            print(f"  {metric:<25} {min_val:>12.4f} {max_val:>12.4f} {range_val:>12.4f} {varies:>10}")
    
    # ========================================================================
    # ANALYSIS 3: Beta effect within each alpha
    # ========================================================================
    print("\n\n3. BETA EFFECT ANALYSIS (β=0 vs β=1 for each α)")
    print("-" * 80)
    
    for alpha in alpha_values:
        beta0 = results_df[(results_df['alpha'] == alpha) & (results_df['beta'] == 0.0)].iloc[0]
        beta1 = results_df[(results_df['alpha'] == alpha) & (results_df['beta'] == 1.0)].iloc[0]
        
        print(f"\nα = {alpha}:")
        print(f"  {'Metric':<25} {'β=0':>12} {'β=1':>12} {'Δ':>12} {'% Change':>12}")
        print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
        
        for metric in ['objective', 'var_z', 'cvar_calculated', 'mean_unmet', 'max_unmet', 'p95_unmet']:
            val0 = beta0[metric]
            val1 = beta1[metric]
            delta = val1 - val0
            pct_change = (delta / val0 * 100) if val0 != 0 else 0
            
            print(f"  {metric:<25} {val0:>12.4f} {val1:>12.4f} {delta:>12.4f} {pct_change:>11.2f}%")
    
    # ========================================================================
    # ANALYSIS 4: Alpha effect within beta=0 and beta=1
    # ========================================================================
    print("\n\n4. ALPHA EFFECT ANALYSIS")
    print("-" * 80)
    
    for beta in [0.0, 1.0]:
        print(f"\nβ = {beta}:")
        beta_results = results_df[results_df['beta'] == beta]
        
        print(f"  {'α':<10} {'Objective':>12} {'VaR':>12} {'CVaR':>12} {'Mean':>12} {'Max':>12}")
        print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
        
        for _, row in beta_results.iterrows():
            print(f"  {row['alpha']:<10.2f} {row['objective']:>12.4f} {row['var_z']:>12.4f} "
                  f"{row['cvar_calculated']:>12.4f} {row['mean_unmet']:>12.4f} {row['max_unmet']:>12.4f}")
    
    # ========================================================================
    # ANALYSIS 5: Check if distribution shapes differ
    # ========================================================================
    print("\n\n5. DISTRIBUTION STATISTICS COMPARISON")
    print("-" * 80)
    
    print(f"\n{'α':<8} {'β':<8} {'Mean':>10} {'Median':>10} {'Std':>10} {'Skew*':>10} "
          f"{'P5':>10} {'P95':>10} {'Range':>10}")
    print("-" * 90)
    
    for _, row in results_df.iterrows():
        # Approximate skewness from percentiles
        skew_proxy = (row['p95_unmet'] - row['median_unmet']) - (row['median_unmet'] - row['p5_unmet'])
        
        print(f"{row['alpha']:<8.2f} {row['beta']:<8.1f} {row['mean_unmet']:>10.2f} "
              f"{row['median_unmet']:>10.2f} {row['std_unmet']:>10.2f} {skew_proxy:>10.2f} "
              f"{row['p5_unmet']:>10.2f} {row['p95_unmet']:>10.2f} "
              f"{row['max_unmet'] - row['min_unmet']:>10.2f}")
    
    print("\n*Skew proxy = (P95-Median) - (Median-P5), positive = right skewed")
    
    # ========================================================================
    # ANALYSIS 6: Pairwise comparisons
    # ========================================================================
    print("\n\n6. DETAILED PAIRWISE COMPARISONS")
    print("-" * 80)
    
    print("\nComparing all pairs to find ANY differences...")
    found_difference = False
    
    for i in range(len(results_df)):
        for j in range(i+1, len(results_df)):
            row_i = results_df.iloc[i]
            row_j = results_df.iloc[j]
            
            # Check if decision variables differ
            if row_i['unmet_array_hash'] != row_j['unmet_array_hash']:
                print(f"\n  ✅ FOUND DIFFERENCE!")
                print(f"     Config 1: α={row_i['alpha']:.2f}, β={row_i['beta']:.1f}")
                print(f"     Config 2: α={row_j['alpha']:.2f}, β={row_j['beta']:.1f}")
                print(f"     Mean unmet: {row_i['mean_unmet']:.4f} vs {row_j['mean_unmet']:.4f}")
                print(f"     Max unmet: {row_i['max_unmet']:.4f} vs {row_j['max_unmet']:.4f}")
                print(f"     CVaR: {row_i['cvar_calculated']:.4f} vs {row_j['cvar_calculated']:.4f}")
                found_difference = True
                break
        
        if found_difference:
            break
    
    if not found_difference:
        print("\n  ⚠️  NO DIFFERENCES found between any parameter combinations")
        print("     All decision variables are identical across all α and β values")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)
    
    if unique_unmet_hashes > 1 or unique_bought_hashes > 1 or unique_stored_hashes > 1:
        print("\n✅ CVaR IS WORKING - Different parameters produce different solutions!")
        print(f"\n   Found {max(unique_unmet_hashes, unique_bought_hashes, unique_stored_hashes)} "
              f"distinct solutions out of {total_runs} runs")
    else:
        print("\n⚠️  CVaR IS NOT WORKING - All parameters produce identical solutions")
        print("\n   Likely causes:")
        print("   1. Model constraints are too tight (over-constrained)")
        print("   2. There exists only one feasible solution")
        print("   3. Budget/ramp rate/storage limits force unique strategy")
        print("\n   The CVaR formulation may be correct, but constraints prevent")
        print("   the optimizer from finding different risk/return trade-offs.")
    
    print("\n" + "="*80)
    
    # Save results to CSV
    output_path = Path(input_path) / "cvar_sensitivity_analysis.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to: {output_path}")
    
    return results_df


# ============================================================================
# RUN THE EXPERIMENT
# ============================================================================

if __name__ == "__main__":
    path = r"C:\Users\alex\OneDrive\Desktop\DTU\Optimistation\Optimisation_Assignment_2\Data"
    
    results_df = comprehensive_cvar_experiment(path)
    
    print("\n\nExperiment complete! Check the output above for detailed analysis.")