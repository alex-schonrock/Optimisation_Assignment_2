"""
Main execution file for all optimization models.
Runs Model 1, Model 2, and Model 3 with various analyses and comparisons.
"""

from pathlib import Path
from Runner.runner_model_1 import RunnerModel1
from Runner.runner_model_2 import RunnerModel2
from Runner.runner_model_3 import RunnerModel3_IndependentStorageBudget
from Utils.utils_and_plotting import *
from Utils.model_3_plotting import *


# def run_model_1_analysis(path):
#     """Run Model 1 analysis with all visualizations."""
#     print("\n" + "="*80)
#     print("MODEL 1: DETERMINISTIC WITHOUT STORAGE OR BUDGET")
#     print("="*80)
    
#     # Run model
#     primal_results, unmet_demand, spending_results, dual_results = RunnerModel1(path).run_model()
    
#     # Print metrics
#     print_model_1_metrics(primal_results, spending_results, unmet_demand)
    
#     # Create visualizations
#     plot_model_1_energy_balance(
#         primal_results=primal_results, 
#         unmet_demand=unmet_demand,
#         save_path=Path(path) / "figures" / "Model_1_energy_balance"
#     )
    
#     plot_model_1_economics(
#         primal_results=primal_results,
#         spending_results=spending_results, 
#         unmet_demand=unmet_demand,
#         save_path=Path(path) / "figures" / "Model_1_economics"
#     )
    
#     print("\nModel 1 analysis complete!")
#     return primal_results, unmet_demand, spending_results, dual_results


# def run_model_2_analysis(path):
#     """Run Model 2 analysis with all visualizations."""
#     print("\n" + "="*80)
#     print("MODEL 2: DETERMINISTIC WITH STORAGE AND BUDGET")
#     print("="*80)
    
#     # Run model
#     flow_results, unmet_demand, spending_results, budget, dual_results = RunnerModel2(path).run_model()
    
#     # Print metrics
#     print_model_2_metrics(flow_results, spending_results, budget, unmet_demand)
    
#     plot_model_2_energy_balance(
#         flow_results=flow_results,
#         spending_results=spending_results,
#         unmet_demand=unmet_demand,
#         save_path=Path(path) / "figures" / "model2_energy_balance"
#     )
    
#     plot_model_2_economics(
#         spending_results=spending_results, 
#         unmet_demand=unmet_demand,
#         save_path=Path(path) / "figures" / "Model_2_economics"
#     )
    
#     print("\nModel 2 analysis complete!")
#     return flow_results, unmet_demand, spending_results, budget, dual_results


# def run_model_2_vs_model_3_comparison(path, alpha=0.9, beta=0.3):
#     """Run comparison between Model 2 and Model 3 with specified beta."""
#     print("\n" + "="*80)
#     print("MODEL 2 vs MODEL 3 COMPARISON")
#     print("="*80)
    
#     # Run Model 2
#     print("\nRunning Model 2...")
#     flow_results_m2, unmet_demand_m2, spending_results_m2, budget_m2, dual_results_m2 = RunnerModel2(path).run_model()
    
#     # Run Model 3
#     print(f"\nRunning Model 3 with α={alpha:.2f}, β={beta:.2f}...")
#     flow_results_m3, spending_results_m3, risk_stats_m3, results_m3 = RunnerModel3_IndependentStorageBudget(
#         path, alpha=alpha, beta=beta).run_model()
    
#     # Print statistics for both
#     print("\n" + "="*80)
#     print("MODEL 2 METRICS:")
#     print("="*80)
#     metrics_m2 = print_model_2_metrics(flow_results_m2, spending_results_m2, budget_m2, unmet_demand_m2)
    
#     print("\n" + "="*80)
#     print("MODEL 3 METRICS:")
#     print("="*80)
#     metrics_m3 = print_model_3_metrics(
#         flow_results=flow_results_m3,
#         spending_results=spending_results_m3,
#         risk_stats=risk_stats_m3,
#         alpha=alpha,
#         beta=beta
#     )
    
#     # Create overlay visualization
#     plot_model2_model3_comparison(
#         flow_results_m2=flow_results_m2,
#         spending_results_m2=spending_results_m2,
#         unmet_demand_m2=unmet_demand_m2,
#         flow_results_m3=flow_results_m3,
#         spending_results_m3=spending_results_m3,
#         risk_stats_m3=risk_stats_m3,
#         alpha=alpha,
#         beta=beta,
#         save_path=Path(path) / "figures" / f"model2_vs_model3_procurement_plan_beta_{beta:.2f}"
#     )

#     print("\nCreating Model 2 vs Model 3 risk distribution comparison...")
#     compare_model2_model3_risk(
#         unmet_m2=unmet_demand_m2,
#         risk_stats_m3=risk_stats_m3,
#         alpha=alpha,
#         beta=beta,
#         save_path=Path(path) / "figures" / f"model2_vs_model3_risk_distribution_beta_{beta:.2f}"
#     )
    
#     print("\n" + "="*80)
#     print("Model 2 vs Model 3 comparison complete!")
#     print("="*80)
    
#     return (flow_results_m2, spending_results_m2, unmet_demand_m2, 
#             flow_results_m3, spending_results_m3, risk_stats_m3)


# def run_three_way_comparison(path, alpha=0.9, beta_1=0.3, beta_2=0.0):
#     """Run three-way comparison: Model 2 vs Model 3 (two beta values)."""
#     print("\n" + "="*80)
#     print("THREE-WAY COMPARISON: Model 2 vs Model 3 (two β values)")
#     print("="*80)
    
#     # Run Model 2
#     print("\nRunning Model 2...")
#     flow_results_m2, unmet_demand_m2, spending_results_m2, budget_m2, dual_results_m2 = RunnerModel2(path).run_model()
    
#     # Run Model 3 with beta_1
#     print(f"\nRunning Model 3 with α={alpha:.2f}, β={beta_1:.2f}...")
#     flow_results_m3_1, spending_results_m3_1, risk_stats_m3_1, results_m3_1 = RunnerModel3_IndependentStorageBudget(
#         path, alpha=alpha, beta=beta_1).run_model()
    
#     # Run Model 3 with beta_2
#     print(f"\nRunning Model 3 with α={alpha:.2f}, β={beta_2:.2f}...")
#     flow_results_m3_2, spending_results_m3_2, risk_stats_m3_2, results_m3_2 = RunnerModel3_IndependentStorageBudget(
#         path, alpha=alpha, beta=beta_2).run_model()
    
#     # Print metrics for all three
#     print("\n" + "="*80)
#     print("MODEL 2 METRICS:")
#     print("="*80)
#     metrics_m2 = print_model_2_metrics(flow_results_m2, spending_results_m2, budget_m2, unmet_demand_m2)
    
#     print("\n" + "="*80)
#     print(f"MODEL 3 METRICS (β={beta_2:.2f}):")
#     print("="*80)
#     metrics_m3_2 = print_model_3_metrics(
#         flow_results=flow_results_m3_2,
#         spending_results=spending_results_m3_2,
#         risk_stats=risk_stats_m3_2,
#         alpha=alpha,
#         beta=beta_2
#     )
    
#     print("\n" + "="*80)
#     print(f"MODEL 3 METRICS (β={beta_1:.2f}):")
#     print("="*80)
#     metrics_m3_1 = print_model_3_metrics(
#         flow_results=flow_results_m3_1,
#         spending_results=spending_results_m3_1,
#         risk_stats=risk_stats_m3_1,
#         alpha=alpha,
#         beta=beta_1
#     )
    
#     # Create three-way comparison visualization
#     compare_three_procurement_strategies(
#         flow_m2=flow_results_m2,
#         spending_m2=spending_results_m2,
#         flow_m3_0=flow_results_m3_2,  # β=beta_2
#         spending_m3_0=spending_results_m3_2,
#         flow_m3_1=flow_results_m3_1,  # β=beta_1
#         spending_m3_1=spending_results_m3_1,
#         alpha=alpha,
#         beta_0=beta_2,
#         beta_1=beta_1,
#         save_path=Path(path) / "figures" / f"three_way_comparison_alpha_{alpha:.2f}_beta_{beta_2:.2f}_vs_{beta_1:.2f}"
#     )
    
#     print("\n" + "="*80)
#     print("Three-way comparison complete!")
#     print("="*80)

# def run_model_3_beta_comparison(path, alpha=0.9, beta_0=0.0, beta_1=0.15):
#     """Run comparison between two Model 3 variants with different beta values."""
#     print("\n" + "="*80)
#     print(f"MODEL 3 COMPARISON: β={beta_0:.2f} vs β={beta_1:.2f} (α={alpha:.2f})")
#     print("="*80)
    
#     # Run Model 3 with beta_0
#     print(f"\nRunning Model 3 with α={alpha:.2f}, β={beta_0:.2f}...")
#     flow_0, spending_0, risk_0, res_0 = RunnerModel3_IndependentStorageBudget(
#         path, alpha=alpha, beta=beta_0).run_model()
    
#     # Run Model 3 with beta_1
#     print(f"\nRunning Model 3 with α={alpha:.2f}, β={beta_1:.2f}...")
#     flow_1, spending_1, risk_1, res_1 = RunnerModel3_IndependentStorageBudget(
#         path, alpha=alpha, beta=beta_1).run_model()
    
#     # # Print metrics for both
#     # print("\n" + "="*80)
#     # print(f"MODEL 3 METRICS (β={beta_0:.2f}):")
#     # print("="*80)
#     # metrics_0 = print_model_3_metrics(
#     #     flow_results=flow_0,
#     #     spending_results=spending_0,
#     #     risk_stats=risk_0,
#     #     alpha=alpha,
#     #     beta=beta_0
#     # )
    
#     # print("\n" + "="*80)
#     # print(f"MODEL 3 METRICS (β={beta_1:.2f}):")
#     # print("="*80)
#     # metrics_1 = print_model_3_metrics(
#     #     flow_results=flow_1,
#     #     spending_results=spending_1,
#     #     risk_stats=risk_1,
#     #     alpha=alpha,
#     #     beta=beta_1
#     # )
    
#     # Create risk distribution comparison
#     print("\nCreating risk distribution comparison...")
#     compare_risk_distributions(
#         risk_stats_0=risk_0,
#         risk_stats_1=risk_1,
#         alpha=alpha,
#         beta_0=beta_0,
#         beta_1=beta_1,
#         save_path=Path(path) / "figures" / f"risk_comparison_alpha_{alpha:.2f}_beta_{beta_0:.2f}_vs_{beta_1:.2f}"
#     )
    
    
#     print("\n" + "="*80)
#     print(f"Model 3 comparison (β={beta_0:.2f} vs β={beta_1:.2f}) complete!")
#     print("="*80)
    
#     return flow_0, spending_0, risk_0, flow_1, spending_1, risk_1


# def main():
#     """Main execution function - runs all analyses."""
#     # Set data path
#     path = r"C:\Users\alex\OneDrive\Desktop\DTU\Optimistation\Optimisation_Assignment_2\Data"
    
#     # Configuration
#     alpha_val = 0.9
#     beta_val_0 = 0.3
#     beta_val_1 = 0
#     # beta_val_2 = 0.3
    
#     print("\n" + "="*80)
#     print("RUNNING ALL MODEL ANALYSES")
#     print("="*80)
    
#     # Run Model 1
#     run_model_1_analysis(path)
    
#     # Run Model 2
#     run_model_2_analysis(path)
    
#     # Run Model 2 vs Model 3 comparison (with β is val 0)
#     run_model_2_vs_model_3_comparison(path, alpha=alpha_val, beta=beta_val_0)
    
#     # Run three-way comparison (Model 2 vs Model 3 with two betas)
#     run_three_way_comparison(path, alpha=alpha_val, beta_1=beta_val_0, beta_2=beta_val_1)

#     # Run Model 3 beta comparison (β val 0 vs β val 1)
#     run_model_3_beta_comparison(path, alpha=alpha_val, beta_0=beta_val_0, beta_1=beta_val_1)
    
#     print("\n" + "="*80)
#     print("ALL ANALYSES COMPLETE!")
#     print("="*80)


# if __name__ == "__main__":
#     main()

def run_model_1_analysis(path):
    """Run Model 1 analysis with all visualizations."""
    print("\n" + "="*80)
    print("MODEL 1: DETERMINISTIC WITHOUT STORAGE OR BUDGET")
    print("="*80)
    
    # Run model
    primal_results, unmet_demand, spending_results, dual_results = RunnerModel1(path).run_model()
    
    # Print metrics
    print_model_1_metrics(primal_results, spending_results, unmet_demand)
    
    # Create visualizations
    plot_model_1_energy_balance(
        primal_results=primal_results, 
        unmet_demand=unmet_demand,
        save_path=Path(path) / "figures" / "Model_1_energy_balance"
    )
    
    plot_model_1_economics(
        primal_results=primal_results,
        spending_results=spending_results, 
        unmet_demand=unmet_demand,
        save_path=Path(path) / "figures" / "Model_1_economics"
    )
    
    print("\nModel 1 analysis complete!")
    return primal_results, unmet_demand, spending_results, dual_results


def run_model_2_analysis(path):
    """Run Model 2 analysis with all visualizations."""
    print("\n" + "="*80)
    print("MODEL 2: DETERMINISTIC WITH STORAGE AND BUDGET")
    print("="*80)
    
    # Run model
    flow_results, unmet_demand, spending_results, budget, dual_results = RunnerModel2(path).run_model()
    
    # Print metrics
    print_model_2_metrics(flow_results, spending_results, budget, unmet_demand)
    
    plot_model_2_energy_balance(
        flow_results=flow_results,
        spending_results=spending_results,
        unmet_demand=unmet_demand,
        save_path=Path(path) / "figures" / "model2_energy_balance"
    )
    
    plot_model_2_economics(
        spending_results=spending_results, 
        unmet_demand=unmet_demand,
        save_path=Path(path) / "figures" / "Model_2_economics"
    )
    
    print("\nModel 2 analysis complete!")
    return flow_results, unmet_demand, spending_results, budget, dual_results


def run_model_3_single(path, alpha, beta):
    """Run a single Model 3 configuration."""
    print(f"\nRunning Model 3 with α={alpha:.2f}, β={beta:.2f}...")
    flow_results, spending_results, risk_stats, results = RunnerModel3_IndependentStorageBudget(
        path, alpha=alpha, beta=beta).run_model()
    
    print("\n" + "="*80)
    print(f"MODEL 3 METRICS (α={alpha:.2f}, β={beta:.2f}):")
    print("="*80)
    print_model_3_metrics(
        flow_results=flow_results,
        spending_results=spending_results,
        risk_stats=risk_stats,
        alpha=alpha,
        beta=beta
    )
    
    return flow_results, spending_results, risk_stats, results


def create_model2_vs_model3_comparison(path, flow_m2, spending_m2, unmet_m2, 
                                        flow_m3, spending_m3, risk_m3, alpha, beta):
    """Create comparison visualizations between Model 2 and Model 3."""
    print("\n" + "="*80)
    print(f"Creating Model 2 vs Model 3 comparison visualizations (β={beta:.2f})...")
    print("="*80)
    
    # Create procurement plan overlay
    plot_model2_model3_comparison(
        flow_results_m2=flow_m2,
        spending_results_m2=spending_m2,
        unmet_demand_m2=unmet_m2,
        flow_results_m3=flow_m3,
        spending_results_m3=spending_m3,
        risk_stats_m3=risk_m3,
        alpha=alpha,
        beta=beta,
        save_path=Path(path) / "figures" / f"model2_vs_model3_procurement_plan_beta_{beta:.2f}"
    )

    # Create risk distribution comparison
    compare_model2_model3_risk(
        unmet_m2=unmet_m2,
        risk_stats_m3=risk_m3,
        alpha=alpha,
        beta=beta,
        save_path=Path(path) / "figures" / f"model2_vs_model3_risk_distribution_beta_{beta:.2f}"
    )
    
    print("Model 2 vs Model 3 comparison visualizations complete!")


def create_three_way_comparison(path, flow_m2, spending_m2,
                                 flow_m3_0, spending_m3_0,
                                 flow_m3_1, spending_m3_1,
                                 alpha, beta_0, beta_1):
    """Create three-way comparison: Model 2 vs Model 3 (two beta values)."""
    print("\n" + "="*80)
    print(f"Creating three-way comparison visualization (β={beta_0:.2f} & β={beta_1:.2f})...")
    print("="*80)
    
    compare_three_procurement_strategies(
        flow_m2=flow_m2,
        spending_m2=spending_m2,
        flow_m3_0=flow_m3_0,
        spending_m3_0=spending_m3_0,
        flow_m3_1=flow_m3_1,
        spending_m3_1=spending_m3_1,
        alpha=alpha,
        beta_0=beta_0,
        beta_1=beta_1,
        save_path=Path(path) / "figures" / f"three_way_comparison_alpha_{alpha:.2f}_beta_{beta_0:.2f}_vs_{beta_1:.2f}"
    )
    
    print("Three-way comparison visualization complete!")


def create_model3_beta_comparison(path, risk_0, risk_1, alpha, beta_0, beta_1):
    """Create risk distribution comparison between two Model 3 beta values."""
    print("\n" + "="*80)
    print(f"Creating Model 3 beta comparison (β={beta_0:.2f} vs β={beta_1:.2f})...")
    print("="*80)
    
    compare_risk_distributions(
        risk_stats_0=risk_0,
        risk_stats_1=risk_1,
        alpha=alpha,
        beta_0=beta_0,
        beta_1=beta_1,
        save_path=Path(path) / "figures" / f"risk_comparison_alpha_{alpha:.2f}_beta_{beta_0:.2f}_vs_{beta_1:.2f}"
    )
    
    print("Model 3 beta comparison complete!")


import time

def main():
    """Main execution function - runs all analyses."""
    # Start timing
    start_time = time.time()
    
    # Set data path
    path = r"C:\Users\alex\OneDrive\Desktop\DTU\Optimistation\Optimisation_Assignment_2\Data"
    
    # Configuration
    alpha_val = 0.9
    beta_val_0 = 0.0
    beta_val_1 = 0.3
    
    print("\n" + "="*80)
    print("RUNNING ALL MODEL ANALYSES")
    print("="*80)
    
    # ========================================================================
    # STEP 1: Run all models once
    # ========================================================================
    
    # Run Model 1
    model1_start = time.time()
    primal_m1, unmet_m1, spending_m1, dual_m1 = run_model_1_analysis(path)
    model1_time = time.time() - model1_start
    print(f"\n⏱️  Model 1 completed in {model1_time:.2f} seconds")
    
    # Run Model 2
    model2_start = time.time()
    flow_m2, unmet_m2, spending_m2, budget_m2, dual_m2 = run_model_2_analysis(path)
    model2_time = time.time() - model2_start
    print(f"⏱️  Model 2 completed in {model2_time:.2f} seconds")
    
    # Run Model 3 with beta_0 (risk-neutral)
    model3_0_start = time.time()
    flow_m3_0, spending_m3_0, risk_m3_0, results_m3_0 = run_model_3_single(
        path, alpha=alpha_val, beta=beta_val_0)
    model3_0_time = time.time() - model3_0_start
    print(f"⏱️  Model 3 (β={beta_val_0:.2f}) completed in {model3_0_time:.2f} seconds")
    
    # Run Model 3 with beta_1 (risk-averse)
    model3_1_start = time.time()
    flow_m3_1, spending_m3_1, risk_m3_1, results_m3_1 = run_model_3_single(
        path, alpha=alpha_val, beta=beta_val_1)
    model3_1_time = time.time() - model3_1_start
    print(f"⏱️  Model 3 (β={beta_val_1:.2f}) completed in {model3_1_time:.2f} seconds")
    
    # ========================================================================
    # STEP 2: Create all comparisons using existing results
    # ========================================================================
    
    comparison_start = time.time()
    
    # Model 2 vs Model 3 (β=0.3 - risk-averse)
    create_model2_vs_model3_comparison(
        path, flow_m2, spending_m2, unmet_m2,
        flow_m3_1, spending_m3_1, risk_m3_1,
        alpha=alpha_val, beta=beta_val_1
    )
    
    # Three-way comparison: Model 2 vs Model 3 (β=0.0) vs Model 3 (β=0.3)
    create_three_way_comparison(
        path, flow_m2, spending_m2,
        flow_m3_0, spending_m3_0,
        flow_m3_1, spending_m3_1,
        alpha=alpha_val, beta_0=beta_val_0, beta_1=beta_val_1
    )
    
    # Model 3 beta comparison: β=0.0 vs β=0.3
    create_model3_beta_comparison(
        path, risk_m3_0, risk_m3_1,
        alpha=alpha_val, beta_0=beta_val_0, beta_1=beta_val_1
    )
    
    comparison_time = time.time() - comparison_start
    print(f"\n⏱️  All comparisons and visualizations completed in {comparison_time:.2f} seconds")
    
    # Calculate and print total time
    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = total_time % 60
    
    print("\n" + "="*80)
    print("ALL ANALYSES COMPLETE!")
    print("="*80)
    print(f"\n⏱️  TOTAL EXECUTION TIME: {minutes} min {seconds:.2f} sec ({total_time:.2f} seconds)")
    print(f"\n   Model 1:              {model1_time:.2f}s")
    print(f"   Model 2:              {model2_time:.2f}s")
    print(f"   Model 3 (β={beta_val_0:.2f}):     {model3_0_time:.2f}s")
    print(f"   Model 3 (β={beta_val_1:.2f}):     {model3_1_time:.2f}s")
    print(f"   Comparisons/Plots:    {comparison_time:.2f}s")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
