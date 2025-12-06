import json
import csv
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np

def print_model_3_metrics(flow_results, spending_results, risk_stats, alpha, beta):
    """
    Print key metrics for Model 3 (stochastic with scenario-independent storage and budget).
    """
    # Extract data
    demand_median = flow_results['Demand_median (MWh)']
    unmet_median = flow_results['Unmet_Demand_median (MWh)']
    stored = flow_results['Stored (MWh)']  # Deterministic
    
    carry_over = spending_results['Carry_over']  # Deterministic
    price_mean = spending_results['Price_mean (dkk/MWh)']
    
    # Calculate metrics from risk_stats
    total_unmet_mean = risk_stats['mean']
    total_unmet_median = risk_stats['median']
    total_unmet_std = risk_stats['std']
    total_unmet_max = risk_stats['max']
    
    # Get VaR (percentile based on alpha)
    percentile = int(alpha * 100)
    p_key = f'p{percentile}' if f'p{percentile}' in risk_stats else 'p95'
    total_unmet_var = risk_stats[p_key]
    
    # Calculate CVaR (conditional expectation beyond VaR)
    unmet_scenarios = risk_stats['total_unmet_per_scenario']
    total_unmet_cvar = np.mean(unmet_scenarios[unmet_scenarios >= total_unmet_var])
    
    # Calculate demand fulfillment (using mean unmet across scenarios)
    total_demand = demand_median.sum() * 30  # Approximate (assumes scenarios have similar total demand)
    demand_fulfillment_rate = ((total_demand - total_unmet_mean) / total_demand) * 100
    
    # Storage and budget metrics (deterministic)
    total_storage_used = stored.sum()
    total_carry_over_used = carry_over.sum()
    
    # Fuel cost (use mean from risk_stats)
    total_fuel_cost_mean = risk_stats['cost_mean']
    
    print("\n" + "="*70)
    print(f"MODEL 3: KEY PERFORMANCE METRICS (α={alpha:.2f}, β={beta:.2f})")
    print("="*70)
    
    print(f"\n  {'UNMET DEMAND (across scenarios):':<45}")
    print(f"    Mean total unmet demand (MWh):    {total_unmet_mean:>12.1f}")
    print(f"    Median total unmet demand (MWh):  {total_unmet_median:>12.1f}")
    print(f"    Std dev (MWh):                     {total_unmet_std:>12.1f}")
    print(f"    VaR{percentile} (MWh):                        {total_unmet_var:>12.1f}")
    print(f"    CVaR{percentile} (MWh):                       {total_unmet_cvar:>12.1f}")
    print(f"    Worst case (MWh):                  {total_unmet_max:>12.1f}")
    
    print(f"\n  {'OPERATIONAL METRICS:':<45}")
    print(f"    Demand fulfillment rate (%):       {demand_fulfillment_rate:>12.1f}")
    print(f"    Total storage used (MWh):          {total_storage_used:>12.1f}")
    print(f"    Total carry-over budget used (DKK): {total_carry_over_used:>12,.0f}")
    
    print(f"\n  {'COST METRICS:':<45}")
    print(f"    Mean total fuel cost (DKK):        {total_fuel_cost_mean:>12,.0f}")
    print(f"    Mean fuel cost std dev (DKK):      {risk_stats['cost_std']:>12,.0f}")
    print(f"    95th %ile fuel cost (DKK):         {risk_stats['cost_p95']:>12,.0f}")
    
    print("\n" + "="*70 + "\n")
    
    # Return metrics as dictionary for easy table population
    return {
        'total_unmet_mean': total_unmet_mean,
        'total_unmet_median': total_unmet_median,
        'total_unmet_std': total_unmet_std,
        'total_unmet_var': total_unmet_var,
        'total_unmet_cvar': total_unmet_cvar,
        'total_unmet_max': total_unmet_max,
        'demand_fulfillment_rate': demand_fulfillment_rate,
        'total_storage_used': total_storage_used,
        'total_carry_over_used': total_carry_over_used,
        'total_fuel_cost_mean': total_fuel_cost_mean,
        'total_fuel_cost_std': risk_stats['cost_std'],
        'total_fuel_cost_p95': risk_stats['cost_p95']
    }

def compare_risk_distributions(risk_stats_0, risk_stats_1, alpha, beta_0, beta_1, save_path=None):
    """
    Compare the unmet demand distributions for two different β values.
    Shows histogram with mean, VaR, and shaded CVaR tail regions.
    
    Parameters:
    - risk_stats_0: Risk statistics for first beta value
    - risk_stats_1: Risk statistics for second beta value
    - alpha: CVaR confidence level (e.g., 0.9)
    - beta_0: First beta value (e.g., 0.0)
    - beta_1: Second beta value (e.g., 0.15)
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Extract data
    unmet_beta_0 = risk_stats_0['total_unmet_per_scenario']
    unmet_beta_1 = risk_stats_1['total_unmet_per_scenario']
    
    # Labels with 2 decimal places
    label_0 = f'β={beta_0:.2f}'
    label_1 = f'β={beta_1:.2f}'
    strategy_0 = "Expected Value" if beta_0 == 0 else "CVaR-Weighted"
    strategy_1 = "CVaR-Weighted" if beta_1 > 0 else "Expected Value"
    
    # Calculate VaR and CVaR
    percentile = int(alpha * 100)
    p_key = f'p{percentile}' if f'p{percentile}' in risk_stats_0 else 'p95'
    
    # VaR is the alpha-percentile
    var_0 = risk_stats_0[p_key]
    var_1 = risk_stats_1[p_key]
    
    # CVaR is the mean of values exceeding VaR (conditional expectation)
    cvar_0 = np.mean(unmet_beta_0[unmet_beta_0 >= var_0])
    cvar_1 = np.mean(unmet_beta_1[unmet_beta_1 >= var_1])
    
    # ========================================================================
    # HISTOGRAM
    # ========================================================================
    
    bins = np.linspace(min(unmet_beta_0.min(), unmet_beta_1.min()),
                       max(unmet_beta_0.max(), unmet_beta_1.max()), 35)
    
    ax.hist(unmet_beta_0, bins=bins, alpha=0.5, color='steelblue', 
            edgecolor='darkblue', linewidth=1.5, 
            label=f'{label_0} ({strategy_0})', density=True)
    ax.hist(unmet_beta_1, bins=bins, alpha=0.5, color='coral', 
            edgecolor='darkred', linewidth=1.5, 
            label=f'{label_1} ({strategy_1})', density=True)
    
    # ========================================================================
    # TAIL SHADING (CVaR regions) - Plot first so lines appear on top
    # ========================================================================
    
    ax.axvspan(var_0, unmet_beta_0.max(), alpha=0.15, color='blue', 
               label=f'{label_0} CVaR Tail (worst {100-percentile}%)', zorder=1)
    ax.axvspan(var_1, unmet_beta_1.max(), alpha=0.15, color='red',
               label=f'{label_1} CVaR Tail (worst {100-percentile}%)', zorder=1)
    
    # ========================================================================
    # BETA_0 LINES - BLUE FAMILY
    # ========================================================================
    
    # Mean - Solid blue, medium thickness
    ax.axvline(risk_stats_0['mean'], color='blue', linestyle='-', 
               linewidth=3, alpha=0.9,
               label=f"{label_0} Mean: {risk_stats_0['mean']:.1f} MWh", zorder=8)
    
    # VaR - Dashed navy, thick
    ax.axvline(var_0, color='navy', linestyle='--', 
               linewidth=4, alpha=0.9,
               label=f"{label_0} VaR{percentile}: {var_0:.1f} MWh", zorder=9)
    
    # ========================================================================
    # BETA_1 LINES - RED/ORANGE FAMILY
    # ========================================================================
    
    # Mean - Solid orange, medium thickness
    ax.axvline(risk_stats_1['mean'], color='darkorange', linestyle='-', 
               linewidth=3, alpha=0.9,
               label=f"{label_1} Mean: {risk_stats_1['mean']:.1f} MWh", zorder=8)
    
    # VaR - Dash-dot red, thick
    ax.axvline(var_1, color='red', linestyle='--', 
               linewidth=4, alpha=0.9,
               label=f"{label_1} VaR{percentile}: {var_1:.1f} MWh", zorder=9)
    
    ax.set_xlabel('Total Unmet Demand (MWh)', fontsize=20, fontweight='bold')
    ax.set_ylabel('Density', fontsize=20, fontweight='bold')
    ax.set_title(f'Risk Profile Comparison: {label_0} vs {label_1} (α={alpha:.2f})\n' +
                 f'Distribution with Mean, VaR, and CVaR Tail Regions', 
                 fontsize=22, fontweight='bold', pad=20)
    ax.legend(fontsize=16, loc='upper left', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=19)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path.with_suffix('.png')}")
    
    plt.show()


def compare_three_procurement_strategies(flow_m2, spending_m2, 
                                         flow_m3_0, spending_m3_0, 
                                         flow_m3_1, spending_m3_1,
                                         alpha, beta_0, beta_1, save_path=None):
    """
    Compare procurement strategies: Model 2 (deterministic) vs two Model 3 variants (different β values).
    
    Parameters:
    - flow_m2, spending_m2: Model 2 results
    - flow_m3_0, spending_m3_0: Model 3 results for first beta value
    - flow_m3_1, spending_m3_1: Model 3 results for second beta value
    - alpha: CVaR confidence level (e.g., 0.9)
    - beta_0: First beta value for Model 3 (e.g., 0.0)
    - beta_1: Second beta value for Model 3 (e.g., 0.15)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    
    days = flow_m2.index
    
    # Labels
    label_m2 = 'Model 2 (Det.)'
    label_m3_0 = f'Model 3: β={beta_0:.2f}'
    label_m3_1 = f'Model 3: β={beta_1:.2f}'
    
    # Extract storage data
    storage_m2 = flow_m2['Stored (MWh)']
    storage_m3_0 = flow_m3_0['Stored (MWh)']
    storage_m3_1 = flow_m3_1['Stored (MWh)']
    
    # Calculate carry-over as Budget - Allowance
    allowance_m2 = spending_m2['Allowance'].iloc[0]
    carry_over_m2 = spending_m2['Carry over budget']
    
    budget_m3_0 = spending_m3_0['Budget']
    allowance_m3_0 = spending_m3_0['Allowance'].iloc[0]
    carry_over_m3_0 = budget_m3_0 - allowance_m3_0
    
    budget_m3_1 = spending_m3_1['Budget']
    allowance_m3_1 = spending_m3_1['Allowance'].iloc[0]
    carry_over_m3_1 = budget_m3_1 - allowance_m3_1
    
    # Calculate totals
    total_stored_m2 = storage_m2.sum()
    total_stored_m3_0 = storage_m3_0.sum()
    total_stored_m3_1 = storage_m3_1.sum()
    
    total_carry_over_m2 = carry_over_m2.sum()
    total_carry_over_m3_0 = carry_over_m3_0.sum()
    total_carry_over_m3_1 = carry_over_m3_1.sum()
    
    # ========================================================================
    # TOP PANEL: STORAGE COMPARISON
    # ========================================================================
    
    # Plot Model 2 storage
    ax1.fill_between(days, 0, storage_m2, color='lightgray', alpha=0.4)
    ax1.plot(days, storage_m2, color='black', linewidth=2.5, marker='o', 
            markersize=4, linestyle='-', label=label_m2, alpha=0.6)
    
    # Plot Model 3 β=beta_0 storage
    ax1.fill_between(days, 0, storage_m3_0, color='steelblue', alpha=0.3)
    ax1.plot(days, storage_m3_0, color='darkblue', linewidth=3, marker='s', 
            markersize=5, linestyle='--', label=label_m3_0)
    
    # Plot Model 3 β=beta_1 storage
    ax1.fill_between(days, 0, storage_m3_1, color='coral', alpha=0.3)
    ax1.plot(days, storage_m3_1, color='darkred', linewidth=3, marker='^', 
            markersize=5, linestyle=':', label=label_m3_1)
    
    ax1.set_ylabel('Storage Level (MWh)', fontsize=19, fontweight='bold')
    ax1.set_title(f'Procurement Strategy Comparison: Model 2 vs Model 3 (α={alpha:.2f}, β={beta_0:.2f} & β={beta_1:.2f})\n' +
                  'Storage and Budget Carry-Over Evolution',
                 fontsize=20, fontweight='bold', pad=15)
    ax1.legend(loc='upper left', fontsize=14, framealpha=0.95)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.tick_params(labelsize=18)
    
    # Statistics - Total Stored
    stats_text_storage = f"""Total Stored Over Period:
{label_m2}: {total_stored_m2:.0f} MWh
{label_m3_0}: {total_stored_m3_0:.0f} MWh
{label_m3_1}: {total_stored_m3_1:.0f} MWh"""
    
    ax1.text(0.98, 0.98, stats_text_storage, transform=ax1.transAxes,
            fontsize=14, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9, pad=0.8),
            family='monospace')
    
    # ========================================================================
    # BOTTOM PANEL: CARRY-OVER BUDGET COMPARISON
    # ========================================================================
    
    # Plot Model 2 carry-over
    ax2.fill_between(days, 0, carry_over_m2, color='lightgray', alpha=0.4)
    ax2.plot(days, carry_over_m2, color='black', linewidth=2.5, marker='o', 
            markersize=4, linestyle='-', label=label_m2, alpha=0.6, zorder=5)
    
    # Plot Model 3 β=beta_0 carry-over
    ax2.fill_between(days, 0, carry_over_m3_0, color='steelblue', alpha=0.3)
    ax2.plot(days, carry_over_m3_0, color='darkblue', linewidth=3, marker='s', 
            markersize=5, linestyle='--', label=label_m3_0, zorder=5)
    
    # Plot Model 3 β=beta_1 carry-over
    ax2.fill_between(days, 0, carry_over_m3_1, color='coral', alpha=0.3)
    ax2.plot(days, carry_over_m3_1, color='darkred', linewidth=3, marker='^', 
            markersize=5, linestyle=':', label=label_m3_1, zorder=5)
    
    # Add zero line
    ax2.axhline(0, color='black', linewidth=1.5, linestyle='-', alpha=0.5, zorder=3)
    
    # Add grid
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax2.set_ylabel('Budget Carry-Over (DKK)', fontsize=19, fontweight='bold')
    ax2.set_xlabel('Day', fontsize=19, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=14, framealpha=0.95)
    ax2.tick_params(labelsize=18)
    
    # Statistics - Total Carry-Over
    stats_text_carryover = f"""Total Carry-Over Over Period:
{label_m2}: {total_carry_over_m2:.0f} DKK
{label_m3_0}: {total_carry_over_m3_0:.0f} DKK
{label_m3_1}: {total_carry_over_m3_1:.0f} DKK"""
    
    ax2.text(0.98, 0.98, stats_text_carryover, transform=ax2.transAxes,
            fontsize=14, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9, pad=0.8),
            family='monospace')
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path.with_suffix('.png')}")
    
    plt.show()
    
    # ========================================================================
    # PRINT DETAILED COMPARISON
    # ========================================================================
    
    print("\n" + "="*80)
    print(f"THREE-WAY PROCUREMENT STRATEGY COMPARISON (α={alpha:.2f})")
    print("="*80)
    
    print("\n📦 STORAGE STRATEGY:")
    print(f"  {label_m2}: Total = {total_stored_m2:.0f} MWh")
    print(f"  {label_m3_0}: Total = {total_stored_m3_0:.0f} MWh (Δ vs M2: {total_stored_m3_0-total_stored_m2:+.0f} MWh)")
    print(f"  {label_m3_1}: Total = {total_stored_m3_1:.0f} MWh (Δ vs M2: {total_stored_m3_1-total_stored_m2:+.0f} MWh)")
    
    print("\n💵 CARRY-OVER BUDGET STRATEGY:")
    print(f"  {label_m2}: Total = {total_carry_over_m2:.0f} DKK")
    print(f"  {label_m3_0}: Total = {total_carry_over_m3_0:.0f} DKK (Δ vs M2: {total_carry_over_m3_0-total_carry_over_m2:+.0f} DKK)")
    print(f"  {label_m3_1}: Total = {total_carry_over_m3_1:.0f} DKK (Δ vs M2: {total_carry_over_m3_1-total_carry_over_m2:+.0f} DKK)")
    
    print("\n" + "="*80)

def plot_model2_model3_comparison(flow_results_m2, spending_results_m2, unmet_demand_m2,
                                   flow_results_m3, spending_results_m3, risk_stats_m3,
                                   alpha, beta, save_path=None):
    """
    Plot Model 2 (deterministic) and Model 3 (stochastic) procurement plans overlaid.
    Shows how robust stochastic planning differs from deterministic planning.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    
    days = flow_results_m2.index
    
    # Extract Model 2 data
    storage_m2 = flow_results_m2['Stored (MWh)']
    carry_over_m2 = spending_results_m2['Carry over budget']
    
    # Extract Model 3 data (scenario-independent)
    storage_m3 = flow_results_m3['Stored (MWh)']
    carry_over_m3 = spending_results_m3['Carry_over']
    
    # Calculate totals
    total_stored_m2 = storage_m2.sum()
    total_stored_m3 = storage_m3.sum()
    total_carry_over_m2 = carry_over_m2.sum()
    total_carry_over_m3 = carry_over_m3.sum()
    
    # Get unmet demand
    mean_unmet_m3 = risk_stats_m3['mean']
    
    # ========================================================================
    # TOP PANEL: STORAGE COMPARISON
    # ========================================================================
    
    # Plot Model 2 storage
    ax1.fill_between(days, 0, storage_m2, color='steelblue', alpha=0.3, label='Model 2 (Deterministic)')
    ax1.plot(days, storage_m2, color='darkblue', linewidth=2.5, marker='o', 
            markersize=4, linestyle='-', label='Model 2')
    
    # Plot Model 3 storage
    ax1.fill_between(days, 0, storage_m3, color='coral', alpha=0.3, label='Model 3 (Stochastic)')
    ax1.plot(days, storage_m3, color='darkred', linewidth=2.5, marker='s', 
            markersize=4, linestyle='--', label=f'Model 3 (α={alpha:.2f}, β={beta:.2f})')
    
    ax1.set_ylabel('Storage Level (MWh)', fontsize=18, fontweight='bold')
    ax1.set_title(f'Deterministic vs Robust Stochastic Procurement Plans\n' +
                  f'Storage and Budget Carry-Over Comparison',
                 fontsize=20, fontweight='bold', pad=15)
    ax1.legend(loc='upper left', fontsize=15, framealpha=0.95)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.tick_params(labelsize=17)
    
    # Add statistics annotation on RIGHT
    stats_text_storage = f"""Total Stored:
Model 2: {total_stored_m2:.0f} MWh
Model 3: {total_stored_m3:.0f} MWh
Δ: {total_stored_m3 - total_stored_m2:+.0f} MWh"""
    
    ax1.text(0.98, 0.98, stats_text_storage, transform=ax1.transAxes,
            fontsize=14, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, pad=0.8),
            family='monospace')
    
    # ========================================================================
    # BOTTOM PANEL: CARRY-OVER BUDGET COMPARISON
    # ========================================================================
    
    # Plot Model 2 carry-over
    ax2.fill_between(days, 0, carry_over_m2, color='steelblue', alpha=0.3)
    ax2.plot(days, carry_over_m2, color='darkblue', linewidth=2.5, marker='o', 
            markersize=4, linestyle='-', label='Model 2', zorder=5)
    
    # Plot Model 3 carry-over
    ax2.fill_between(days, 0, carry_over_m3, color='coral', alpha=0.3)
    ax2.plot(days, carry_over_m3, color='darkred', linewidth=2.5, marker='s', 
            markersize=4, linestyle='--', label=f'Model 3 (α={alpha:.2f}, β={beta:.2f})', zorder=5)
    
    # Add zero line
    ax2.axhline(0, color='black', linewidth=1.5, linestyle='-', alpha=0.5, zorder=3)
    
    # Add grid
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax2.set_ylabel('Budget Carry-Over (DKK)', fontsize=19, fontweight='bold')
    ax2.set_xlabel('Day', fontsize=19, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=14, framealpha=0.95)
    ax2.tick_params(labelsize=18)
    
    # Add statistics annotation on RIGHT
    stats_text_carryover = f"""Total Carry-Over:
Model 2: {total_carry_over_m2:.0f} DKK
Model 3: {total_carry_over_m3:.0f} DKK
Δ: {total_carry_over_m3 - total_carry_over_m2:+.0f} DKK"""
    
    ax2.text(0.98, 0.98, stats_text_carryover, transform=ax2.transAxes,
            fontsize=14, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, pad=0.8),
            family='monospace')
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path.with_suffix('.png')}")
    
    plt.show()
    
    # ========================================================================
    # PRINT COMPARISON SUMMARY
    # ========================================================================
    
    print("\n" + "="*80)
    print(f"MODEL 2 vs MODEL 3 COMPARISON (Model 3: α={alpha:.2f}, β={beta:.2f})")
    print("="*80)
    
    print("\n📦 STORAGE STRATEGY:")
    print(f"  Model 2: Total = {total_stored_m2:.0f} MWh, Max = {storage_m2.max():.0f} MWh")
    print(f"  Model 3: Total = {total_stored_m3:.0f} MWh, Max = {storage_m3.max():.0f} MWh")
    print(f"  Δ: {total_stored_m3 - total_stored_m2:+.0f} MWh ({((total_stored_m3/total_stored_m2-1)*100):+.1f}%)")
    
    print("\n💵 CARRY-OVER BUDGET STRATEGY:")
    print(f"  Model 2: Total = {total_carry_over_m2:.0f} DKK, Max = {carry_over_m2.max():.0f} DKK")
    print(f"  Model 3: Total = {total_carry_over_m3:.0f} DKK, Max = {carry_over_m3.max():.0f} DKK")
    print(f"  Δ: {total_carry_over_m3 - total_carry_over_m2:+.0f} DKK ({((total_carry_over_m3/total_carry_over_m2-1)*100):+.1f}%)")
    
    print("\n📊 UNMET DEMAND:")
    print(f"  Model 2: {unmet_demand_m2:.0f} MWh (deterministic)")
    print(f"  Model 3: {mean_unmet_m3:.0f} MWh (mean), {risk_stats_m3['p95']:.0f} MWh (95th %ile)")
    
    print("\n" + "="*80)


def compare_model2_model3_risk(unmet_m2, risk_stats_m3, alpha, beta, save_path=None):
    """
    Compare risk distributions between Model 2 (deterministic) and Model 3 (stochastic).
    Model 2 has a single deterministic outcome, Model 3 has a distribution.
    
    Parameters:
    - unmet_m2: Total unmet demand from Model 2 (scalar)
    - risk_stats_m3: Risk statistics dictionary from Model 3
    - alpha: CVaR confidence level (e.g., 0.9)
    - beta: Beta value for Model 3 (e.g., 0.15)
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Extract Model 3 data
    unmet_m3 = risk_stats_m3['total_unmet_per_scenario']
    
    # Labels
    label_m2 = 'Model 2 (Deterministic)'
    label_m3 = f'Model 3 (Stochastic, β={beta:.2f})'
    
    # Calculate VaR and CVaR for Model 3
    percentile = int(alpha * 100)
    p_key = f'p{percentile}' if f'p{percentile}' in risk_stats_m3 else 'p95'
    
    var_m3 = risk_stats_m3[p_key]
    cvar_m3 = np.mean(unmet_m3[unmet_m3 >= var_m3])
    
    # ========================================================================
    # HISTOGRAM FOR MODEL 3
    # ========================================================================
    
    bins = np.linspace(min(unmet_m2, unmet_m3.min()),
                       max(unmet_m2, unmet_m3.max()), 35)
    
    ax.hist(unmet_m3, bins=bins, alpha=0.6, color='coral', 
            edgecolor='darkred', linewidth=1.5, 
            label=f'{label_m3} Distribution', density=True)
    
    # ========================================================================
    # TAIL SHADING (CVaR region) - Plot first so lines appear on top
    # ========================================================================
    
    ax.axvspan(var_m3, unmet_m3.max(), alpha=0.15, color='red',
               label=f'M3 CVaR Tail (worst {100-percentile}%)', zorder=1)
    
    # ========================================================================
    # VERTICAL LINE FOR MODEL 2 (deterministic outcome)
    # ========================================================================
    
    ax.axvline(unmet_m2, color='black', linestyle='-', 
               linewidth=5, alpha=0.9,
               label=f'{label_m2}: {unmet_m2:.1f} MWh', zorder=10)
    
    # ========================================================================
    # MODEL 3 STATISTICS
    # ========================================================================
    
    # Mean - Solid blue, medium thickness
    ax.axvline(risk_stats_m3['mean'], color='blue', linestyle='-', 
               linewidth=3, alpha=0.9,
               label=f"M3 Mean: {risk_stats_m3['mean']:.1f} MWh", zorder=8)
    
    # VaR - Dashed orange, thick
    ax.axvline(var_m3, color='orange', linestyle='--', 
               linewidth=4, alpha=0.9,
               label=f"M3 VaR{percentile}: {var_m3:.1f} MWh", zorder=9)
    
    ax.set_xlabel('Total Unmet Demand (MWh)', fontsize=20, fontweight='bold')
    ax.set_ylabel('Density', fontsize=20, fontweight='bold')
    ax.set_title(f'Risk Profile Comparison: Model 2 (Deterministic) vs Model 3 (Stochastic, α={alpha:.2f}, β={beta:.2f})\n' +
                 f'Deterministic Outcome vs Scenario Distribution with Mean, VaR, and CVaR Tail', 
                 fontsize=22, fontweight='bold', pad=20)
    ax.legend(fontsize=16, loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=19)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path.with_suffix('.png')}")
    
    plt.show()
    






