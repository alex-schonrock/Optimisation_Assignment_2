
import json
import csv
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
def plot_model_3_energy_balance(flow_results, save_path=None):
    """
    Plot energy balance for Model 3 with median values, percentile bands, and storage on secondary axis.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Extract data
    days = flow_results.index
    demand_median = flow_results['Demand_median (MWh)']
    demand_p5 = flow_results['Demand_p5 (MWh)']
    demand_p95 = flow_results['Demand_p95 (MWh)']
    
    bought_median = flow_results['Bought_median (MWh)']
    bought_p5 = flow_results['Bought_p5 (MWh)']
    bought_p95 = flow_results['Bought_p95 (MWh)']
    
    unmet_median = flow_results['Unmet_Demand_median (MWh)']
    unmet_p5 = flow_results['Unmet_Demand_p5 (MWh)']
    unmet_p95 = flow_results['Unmet_Demand_p95 (MWh)']
    
    stored_median = flow_results['Stored_median (MWh)']
    stored_p5 = flow_results['Stored_p5 (MWh)']
    stored_p95 = flow_results['Stored_p95 (MWh)']
    
    # DEMAND SIDE (negative)
    ax.bar(days, -demand_median, width=0.8, color='lightgreen', 
           edgecolor='darkgreen', linewidth=1.5, label='Demand (median)', alpha=0.7)
    
    # Add uncertainty band for demand with edges
    ax.fill_between(days, -demand_p5, -demand_p95, color='lightgreen', alpha=0.3,
                    edgecolor='green', linewidth=1.5, linestyle='--')
    
    # SUPPLY SIDE (positive)
    # Base: Bought energy
    ax.bar(days, bought_median, width=0.8, color='steelblue', 
           edgecolor='darkblue', linewidth=1.5, label='Bought (median)', alpha=0.8)
    
    # Add uncertainty band for bought with edges
    ax.fill_between(days, bought_p5, bought_p95, color='steelblue', alpha=0.3,
                    edgecolor='steelblue', linewidth=1.5, linestyle='--')
    
    # Add: Unmet demand (on top of bought)
    ax.bar(days, unmet_median, bottom=bought_median, width=0.8, color='coral', 
           edgecolor='darkred', linewidth=1.5, label='Unmet Demand (median)', alpha=0.8)
    
    # Add uncertainty band for unmet with edges
    ax.fill_between(days, bought_median + unmet_p5, 
                    bought_median + unmet_p95, 
                    color='coral', alpha=0.3,
                    edgecolor='coral', linewidth=1.5, linestyle='--')
    
    # Add zero line
    ax.axhline(0, color='black', linewidth=1.5, linestyle='-')
    
    # Add storage line on secondary axis
    ax_storage = ax.twinx()
    ax_storage.plot(days, stored_median, color='purple', marker='o', linewidth=2.5, 
                    markersize=6, label='Stored (median)', alpha=0.8)
    
    # Add uncertainty band for storage with edges
    ax_storage.fill_between(days, stored_p5, stored_p95, color='purple', alpha=0.3,
                            edgecolor='purple', linewidth=1.5, linestyle='--')
    
    ax_storage.set_ylabel('Stored Energy (MWh)', fontsize=13, color='purple', fontweight='bold')
    ax_storage.tick_params(axis='y', labelcolor='purple', labelsize=11)
    
    ax.set_xlabel('Day', fontsize=13, fontweight='bold')
    ax.set_ylabel('Energy (MWh)', fontsize=13, fontweight='bold')
    ax.set_title('Model 3: Energy Balance (Stochastic)', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.tick_params(labelsize=11)
    
    # Create custom legend with clear explanation
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Patch(facecolor='lightgreen', edgecolor='darkgreen', label='Demand (median)', alpha=0.7),
        Patch(facecolor='steelblue', edgecolor='darkblue', label='Bought (median)', alpha=0.8),
        Patch(facecolor='coral', edgecolor='darkred', label='Unmet Demand (median)', alpha=0.8),
        Patch(facecolor='gray', edgecolor='black', linestyle='--', linewidth=1.5,
              label='Dashed edges = Scenario variation (5th-95th %ile)', alpha=0.3)
    ]
    
    storage_legend = [
        Line2D([0], [0], color='purple', linewidth=2.5, marker='o', label='Stored (median)')
    ]
    
    # Place legends outside
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, 
              bbox_to_anchor=(0, -0.12), ncol=4, frameon=True)
    ax_storage.legend(handles=storage_legend, loc='upper right', fontsize=10,
                      bbox_to_anchor=(1, -0.12), frameon=True)
    
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path.with_suffix('.png')}")
    
    plt.show()

def plot_model_3_economics(spending_results, risk_stats, save_path=None):
    """
    Plot economic view for Model 3 - spending (negative) vs available budget (positive).
    Now handles scenario-dependent prices with MEAN instead of median.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Extract data
    days = spending_results.index
    bought_fuel_median = spending_results['Bought_fuel_median']
    bought_fuel_p5 = spending_results['Bought_fuel_p5']
    bought_fuel_p95 = spending_results['Bought_fuel_p95']
    
    storage_cost_median = spending_results['Storage_cost_median']
    storage_cost_p5 = spending_results['Storage_cost_p5']
    storage_cost_p95 = spending_results['Storage_cost_p95']
    
    carry_over_median = spending_results['Carry_over_median']
    allowance = spending_results['Allowance'].iloc[0]
    
    # UPDATED: Use MEAN for price (not median)
    price_mean = spending_results['Price_mean (dkk/MWh)']
    price_p5 = spending_results['Price_p5 (dkk/MWh)']
    price_p95 = spending_results['Price_p95 (dkk/MWh)']
    
    # ========================================================================
    # SPENDING (negative bars) - median
    # ========================================================================
    ax.bar(days, -bought_fuel_median, width=0.8, color='steelblue', 
           edgecolor='darkblue', linewidth=1, label='Fuel Cost (median)', alpha=0.8)
    ax.bar(days, -storage_cost_median, bottom=-bought_fuel_median, width=0.8, 
           color='orange', edgecolor='darkorange', linewidth=1, 
           label='Storage Cost (median)', alpha=0.7)
    
    # Add uncertainty band for fuel spending with edges
    ax.fill_between(days, -bought_fuel_p5, -bought_fuel_p95, 
                     color='steelblue', alpha=0.3, edgecolor='steelblue', 
                     linewidth=1.5, linestyle='--')
    
    # Add uncertainty band for storage cost with edges
    ax.fill_between(days, -bought_fuel_median - storage_cost_p5, 
                    -bought_fuel_median - storage_cost_p95, 
                     color='orange', alpha=0.3, edgecolor='orange', 
                     linewidth=1.5, linestyle='--')
    
    # ========================================================================
    # AVAILABLE BUDGET (positive bars) - median
    # ========================================================================
    ax.bar(days, carry_over_median, width=0.8, color='lightgreen', 
           edgecolor='darkgreen', linewidth=1, label='Carry Over Budget (median)', alpha=0.7)
    ax.bar(days, allowance, bottom=carry_over_median, width=0.8, color='mediumpurple', 
           edgecolor='purple', linewidth=1, label='Allowance', alpha=0.7)
    
    # Add uncertainty band for carry over with edges
    carry_over_p5 = carry_over_median - (bought_fuel_p95 - bought_fuel_median + storage_cost_p95 - storage_cost_median)
    carry_over_p95 = carry_over_median + (bought_fuel_median - bought_fuel_p5 + storage_cost_median - storage_cost_p5)
    
    ax.fill_between(days, carry_over_p5, carry_over_p95, 
                     color='lightgreen', alpha=0.3, edgecolor='green', 
                     linewidth=1.5, linestyle='--')
    
    # Add zero line
    ax.axhline(0, color='black', linewidth=1.5, linestyle='-')
    
    # ========================================================================
    # PRICE (secondary axis) - UPDATED: Use MEAN
    # ========================================================================
    ax_price = ax.twinx()
    
    # Plot mean price (was median)
    ax_price.plot(days, price_mean, color='red', marker='x', linewidth=2.5, 
                  markersize=8, label='Price (mean)', linestyle='--', alpha=0.8, zorder=5)
    
    # Add price uncertainty band
    ax_price.fill_between(days, price_p5, price_p95, 
                          color='red', alpha=0.2, edgecolor='red', 
                          linewidth=1, linestyle=':', label='Price range (5th-95th %ile)', zorder=3)
    
    ax_price.set_ylabel('Price (DKK/MWh)', fontsize=13, color='red', fontweight='bold')
    ax_price.tick_params(axis='y', labelcolor='red', labelsize=11)
    
    # Styling
    ax.set_xlabel('Day', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cash Flow (DKK)', fontsize=13, fontweight='bold')
    ax.set_title(f'Model 3: Budget Dynamics - Spending vs Available Funds (Mean Unmet = {risk_stats["mean"]:.2f} MWh)', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.tick_params(labelsize=11)
    
    # Create custom legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Patch(facecolor='steelblue', edgecolor='darkblue', label='Fuel Cost (median)', alpha=0.8),
        Patch(facecolor='orange', edgecolor='darkorange', label='Storage Cost (median)', alpha=0.7),
        Patch(facecolor='lightgreen', edgecolor='darkgreen', label='Carry Over Budget (median)', alpha=0.7),
        Patch(facecolor='mediumpurple', edgecolor='purple', label='Allowance', alpha=0.7),
        Patch(facecolor='gray', edgecolor='black', linestyle='--', linewidth=1.5, 
              label='Dashed edges = Scenario variation (5th-95th %ile)', alpha=0.3)
    ]
    
    # UPDATED: Price legend says "mean" now
    price_legend_elements = [
        Line2D([0], [0], color='red', marker='x', linewidth=2.5, markersize=8,
               linestyle='--', label='Price (mean)', alpha=0.8),
        Patch(facecolor='red', edgecolor='red', linestyle=':', 
              label='Price uncertainty (5th-95th %ile)', alpha=0.2)
    ]
    
    # Place legends outside
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, 
              bbox_to_anchor=(0, -0.12), ncol=3, frameon=True)
    ax_price.legend(handles=price_legend_elements, loc='upper right', fontsize=10, 
                    bbox_to_anchor=(1, -0.12), frameon=True)
    
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path.with_suffix('.png')}")
    
    plt.show()

def plot_model_3_risk_distribution(risk_stats, save_path=None):
    """
    Plot risk distribution showing total unmet demand across all scenarios.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Extract data
    total_unmet = risk_stats['total_unmet_per_scenario']
    mean_val = risk_stats['mean']
    median_val = risk_stats['median']
    p5 = risk_stats['p5']
    p25 = risk_stats['p25']
    p75 = risk_stats['p75']
    p95 = risk_stats['p95']
    min_val = risk_stats['min']
    max_val = risk_stats['max']
    
    # Create histogram
    counts, bins, patches = ax.hist(total_unmet, bins=50, density=True, 
                                     alpha=0.7, color='steelblue', edgecolor='black')
    
    # Add vertical lines for key statistics
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2.5, 
               label=f'Mean: {mean_val:.1f} MWh', alpha=0.8)
    ax.axvline(median_val, color='green', linestyle='--', linewidth=2.5, 
               label=f'Median: {median_val:.1f} MWh', alpha=0.8)
    ax.axvline(p5, color='orange', linestyle=':', linewidth=2, 
               label=f'5th percentile: {p5:.1f} MWh', alpha=0.7)
    ax.axvline(p95, color='purple', linestyle=':', linewidth=2, 
               label=f'95th percentile: {p95:.1f} MWh', alpha=0.7)
    
    # Add shaded regions
    ax.axvspan(p5, p25, alpha=0.1, color='orange', label='5th-25th percentile')
    ax.axvspan(p75, p95, alpha=0.1, color='purple', label='75th-95th percentile')
    
    ax.set_xlabel('Total Unmet Demand (MWh)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Density', fontsize=13, fontweight='bold')
    ax.set_title('Model 3: Risk Distribution - Total Unmet Demand Across Scenarios', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.tick_params(labelsize=11)
    
    # Add text box with statistics
    stats_text = f"""Statistics (1000 scenarios):
Mean: {mean_val:.2f} MWh
Median: {median_val:.2f} MWh
Std Dev: {risk_stats['std']:.2f} MWh
Min: {min_val:.2f} MWh
Max: {max_val:.2f} MWh
Range: {max_val - min_val:.2f} MWh"""
    
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontfamily='monospace')
    
    ax.legend(loc='upper left', fontsize=10, frameon=True)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path.with_suffix('.png')}")
    
    plt.show()

def plot_model_3_cvar_distribution(risk_stats, results, alpha, beta, save_path=None):
    """
    Plot risk distribution with CVaR-specific metrics highlighted.
    Shows VaR, CVaR, and the tail region that CVaR optimizes.
    """
    fig, ax = plt.subplots(figsize=(15, 7))
    
    # Extract data - check if dict or Expando
    try:
        # Try dictionary access first
        total_unmet = risk_stats['total_unmet_per_scenario']
        mean_val = risk_stats['mean']
        median_val = risk_stats['median']
        std_val = risk_stats['std']
        min_val = risk_stats['min']
        max_val = risk_stats['max']
    except (TypeError, KeyError):
        # Fall back to attribute access (Expando)
        total_unmet = risk_stats.total_unmet_per_scenario
        mean_val = risk_stats.mean
        median_val = risk_stats.median
        std_val = risk_stats.std
        min_val = risk_stats.min
        max_val = risk_stats.max
    
    # CVaR-specific metrics
    var_value = results.z  # VaR from the model
    alpha_percentile = np.percentile(total_unmet, alpha * 100)
    
    # Calculate CVaR (mean of worst (1-alpha)% scenarios)
    tail_threshold = np.percentile(total_unmet, alpha * 100)
    tail_scenarios = total_unmet[total_unmet >= tail_threshold]
    cvar_value = np.mean(tail_scenarios)
    
    # Create histogram
    counts, bins, patches = ax.hist(total_unmet, bins=50, density=True, 
                                     alpha=0.7, color='steelblue', edgecolor='black',
                                     label='Unmet demand distribution')
    
    # Highlight the CVaR tail region (worst (1-alpha)% scenarios)
    for i, patch in enumerate(patches):
        if bins[i] >= tail_threshold:
            patch.set_facecolor('darkred')
            patch.set_alpha(0.7)
    
    # Add vertical lines for key statistics
    ax.axvline(mean_val, color='green', linestyle='--', linewidth=2.5, 
               label=f'Mean: {mean_val:.1f} MWh', alpha=0.8, zorder=5)
    
    ax.axvline(median_val, color='blue', linestyle='--', linewidth=2.5, 
               label=f'Median: {median_val:.1f} MWh', alpha=0.8, zorder=5)
    
    # VaR line (α-quantile)
    ax.axvline(alpha_percentile, color='orange', linestyle='-', linewidth=3, 
               label=f'VaR ({alpha*100:.0f}th %ile): {alpha_percentile:.1f} MWh', 
               alpha=0.9, zorder=6)
    
    # CVaR line (conditional mean above VaR)
    ax.axvline(cvar_value, color='red', linestyle='-', linewidth=3, 
               label=f'CVaR (tail mean): {cvar_value:.1f} MWh', 
               alpha=0.9, zorder=6)
    
    # Shade the CVaR tail region
    ax.axvspan(tail_threshold, max_val * 1.05, alpha=0.2, color='darkred', 
               label=f'CVaR tail region (worst {(1-alpha)*100:.0f}%)', zorder=1)
    
    # Labels and title
    ax.set_xlabel('Total Unmet Demand (MWh)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Density', fontsize=13, fontweight='bold')
    ax.set_title(f'Model 3 CVaR Analysis: Risk Distribution (α={alpha}, β={beta})', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.tick_params(labelsize=11)
    
    # Add comprehensive text box with statistics
    stats_text = f"""Risk Metrics (1000 scenarios):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Central Tendency:
  Mean:   {mean_val:.2f} MWh
  Median: {median_val:.2f} MWh
  Std Dev: {std_val:.2f} MWh

Extreme Values:
  Min:    {min_val:.2f} MWh
  Max:    {max_val:.2f} MWh

CVaR Metrics (α={alpha}):
  VaR:    {alpha_percentile:.2f} MWh
  CVaR:   {cvar_value:.2f} MWh
  Δ:      {cvar_value - alpha_percentile:.2f} MWh
  
Tail: {len(tail_scenarios)} scenarios
      ({len(tail_scenarios)/len(total_unmet)*100:.1f}%)"""
    
    ax.text(0.02, 0.97, stats_text, transform=ax.transAxes,
            fontsize=9.5, verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, 
                     edgecolor='darkgoldenrod', linewidth=2),
            fontfamily='monospace')
    
    # Add interpretation box
    interpretation_text = f"""CVaR Interpretation:
━━━━━━━━━━━━━━━━━━━━━━━
β = {beta} means:
- {(1-beta)*100:.0f}% weight on expected cost
- {beta*100:.0f}% weight on tail risk (CVaR)

The model hedges against the
worst {(1-alpha)*100:.0f}% of scenarios (red region)"""
    
    ax.text(0.98, 0.60, interpretation_text, transform=ax.transAxes,
            fontsize=9.5, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8, 
                     edgecolor='darkred', linewidth=2),
            fontfamily='monospace')
    
    ax.legend(loc='upper right', fontsize=9.5, frameon=True, 
              fancybox=True, shadow=True, ncol=1)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    
    # Save
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path.with_suffix('.png')}")
    
    plt.show()
    
    # Print detailed CVaR analysis to console
    print("\n" + "="*70)
    print(f"DETAILED CVaR ANALYSIS (α={alpha}, β={beta})")
    print("="*70)
    print(f"\nRisk Measures:")
    print(f"  Expected unmet demand:        {mean_val:.2f} MWh")
    print(f"  VaR (α={alpha}):              {alpha_percentile:.2f} MWh")
    print(f"  CVaR (conditional tail mean): {cvar_value:.2f} MWh")
    print(f"  Excess tail risk (CVaR - VaR): {cvar_value - alpha_percentile:.2f} MWh")
    print(f"\nTail Statistics:")
    print(f"  Number of tail scenarios:      {len(tail_scenarios)}")
    print(f"  Percentage in tail:            {len(tail_scenarios)/len(total_unmet)*100:.1f}%")
    print(f"  Worst scenario:                {max_val:.2f} MWh")
    print(f"  Best scenario in tail:         {np.min(tail_scenarios):.2f} MWh")
    print(f"\nOptimization Focus:")
    print(f"  Weight on expected cost:       {(1-beta)*100:.0f}%")
    print(f"  Weight on CVaR (tail risk):    {beta*100:.0f}%")
    print("="*70 + "\n")




def print_model_3_metrics(flow_results, spending_results, risk_stats):
    """
    Print key metrics for Model 3.
    """
    # Extract summary statistics
    days = flow_results.index
    n_days = len(days)
    
    bought_median = flow_results['Bought_median (MWh)'].sum()
    unmet_median = flow_results['Unmet_Demand_median (MWh)'].sum()
    demand_median = flow_results['Demand_median (MWh)'].sum()
    
    fuel_cost_median = spending_results['Bought_fuel_median'].sum()
    storage_cost_median = spending_results['Storage_cost_median'].sum()
    total_cost_median = fuel_cost_median + storage_cost_median
    
    print("\n" + "="*70)
    print("MODEL 3: KEY METRICS (STOCHASTIC)")
    print("="*70)
    print(f"\n{'ENERGY METRICS (MEDIAN):':<30}")
    print(f"  Total Demand:              {demand_median:>12.1f} MWh")
    print(f"  Total Bought:              {bought_median:>12.1f} MWh")
    print(f"  Total Unmet Demand:        {unmet_median:>12.1f} MWh")
    print(f"  Demand Met:                {(demand_median-unmet_median)/demand_median*100:>12.1f} %")
    
    print(f"\n{'RISK METRICS (ACROSS SCENARIOS):':<30}")
    print(f"  Mean Total Unmet:          {risk_stats['mean']:>12.1f} MWh")
    print(f"  Median Total Unmet:        {risk_stats['median']:>12.1f} MWh")
    print(f"  Std Dev:                   {risk_stats['std']:>12.1f} MWh")
    print(f"  Best Case (min):           {risk_stats['min']:>12.1f} MWh")
    print(f"  Worst Case (max):          {risk_stats['max']:>12.1f} MWh")
    print(f"  5th percentile:            {risk_stats['p5']:>12.1f} MWh")
    print(f"  95th percentile:           {risk_stats['p95']:>12.1f} MWh")
    
    print(f"\n{'ECONOMIC METRICS (MEDIAN):':<30}")
    print(f"  Total Fuel Cost:           {fuel_cost_median:>12.1f} DKK")
    print(f"  Total Storage Cost:        {storage_cost_median:>12.1f} DKK")
    print(f"  Total Cost:                {total_cost_median:>12.1f} DKK")
    
    print("="*70 + "\n")