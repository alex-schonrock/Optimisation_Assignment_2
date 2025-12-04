
import json
import csv
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
def plot_model_3_energy_balance(flow_results, save_path=None):
    """
    Plot energy balance for Model 3 with median values, percentile bands, and storage flows.
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
    
    # Calculate storage changes for median
    storage_change_median = np.zeros(len(days))
    storage_change_median[0] = stored_median[0]
    for i in range(1, len(days)):
        storage_change_median[i] = stored_median[i] - stored_median[i-1]
    
    # Calculate storage changes for p5 and p95
    storage_change_p5 = np.zeros(len(days))
    storage_change_p5[0] = stored_p5[0]
    for i in range(1, len(days)):
        storage_change_p5[i] = stored_p5[i] - stored_p5[i-1]
    
    storage_change_p95 = np.zeros(len(days))
    storage_change_p95[0] = stored_p95[0]
    for i in range(1, len(days)):
        storage_change_p95[i] = stored_p95[i] - stored_p95[i-1]
    
    # Released from storage (negative change = energy added to supply)
    storage_released_median = np.maximum(-storage_change_median, 0)
    storage_released_p5 = np.maximum(-storage_change_p5, 0)
    storage_released_p95 = np.maximum(-storage_change_p95, 0)
    
    # Added to storage (positive change = energy taken from supply)
    storage_added_median = np.maximum(storage_change_median, 0)
    storage_added_p5 = np.maximum(storage_change_p5, 0)
    storage_added_p95 = np.maximum(storage_change_p95, 0)
    
    # DEMAND SIDE (negative)
    ax.bar(days, -demand_median, width=0.8, color='lightgreen', 
           edgecolor='darkgreen', linewidth=1.5, label='Demand (median)', alpha=0.7)
    
    # Add uncertainty band for demand
    ax.fill_between(days, -demand_p5, -demand_p95, color='lightgreen', alpha=0.2)
    
    # SUPPLY SIDE (positive)
    # Base: Bought energy
    ax.bar(days, bought_median, width=0.8, color='steelblue', 
           edgecolor='darkblue', linewidth=1.5, label='Bought (median)', alpha=0.8)
    
    # Add uncertainty band for bought
    ax.fill_between(days, bought_p5, bought_p95, color='steelblue', alpha=0.2)
    
    # Add: Energy released from storage (on top of bought)
    ax.bar(days, storage_released_median, bottom=bought_median, width=0.8, color='gold', 
           edgecolor='darkgoldenrod', linewidth=1.5, label='Released from Storage (median)', alpha=0.8)
    
    # Add uncertainty band for released storage
    ax.fill_between(days, bought_median + storage_released_p5, 
                    bought_median + storage_released_p95, 
                    color='gold', alpha=0.2)
    
    # Add: Unmet demand (on top of bought + released)
    ax.bar(days, unmet_median, bottom=bought_median + storage_released_median, width=0.8, color='coral', 
           edgecolor='darkred', linewidth=1.5, label='Unmet Demand (median)', alpha=0.8)
    
    # Add uncertainty band for unmet
    ax.fill_between(days, bought_median + storage_released_median + unmet_p5, 
                    bought_median + storage_released_median + unmet_p95, 
                    color='coral', alpha=0.2)
    
    # Show energy added to storage (as negative bars below zero)
    ax.bar(days, -storage_added_median, width=0.8, color='purple', 
           edgecolor='darkviolet', linewidth=1.5, label='Added to Storage (median)', alpha=0.6)
    
    # Add uncertainty band for added storage
    ax.fill_between(days, -storage_added_p5, -storage_added_p95, 
                    color='purple', alpha=0.2)
    
    # Add zero line
    ax.axhline(0, color='black', linewidth=1.5, linestyle='-')
    
    ax.set_xlabel('Day', fontsize=13, fontweight='bold')
    ax.set_ylabel('Energy (MWh)', fontsize=13, fontweight='bold')
    ax.set_title('Model 3: Energy Balance with Storage Flows (Stochastic)', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.tick_params(labelsize=11)
    
    # Create custom legend with clear explanation
    from matplotlib.patches import Patch
    
    legend_elements = [
        Patch(facecolor='lightgreen', edgecolor='darkgreen', label='Demand (median)', alpha=0.7),
        Patch(facecolor='steelblue', edgecolor='darkblue', label='Bought (median)', alpha=0.8),
        Patch(facecolor='gold', edgecolor='darkgoldenrod', label='Released from Storage (median)', alpha=0.8),
        Patch(facecolor='coral', edgecolor='darkred', label='Unmet Demand (median)', alpha=0.8),
        Patch(facecolor='purple', edgecolor='darkviolet', label='Added to Storage (median)', alpha=0.6),
        Patch(facecolor='gray', alpha=0.3, label='Shaded areas = Scenario variation (5th-95th percentile)')
    ]
    
    # Place legend outside
    ax.legend(handles=legend_elements, loc='upper center', fontsize=10, 
              bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=True)
    
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
    price = spending_results['Price (dkk/MWh)']
    
    # ========================================================================
    # SPENDING (negative bars) - median
    # ========================================================================
    ax.bar(days, -bought_fuel_median, width=0.8, color='steelblue', 
           edgecolor='darkblue', linewidth=1, label='Fuel Cost (median)', alpha=0.8)
    ax.bar(days, -storage_cost_median, bottom=-bought_fuel_median, width=0.8, 
           color='purple', edgecolor='darkviolet', linewidth=1, 
           label='Storage Cost (median)', alpha=0.7)
    
    # Add uncertainty band for spending
    total_spending_median = -(bought_fuel_median + storage_cost_median)
    total_spending_p5 = -(bought_fuel_p5 + storage_cost_p5)
    total_spending_p95 = -(bought_fuel_p95 + storage_cost_p95)
    
    ax.fill_between(days, total_spending_p5, total_spending_p95, 
                     color='gray', alpha=0.2, label='Spending uncertainty (5th-95th %ile)')
    
    # ========================================================================
    # AVAILABLE BUDGET (positive bars) - median
    # ========================================================================
    ax.bar(days, carry_over_median, width=0.8, color='lightgreen', 
           edgecolor='darkgreen', linewidth=1, label='Carry Over Budget (median)', alpha=0.7)
    ax.bar(days, allowance, bottom=carry_over_median, width=0.8, color='mediumpurple', 
           edgecolor='purple', linewidth=1, label='Allowance', alpha=0.7)
    
    # Add zero line
    ax.axhline(0, color='black', linewidth=1.5, linestyle='-')
    
    # ========================================================================
    # PRICE (secondary axis)
    # ========================================================================
    ax_price = ax.twinx()
    ax_price.plot(days, price, color='red', marker='x', linewidth=2.5, 
                  markersize=8, label='Price', linestyle='--', alpha=0.7)
    ax_price.set_ylabel('Price (DKK/MWh)', fontsize=13, color='red', fontweight='bold')
    ax_price.tick_params(axis='y', labelcolor='red', labelsize=11)
    
    # Styling
    ax.set_xlabel('Day', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cash Flow (DKK)', fontsize=13, fontweight='bold')
    ax.set_title(f'Model 3: Budget Dynamics - Spending vs Available Funds (Mean Unmet = {risk_stats["mean"]:.2f} MWh)', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.tick_params(labelsize=11)
    
    # Place legends outside
    ax.legend(loc='upper left', fontsize=10, bbox_to_anchor=(0, -0.12), ncol=5, frameon=True)
    ax_price.legend(loc='upper right', fontsize=10, bbox_to_anchor=(1, -0.12), frameon=True)
    
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