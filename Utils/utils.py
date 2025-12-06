import json
import csv
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
### load data file function
def load_datafile(file_name, input_path):
    base_path = Path(input_path) / file_name
    result = pd.read_csv(base_path, sep=";")
    # print(result)
    return result


def plot_all_duals(dual_dict, save_path: str | None = None, show: bool = True, title: str = "Dual values for the different price scenarios"):
    """Plots the duals in a line chart."""
    x = np.arange(len(dual_dict.index))
    x_labels = dual_dict.index
    plt.figure(figsize=(12, 6))
    markers = ['o', 'x', 's', '^', 'v', '*', 'D', 'p', '+', '>', 'h']
    linestyles = ['-', '--', '-.', ':']
    for idx, key in enumerate(dual_dict):
        marker = markers[idx % len(markers)]
        linestyle = linestyles[idx % len(linestyles)]
        plt.plot(x, dual_dict[key], marker=marker,ms=8, linestyle=linestyle, alpha=0.7, label=key)
    # Graph features
    plt.xlabel('Day')
    plt.xticks(x_labels)
    plt.ylabel('Dual Value')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc="upper left", bbox_to_anchor=(1,1.07))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()



import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_model_1_energy_balance(primal_results, unmet_demand, save_path=None):
    """
    Plot energy balance for Model 1.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Extract data
    days = primal_results.index
    demand = primal_results['Demand (MWh)']
    bought = primal_results['Bought (MWh)']
    unmet = primal_results['Unmet_Demand (MWh)']
    price = primal_results['Price (DKK/MWh)']
    
    # Plot demand as negative (baseline)
    ax.bar(days, -demand, width=0.8, color='lightgreen', 
           edgecolor='darkgreen', linewidth=1.5, label='Demand', alpha=0.7)
    
    # Plot bought as positive
    ax.bar(days, bought, width=0.8, color='steelblue', 
           edgecolor='darkblue', linewidth=1.5, label='Bought', alpha=0.8)
    
    # Plot unmet demand on top of bought
    ax.bar(days, unmet, bottom=bought, width=0.8, color='coral', 
           edgecolor='darkred', linewidth=1.5, label='Unmet Demand', alpha=0.8)
    
    # Add zero line
    ax.axhline(0, color='black', linewidth=1.5, linestyle='-')
    
    # Add price line on secondary axis
    ax_price = ax.twinx()
    ax_price.plot(days, price, color='red', marker='x', linewidth=2.5, 
                  markersize=8, label='Price', linestyle='--', alpha=0.7)
    ax_price.set_ylabel('Price (DKK/MWh)', fontsize=13, color='red', fontweight='bold')
    ax_price.tick_params(axis='y', labelcolor='red', labelsize=11)
    
    ax.set_xlabel('Day', fontsize=13, fontweight='bold')
    ax.set_ylabel('Gas (MWh)', fontsize=13, fontweight='bold')
    ax.set_title(f'Model 1: Gas flows. Total Unmet Demand = {unmet_demand:.2f} MWh', fontsize=15, fontweight='bold', pad=20)
    ax.tick_params(labelsize=11)
    
    # Place legends outside
    ax.legend(loc='upper left', fontsize=11, bbox_to_anchor=(0, -0.12), ncol=3, frameon=True)
    ax_price.legend(loc='upper right', fontsize=11, bbox_to_anchor=(1, -0.12), frameon=True)
    
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path.with_suffix('.png')}")
    
    plt.show()


def plot_model_1_economics(primal_results, spending_results, unmet_demand, save_path=None):
    """
    Plot economic view for Model 1.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Extract data
    days = spending_results.index
    daily_spending = spending_results['Bought fuel (1000 DKK)']
    allowance = spending_results['Allowance (1000 DKK)'].iloc[0]
    price = spending_results['Price (dkk/MWh)']
    
    # Plot daily spending as bars
    ax.bar(days, daily_spending, width=0.8, color='steelblue', 
           edgecolor='black', linewidth=1, alpha=0.7, label='Fuel Spending')
    
    # Add allowance line
    ax.axhline(allowance, color='orange', linewidth=2.5, 
               linestyle='--', label=f'Allowance ({allowance:.0f} DKK)')
    
    # Add price line on secondary axis
    ax_price = ax.twinx()
    ax_price.plot(days, price, color='red', marker='x', linewidth=2.5, 
                  markersize=8, label='Price', linestyle='--', alpha=0.7)
    ax_price.set_ylabel('Price (DKK/MWh)', fontsize=13, color='red', fontweight='bold')
    ax_price.tick_params(axis='y', labelcolor='red', labelsize=11)
    
    ax.set_xlabel('Day', fontsize=13, fontweight='bold')
    ax.set_ylabel('Spending', fontsize=13, fontweight='bold')
    ax.set_title(f'Model 1: Daily Spending vs Budget Allowance', fontsize=15, fontweight='bold', pad=20)
    ax.tick_params(labelsize=11)
    
    # Place legends outside
    ax.legend(loc='upper left', fontsize=11, bbox_to_anchor=(0, -0.12), ncol=2, frameon=True)
    ax_price.legend(loc='upper right', fontsize=11, bbox_to_anchor=(1, -0.12), frameon=True)
    
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path.with_suffix('.png')}")
    
    plt.show()


def print_model_1_metrics(primal_results, spending_results, expenditure):
    """
    Print key metrics for Model 1.
    """
    # Extract data
    demand = primal_results['Demand (MWh)']
    bought = primal_results['Bought (MWh)']
    unmet = primal_results['Unmet_Demand (MWh)']
    price = primal_results['Price (DKK/MWh)']
    
    # Calculate metrics
    total_unmet = unmet.sum()
    max_daily_unmet = unmet.max()
    demand_fulfillment_rate = ((demand - unmet).sum() / demand.sum()) * 100
    days_with_unmet = (unmet > 0.01).sum()  # Days with unmet > 0 (tolerance for numerical noise)
    total_fuel_cost = (bought * price).sum()
    
    print("\n" + "="*70)
    print("MODEL 1: KEY PERFORMANCE METRICS")
    print("="*70)
    
    print(f"\n  Total unmet demand (MWh):        {total_unmet:>12.1f}")
    print(f"  Maximum daily unmet demand (MWh): {max_daily_unmet:>12.1f}")
    print(f"  Demand fulfillment rate (%):      {demand_fulfillment_rate:>12.1f}")
    print(f"  Days with unmet demand:           {days_with_unmet:>12d}")
    print(f"  Total fuel cost (DKK):            {total_fuel_cost:>12,.0f}")
    
    print("\n" + "="*70 + "\n")
    
    # Return metrics as dictionary for easy table population
    return {
        'total_unmet': total_unmet,
        'max_daily_unmet': max_daily_unmet,
        'demand_fulfillment_rate': demand_fulfillment_rate,
        'days_with_unmet': days_with_unmet,
        'total_fuel_cost': total_fuel_cost
    }

def plot_model_2_energy_balance(flow_results, spending_results, unmet_demand, save_path=None):
    """
    Plot energy balance for Model 2 with storage integrated into bars.
    Shows storage changes as part of the energy supply/sink.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Extract data
    days = flow_results.index
    demand = flow_results['Demand (MWh)']
    bought = flow_results['Bought (MWh)']
    unmet = flow_results['Unmet_Demand (MWh)']
    stored = flow_results['Stored (MWh)']
    price = spending_results['Price (dkk/MWh)']
    
    # Calculate storage changes (positive = energy added to storage, negative = released from storage)
    storage_change = np.zeros(len(days))
    storage_change[0] = stored[0]  # First day: all storage is added
    for i in range(1, len(days)):
        storage_change[i] = stored[i] - stored[i-1]
    
    # Released from storage (negative change = energy added to supply)
    storage_released = np.maximum(-storage_change, 0)
    
    # Added to storage (positive change = energy taken from supply)
    storage_added = np.maximum(storage_change, 0)
    
    # DEMAND SIDE (negative)
    ax.bar(days, -demand, width=0.8, color='lightgreen', 
           edgecolor='darkgreen', linewidth=1.5, label='Demand', alpha=0.7)
    
    # SUPPLY SIDE (positive)
    # Base: Bought energy
    ax.bar(days, bought, width=0.8, color='steelblue', 
           edgecolor='darkblue', linewidth=1.5, label='Bought', alpha=0.8)
    
    # Add: Energy released from storage (on top of bought)
    ax.bar(days, storage_released, bottom=bought, width=0.8, color='gold', 
           edgecolor='darkgoldenrod', linewidth=1.5, label='Released from Storage', alpha=0.8)
    
    # Add: Unmet demand (on top of bought + released)
    ax.bar(days, unmet, bottom=bought + storage_released, width=0.8, color='coral', 
           edgecolor='darkred', linewidth=1.5, label='Unmet Demand', alpha=0.8)
    
    # Show energy added to storage (as negative bars below bought)
    ax.bar(days, -storage_added, width=0.8, color='purple', 
           edgecolor='darkviolet', linewidth=1.5, label='Added to Storage', alpha=0.6)
    
    # Add zero line
    ax.axhline(0, color='black', linewidth=1.5, linestyle='-')
    
    ax.set_xlabel('Day', fontsize=13, fontweight='bold')
    ax.set_ylabel('Energy (MWh)', fontsize=13, fontweight='bold')
    ax.set_title(f'Model 2: Energy Balance with Storage Flows. Total unmet demand = {unmet_demand:.2f} MWh', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.tick_params(labelsize=11)
    
    # ========================================================================
    # PRICE (secondary axis)
    # ========================================================================
    ax_price = ax.twinx()
    
    ax_price.plot(days, price, color='red', marker='x', linewidth=2.5, 
                  markersize=8, label='Price', linestyle='--', alpha=0.8, zorder=5)
    
    ax_price.set_ylabel('Price (DKK/MWh)', fontsize=13, color='red', fontweight='bold')
    ax_price.tick_params(axis='y', labelcolor='red', labelsize=11)
    
    # Create combined legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    
    energy_legend_elements = [
        Patch(facecolor='lightgreen', edgecolor='darkgreen', label='Demand', alpha=0.7),
        Patch(facecolor='steelblue', edgecolor='darkblue', label='Bought', alpha=0.8),
        Patch(facecolor='gold', edgecolor='darkgoldenrod', label='Released from Storage', alpha=0.8),
        Patch(facecolor='coral', edgecolor='darkred', label='Unmet Demand', alpha=0.8),
        Patch(facecolor='purple', edgecolor='darkviolet', label='Added to Storage', alpha=0.6)
    ]
    
    price_legend_elements = [
        Line2D([0], [0], color='red', marker='x', linewidth=2.5, markersize=8,
               linestyle='--', label='Price', alpha=0.8)
    ]
    
    # Place legends outside
    ax.legend(handles=energy_legend_elements, loc='upper left', fontsize=10, 
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


def plot_model_2_economics(spending_results, unmet_demand, save_path=None):
    """
    Plot economic view for Model 2 - simplified and clearer.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Extract data
    days = spending_results.index
    bought_fuel = spending_results['Bought fuel']
    storage_cost = spending_results['Storage cost']
    carry_over = spending_results['Carry over budget']
    allowance = spending_results['Allowance'].iloc[0]
    price = spending_results['Price (dkk/MWh)']
    
    # Plot spent (negative) and available budget (positive)
    # Spending (negative bars)
    ax.bar(days, -bought_fuel, width=0.8, color='steelblue', 
           edgecolor='darkblue', linewidth=1, label='Fuel Cost', alpha=0.8)
    ax.bar(days, -storage_cost, bottom=-bought_fuel, width=0.8, color='orange', 
           edgecolor='darkorange', linewidth=1, label='Storage Cost', alpha=0.8)
    
    # Available budget (positive bars)
    ax.bar(days, carry_over, width=0.8, color='lightgreen', 
           edgecolor='darkgreen', linewidth=1, label='Carry Over Budget', alpha=0.7)
    ax.bar(days, allowance, bottom=carry_over, width=0.8, color='mediumpurple', 
           edgecolor='purple', linewidth=1, label='Allowance', alpha=0.7)
    
    # Add zero line
    ax.axhline(0, color='black', linewidth=1.5, linestyle='-')
    
    # Add price line on secondary axis
    ax_price = ax.twinx()
    ax_price.plot(days, price, color='red', marker='x', linewidth=2.5, 
                  markersize=8, label='Price', linestyle='--', alpha=0.7)
    ax_price.set_ylabel('Price (DKK/MWh)', fontsize=13, color='red', fontweight='bold')
    ax_price.tick_params(axis='y', labelcolor='red', labelsize=11)
    
    ax.set_xlabel('Day', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cash Flow (DKK)', fontsize=13, fontweight='bold')
    ax.set_title(f'Model 2: Budget Dynamics - Spending vs Available Funds', fontsize=15, fontweight='bold', pad=20)
    ax.tick_params(labelsize=11)
    
    # Place legends outside
    ax.legend(loc='upper left', fontsize=11, bbox_to_anchor=(0, -0.12), ncol=4, frameon=True)
    ax_price.legend(loc='upper right', fontsize=11, bbox_to_anchor=(1, -0.12), frameon=True)
    
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path.with_suffix('.png')}")
    
    plt.show()



def print_model_2_metrics(flow_results, spending_results, budget, unmet_demand):
    """
    Print key metrics for Model 2.
    """
    # Extract data
    demand = flow_results['Demand (MWh)']
    bought = flow_results['Bought (MWh)']
    unmet = flow_results['Unmet_Demand (MWh)']
    stored = flow_results['Stored (MWh)']
    
    price = spending_results['Price (dkk/MWh)']
    carry_over = spending_results['Carry over budget']
    allowance = spending_results['Allowance'].iloc[0]
    
    # Calculate metrics
    total_unmet = unmet.sum()
    max_daily_unmet = unmet.max()
    demand_fulfillment_rate = ((demand - unmet).sum() / demand.sum()) * 100
    days_with_unmet = (unmet > 0.01).sum()
    total_fuel_cost = (bought * price).sum()
    total_storage_used = stored.sum()
    total_carry_over_used = carry_over.sum()
    
    print("\n" + "="*70)
    print("MODEL 2: KEY PERFORMANCE METRICS")
    print("="*70)
    
    print(f"\n  Total unmet demand (MWh):        {total_unmet:>12.1f}")
    print(f"  Maximum daily unmet demand (MWh): {max_daily_unmet:>12.1f}")
    print(f"  Demand fulfillment rate (%):      {demand_fulfillment_rate:>12.1f}")
    print(f"  Days with unmet demand:           {days_with_unmet:>12d}")
    print(f"  Total fuel cost (DKK):            {total_fuel_cost:>12,.0f}")
    print(f"  Total storage used (MWh):         {total_storage_used:>12.1f}")
    print(f"  Total carry-over budget used (DKK): {total_carry_over_used:>12,.0f}")
    
    print("\n" + "="*70 + "\n")
    
    # Return metrics as dictionary for easy table population
    return {
        'total_unmet': total_unmet,
        'max_daily_unmet': max_daily_unmet,
        'demand_fulfillment_rate': demand_fulfillment_rate,
        'days_with_unmet': days_with_unmet,
        'total_fuel_cost': total_fuel_cost,
        'total_storage_used': total_storage_used,
        'total_carry_over_used': total_carry_over_used
    }

def plot_model_2_storage_vs_budget_over_time(flow_results, spending_results, save_path=None):
    """
    Plot storage and budget dynamics over time with two-level demand context.
    """
    fig, ax = plt.subplots(figsize=(14, 7))  # Increased height for legend below
    
    # Extract data
    days = flow_results.index
    stored = flow_results['Stored (MWh)']
    demand = flow_results['Demand (MWh)']
    carry_over = spending_results['Carry over budget']
    
    # Plot storage on primary axis
    ax.plot(days, stored, color='goldenrod', marker='o', linewidth=2.5, 
            markersize=6, label='Storage Level', alpha=0.8, zorder=5)
    
    # Plot carry-over budget on secondary axis
    ax_budget = ax.twinx()
    ax_budget.plot(days, carry_over, color='darkgreen', marker='s', linewidth=2.5,
                   markersize=6, label='Carry Over Budget', alpha=0.8, zorder=5)
    
    # Two-level demand shading
    high_demand_threshold = demand.quantile(0.67)    # Top 33% = "High"
    very_high_demand_threshold = demand.quantile(0.85)  # Top 15% = "Very High"
    
    # Shade high-demand periods (light yellow)
    for i in range(len(days)):
        if demand[i] > high_demand_threshold and demand[i] <= very_high_demand_threshold:
            ax.axvspan(days[i]-0.5, days[i]+0.5, alpha=0.25, color='yellow', 
                      hatch='---', edgecolor='orange', linewidth=0, zorder=2)
    
    # Shade very high-demand periods (darker orange)
    for i in range(len(days)):
        if demand[i] > very_high_demand_threshold:
            ax.axvspan(days[i]-0.5, days[i]+0.5, alpha=0.35, color='darkorange', 
                      hatch='xxx', edgecolor='darkred', linewidth=0, zorder=3)
    
    ax.set_xlabel('Day', fontsize=13, fontweight='bold')
    ax.set_ylabel('Storage Level (MWh)', fontsize=13, color='goldenrod', fontweight='bold')
    ax_budget.set_ylabel('Carry Over Budget (DKK)', fontsize=13, color='darkgreen', fontweight='bold')
    ax.set_title('Model 2: Storage vs Carry Over budget usage in anticipation of high demand', 
                 fontsize=15, fontweight='bold', pad=20)
    
    ax.tick_params(axis='y', labelcolor='goldenrod', labelsize=11)
    ax_budget.tick_params(axis='y', labelcolor='darkgreen', labelsize=11)
    ax.tick_params(axis='x', labelsize=11)
    
    ax.legend(loc='upper left', fontsize=11, frameon=True)
    ax_budget.legend(loc='upper right', fontsize=11, frameon=True)
    ax.grid(True, alpha=0.3)
    
    # Add shading legend below the plot
    fig.text(0.5, 0.02, 
             f'Yellow (- - -) = High Demand (> {high_demand_threshold:.1f} MWh)  |  '
             f'Orange (×××) = Very High Demand (> {very_high_demand_threshold:.1f} MWh)',
             ha='center', fontsize=11, 
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, 
                      edgecolor='orange', linewidth=1.5))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)  # Make room for legend below
    
    # Save
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path.with_suffix('.png')}")
    
    plt.show()

def plot_model_2_storage_strategy(flow_results, spending_results, save_path=None):
    """
    Simplified view focusing on storage strategy vs demand and price.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    days = flow_results.index
    stored = flow_results['Stored (MWh)']
    demand = flow_results['Demand (MWh)']
    price = spending_results['Price (dkk/MWh)']
    
    # ========================================================================
    # TOP: Storage vs Demand
    # ========================================================================
    ax1_demand = ax1.twinx()
    
    # Plot storage as area
    ax1.fill_between(days, 0, stored, color='goldenrod', alpha=0.5, label='Storage Level')
    ax1.plot(days, stored, color='darkgoldenrod', linewidth=3, marker='o', markersize=6)
    
    # Plot demand as bars
    ax1_demand.bar(days, demand, alpha=0.3, color='steelblue', width=0.8, 
                   label='Demand', edgecolor='darkblue', linewidth=1)
    
    ax1.set_ylabel('Storage Level (MWh)', fontsize=13, fontweight='bold', color='darkgoldenrod')
    ax1_demand.set_ylabel('Demand (MWh)', fontsize=13, fontweight='bold', color='steelblue')
    ax1.set_title('Model 2: Storage Strategy - Building Reserves Before High Demand', 
                  fontsize=15, fontweight='bold', pad=15)
    
    ax1.tick_params(axis='y', labelcolor='darkgoldenrod', labelsize=11)
    ax1_demand.tick_params(axis='y', labelcolor='steelblue', labelsize=11)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_demand.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # ========================================================================
    # BOTTOM: Storage vs Price
    # ========================================================================
    ax2_price = ax2.twinx()
    
    # Plot storage as area
    ax2.fill_between(days, 0, stored, color='goldenrod', alpha=0.5, label='Storage Level')
    ax2.plot(days, stored, color='darkgoldenrod', linewidth=3, marker='o', markersize=6)
    
    # Plot price as line
    ax2_price.plot(days, price, color='crimson', linewidth=2.5, marker='^', 
                   markersize=6, linestyle='--', label='Price', alpha=0.8)
    ax2_price.axhline(price.mean(), color='darkred', linestyle=':', linewidth=1.5,
                      alpha=0.5, label=f'Avg Price ({price.mean():.1f})')
    
    ax2.set_xlabel('Day', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Storage Level (MWh)', fontsize=13, fontweight='bold', color='darkgoldenrod')
    ax2_price.set_ylabel('Price (DKK/MWh)', fontsize=13, fontweight='bold', color='crimson')
    
    ax2.tick_params(axis='y', labelcolor='darkgoldenrod', labelsize=11)
    ax2_price.tick_params(axis='y', labelcolor='crimson', labelsize=11)
    
    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_price.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path.with_suffix('.png')}")
    
    plt.show()


def create_decision_matrix(flow_results, spending_results, save_path=None):
    """
    Create a decision matrix showing storage charging and budget carry-over decisions
    based on discretized price and demand levels (4 levels each).
    """
    # Extract data
    demand = flow_results['Demand (MWh)']
    stored = flow_results['Stored (MWh)']
    price = spending_results['Price (dkk/MWh)']
    bought_fuel = spending_results['Bought fuel']
    storage_cost = spending_results['Storage cost']
    carry_over = spending_results['Carry over budget']
    allowance = spending_results['Allowance']
    
    # Calculate storage change (charging vs discharging)
    storage_change = np.zeros(len(stored))
    storage_change[0] = stored[0]  # Day 0: any storage is charging
    for t in range(1, len(stored)):
        storage_change[t] = stored[t] - stored[t-1]
    
    # DECISION 1: Is storage being CHARGED? (increased)
    charges_storage = (storage_change > 0.1).astype(int)  # Threshold for numerical noise
    
    # DECISION 2: Was budget CARRIED OVER? (spent less than available)
    carries_budget = np.zeros(len(stored), dtype=int)
    
    for t in range(len(stored)):
        if t == 0:
            # Day 0: available budget = allowance only
            available_budget = allowance.iloc[0]
        else:
            # Later days: available budget = carry_over from previous + allowance
            available_budget = carry_over[t-1] + allowance.iloc[0]
        
        # Total spending this day
        total_spending = bought_fuel[t] + storage_cost[t]
        
        # If spent less than available, they carried over budget
        if available_budget - total_spending > 1.0:  # Carried over at least 1 DKK
            carries_budget[t] = 1
        else:
            carries_budget[t] = 0
    
    # Create combined decision categories
    decision_type = charges_storage + 2 * carries_budget
    
    decision_labels = ['Neither', 'Charge Storage', 'Carry Budget', 'Both']
    
    # DETAILED DIAGNOSTICS
    print("\n" + "="*80)
    print("DETAILED DECISION DIAGNOSTICS - CHECKING FOR 'BOTH'")
    print("="*80)
    
    print(f"\nTotal days: {len(stored)}")
    print(f"Days charging storage: {np.sum(charges_storage)}")
    print(f"Days carrying budget: {np.sum(carries_budget)}")
    print(f"Days doing BOTH: {np.sum((charges_storage == 1) & (carries_budget == 1))}")
    print(f"Days doing NEITHER: {np.sum((charges_storage == 0) & (carries_budget == 0))}")
    
    print("\nDecision type distribution:")
    from collections import Counter
    overall_counts = Counter(decision_type)
    for dec_val in range(4):
        count = overall_counts.get(dec_val, 0)
        print(f"  {decision_labels[dec_val]:<20}: {count} days (decision_type={dec_val})")
    
    print("\n" + "-"*80)
    print("ALL DAYS BREAKDOWN:")
    print("-"*80)
    print(f"{'Day':>3} {'Storage':>8} {'Δ Storage':>10} {'Charge?':>8} {'Available':>12} "
          f"{'Spending':>10} {'Leftover':>10} {'Carry?':>7} {'Decision':>15}")
    print("-"*80)
    
    for t in range(len(stored)):
        if t == 0:
            available = allowance.iloc[0]
        else:
            available = carry_over[t-1] + allowance.iloc[0]
        
        spending = bought_fuel[t] + storage_cost[t]
        leftover = available - spending
        
        print(f"{t:>3} {stored[t]:>8.1f} {storage_change[t]:>10.1f} {charges_storage[t]:>8} "
              f"{available:>12.1f} {spending:>10.1f} {leftover:>10.1f} {carries_budget[t]:>7} "
              f"{decision_labels[decision_type[t]]:>15}")
    
    print("\n" + "="*80)
    print("DAYS WHERE BOTH SHOULD BE TRUE:")
    print("="*80)
    
    both_mask = (charges_storage == 1) & (carries_budget == 1)
    if np.sum(both_mask) > 0:
        print(f"Found {np.sum(both_mask)} days with BOTH decisions:")
        both_days = np.where(both_mask)[0]
        for day in both_days:
            print(f"\n  Day {day}:")
            print(f"    Storage change: {storage_change[day]:+.1f} (charges={charges_storage[day]})")
            if day == 0:
                avail = allowance.iloc[0]
            else:
                avail = carry_over[day-1] + allowance.iloc[0]
            spend = bought_fuel[day] + storage_cost[day]
            left = avail - spend
            print(f"    Budget: available={avail:.1f}, spent={spend:.1f}, leftover={left:.1f} (carries={carries_budget[day]})")
            print(f"    Decision type value: {decision_type[day]} (should be 3)")
    else:
        print("NO days found where both decisions are true!")
        print("\nPossible issues:")
        print("  1. Thresholds too strict (storage > 0.1, budget leftover > 1.0)")
        print("  2. Model never does both simultaneously")
        print("\nLet's check with looser thresholds:")
        
        loose_storage = (storage_change > 0.01).astype(int)
        loose_budget = np.zeros(len(stored), dtype=int)
        for t in range(len(stored)):
            if t == 0:
                avail = allowance.iloc[0]
            else:
                avail = carry_over[t-1] + allowance.iloc[0]
            spend = bought_fuel[t] + storage_cost[t]
            loose_budget[t] = 1 if (avail - spend > 0.01) else 0
        
        loose_both = np.sum((loose_storage == 1) & (loose_budget == 1))
        print(f"  With loose thresholds (>0.01): {loose_both} days with BOTH")
    
    # Discretize demand and price into 4 levels
    demand_bins = pd.qcut(demand, q=4, labels=['Low', 'Med-Low', 'Med-High', 'High'], duplicates='drop')
    price_bins = pd.qcut(price, q=4, labels=['Low', 'Med-Low', 'Med-High', 'High'], duplicates='drop')
    
    # Aggregate by (demand_bin, price_bin)
    matrix_data = {}
    demand_levels = ['Low', 'Med-Low', 'Med-High', 'High']
    price_levels = ['Low', 'Med-Low', 'Med-High', 'High']
    
    for d_level in demand_levels:
        for p_level in price_levels:
            mask = (demand_bins == d_level) & (price_bins == p_level)
            decisions_in_cell = decision_type[mask]
            
            if len(decisions_in_cell) > 0:
                decision_counts = Counter(decisions_in_cell)
                most_common = decision_counts.most_common(1)[0][0]
                matrix_data[(d_level, p_level)] = most_common
            else:
                matrix_data[(d_level, p_level)] = -1
    
    # Create matrix for heatmap (4x4)
    matrix = np.zeros((len(price_levels), len(demand_levels)))
    for i, p_level in enumerate(price_levels):
        for j, d_level in enumerate(demand_levels):
            matrix[i, j] = matrix_data.get((d_level, p_level), -1)
    
    # ========================================================================
    # PLOT
    # ========================================================================
    fig, ax = plt.subplots(figsize=(12, 10))
    
    from matplotlib.colors import ListedColormap, BoundaryNorm
    colors = ['lightgray', 'steelblue', 'lightgreen', 'mediumpurple']
    cmap = ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = BoundaryNorm(bounds, cmap.N)
    
    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect='auto', origin='lower')
    
    ax.set_xticks(np.arange(len(demand_levels)))
    ax.set_yticks(np.arange(len(price_levels)))
    ax.set_xticklabels(demand_levels, fontsize=11, fontweight='bold')
    ax.set_yticklabels(price_levels, fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Demand Level', fontsize=14, fontweight='bold')
    ax.set_ylabel('Price Level', fontsize=14, fontweight='bold')
    ax.set_title('Model 2: Decision Matrix - Storage Charging & Budget Carry-Over Strategy', 
                 fontsize=15, fontweight='bold', pad=20)
    
    # Annotations
    for i, p_level in enumerate(price_levels):
        for j, d_level in enumerate(demand_levels):
            decision_val = matrix[i, j]
            
            mask = (demand_bins == d_level) & (price_bins == p_level)
            n_days = np.sum(mask)
            
            if decision_val >= 0:
                label = decision_labels[int(decision_val)]
                short_label = label.replace('Charge Storage', 'Storage').replace('Carry Budget', 'Budget')
                text = f"{short_label}\n({n_days}d)"
                color = 'white' if decision_val == 3 else 'black'
                ax.text(j, i, text, ha='center', va='center', 
                       fontsize=9, fontweight='bold', color=color)
            else:
                ax.text(j, i, 'No\ndata', ha='center', va='center', 
                       fontsize=8, style='italic', color='gray')
    
    ax.set_xticks(np.arange(len(demand_levels)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(price_levels)) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
    
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3], boundaries=bounds)
    cbar.set_label('Decision Type', fontsize=12, fontweight='bold')
    cbar.ax.set_yticklabels(decision_labels, fontsize=10)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightgray', edgecolor='black', label='Neither'),
        Patch(facecolor='steelblue', edgecolor='black', label='Storage only'),
        Patch(facecolor='lightgreen', edgecolor='black', label='Budget only'),
        Patch(facecolor='mediumpurple', edgecolor='black', label='Both')
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9, 
              bbox_to_anchor=(0, -0.12), ncol=4, frameon=True)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path.with_suffix('.png')}")
    
    plt.show()

def plot_model2_procurement_plan(flow_results, spending_results, unmet_demand, save_path=None):
    """
    Plot the procurement plan for Model 2 (deterministic - no scenarios).
    Shows storage and budget carry-over evolution over time.
    Designed to match Model 3 visualization for comparison.
    
    Parameters:
    - flow_results: Flow results dataframe from Model 2
    - spending_results: Spending results dataframe from Model 2
    - unmet_demand: Total unmet demand (scalar)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    
    days = flow_results.index
    
    # Extract data
    storage = flow_results['Stored (MWh)']
    carry_over = spending_results['Carry over budget']
    allowance = spending_results['Allowance'].iloc[0]
    
    # Calculate total stored and total carry-over
    total_stored = storage.sum()
    total_carry_over = carry_over.sum()
    
    # ========================================================================
    # TOP PANEL: STORAGE
    # ========================================================================
    
    # Plot storage strategy
    ax1.fill_between(days, 0, storage, color='steelblue', alpha=0.4)
    ax1.plot(days, storage, color='darkblue', linewidth=3, marker='o', 
            markersize=5, linestyle='-', label='Storage Level')
    
    ax1.set_ylabel('Storage Level (MWh)', fontsize=13, fontweight='bold')
    ax1.set_title(f'Model 2: Deterministic Procurement Plan\n' +
                  f'Storage and Budget Carry-Over Evolution (Total Unmet Demand: {unmet_demand:.1f} MWh)',
                 fontsize=15, fontweight='bold', pad=15)
    ax1.legend(loc='upper left', fontsize=11, framealpha=0.95)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.tick_params(labelsize=11)
    
    # Add statistics annotation on RIGHT
    stats_text_storage = f"""Total Stored Over Period:
{total_stored:.0f} MWh"""
    
    ax1.text(0.98, 0.98, stats_text_storage, transform=ax1.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9, pad=0.8),
            family='monospace')
    
    # ========================================================================
    # BOTTOM PANEL: CARRY-OVER BUDGET
    # ========================================================================
    
    # Plot carry-over strategy
    ax2.fill_between(days, 0, carry_over, color='steelblue', alpha=0.4)
    ax2.plot(days, carry_over, color='darkblue', linewidth=3.5, marker='o', 
            markersize=6, linestyle='-', label='Budget Carry-Over', zorder=5)
    
    # Add zero line
    ax2.axhline(0, color='black', linewidth=1.5, linestyle='-', alpha=0.5, zorder=3)
    
    # Add grid
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax2.set_ylabel('Budget Carry-Over (DKK)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Day', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=11, framealpha=0.95)
    ax2.tick_params(labelsize=11)
    
    # Add statistics annotation on RIGHT
    stats_text_carryover = f"""Total Carry-Over Over Period:
{total_carry_over:.0f} DKK"""
    
    ax2.text(0.98, 0.98, stats_text_carryover, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
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
    # PRINT SUMMARY
    # ========================================================================
    
    print("\n" + "="*80)
    print("MODEL 2: DETERMINISTIC PROCUREMENT PLAN SUMMARY")
    print("="*80)
    
    print("\n📦 STORAGE STRATEGY:")
    print(f"  Total Stored: {total_stored:.0f} MWh")
    print(f"  Maximum Storage: {storage.max():.0f} MWh on day {storage.idxmax()}")
    print(f"  Average Storage: {storage.mean():.0f} MWh")
    print(f"  Days with storage > 0: {np.sum(storage > 1)}")
    
    print("\n💵 CARRY-OVER BUDGET STRATEGY:")
    print(f"  Total Carry-Over: {total_carry_over:.0f} DKK")
    print(f"  Maximum Carry-Over: {carry_over.max():.0f} DKK on day {carry_over.idxmax()}")
    print(f"  Average Carry-Over: {carry_over.mean():.0f} DKK")
    print(f"  Days with positive carry-over: {np.sum(carry_over > 1)}")
    
    print("\n📊 PERFORMANCE:")
    print(f"  Total Unmet Demand: {unmet_demand:.2f} MWh")
    
    print("\n" + "="*80)


# ============================================================================
# USAGE
# ============================================================================


# ============================================================================
# USAGE
# ============================================================================


