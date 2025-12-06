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
    ax_price.set_ylabel('Price (DKK/MWh)', fontsize=18, color='red', fontweight='bold')
    ax_price.tick_params(axis='y', labelcolor='red', labelsize=17)
    
    ax.set_xlabel('Day', fontsize=18, fontweight='bold')
    ax.set_ylabel('Gas (MWh)', fontsize=18, fontweight='bold')
    ax.set_title(f'Model 1: Gas flows. Total Unmet Demand = {unmet_demand:.2f} MWh', fontsize=20, fontweight='bold', pad=20)
    ax.tick_params(labelsize=17)
    
    # Place legends outside
    ax.legend(loc='upper left', fontsize=15, bbox_to_anchor=(0, -0.12), ncol=3, frameon=True)
    ax_price.legend(loc='upper right', fontsize=15, bbox_to_anchor=(1, -0.12), frameon=True)
    
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
    ax_price.set_ylabel('Price (DKK/MWh)', fontsize=18, color='red', fontweight='bold')
    ax_price.tick_params(axis='y', labelcolor='red', labelsize=17)
    
    ax.set_xlabel('Day', fontsize=18, fontweight='bold')
    ax.set_ylabel('Spending', fontsize=18, fontweight='bold')
    ax.set_title(f'Model 1: Daily Spending vs Budget Allowance', fontsize=20, fontweight='bold', pad=20)
    ax.tick_params(labelsize=17)
    
    # Place legends outside
    ax.legend(loc='upper left', fontsize=15, bbox_to_anchor=(0, -0.12), ncol=2, frameon=True)
    ax_price.legend(loc='upper right', fontsize=15, bbox_to_anchor=(1, -0.12), frameon=True)
    
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path.with_suffix('.png')}")
    
    plt.show()


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
    
    ax.set_xlabel('Day', fontsize=18, fontweight='bold')
    ax.set_ylabel('Energy (MWh)', fontsize=18, fontweight='bold')
    ax.set_title(f'Model 2: Energy Balance with Storage Flows. Total unmet demand = {unmet_demand:.2f} MWh', 
                 fontsize=20, fontweight='bold', pad=20)
    ax.tick_params(labelsize=17)
    
    # ========================================================================
    # PRICE (secondary axis)
    # ========================================================================
    ax_price = ax.twinx()
    
    ax_price.plot(days, price, color='red', marker='x', linewidth=2.5, 
                  markersize=8, label='Price', linestyle='--', alpha=0.8, zorder=5)
    
    ax_price.set_ylabel('Price (DKK/MWh)', fontsize=18, color='red', fontweight='bold')
    ax_price.tick_params(axis='y', labelcolor='red', labelsize=17)
    
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
    ax.legend(handles=energy_legend_elements, loc='upper left', fontsize=15, 
              bbox_to_anchor=(0, -0.12), ncol=3, frameon=True)
    ax_price.legend(handles=price_legend_elements, loc='upper right', fontsize=15, 
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
    ax_price.set_ylabel('Price (DKK/MWh)', fontsize=18, color='red', fontweight='bold')
    ax_price.tick_params(axis='y', labelcolor='red', labelsize=17)
    
    ax.set_xlabel('Day', fontsize=18, fontweight='bold')
    ax.set_ylabel('Cash Flow (DKK)', fontsize=18, fontweight='bold')
    ax.set_title(f'Model 2: Budget Dynamics - Spending vs Available Funds', fontsize=20, fontweight='bold', pad=20)
    ax.tick_params(labelsize=17)
    
    # Place legends outside
    ax.legend(loc='upper left', fontsize=15, bbox_to_anchor=(0, -0.12), ncol=4, frameon=True)
    ax_price.legend(loc='upper right', fontsize=15, bbox_to_anchor=(1, -0.12), frameon=True)
    
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
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
    
    ax1.set_ylabel('Storage Level (MWh)', fontsize=18, fontweight='bold', color='darkgoldenrod')
    ax1_demand.set_ylabel('Demand (MWh)', fontsize=18, fontweight='bold', color='steelblue')
    ax1.set_title('Model 2: Storage Strategy - Building Reserves Before High Demand', 
                  fontsize=20, fontweight='bold', pad=15)
    
    ax1.tick_params(axis='y', labelcolor='darkgoldenrod', labelsize=17)
    ax1_demand.tick_params(axis='y', labelcolor='steelblue', labelsize=17)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_demand.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=15)
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
    
    ax2.set_xlabel('Day', fontsize=18, fontweight='bold')
    ax2.set_ylabel('Storage Level (MWh)', fontsize=18, fontweight='bold', color='darkgoldenrod')
    ax2_price.set_ylabel('Price (DKK/MWh)', fontsize=18, fontweight='bold', color='crimson')
    
    ax2.tick_params(axis='y', labelcolor='darkgoldenrod', labelsize=17)
    ax2_price.tick_params(axis='y', labelcolor='crimson', labelsize=17)
    
    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_price.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=15)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path.with_suffix('.png')}")
    
    plt.show()

