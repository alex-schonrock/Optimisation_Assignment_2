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

def plot_primal_results(df: pd.DataFrame, save_path: str | None = None, show: bool = True, 
                               show_price_line = True, title: str = "Stacked flows vs time (with gas price)",
                               line_label: str = "Price (DKK/kWh)", y_axis_label: str = "Gas Flows (kWh)"):
    """Plots the primals in a stacked bar chart, with an optional line for price"""
    if show_price_line == True:
        price_ser = df.pop(line_label)
        # if line_label == "Deviation Down (kWh)":
        #     price_ser2 = df.pop("Deviation Up (kWh)")
    cols = [c for c in df.columns if not df[c].dropna().empty]
    if not cols:
        raise ValueError("No numeric columns to plot.")

    # to_neg = [c for c in cols if "demand (btu)" in c.lower()]
    cols_to_neg = [
    "Demand (MWh)",
    "Bought fuel",
    "Unmet demand",
    "Storage cost"
    ]

    to_neg = [c for c in cols if c in cols_to_neg]
    # to_neg = [c for c in cols if c == "Demand (Btu)"]
    df_plot = df.copy()
    for c in to_neg:
        df_plot[c] = -df_plot[c]
    
    x = np.arange(len(df_plot.index))
    x_labels = df_plot.index
    x_label = df.index.name or "Time"

    fig, ax = plt.subplots(figsize=(10, 6))
    cum_pos = np.zeros(len(df_plot), dtype=float)
    cum_neg = np.zeros(len(df_plot), dtype=float)

    for col in cols:
        vals = df_plot[col].fillna(0).to_numpy(dtype=float)
        pos = np.where(vals > 0, vals, 0.0)
        neg = np.where(vals < 0, vals, 0.0)

        # positives stack upward
        if pos.any():
            ax.bar(x, pos, bottom=cum_pos, label=str(col), alpha = 0.7)
            cum_pos += pos

        # negatives stack downward (skip legend duplication)
        if neg.any():
            ax.bar(x, neg, bottom=cum_neg, label=str(col))
            cum_neg += neg
    handles, labels = ax.get_legend_handles_labels()
    ax2 = None
    if show_price_line:
        ax2 = ax.twinx()
        price_line, = ax2.plot(x, price_ser.reindex(df.index).to_numpy(dtype=float), marker="x", color="red", linestyle="--" ,label=line_label)
        price_line.set_zorder(3)  # draw line above bars
        ax2.tick_params(axis="y", colors="red")
        ax2.spines["right"].set_color("red")
        ax2.set_ylabel(line_label, color="red")
        # if line_label == "Price (DKK/kWh)":
        #     ax2.set_ylim(-1.5, 3)
        # Merge legends
        h2, l2 = ax2.get_legend_handles_labels()
        handles += h2
        labels += l2
    else:
        for extra_ax in getattr(fig, "axes", [])[1:]:
            extra_ax.remove()
        # keep legend for the bars only
    ax.legend(handles, labels, loc="best")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_axis_label)
    ax.set_title(title)
    ax.yaxis.set_minor_locator(AutoMinorLocator(4)) 
    ax.set_xticks(x, labels=x_labels)
    # ax.set_ylim(-3,3)
    ax.grid(True, which="major", axis="y")
    ax.grid(True, which="minor", axis="y", alpha=0.4)
    ax.legend()
    fig.tight_layout()

    if handles:
        ax.legend(handles, labels, bbox_to_anchor=(1.07, 1), loc='upper left')
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

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
    ax.set_ylabel('Energy (MWh)', fontsize=13, fontweight='bold')
    ax.set_title(f'Model 1: Energy Balance. Total unmet demand = {unmet_demand:.2f} MWh', fontsize=15, fontweight='bold', pad=20)
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
    colors = ['steelblue' if spend <= allowance else 'red' for spend in daily_spending]
    ax.bar(days, daily_spending, width=0.8, color=colors, 
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
    ax.set_title(f'Model 1: Daily Spending vs Budget Allowance. Total unmet demand = {unmet_demand:.2f} MWh', fontsize=15, fontweight='bold', pad=20)
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
    daily_spending = spending_results['Bought fuel (1000 DKK)']
    allowance = spending_results['Allowance (1000 DKK)'].iloc[0]
    
    # Calculate metrics
    total_unmet = unmet.sum()
    total_cost = (bought * price).sum()
    avg_price = price.mean()
    demand_met_pct = ((demand - unmet).sum() / demand.sum()) * 100
    days_over_budget = (daily_spending > allowance).sum()
    days_under_budget = len(daily_spending) - days_over_budget
    
    print("\n" + "="*70)
    print("MODEL 1: KEY METRICS")
    print("="*70)
    print(f"\n{'ENERGY METRICS:':<30}")
    print(f"  Total Demand:              {demand.sum():>12.1f} MWh")
    print(f"  Total Bought:              {bought.sum():>12.1f} MWh")
    print(f"  Total Unmet Demand:        {total_unmet:>12.1f} MWh")
    print(f"  Demand Met:                {demand_met_pct:>12.1f} %")
    print(f"  Peak Unmet:                {unmet.max():>12.1f} MWh (day {unmet.idxmax()})")
    
    print(f"\n{'ECONOMIC METRICS:':<30}")
    print(f"  Total Cost:                {total_cost:>12.1f} DKK")
    print(f"  Daily Allowance:           {allowance:>12.1f} DKK")
    print(f"  Average Price:             {avg_price:>12.1f} DKK/MWh")
    print(f"  Peak Price:                {price.max():>12.1f} DKK/MWh (day {price.idxmax()})")
    print(f"  Min Price:                 {price.min():>12.1f} DKK/MWh (day {price.idxmin()})")
    
    print(f"\n{'BUDGET UTILIZATION:':<30}")
    print(f"  Days within budget:        {days_under_budget:>12d} days")
    print(f"  Days over budget:          {days_over_budget:>12d} days")
    print(f"  Budget utilization:        {days_under_budget/len(daily_spending)*100:>12.1f} %")
    
    print("="*70 + "\n")

def plot_model_2_energy_balance(flow_results, unmet_demand, save_path=None):
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
    
    # Show energy added to storage (as negative bars below bought, or as markers)
    # Option A: Show as negative component
    ax.bar(days, -storage_added, width=0.8, color='purple', 
           edgecolor='darkviolet', linewidth=1.5, label='Added to Storage', alpha=0.6)
    
    # Add zero line
    ax.axhline(0, color='black', linewidth=1.5, linestyle='-')
    
    ax.set_xlabel('Day', fontsize=13, fontweight='bold')
    ax.set_ylabel('Energy (MWh)', fontsize=13, fontweight='bold')
    ax.set_title(f'Model 2: Energy Balance with Storage Flows. Total unmet demand = {unmet_demand:.2f} MWh', fontsize=15, fontweight='bold', pad=20)
    ax.tick_params(labelsize=11)
    
    # Place legend outside
    ax.legend(loc='upper center', fontsize=11, bbox_to_anchor=(0.5, -0.12), ncol=5, frameon=True)
    
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
    ax.set_title(f'Model 2: Budget Dynamics - Spending vs Available Funds. Total unmet demand = {unmet_demand:.2f} MWh', fontsize=15, fontweight='bold', pad=20)
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

def plot_model_2_economics_alternative(spending_results, unmet_demand, save_path=None):
    """
    Plot economic view for Model 2 - spending bars with total budget line.
    Price shown in separate subplot below.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True, 
                                     gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.15})
    
    # Extract data
    days = spending_results.index
    bought_fuel = spending_results['Bought fuel']
    storage_cost = spending_results['Storage cost']
    carry_over = spending_results['Carry over budget']
    allowance = spending_results['Allowance'].iloc[0]
    price = spending_results['Price (dkk/MWh)']
    
    # Calculate total budget
    total_budget = carry_over + allowance
    
    # ========================================================================
    # TOP PANEL: SPENDING AND BUDGET
    # ========================================================================
    
    # SPENDING BARS (stacked)
    ax1.bar(days, bought_fuel, width=0.8, color='steelblue', 
            edgecolor='darkblue', linewidth=1, label='Fuel Cost', alpha=0.8)
    ax1.bar(days, storage_cost, bottom=bought_fuel, width=0.8, 
            color='purple', edgecolor='darkviolet', linewidth=1, 
            label='Storage Cost', alpha=0.7)
    
    # TOTAL BUDGET LINE (orange like Model 1 allowance)
    ax1.plot(days, total_budget, color='orange', linewidth=2.5, 
             linestyle='--', marker='o', markersize=6, 
             label='Total Budget (Allowance + Carry Over)', alpha=0.8, zorder=10)
    
    # Styling
    ax1.set_ylabel('Budget (DKK)', fontsize=13, fontweight='bold')
    ax1.set_title(f'Model 2: Daily Spending vs Total Budget. Total unmet demand = {unmet_demand:.2f} MWh', 
                  fontsize=15, fontweight='bold', pad=20)
    ax1.tick_params(labelsize=11)
    ax1.legend(loc='upper left', fontsize=11, frameon=True)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # ========================================================================
    # BOTTOM PANEL: PRICE
    # ========================================================================
    
    ax2.plot(days, price, color='red', marker='x', linewidth=2.5, 
             markersize=8, label='Price', linestyle='--', alpha=0.7)
    
    ax2.set_xlabel('Day', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Price (DKK/MWh)', fontsize=12, fontweight='bold', color='red')
    ax2.tick_params(axis='y', labelcolor='red', labelsize=11)
    ax2.tick_params(axis='x', labelsize=11)
    ax2.set_xticks(days)
    ax2.legend(loc='upper right', fontsize=10, frameon=True)
    ax2.grid(True, alpha=0.3)
    
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
    
    bought_fuel = spending_results['Bought fuel']
    storage_cost = spending_results['Storage cost']
    price = spending_results['Price (dkk/MWh)']
    allowance = spending_results['Allowance'].iloc[0]
    
    # Calculate metrics
    total_unmet = unmet.sum()
    total_fuel_cost = bought_fuel.sum()
    total_storage_cost = storage_cost.sum()
    total_cost = total_fuel_cost + total_storage_cost
    avg_price = price.mean()
    demand_met_pct = ((demand - unmet).sum() / demand.sum()) * 100
    avg_stored = stored.mean()
    max_stored = stored.max()
    storage_utilization = (stored > 0).sum()
    
    print("\n" + "="*70)
    print("MODEL 2: KEY METRICS")
    print("="*70)
    print(f"\n{'ENERGY METRICS:':<30}")
    print(f"  Total Demand:              {demand.sum():>12.1f} MWh")
    print(f"  Total Bought:              {bought.sum():>12.1f} MWh")
    print(f"  Total Unmet Demand:        {total_unmet:>12.1f} MWh")
    print(f"  Demand Met:                {demand_met_pct:>12.1f} %")
    print(f"  Peak Unmet:                {unmet.max():>12.1f} MWh (day {unmet.idxmax()})")
    
    print(f"\n{'STORAGE METRICS:':<30}")
    print(f"  Average Stored:            {avg_stored:>12.1f} MWh")
    print(f"  Peak Stored:               {max_stored:>12.1f} MWh (day {stored.idxmax()})")
    print(f"  Days with Storage:         {storage_utilization:>12d} days")
    print(f"  Storage Utilization:       {storage_utilization/len(stored)*100:>12.1f} %")
    
    print(f"\n{'ECONOMIC METRICS:':<30}")
    print(f"  Total Fuel Cost:           {total_fuel_cost:>12.1f} DKK")
    print(f"  Total Storage Cost:        {total_storage_cost:>12.1f} DKK")
    print(f"  Total Cost:                {total_cost:>12.1f} DKK")
    print(f"  Daily Allowance:           {allowance:>12.1f} DKK")
    print(f"  Average Price:             {avg_price:>12.1f} DKK/MWh")
    print(f"  Peak Price:                {price.max():>12.1f} DKK/MWh (day {price.idxmax()})")
    print(f"  Min Price:                 {price.min():>12.1f} DKK/MWh (day {price.idxmin()})")
    
    print(f"\n{'BUDGET METRICS:':<30}")
    print(f"  Final Budget:              {budget[-1]:>12.1f} DKK")
    print(f"  Average Budget:            {budget.mean():>12.1f} DKK")
    print(f"  Peak Budget:               {budget.max():>12.1f} DKK (day {budget.argmax()})")
    
    print("="*70 + "\n")

def plot_model_2_storage_vs_budget_over_time(flow_results, spending_results, save_path=None):
    """
    Plot storage and budget dynamics over time with price and demand context.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Extract data
    days = flow_results.index
    stored = flow_results['Stored (MWh)']
    demand = flow_results['Demand (MWh)']
    carry_over = spending_results['Carry over budget']
    price = spending_results['Price (dkk/MWh)']
    
    # Plot storage on primary axis
    ax.plot(days, stored, color='goldenrod', marker='o', linewidth=2.5, 
            markersize=6, label='Storage Level', alpha=0.8, zorder=5)
    
    # Plot carry-over budget on secondary axis
    ax_budget = ax.twinx()
    ax_budget.plot(days, carry_over, color='darkgreen', marker='s', linewidth=2.5,
                   markersize=6, label='Carry Over Budget', alpha=0.8, zorder=5)
    
    # Shade high-price periods (red with vertical lines pattern)
    high_price_threshold = price.quantile(0.75)
    for i in range(len(days)):
        if price[i] > high_price_threshold:
            ax.axvspan(days[i]-0.5, days[i]+0.5, alpha=0.3, color='red', 
                      hatch='///', edgecolor='darkred', linewidth=0, zorder=1)
    
    # Shade high-demand periods (yellow with horizontal lines pattern)
    high_demand_threshold = demand.quantile(0.75)
    for i in range(len(days)):
        if demand[i] > high_demand_threshold:
            ax.axvspan(days[i]-0.5, days[i]+0.5, alpha=0.3, color='yellow', 
                      hatch='---', edgecolor='orange', linewidth=0, zorder=2)
    
    # Add annotations for shading
    ax.text(0.98, 0.08, f'Red diagonal = High Price (> {high_price_threshold:.1f} DKK/MWh)',
            transform=ax.transAxes, fontsize=10, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='red'))
    
    ax.text(0.98, 0.02, f'Yellow horizontal = High Demand (> {high_demand_threshold:.1f} MWh)',
            transform=ax.transAxes, fontsize=10, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='orange'))
    
    ax.set_xlabel('Day', fontsize=13, fontweight='bold')
    ax.set_ylabel('Storage Level (MWh)', fontsize=13, color='goldenrod', fontweight='bold')
    ax_budget.set_ylabel('Carry Over Budget (DKK)', fontsize=13, color='darkgreen', fontweight='bold')
    ax.set_title('Model 2: Storage vs Budget Over Time', 
                 fontsize=15, fontweight='bold', pad=20)
    
    ax.tick_params(axis='y', labelcolor='goldenrod', labelsize=11)
    ax_budget.tick_params(axis='y', labelcolor='darkgreen', labelsize=11)
    ax.tick_params(axis='x', labelsize=11)
    
    ax.legend(loc='upper left', fontsize=11, frameon=True)
    ax_budget.legend(loc='upper right', fontsize=11, frameon=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path.with_suffix('.png')}")
    
    plt.show()


# Create plot




### plotting functions