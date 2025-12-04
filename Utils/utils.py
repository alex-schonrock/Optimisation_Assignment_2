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



def plot_combined_results(df: pd.DataFrame, spending_results: pd.DataFrame, expenditure: float, save_path: str | None = None, show: bool = True, 
                          title: str = "Stacked Flows vs Time (with Gas Price)", line_label: str = "Price (DKK/kWh)", 
                          y_axis_label: str = "Gas Flows (kWh)"):

    # Extract price data if line is needed
    price_ser = spending_results.pop(line_label) if line_label in spending_results else None
    
    cols = [c for c in df.columns if not df[c].dropna().empty]
    if not cols:
        raise ValueError("No numeric columns to plot.")

    # Columns to flip (make negative for correct stacking)
    cols_to_neg = ["Demand (MWh)", "Bought fuel", "Unmet demand", "Storage"]
    to_neg = [c for c in cols if c in cols_to_neg]
    df_plot = df.copy()

    # Flip the selected columns to negative
    for c in to_neg:
        df_plot[c] = -df_plot[c]
    
    x = np.arange(len(df_plot.index))
    x_labels = df_plot.index
    x_label = df.index.name or "Time"
    
    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Initialize cumulative values for stacking
    cum_pos = np.zeros(len(df_plot), dtype=float)
    cum_neg = np.zeros(len(df_plot), dtype=float)

    # Plot stacked bars for each column in gas flows
    for col in cols:
        vals = df_plot[col].fillna(0).to_numpy(dtype=float)
        pos = np.where(vals > 0, vals, 0.0)
        neg = np.where(vals < 0, vals, 0.0)

        # Stack positive values (gas flows)
        if pos.any():
            ax1.bar(x, pos, bottom=cum_pos, label=str(col), alpha=0.7)
            cum_pos += pos

        # Stack negative values (for demands and storage)
        if neg.any():
            ax1.bar(x, neg, bottom=cum_neg, label=str(col), alpha=0.7)
            cum_neg += neg

    # Set labels and title for the stacked bar chart
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_axis_label)
    ax1.set_title(title)

    # Create a secondary y-axis for spending and price
    ax2 = ax1.twinx()

    # Plot spending components (Bought Fuel, Storage Cost, Carry Over Budget)
    ax2.plot(x, spending_results['Bought fuel'], label='Bought Fuel Spending (DKK)', color='b', linestyle='-', marker='o')
    ax2.plot(x, spending_results['Storage cost'], label='Storage Cost (DKK)', color='g', linestyle=':', marker='x')
    ax2.plot(x, spending_results['Carry over budget'], label='Carry Over Budget (DKK)', color='r', linestyle='--', marker='^')

    # Plot price on the secondary axis if required
    if price_ser is not None:
        ax2.plot(x, price_ser.reindex(df.index).to_numpy(dtype=float), label=line_label, color='purple', linestyle='--', marker='x')

    # Set secondary axis labels
    ax2.set_ylabel('Spending & Price (DKK)', color='purple')
    ax2.tick_params(axis='y', colors='purple')

    # Combine legends from both axes
    handles, labels = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    handles += h2
    labels += l2

    # Display the combined legend
    ax1.legend(handles, labels, bbox_to_anchor=(1.07, 1), loc='upper left')

    # Adjust grid and minor ticks for readability
    ax1.yaxis.set_minor_locator(plt.AutoLocator())  # Minor ticks
    ax1.grid(True, which="major", axis="y")
    ax1.grid(True, which="minor", axis="y", alpha=0.4)

    # Adjust layout
    fig.tight_layout()

    # Save or display the plot
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


### plotting functions