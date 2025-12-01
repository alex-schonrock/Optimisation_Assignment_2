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
                               line_label: str = "Price (DKK/kWh)"):
    """Plots the primals in a stacked bar chart, with an optional line for price"""
    if show_price_line == True:
        price_ser = df.pop(line_label)
        # if line_label == "Deviation Down (kWh)":
        #     price_ser2 = df.pop("Deviation Up (kWh)")
    cols = [c for c in df.columns if not df[c].dropna().empty]
    if not cols:
        raise ValueError("No numeric columns to plot.")

    # to_neg = [c for c in cols if "demand (btu)" in c.lower()]
    to_neg = [c for c in cols if c == "Demand (Btu)"]
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
            ax.bar(x, pos, bottom=cum_pos, label=str(col))
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
    ax.set_ylabel("Flows (Btu)")
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
### plotting functions