import json
import csv
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np




def compare_risk_distributions(risk_stats_0, risk_stats_1, alpha, beta_0, beta_1, save_path=None):
    """
    Compare the unmet demand distributions for two different β values.
    Shows only histogram comparison.
    
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
    
    # ========================================================================
    # HISTOGRAM
    # ========================================================================
    
    bins = np.linspace(min(unmet_beta_0.min(), unmet_beta_1.min()),
                       max(unmet_beta_0.max(), unmet_beta_1.max()), 35)
    
    ax.hist(unmet_beta_0, bins=bins, alpha=0.6, color='steelblue', 
            edgecolor='darkblue', linewidth=1.5, 
            label=f'{label_0} ({strategy_0})', density=True)
    ax.hist(unmet_beta_1, bins=bins, alpha=0.6, color='coral', 
            edgecolor='darkred', linewidth=1.5, 
            label=f'{label_1} ({strategy_1})', density=True)
    
    # Add mean lines
    ax.axvline(risk_stats_0['mean'], color='darkblue', linestyle='--', 
               linewidth=3, label=f"{label_0} Mean: {risk_stats_0['mean']:.1f} MWh")
    ax.axvline(risk_stats_1['mean'], color='darkred', linestyle='--', 
               linewidth=3, label=f"{label_1} Mean: {risk_stats_1['mean']:.1f} MWh")
    
    # Add percentile lines
    percentile = int(alpha * 100)
    p_key = f'p{percentile}' if f'p{percentile}' in risk_stats_0 else 'p95'
    
    ax.axvline(risk_stats_0[p_key], color='darkblue', linestyle=':', 
               linewidth=2.5, alpha=0.7, 
               label=f"{label_0} {percentile}th %ile: {risk_stats_0[p_key]:.1f} MWh")
    ax.axvline(risk_stats_1[p_key], color='darkred', linestyle=':', 
               linewidth=2.5, alpha=0.7,
               label=f"{label_1} {percentile}th %ile: {risk_stats_1[p_key]:.1f} MWh")
    
    ax.set_xlabel('Total Unmet Demand (MWh)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Density', fontsize=13, fontweight='bold')
    ax.set_title(f'Risk Profile Comparison: {label_0} vs {label_1} (α={alpha:.2f})\n' +
                 f'Distribution of Total Unmet Demand Across Scenarios', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=11)
    
    # Add statistics box
    stats_text = f"""Key Statistics:

{label_0}:  Mean={risk_stats_0['mean']:.1f}, Std={risk_stats_0['std']:.1f}, Max={risk_stats_0['max']:.1f}
{label_1}:  Mean={risk_stats_1['mean']:.1f}, Std={risk_stats_1['std']:.1f}, Max={risk_stats_1['max']:.1f}

Δ (Improvement):
  Mean: {risk_stats_0['mean'] - risk_stats_1['mean']:+.1f} MWh ({((risk_stats_0['mean']/risk_stats_1['mean']-1)*100):+.1f}%)
  {percentile}th %ile: {risk_stats_0[p_key] - risk_stats_1[p_key]:+.1f} MWh ({((risk_stats_0[p_key]/risk_stats_1[p_key]-1)*100):+.1f}%)
  Max: {risk_stats_0['max'] - risk_stats_1['max']:+.1f} MWh"""
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, pad=1))
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path.with_suffix('.png')}")
    
    plt.show()


# ============================================================================
# USAGE - α=0.9, β=0.0 vs β=0.15
# ============================================================================

def compare_procurement_strategies(flow_0, spending_0, flow_1, spending_1, 
                                   alpha, beta_0, beta_1, save_path=None):
    """
    Compare the procurement strategies (storage and carry-over budget) for two different β values.
    
    Parameters:
    - flow_0, spending_0: Results for first beta value
    - flow_1, spending_1: Results for second beta value
    - alpha: CVaR confidence level (e.g., 0.9)
    - beta_0: First beta value (e.g., 0.0)
    - beta_1: Second beta value (e.g., 0.15)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    
    days = flow_0.index
    
    # Labels with 2 decimal places
    label_0 = f'β={beta_0:.2f}'
    label_1 = f'β={beta_1:.2f}'
    
    # Extract data
    storage_0 = flow_0['Stored (MWh)']
    storage_1 = flow_1['Stored (MWh)']
    
    # Calculate carry-over as Budget - Allowance
    budget_0 = spending_0['Budget']
    budget_1 = spending_1['Budget']
    allowance = spending_0['Allowance'].iloc[0]
    carry_over_0 = budget_0 - allowance
    carry_over_1 = budget_1 - allowance
    
    # Calculate total stored over entire period
    total_stored_0 = storage_0.sum()
    total_stored_1 = storage_1.sum()
    
    # Calculate total carry-over over entire period
    total_carry_over_0 = carry_over_0.sum()
    total_carry_over_1 = carry_over_1.sum()
    
    # ========================================================================
    # TOP PANEL: STORAGE COMPARISON
    # ========================================================================
    
    # Plot both storage strategies
    ax1.fill_between(days, 0, storage_0, color='steelblue', alpha=0.4)
    ax1.plot(days, storage_0, color='darkblue', linewidth=3, marker='o', 
            markersize=5, linestyle='-', label=f'{label_0}')
    
    ax1.fill_between(days, 0, storage_1, color='coral', alpha=0.4)
    ax1.plot(days, storage_1, color='darkred', linewidth=3, marker='s', 
            markersize=5, linestyle='--', label=f'{label_1}')
    
    ax1.set_ylabel('Storage Level (MWh)', fontsize=13, fontweight='bold')
    ax1.set_title(f'Procurement Strategy Comparison: {label_0} vs {label_1} (α={alpha:.2f})\n' +
                  'Storage and Budget Carry-Over Evolution',
                 fontsize=15, fontweight='bold', pad=15)
    ax1.legend(loc='upper left', fontsize=11, framealpha=0.95)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.tick_params(labelsize=11)
    
    # Statistics - Total Stored
    storage_delta = total_stored_1 - total_stored_0
    storage_pct = (storage_delta / total_stored_0 * 100) if total_stored_0 > 0 else 0
    
    stats_text_storage = f"""Total Stored Over Period:
{label_0}: {total_stored_0:.0f} MWh
{label_1}: {total_stored_1:.0f} MWh
Δ: {storage_delta:+.0f} MWh ({storage_pct:+.1f}%)"""
    
    ax1.text(0.98, 0.98, stats_text_storage, transform=ax1.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9, pad=0.8),
            family='monospace')
    
    # ========================================================================
    # BOTTOM PANEL: CARRY-OVER BUDGET COMPARISON
    # ========================================================================
    
    # Plot both carry-over strategies
    ax2.fill_between(days, 0, carry_over_0, color='steelblue', alpha=0.4)
    ax2.plot(days, carry_over_0, color='darkblue', linewidth=3.5, marker='o', 
            markersize=6, linestyle='-', label=f'{label_0}', zorder=5)
    
    ax2.fill_between(days, 0, carry_over_1, color='coral', alpha=0.4)
    ax2.plot(days, carry_over_1, color='darkred', linewidth=3.5, marker='s', 
            markersize=6, linestyle='--', label=f'{label_1}', zorder=5)
    
    # Add zero line
    ax2.axhline(0, color='black', linewidth=1.5, linestyle='-', alpha=0.5, zorder=3)
    
    # Add grid
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax2.set_ylabel('Budget Carry-Over (DKK)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Day', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=11, framealpha=0.95)
    ax2.tick_params(labelsize=11)
    
    # Statistics - Total Carry-Over
    carry_over_delta = total_carry_over_1 - total_carry_over_0
    carry_over_pct = (carry_over_delta / total_carry_over_0 * 100) if total_carry_over_0 > 0 else 0
    
    stats_text_carryover = f"""Total Carry-Over Over Period:
{label_0}: {total_carry_over_0:.0f} DKK
{label_1}: {total_carry_over_1:.0f} DKK
Δ: {carry_over_delta:+.0f} DKK ({carry_over_pct:+.1f}%)"""
    
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
    # PRINT DETAILED COMPARISON
    # ========================================================================
    
    print("\n" + "="*80)
    print(f"PROCUREMENT STRATEGY COMPARISON: {label_0} vs {label_1} (α={alpha:.2f})")
    print("="*80)
    
    print("\n📦 STORAGE STRATEGY:")
    print(f"  {label_0}: Total Stored = {total_stored_0:.0f} MWh")
    print(f"  {label_1}: Total Stored = {total_stored_1:.0f} MWh")
    print(f"  Δ: {storage_delta:+.0f} MWh ({storage_pct:+.1f}%)")
    
    print("\n💵 CARRY-OVER BUDGET STRATEGY:")
    print(f"  {label_0}: Total Carry-Over = {total_carry_over_0:.0f} DKK")
    print(f"  {label_1}: Total Carry-Over = {total_carry_over_1:.0f} DKK")
    print(f"  Δ: {carry_over_delta:+.0f} DKK ({carry_over_pct:+.1f}%)")
    
    print("\n" + "="*80)


# ============================================================================
# USAGE - α=0.9, β=0.0 vs β=0.3
# ============================================================================




