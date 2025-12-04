# from Runner.opt_model_1 import InputData
# from Runner.opt_model_1 import DataProcessor
from pathlib import Path
from Data_ops.data_loader_processor import *
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

class Expando(object):
    pass

import gurobipy as gp
from gurobipy import GRB
import numpy as np


class OptModelCVaR:
    def __init__(self, input_data, model_type: str, alpha: float = 0.8, beta: float = 0):
        # Input data for the model (e.g., demand, prices, costs, etc.)
        self.data = input_data
        self.model_type = model_type
        self.alpha = alpha  # Confidence level for CVaR
        self.beta = beta  # Weight for expected unmet demand
        self.results = Expando()
        self.model = gp.Model()
        self.vars = Expando()
        self.cons = Expando()
        self.cons.balance = {}
        self.cons.unmet_demand_max = {}
        self.cons.plant_min = {}
        self.cons.ramp_up = {}
        self.cons.ramp_down = {}
        self.cons.depreciation = {}
        self.cons.budget = {}
        self.cons.initial_budget = {}
        self.cons.eta_up = {}
        # self.cons.eta_down = {}
        self.T = list(range(len(self.data.demand_scenarios[0])))  # Time periods
        self.S = list(range(len(self.data.demand_scenarios)))     # Scenarios
        self.probs = self.data.probabilities  # Probabilities of each scenario
        self.expected_unmet_demand = self.data.expected_unmet_demand

    def _set_objective_with_cvar(self):
        # CVaR penalty based on `eta`
        cvar_penalty = self.vars.z + (1 / (1 - self.alpha))  * gp.quicksum(self.probs[s] * self.vars.eta[s] for s in self.S)
        
        unmet_demand = gp.quicksum(self.probs[s] * gp.quicksum(self.vars.unmet_demand[t, s] for t in self.T) for s in self.S)

        # Objective function: Minimize expected unmet demand + CVaR penalty
        obj_fn = (1-self.beta)*unmet_demand + self.beta * cvar_penalty
        self.model.setObjective(obj_fn, GRB.MINIMIZE)

    def _build(self):
        # Add variables for each time period and scenario
        v_bought = self.model.addVars(self.T, self.S, name="bought", lb=0)
        v_unmet_demand = self.model.addVars(self.T, self.S, name="unmet_demand", lb=0)
        v_stored = self.model.addVars(self.T, self.S, name="stored", lb=0, ub=self.data.max_storage_capacity)
        v_budget = self.model.addVars(self.T, self.S, name="budget", lb=0)
        z = self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name='z')
        eta = self.model.addVars(self.S, name="eta", lb=0)
        
        # Store variable handles in self.vars
        self.vars.bought = v_bought
        self.vars.unmet_demand = v_unmet_demand
        self.vars.stored = v_stored
        self.vars.budget = v_budget
        self.vars.z = z
        self.vars.eta = eta

        
        # Define constraints for each time period and scenario
        for t in self.T:
            for s in self.S:
                # Constraints for unmet demand (max demand that can be unmet)
                self.cons.unmet_demand_max[t, s] = self.model.addConstr(self.vars.unmet_demand[t, s] <= self.data.demand_scenarios[s, t], name=f"unmet_demand_max_{t}_{s}")
                # Constraints for plant minimum
                self.cons.plant_min[t, s] = self.model.addConstr(self.data.demand_scenarios[s, t] - self.vars.unmet_demand[t, s] >= self.data.demand_min, name=f"bought_min_{t}_{s}")
                # Constraints for the budget (bought energy + storage cost)
                self.cons.budget[t, s] = self.model.addConstr(self.vars.bought[t, s] * self.data.price[t] + self.vars.stored[t, s] * self.data.storage_cost <= self.vars.budget[t, s], name=f"budget_{t}_{s}")

        # Constraints that don't apply in the first day due to intertemporal nature
        for t in list(range(1, len(self.data.demand_scenarios[0]))):  # Loop over time periods
            for s in self.S:  # Loop over scenarios
                # Ramp-up and ramp-down constraints (capacity change between time periods)
                self.cons.ramp_up[t, s] = self.model.addConstr(self.data.demand_scenarios[s, t] - self.vars.unmet_demand[t, s] - (self.data.demand_scenarios[s, t-1] - self.vars.unmet_demand[t-1, s]) <= self.data.ramp_rate)
                self.cons.ramp_down[t, s] = self.model.addConstr(self.data.demand_scenarios[s, t-1] - self.vars.unmet_demand[t-1, s] - (self.data.demand_scenarios[s, t] - self.vars.unmet_demand[t, s]) <= self.data.ramp_rate)
                # Depreciation constraint
                self.cons.depreciation[t, s] = self.model.addConstr(self.vars.budget[t, s] == self.data.exp_allowance + self.data.depreciation * (self.vars.budget[t-1, s] - self.vars.bought[t-1, s] * self.data.price[t-1] - self.vars.stored[t-1, s] * self.data.storage_cost))
                # Balance constraint (ensuring demand is met or stored)
                self.cons.balance[t, s] = self.model.addConstr(self.vars.bought[t, s] + self.vars.stored[t-1, s] == self.data.demand_scenarios[s, t] - self.vars.unmet_demand[t, s] + self.vars.stored[t, s], name=f"balance_{t}_{s}")
        
        # Balance and depreciation for the first time period (index 0)
        for s in self.S:
            self.cons.balance[0, s] = self.model.addConstr(self.vars.bought[0, s] == self.data.demand_scenarios[s, 0] - self.vars.unmet_demand[0, s] + self.vars.stored[0, s], name=f"balance_0_{s}")
            self.cons.depreciation[0, s] = self.model.addConstr(self.vars.budget[0, s] == self.data.exp_allowance, name=f"initial_budget_{s}")
            self.cons.ramp_up[0, s] = self.model.addConstr(self.data.demand_scenarios[s, 0] - self.vars.unmet_demand[0, s] <= self.data.ramp_rate, name=f"ramp_up_0_{s}")
            self.cons.ramp_down[0, s] = self.model.addConstr(self.data.demand_scenarios[s, 0] - self.vars.unmet_demand[0, s] <= self.data.ramp_rate, name=f"ramp_down_0_{s}")
            self.cons.eta_up[s] = self.model.addConstr(eta[s] >= gp.quicksum(self.vars.unmet_demand[t, s] for t in self.T) - self.vars.z, name=f"eta_abs_upper_{s}")
            # self.cons.eta_down[s] = self.model.addConstr(eta[s] >= gp.quicksum(-(self.vars.unmet_demand[t, s] - self.data.expected_unmet_demand[t]) for t in self.T) , name=f"eta_abs_lower_{s}")
        
        self._set_objective_with_cvar()

    def solve(self, verbose: bool = False):
        if not verbose:
            self.model.Params.OutputFlag = 0
        
        # Optimize the model
        self.model.optimize()

        v = self.vars
        self.results.v_bought = np.array([[v.bought[t, s].X for s in self.S] for t in self.T])
        self.results.v_unmet_demand = np.array([[v.unmet_demand[t, s].X for s in self.S] for t in self.T])
        self.results.prices = np.asarray(self.data.price, dtype=float).reshape(-1)
        self.results.demand_scenarios = np.asarray(self.data.demand_scenarios, dtype=float)
        self.results.v_stored = np.array([[v.stored[t, s].X for s in self.S] for t in self.T])
        self.results.v_budget = np.array([[v.budget[t, s].X for s in self.S] for t in self.T])
        self.results.allowance = np.asarray(self.data.exp_allowance, dtype=float).reshape(-1)
        self.results.z = v.z.X
        self.results.eta = np.array([v.eta[s].X for s in self.S])

        self.results.obj = self.model.ObjVal
        
        # ... existing code ...
        
        # ADD THIS DEBUGGING OUTPUT:
        # print(f"\n{'='*60}")
        # print(f"DETAILED DIAGNOSTICS (β={self.beta}, α={self.alpha})")
        # print(f"{'='*60}")
        
        # # Check z value and its impact
        # print(f"z (VaR): {self.results.z:.4f}")
        
        # # Check how many scenarios have loss > z
        # scenario_losses = [np.sum(self.results.v_unmet_demand[:, s]) for s in self.S]
        # exceeds_var = sum(1 for loss in scenario_losses if loss > self.results.z)
        # print(f"Scenarios with loss > VaR: {exceeds_var} / {len(self.S)}")
        
        # # Check eta values
        # eta_sum = np.sum([self.results.eta[s] for s in self.S])
        # print(f"Sum of eta values: {eta_sum:.4f}")
        # print(f"Non-zero etas: {sum(1 for s in self.S if self.results.eta[s] > 0.001)}")
        
        # # Check objective components
        # expected = np.sum(self.probs[s] * sum(self.results.v_unmet_demand[:, s]) for s in self.S)
        # cvar = self.results.z + (1/(1-self.alpha)) * np.sum(self.probs[s] * self.results.eta[s] for s in self.S)
        
        # print(f"\nObjective components:")
        # print(f"  Expected unmet demand: {expected:.4f}")
        # print(f"  CVaR value: {cvar:.4f}")
        # print(f"  (1-β) × Expected = {(1-self.beta) * expected:.4f}")
        # print(f"  β × CVaR = {self.beta * cvar:.4f}")
        # print(f"  Total objective: {self.results.obj:.4f}")
        # print(f"  Verification: {(1-self.beta)*expected + self.beta*cvar:.4f}")
        
        # # Check if budget is binding
        # budget_slack = np.mean(self.results.v_budget - (
        #     self.results.v_bought * self.data.price.reshape(-1, 1) + 
        #     self.results.v_stored * self.data.storage_cost
        # ))
        # print(f"\nAverage budget slack: {budget_slack:.4f}")
        
        # print(f"{'='*60}\n")
        

        duals = Expando()
        duals.balance = np.array([[self.cons.balance[t, s].Pi for s in self.S] for t in self.T], dtype=float)
        duals.unmet_demand_max = np.array([[self.cons.unmet_demand_max[t, s].Pi for s in self.S] for t in self.T], dtype=float)
        duals.plant_min = np.array([[self.cons.plant_min[t, s].Pi for s in self.S] for t in self.T], dtype=float)
        duals.ramp_up = np.array([[self.cons.ramp_up[t, s].Pi for s in self.S] for t in self.T], dtype=float)
        duals.ramp_down = np.array([[self.cons.ramp_down[t, s].Pi for s in self.S] for t in self.T], dtype=float)
        duals.depreciation = np.array([[self.cons.depreciation[t, s].Pi for s in self.S] for t in self.T], dtype=float)
        duals.budget = np.array([[self.cons.budget[t, s].Pi for s in self.S] for t in self.T], dtype=float)

        self.results.duals = duals
        return self.results
    


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats

def plot_cvar_analysis(model, results, save_prefix="cvar_analysis"):
    """
    Comprehensive plotting function for CVaR model results with improved visualizations
    """
    # Calculate scenario losses
    scenario_losses = [np.sum(results.v_unmet_demand[:, s]) for s in model.S]
    sorted_indices = np.argsort(scenario_losses)
    sorted_losses = np.array(scenario_losses)[sorted_indices]
    sorted_probs = np.array(model.probs)[sorted_indices]
    
    # Calculate CVaR components
    expected_loss = np.sum(model.probs[s] * scenario_losses[s] for s in model.S)
    cvar_value = results.z + (1/(1-model.alpha)) * np.sum(model.probs[s] * results.eta[s] for s in model.S)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)
    
    # 1. KDE (Kernel Density Estimate) instead of histogram - smoother
    ax1 = fig.add_subplot(gs[0, 0])
    if len(scenario_losses) > 1:
        kde = stats.gaussian_kde(scenario_losses, weights=model.probs)
        x_range = np.linspace(min(scenario_losses)*0.8, max(scenario_losses)*1.2, 300)
        density = kde(x_range)
        ax1.fill_between(x_range, density, alpha=0.3, color='steelblue')
        ax1.plot(x_range, density, color='steelblue', linewidth=2, label='Loss Distribution')
    else:
        ax1.hist(scenario_losses, bins=20, density=True, alpha=0.6, color='steelblue', edgecolor='black')
    
    ax1.axvline(expected_loss, color='red', linestyle='--', linewidth=2.5, label=f'Expected: {expected_loss:.2f}')
    ax1.axvline(results.z, color='orange', linestyle='--', linewidth=2.5, label=f'VaR (z): {results.z:.2f}')
    ax1.axvline(cvar_value, color='purple', linestyle='--', linewidth=2.5, label=f'CVaR: {cvar_value:.2f}')
    ax1.set_xlabel('Total Unmet Demand', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title(f'Loss Distribution (β={model.beta}, α={model.alpha})', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. CDF (Cumulative Distribution) - much clearer than overlapping histograms
    ax2 = fig.add_subplot(gs[0, 1])
    cumsum_probs = np.cumsum(sorted_probs)
    ax2.plot(sorted_losses, cumsum_probs, linewidth=3, color='steelblue', marker='o', 
             markersize=3, alpha=0.7, label='Empirical CDF')
    ax2.axvline(results.z, color='orange', linestyle='--', linewidth=2.5, label=f'VaR: {results.z:.2f}')
    ax2.axhline(model.alpha, color='green', linestyle=':', linewidth=2, label=f'α = {model.alpha}')
    
    # Highlight CVaR region
    cvar_mask = sorted_losses >= results.z
    if any(cvar_mask):
        ax2.fill_between(sorted_losses[cvar_mask], 0, cumsum_probs[cvar_mask], 
                         alpha=0.3, color='red', label='CVaR Tail')
    
    ax2.set_xlabel('Total Unmet Demand', fontsize=11)
    ax2.set_ylabel('Cumulative Probability', fontsize=11)
    ax2.set_title('Cumulative Distribution Function', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    # 3. Sorted scenario losses with better visualization
    ax3 = fig.add_subplot(gs[0, 2])
    colors = ['red' if loss >= results.z else 'lightblue' for loss in sorted_losses]
    ax3.bar(range(len(sorted_losses)), sorted_losses, color=colors, edgecolor='black', 
            alpha=0.7, linewidth=0.5)
    ax3.axhline(results.z, color='orange', linestyle='--', linewidth=2.5, label=f'VaR: {results.z:.2f}')
    ax3.set_xlabel('Scenario (sorted by loss)', fontsize=11)
    ax3.set_ylabel('Total Unmet Demand', fontsize=11)
    ax3.set_title(f'Sorted Scenario Losses', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Box plot showing distribution statistics
    ax4 = fig.add_subplot(gs[1, 0])
    bp = ax4.boxplot([scenario_losses], vert=True, widths=0.5, patch_artist=True,
                      labels=['Loss Distribution'])
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][0].set_alpha(0.7)
    ax4.axhline(results.z, color='orange', linestyle='--', linewidth=2, label=f'VaR: {results.z:.2f}')
    ax4.axhline(expected_loss, color='red', linestyle='--', linewidth=2, label=f'Expected: {expected_loss:.2f}')
    ax4.set_ylabel('Total Unmet Demand', fontsize=11)
    ax4.set_title('Distribution Box Plot', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Eta values (CVaR contributions) - better visualization
    ax5 = fig.add_subplot(gs[1, 1])
    eta_vals = [results.eta[s] for s in model.S]
    eta_sorted_indices = np.argsort(eta_vals)[::-1]  # Sort descending
    eta_sorted = np.array(eta_vals)[eta_sorted_indices]
    
    # Only plot non-zero or significant etas for clarity
    significant_etas = eta_sorted[eta_sorted > 0.01]
    if len(significant_etas) > 0:
        colors_eta = plt.cm.Reds(np.linspace(0.4, 0.9, len(significant_etas)))
        ax5.bar(range(len(significant_etas)), significant_etas, color=colors_eta, 
                edgecolor='black', alpha=0.8)
        ax5.set_xlabel('Scenario (sorted by η)', fontsize=11)
        ax5.set_title(f'CVaR Contributions (η > 0.01): {len(significant_etas)} scenarios', 
                      fontsize=12, fontweight='bold')
    else:
        ax5.bar(range(len(eta_sorted)), eta_sorted, color='lightblue', 
                edgecolor='black', alpha=0.7)
        ax5.set_xlabel('Scenario (sorted by η)', fontsize=11)
        ax5.set_title(f'CVaR Contributions: All Zero', fontsize=12, fontweight='bold')
    
    ax5.set_ylabel('η (CVaR contribution)', fontsize=11)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Scatter plot: Scenario loss vs probability
    ax6 = fig.add_subplot(gs[1, 2])
    colors_scatter = ['red' if loss >= results.z else 'blue' for loss in scenario_losses]
    ax6.scatter(scenario_losses, model.probs, c=colors_scatter, s=50, alpha=0.6, edgecolors='black')
    ax6.axvline(results.z, color='orange', linestyle='--', linewidth=2, label=f'VaR: {results.z:.2f}')
    ax6.set_xlabel('Total Unmet Demand', fontsize=11)
    ax6.set_ylabel('Probability', fontsize=11)
    ax6.set_title('Scenario Loss vs Probability', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    # 7. Time series of unmet demand - improved with percentiles
    ax7 = fig.add_subplot(gs[2, :2])
    
    # Calculate percentiles across scenarios for each time period
    percentiles = [5, 25, 50, 75, 95]
    percentile_values = np.percentile(results.v_unmet_demand, percentiles, axis=1)
    
    # Plot median
    ax7.plot(model.T, percentile_values[2, :], linewidth=3, color='blue', 
             label='Median (50th percentile)', zorder=3)
    
    # Plot shaded regions for percentiles
    ax7.fill_between(model.T, percentile_values[0, :], percentile_values[4, :], 
                     alpha=0.2, color='blue', label='5th-95th percentile')
    ax7.fill_between(model.T, percentile_values[1, :], percentile_values[3, :], 
                     alpha=0.3, color='blue', label='25th-75th percentile')
    
    # Optionally overlay worst scenarios
    worst_scenarios = sorted_indices[-3:]  # 3 worst
    for s in worst_scenarios:
        ax7.plot(model.T, results.v_unmet_demand[:, s], linestyle='--', alpha=0.5, 
                linewidth=1.5, label=f'Scenario {s} (loss={scenario_losses[s]:.1f})')
    
    ax7.set_xlabel('Time Period', fontsize=11)
    ax7.set_ylabel('Unmet Demand', fontsize=11)
    ax7.set_title('Unmet Demand Over Time (Percentiles + Worst Scenarios)', fontsize=12, fontweight='bold')
    ax7.legend(fontsize=8, ncol=2)
    ax7.grid(True, alpha=0.3)
    
    # 8. Summary statistics table
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    stats_text = f"""MODEL SUMMARY
β={model.beta}, α={model.alpha}
{'='*35}
Objective: {results.obj:.4f}

Expected Loss: {expected_loss:.4f}
VaR (z): {results.z:.4f}
CVaR: {cvar_value:.4f}

Scenario Stats:
  Min: {min(scenario_losses):.2f}
  25%: {np.percentile(scenario_losses, 25):.2f}
  50%: {np.percentile(scenario_losses, 50):.2f}
  75%: {np.percentile(scenario_losses, 75):.2f}
  Max: {max(scenario_losses):.2f}
  Std: {np.std(scenario_losses):.2f}

CVaR tail: {sum(1 for l in scenario_losses if l > results.z)} scenarios
Non-zero η: {sum(1 for e in eta_vals if e > 0.01)}

Objective Breakdown:
  (1-β)×E[L] = {(1-model.beta)*expected_loss:.4f}
  β×CVaR = {model.beta*cvar_value:.4f}
    """
    
    ax8.text(0.05, 0.95, stats_text, transform=ax8.transAxes, 
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'CVaR Model Analysis (β={model.beta}, α={model.alpha})', 
                 fontsize=14, fontweight='bold', y=0.998)
    
    # Save the figure
    filename = f'{save_prefix}_beta_{model.beta}_alpha_{model.alpha}.pdf'
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"Saved plot: {filename}")
    
    plt.show()
    
    return {
        'expected_loss': expected_loss,
        'var': results.z,
        'cvar': cvar_value,
        'scenario_losses': scenario_losses,
        'eta_values': eta_vals
    }


def plot_beta_comparison(results_comparison):
    """
    Improved comparison plot with better visibility
    """
    betas = [r['beta'] for r in results_comparison]
    expected_losses = [r['expected_loss'] for r in results_comparison]
    vars = [r['var'] for r in results_comparison]
    cvars = [r['cvar'] for r in results_comparison]
    
    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(1, 4, figure=fig, wspace=0.3)
    
    # Plot 1: Metrics vs Beta
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(betas, expected_losses, marker='o', linewidth=2.5, markersize=8, label='Expected Loss', color='blue')
    ax1.plot(betas, vars, marker='s', linewidth=2.5, markersize=8, label='VaR', color='orange')
    ax1.plot(betas, cvars, marker='^', linewidth=2.5, markersize=8, label='CVaR', color='purple')
    ax1.set_xlabel('β (CVaR weight)', fontsize=12)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.set_title('Risk Metrics vs β', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: KDE comparison (much clearer than overlapping histograms)
    ax2 = fig.add_subplot(gs[0, 1])
    colors_kde = plt.cm.viridis(np.linspace(0, 1, len(results_comparison)))
    
    for i, r in enumerate(results_comparison):
        if len(r['scenario_losses']) > 1:
            kde = stats.gaussian_kde(r['scenario_losses'])
            x_range = np.linspace(min(r['scenario_losses'])*0.8, max(r['scenario_losses'])*1.2, 300)
            density = kde(x_range)
            ax2.plot(x_range, density, linewidth=2.5, color=colors_kde[i], 
                    label=f"β={r['beta']:.2f}", alpha=0.8)
    
    ax2.set_xlabel('Total Unmet Demand', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Loss Distributions (KDE)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: CDF comparison (best for comparing distributions!)
    ax3 = fig.add_subplot(gs[0, 2])
    
    for i, r in enumerate(results_comparison):
        sorted_losses = np.sort(r['scenario_losses'])
        # Assuming equal probabilities; adjust if you have probs stored
        cdf = np.arange(1, len(sorted_losses) + 1) / len(sorted_losses)
        ax3.plot(sorted_losses, cdf, linewidth=2.5, color=colors_kde[i], 
                label=f"β={r['beta']:.2f}", alpha=0.8, marker='o', markersize=3)
    
    ax3.set_xlabel('Total Unmet Demand', fontsize=12)
    ax3.set_ylabel('Cumulative Probability', fontsize=12)
    ax3.set_title('Cumulative Distributions', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Non-zero eta count
    ax4 = fig.add_subplot(gs[0, 3])
    eta_counts = [sum(1 for e in r['eta_values'] if e > 0.01) for r in results_comparison]
    bars = ax4.bar(range(len(betas)), eta_counts, tick_label=[f"{b:.2f}" for b in betas],
                   color=colors_kde, edgecolor='black', alpha=0.8, linewidth=1.5)
    ax4.set_xlabel('β (CVaR weight)', fontsize=12)
    ax4.set_ylabel('Count', fontsize=12)
    ax4.set_title('Scenarios in CVaR Tail', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, eta_counts)):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(eta_counts)*0.02, 
                str(count), ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle('Beta Comparison Summary', fontsize=15, fontweight='bold', y=0.98)
    plt.savefig('beta_comparison_summary.pdf', dpi=300, bbox_inches='tight')
    print("Saved: beta_comparison_summary.pdf")
    plt.show()


def compare_beta_values(path, model_type, beta_values, alpha=0.85):
    """
    Run model for multiple beta values and compare results
    """
    results_comparison = []
    
    for beta in beta_values:
        print(f"\n{'='*60}")
        print(f"Running model with β={beta}, α={alpha}")
        print(f"{'='*60}")
        
        # Load data and run model
        data_processor = DataProcessor(path, model_type)
        input_data = data_processor.get_coefficients()
        
        model = OptModelCVaR(input_data, model_type=model_type, alpha=alpha, beta=beta)
        model._build()
        results = model.solve(verbose=False)
        
        # Plot and analyze
        stats = plot_cvar_analysis(model, results, save_prefix=f"cvar_{model_type}")
        stats['beta'] = beta
        stats['alpha'] = alpha
        results_comparison.append(stats)
    
    # Create comparison plot
    plot_beta_comparison(results_comparison)
    
    # Print comparison table
    print("\n" + "="*80)
    print("BETA COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Beta':>8} {'Expected':>12} {'VaR':>12} {'CVaR':>12} {'CVaR Tail':>12}")
    print("-"*80)
    for r in results_comparison:
        tail_count = sum(1 for e in r['eta_values'] if e > 0.01)
        print(f"{r['beta']:>8.2f} {r['expected_loss']:>12.4f} {r['var']:>12.4f} "
              f"{r['cvar']:>12.4f} {tail_count:>12d}")
    print("="*80)
    
    return results_comparison


# ============================================================================
# USAGE: Run this code
# ============================================================================

# path = r"C:\Users\alex\OneDrive\Desktop\DTU\Optimistation\Optimisation_Assignment_2\Data"

# # Compare multiple beta values
# beta_values = [0.0, 0.25, 0.5, 0.75, 1.0]
# results_comparison = compare_beta_values(path, "Model_3", beta_values, alpha=0.85)


    
    
# path = r"C:\Users\alex\OneDrive\Desktop\DTU\Optimistation\Optimisation_Assignment_2\Data"
# data_processor = DataProcessor(path, "Model_3")
# input_data = data_processor.get_coefficients()
# # print(input_data.demand_scenarios)

# # Initialize the OptModelCVaR model
# model = OptModelCVaR(input_data, model_type="Model_3")
# # print("model probs:")
# # print(model.probs)
# model._build()
# results = model.solve()

# # Access the results
# v_bought = results.v_bought  # Procurement (bought) values for each time period and scenario
# v_unmet_demand = results.v_unmet_demand  # Unmet demand for each time period and scenario
# prices = results.prices  # Prices for each time period
# demand_scenarios = results.demand_scenarios  # Demand scenarios
# v_stored = results.v_stored  # Stored energy values for each time period and scenario
# v_budget = results.v_budget  # Budget values for each time period and scenario
# allowance = results.allowance  # Allowance values for each time period


# Print the total objective value
# print(f"Objective Value: {results.obj}")

# Print CVaR-related results
# if model.alpha > 0:  # If CVaR is included
#     # print(f"VaR (zeta): {results.z}")
#     # print("Eta values (deviation for each scenario):")
#     for s in model.S:
        # print(f"  Scenario {s}: {results.eta[s]}")

# # Now, display the results for each scenario
# total_unmet_demand_list = []
# for s in model.S:
#     # print(f"Results for Scenario {s}:")
#     # print("Total unmet demand in this scenario:", sum(v_unmet_demand[t, s] for t in model.T))
#     # print("Cvar contribution in this scenario:", results.eta[s] * model.probs[s]/ (1 - model.alpha))
#     total_unmet_demand_list.append(sum(v_unmet_demand[t, s] for t in model.T))


# import numpy as np
# import matplotlib.pyplot as plt

# # Plotting the Unmet Demand distribution as a histogram
# plt.figure(figsize=(10, 6))
# plt.hist(total_unmet_demand_list, bins=20, density=True, alpha=0.6, color='g', edgecolor='black')

# # Adding title and labels
# plt.title('Probability Density Function (PDF) of Unmet Demand Across Scenarios', fontsize=14)
# plt.xlabel('Total Unmet Demand', fontsize=12)
# plt.ylabel('Density', fontsize=12)

# # Show grid
# plt.grid(True)

# # Save the plot as a PDF file
# plt.savefig(f'unmet_demand_pdf_plot_beta_{model.beta}_alpha_{model.alpha}.pdf')

# # Display the plot
# plt.show()
# # Calculate average unmet demand across all scenarios
# average_unmet_demand = np.sum([total_unmet_demand_list[s] * model.probs[s] for s in model.S])

# # Calculate the 10th percentile of unmet demand across all scenarios
# percentile_10th = np.percentile(total_unmet_demand_list, 10)

# # Calculate the 90th percentile of unmet demand across all scenarios
# percentile_90th = np.percentile(total_unmet_demand_list, 90)

# # Calculate the lower quartile (25th percentile)
# lower_quartile = np.percentile(total_unmet_demand_list, 25)

# # Calculate the upper quartile (75th percentile)
# upper_quartile = np.percentile(total_unmet_demand_list, 75)

# # Calculate the interquartile range (IQR)
# iqr = upper_quartile - lower_quartile

# # Calculate the median (50th percentile)
# median_unmet_demand = np.percentile(total_unmet_demand_list, 50)

# # Print the statistics
# print("Average unmet demand across scenarios:", average_unmet_demand)
# print("Lower quartile (25th percentile) unmet demand:", lower_quartile)
# print("Upper quartile (75th percentile) unmet demand:", upper_quartile)
# print("10th percentile of unmet demand:", percentile_10th)
# print("90th percentile of unmet demand:", percentile_90th)
# print("Interquartile range (IQR) of unmet demand:", iqr)
# print("Median unmet demand:", median_unmet_demand)
