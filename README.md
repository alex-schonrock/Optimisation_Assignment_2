# Fuel Procurement Optimization Under Uncertainty

Progressive optimization models for 30-day fuel procurement planning under demand and price uncertainty, comparing deterministic baseline, storage-enabled, and stochastic CVaR approaches.

## File Structure
```
├── Data/
│   ├── demand_data.csv              # Input: demand scenarios
│   ├── price_data.csv               # Input: price scenarios
│   └── figures/                     # Output: all generated plots
│
├── Data_ops/
│   └── data_loader_processor.py     # Data loading and preprocessing
│
├── Runner/
│   ├── opt_model_1.py              # Model 1: Deterministic baseline (no storage)
│   ├── opt_model_2.py              # Model 2: With storage & budget carryover
│   ├── opt_model_3.py              # Model 3: Stochastic with CVaR
│   ├── runner_model_1.py           # Model 1 execution wrapper
│   ├── runner_model_2.py           # Model 2 execution wrapper
│   └── runner_model_3.py           # Model 3 execution wrapper
│
├── Utils/
│   ├── model_3_plotting.py         # Plotting functions (all models)
│   ├── utils.py                    # Shared utilities
│
├── main.py                         # Run all analyses
└── README.md
```

## Quick Start

### Requirements
```bash
pip install numpy pandas matplotlib gurobipy
```
Requires Gurobi license from [gurobi.com](https://www.gurobi.com/)

### Run Everything
```bash
python main.py
```

This runs:
1. Model 1 (baseline)
2. Model 2 (with storage)
3. Model 3 (stochastic, β=0.3)
4. All comparisons and visualizations

Results saved to `Data/figures/`

### Customize Risk Parameters
Edit `main.py`:
```python
alpha_val = 0.9      # CVaR confidence level
beta_val_0 = 0.0     # Risk-neutral
beta_val_1 = 0.3     # Risk-averse
```

## Models

- **Model 1**: Deterministic baseline (no storage, fixed budget)
- **Model 2**: Storage + budget carryover with depreciation
- **Model 3**: Two-stage stochastic with CVaR risk management (1,000 scenarios)
