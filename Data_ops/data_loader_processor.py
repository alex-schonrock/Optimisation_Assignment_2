import pandas as pd
from pathlib import Path
from pathlib import Path
from dataclasses import dataclass
from logging import Logger
import pandas as pd
from Utils.utils import load_datafile
import numpy as np
import csv

#### data loader class
class DataLoader:
    def __init__(self, input_path: str, model_type: str):
        self.input_path = Path(input_path).resolve()
        self.model_type = model_type
    def load_data_file(self, file_name: str):
        data = load_datafile(file_name, self.input_path)
        return data
    
class InputData:
    """Class to hold input data for the optimization model."""
    def __init__(
        self,
        price: list[float],
        demand: list[float],
        demand_min: float,
        exp_allowance: float,
        unmet_demand_cost: float,

        # optional for model 2 and 3
        storage_cost: float | None = None,  
        ramp_rate: float | None = None,
        depreciation: float | None = None,
        max_storage_capacity: float | None = None,
        demand_scenarios: list[list[float]] | None = None,
        scenario_probabilities: list[float] | None = None,
        price_scenarios: list[list[float]] | None = None
        ):

        # defining model 1 data inputs
        self.price = price
        self.demand = demand
        self.demand_min = demand_min
        self.exp_allowance = exp_allowance
        self.unmet_demand_cost = unmet_demand_cost

        # defining model 2 and 3 data inputs
        self.storage_cost = storage_cost
        self.ramp_rate = ramp_rate
        self.depreciation = depreciation
        self.max_storage_capacity = max_storage_capacity
        self.demand_scenarios = demand_scenarios
        self.probabilities = scenario_probabilities
        self.price_scenarios = price_scenarios
        

class DataProcessor():

    def __init__(self, input_path: str, model_type: str):
        self.input_path = input_path
        self.model_type = model_type
        data = DataLoader(input_path, model_type)
        self.data_loader = data

    def get_coefficients (self):
        """
        Extract and process data from the data files to create InputData instance
        """
        price_df = self.data_loader.load_data_file("price_data.csv")

        # 1) Convert Date to datetime and set as index
        price_df["Date"] = pd.to_datetime(price_df["Date"], dayfirst=True)
        price_df = price_df.set_index("Date").sort_index()

        # 2) Filter by your desired window
        start_date = "2016-04-01"
        end_date = "2016-04-30"
        price_df = price_df.loc[start_date:end_date]

        # 3) Create full daily date range and reindex
        full_idx = pd.date_range(start=start_date, end=end_date, freq="D")
        price_df_full = price_df.reindex(full_idx)

        # 4) Fill missing prices with interpolation
        col = "Gas Spot Price ($/mil Btu)"   # adjust if your column name is different
        price_df_full[col] = price_df_full[col].interpolate(method="time")

        price_btu = price_df_full[col].astype(float).to_numpy()
        price_MWh = price_btu * 6.44 *3.412  # convert $/btu to dkk/MWh
        # print("prices are", price_MWh)
        # print("average price is", np.mean(price_MWh))

        ################# NEED TO CONVERT PRICE TO DKK/MWh  ###############################

        demand_df = self.data_loader.load_data_file("demand_data.csv")
        demand_df["Date"] = pd.to_datetime(demand_df["Date"], dayfirst=True)
        demand_df = demand_df.set_index("Date").sort_index()
        demand_df = demand_df.loc[start_date:end_date]
        demand_MW = demand_df['load MW'].astype(float).to_numpy()
        ### Need to scale demand data to be relative to our power plant size 
        demand_MW_max = demand_MW.max()
        plant_max_fuel_capacity_MWh = 9265.35
        demand_MW_plant = demand_MW * (plant_max_fuel_capacity_MWh / demand_MW_max)  # scale to 80% of max plant capacity
        # print("average demand is", np.mean(demand_MW_plant))
        # print(demand_MW_plant)


        demand_min = 4169.41 # in MWh/day
        exp_allowance =  np.mean(price_MWh)*np.mean(demand_MW_plant) # in dkk/day
        ############## need to set unment demand cost ##################
        unmet_demand_cost = 100 # in dkk/MWh
        depreciation = 0.95

        
        if self.model_type == "Model_1":
            return InputData(
             price_MWh,
             demand_MW_plant,
             demand_min,
             exp_allowance,
             unmet_demand_cost,
            )
        
        if self.model_type == "Model_2":
            storage_cost = 0.05*np.mean(price_MWh) # dkk/MWh stored per day
            ramp_rate = 7.2 * 10**4 # MWh per day
            depreciation = depreciation # 2% depreciation per day
            max_storage_capacity = 18530.7 # MWh

            return InputData(
            price_MWh,
             demand_MW_plant,
             demand_min,
             exp_allowance,
             unmet_demand_cost,
             storage_cost,
             ramp_rate,
             depreciation,
             max_storage_capacity
            )
        
        if self.model_type == "Model_3":
            storage_cost = 0.05*np.mean(price_MWh) # dkk/MWh stored per day
            ramp_rate = 7.2 * 10**4 # MWh per day
            depreciation = depreciation # 2% depreciation per day
            max_storage_capacity = 18530.7 # MWh
            

            # def generate_scenarios(demand_MW_plant, demand_factor_scenarios, K):
            #     """
            #     Generate K demand scenarios where each day is independently high or low.
                
            #     Parameters:
            #     - demand_MW_plant: array of base demand for each day (shape: 30,)
            #     - demand_factor_scenarios: [low_factor, high_factor], e.g., [0.9, 1.1]
            #     - K: number of scenarios to sample
                
            #     Returns:
            #     - sampled_scenarios: array of shape (K, 30) with demand scenarios
            #     """
            #     np.random.seed(42)
                
            #     n_days = len(demand_MW_plant)
            #     sampled_scenarios = np.zeros((K, n_days))
                
            #     # For each scenario, randomly choose high or low for each day
            #     for k in range(K):
            #         # For each day, randomly pick 0 (low=0.9) or 1 (high=1.1) with equal probability
            #         random_choices = np.random.choice([0, 1], size=n_days, p=[0.5, 0.5])
                    
            #         # Apply the factors
            #         for t in range(n_days):
            #             factor = demand_factor_scenarios[random_choices[t]]
            #             sampled_scenarios[k, t] = demand_MW_plant[t] * factor
                
            #     return sampled_scenarios


            # demand_factor_scenarios = [0.9, 1.1]  # Low and high scenarios
            # K = 1000  # Number of scenarios

            def generate_scenarios_demand_and_price(demand_MW_plant, price_base, demand_factor_scenarios, price_factor_scenarios, K):
                """
                Generate K scenarios where BOTH demand and price vary independently each day.
                
                Each day has 4 possible states:
                1. Low demand, Low price (prob = 0.25)
                2. Low demand, High price (prob = 0.25)
                3. High demand, Low price (prob = 0.25)
                4. High demand, High price (prob = 0.25)
                
                Parameters:
                - demand_MW_plant: array of base demand for each day (shape: 30,)
                - price_base: array of base price for each day (shape: 30,)
                - demand_factor_scenarios: [low_factor, high_factor], e.g., [0.9, 1.1]
                - price_factor_scenarios: [low_factor, high_factor], e.g., [0.9, 1.1]
                - K: number of scenarios to sample
                
                Returns:
                - demand_scenarios: array of shape (K, 30) with demand scenarios
                - price_scenarios: array of shape (K, 30) with price scenarios
                """
                np.random.seed(42)
                
                n_days = len(demand_MW_plant)
                demand_scenarios = np.zeros((K, n_days))
                price_scenarios = np.zeros((K, n_days))
                
                # For each scenario, randomly choose high or low for demand AND price for each day
                for k in range(K):
                    # Independent random choices for demand and price
                    # 0 = low (0.9×), 1 = high (1.1×)
                    demand_choices = np.random.choice([0, 1], size=n_days, p=[0.5, 0.5])
                    price_choices = np.random.choice([0, 1], size=n_days, p=[0.5, 0.5])
                    
                    # Apply the factors
                    for t in range(n_days):
                        demand_factor = demand_factor_scenarios[demand_choices[t]]
                        price_factor = price_factor_scenarios[price_choices[t]]
                        
                        demand_scenarios[k, t] = demand_MW_plant[t] * demand_factor
                        price_scenarios[k, t] = price_base[t] * price_factor
                
                return demand_scenarios, price_scenarios
            
            demand_factor_scenarios = [0.9, 1.1]  # Low and high demand factors
            price_factor_scenarios = [0.9, 1.1]   # Low and high price factors
            K = 1000  # Number of scenarios

            demand_scenarios, price_scenarios = generate_scenarios_demand_and_price(
                demand_MW_plant=demand_MW_plant,
                price_base=price_MWh,  # Your base price array
                demand_factor_scenarios=demand_factor_scenarios,
                price_factor_scenarios=price_factor_scenarios,
                K=K
            )

            scenario_probabilities = 1 / K * np.ones(K)  # Equal probabilities

            return InputData(
            price_MWh,
             demand_MW_plant,
             demand_min,
             exp_allowance,
             unmet_demand_cost,
             storage_cost,
             ramp_rate,
             depreciation,
             max_storage_capacity,
             demand_scenarios,
             scenario_probabilities,
             price_scenarios
            )    
        pass

