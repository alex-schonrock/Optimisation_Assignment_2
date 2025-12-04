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
        expected_unmet_demand: list[float] | None = None
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
        self.expected_unmet_demand = expected_unmet_demand
        

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
        print("average price is", np.mean(price_MWh))

        ################# NEED TO CONVERT PRICE TO DKK/MWh  ###############################

        demand_df = self.data_loader.load_data_file("demand_data.csv")
        demand_df["Date"] = pd.to_datetime(demand_df["Date"], dayfirst=True)
        demand_df = demand_df.set_index("Date").sort_index()
        demand_df = demand_df.loc[start_date:end_date]
        demand_MW = demand_df['load MW'].astype(float).to_numpy()
        ### Need to scale demand data to be relative to our power plant size 
        demand_MW_max = demand_MW.max()
        plant_max_fuel_capacity_MWh = 386.06
        demand_MW_plant = demand_MW * (plant_max_fuel_capacity_MWh / demand_MW_max)*0.8  # scale to 80% of max plant capacity
        print("average demand is", np.mean(demand_MW_plant))
        # print(demand_MW_plant)


        demand_min = 173.73 # in MWh/day
        exp_allowance =  np.mean(price_MWh)*np.mean(demand_MW_plant) # in dkk/day
        ############## need to set unment demand cost ##################
        unmet_demand_cost = 100 # in dkk/MWh

        
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
            depreciation = 0.95 # 2% depreciation per day
            max_storage_capacity = 9000 # MWh

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
            depreciation = 0.95 # 2% depreciation per day
            max_storage_capacity = 9000 # MWh
            # demand_factor_scenarios = [0.9, 1.1]  # Example factors for scenarios
            # demand_scenarios = [demand_factor_scenarios[i] * demand_MW_plant for i in range(len(demand_factor_scenarios))]
            # scenario_probabilities = [1/2, 1/2]  # Equal probabilities for simplicity

            def generate_scenarios(demand_MW_plant, demand_factor_scenarios, K):
                # Multiply each factor by the demand for each time period
                demand_scenarios = np.array([demand_MW_plant * factor for factor in demand_factor_scenarios])

                # Sample from the demand scenarios for K time periods (assuming 30 time periods in total)
                sampled_scenarios = np.random.choice(demand_scenarios.flatten(), size=(K, 30))  # K samples, 30 time periods
                return sampled_scenarios

            # Example parameters
            # demand_MW_plant = np.array([100] * 30)  # Example: Total demand for the plant in MW for 30 time periods
            demand_factor_scenarios = [0.95, 1.05]  # Scenario factors
            K = 2  # Number of sampled scenarios

            sampled_scenarios = generate_scenarios(demand_MW_plant, demand_factor_scenarios, K)
            scenario_probabilities = 1 / K * np.ones(K)  # Equal probabilities for simplicity
            print("Sampled Scenarios:\n", sampled_scenarios)
            with open('unmet_demand_model_2.csv', 'r') as f:
                reader = csv.reader(f)
                loaded_list = next(reader)
            expected_unmet_demand = [float(x) for x in loaded_list]
            print("Expected unmet demand:\n", expected_unmet_demand)

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
             sampled_scenarios,
             scenario_probabilities,
             expected_unmet_demand
            )    
        pass

