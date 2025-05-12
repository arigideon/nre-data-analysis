"""
data_merger.py

Merges hourly residential profiles with PEV profiles (L1 and L2),
adding vehicle demands to their respective households.

Input files (from /data/processed/):
- Residential-Profiles-By-Hours.csv
- PEV-Profiles-L1-By-Hours.csv
- PEV-Profiles-L2-By-Hours.csv

Output files:
- Residential-Profiles-With-PEV-L1.csv
- Residential-Profiles-With-PEV-L2.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DataMerger:
    def __init__(self, project_root: Path):
        """Initialize the DataMerger."""
        self.project_root = project_root
        self.input_dir = project_root / 'data/processed'
        self.output_dir = project_root / 'data/processed'
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all hourly profile data.

        Returns:
            tuple: (residential_df, pev_l1_df, pev_l2_df)
        """
        try:
            # Load residential profiles
            res_df = pd.read_csv(self.input_dir / 'Residential-Profiles-By-Hours.csv')
            res_df['Time'] = pd.to_datetime(res_df['Time'])
            logging.info("Loaded residential profiles")

            # Load PEV L1 profiles
            pev_l1_df = pd.read_csv(self.input_dir / 'PEV-Profiles-L1-By-Hours.csv')
            pev_l1_df['Time'] = pd.to_datetime(pev_l1_df['Time'])
            logging.info("Loaded PEV L1 profiles")

            # Load PEV L2 profiles
            pev_l2_df = pd.read_csv(self.input_dir / 'PEV-Profiles-L2-By-Hours.csv')
            pev_l2_df['Time'] = pd.to_datetime(pev_l2_df['Time'])
            logging.info("Loaded PEV L2 profiles")

            return res_df, pev_l1_df, pev_l2_df

        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise

    def get_household_vehicle_mapping(self, pev_df: pd.DataFrame) -> dict:
        """
        Create mapping between households and their vehicles.

        Args:
            pev_df: PEV profiles DataFrame

        Returns:
            dict: Mapping of household number to list of vehicle column names
        """
        mapping = {}
        for col in pev_df.columns:
            if 'Vehicle' in col:
                # Extract household number using regex
                match = re.search(r'Household (\d+)', col)
                if match:
                    household_num = int(match.group(1))
                    if household_num not in mapping:
                        mapping[household_num] = []
                    mapping[household_num].append(col)

        return mapping

    def merge_profiles(self, res_df: pd.DataFrame, pev_df: pd.DataFrame,
                      charge_level: str) -> pd.DataFrame:
        """
        Merge residential and PEV profiles by summing vehicle demands per household.

        Args:
            res_df: Residential profiles DataFrame
            pev_df: PEV profiles DataFrame
            charge_level: 'L1' or 'L2'

        Returns:
            pd.DataFrame: Merged profiles with total demand per household
        """
        try:
            # Create copy of residential data
            merged_df = res_df.copy()

            # Get household-vehicle mapping
            vehicle_mapping = self.get_household_vehicle_mapping(pev_df)

            # For each household, sum all its vehicle demands
            for household_num, vehicle_cols in vehicle_mapping.items():
                household_col = f'Household {household_num}'

                if household_col in merged_df.columns:
                    # Sum all vehicles for this household
                    vehicles_sum = pev_df[vehicle_cols].sum(axis=1)

                    # Add vehicle demand to household demand
                    merged_df[household_col] = merged_df[household_col] + vehicles_sum

            # Log statistics
            logging.info(f"\nMerged profiles statistics ({charge_level}):")
            logging.info(f"Number of households processed: {len(vehicle_mapping)}")

            # Calculate and log some validation statistics
            total_res_demand = merged_df[[col for col in merged_df.columns
                                        if 'Household' in col]].sum().sum()
            logging.info(f"Total demand after merge: {total_res_demand:,.2f}")

            return merged_df

        except Exception as e:
            logging.error(f"Error merging profiles: {str(e)}")
            raise

    def save_merged_data(self, merged_l1_df: pd.DataFrame, merged_l2_df: pd.DataFrame):
        """
        Save the merged profile data.

        Args:
            merged_l1_df: Merged profiles with L1 charging
            merged_l2_df: Merged profiles with L2 charging
        """
        try:
            # Save L1 merged profiles
            output_l1 = self.output_dir / 'Residential-Profiles-With-PEV-L1.csv'
            merged_l1_df.to_csv(output_l1, index=False)
            logging.info(f"Saved L1 merged profiles to: {output_l1}")

            # Save L2 merged profiles
            output_l2 = self.output_dir / 'Residential-Profiles-With-PEV-L2.csv'
            merged_l2_df.to_csv(output_l2, index=False)
            logging.info(f"Saved L2 merged profiles to: {output_l2}")

            # Log sample of results
            logging.info("\nSample of merged data:")
            sample_cols = ['Time'] + [col for col in merged_l1_df.columns
                                    if 'Household' in col][:3]
            logging.info("\nL1 sample (first 3 households):")
            logging.info(merged_l1_df[sample_cols].head())
            logging.info("\nL2 sample (first 3 households):")
            logging.info(merged_l2_df[sample_cols].head())

        except Exception as e:
            logging.error(f"Error saving merged data: {str(e)}")
            raise

    def validate_merges(self, merged_l1_df: pd.DataFrame, merged_l2_df: pd.DataFrame):
        """
        Validate the merged data.

        Args:
            merged_l1_df: Merged profiles with L1 charging
            merged_l2_df: Merged profiles with L2 charging
        """
        try:
            logging.info("\nValidation Results:")

            # Check household counts
            l1_households = len([col for col in merged_l1_df.columns
                               if 'Household' in col])
            l2_households = len([col for col in merged_l2_df.columns
                               if 'Household' in col])
            logging.info(f"Number of households (L1): {l1_households}")
            logging.info(f"Number of households (L2): {l2_households}")

            # Check time range
            logging.info(f"Time range: {merged_l1_df['Time'].min()} to {merged_l1_df['Time'].max()}")

            # Check for any missing values
            logging.info("\nMissing values check:")
            logging.info(f"L1 missing values: {merged_l1_df.isnull().sum().sum()}")
            logging.info(f"L2 missing values: {merged_l2_df.isnull().sum().sum()}")

        except Exception as e:
            logging.error(f"Error in validation: {str(e)}")
            raise

def main():
    """Main function to execute the data merging pipeline."""
    try:
        # Get project root directory
        project_root = Path(__file__).parent.parent.parent

        # Create merger instance
        merger = DataMerger(project_root)

        # Load data
        res_df, pev_l1_df, pev_l2_df = merger.load_data()

        # Merge profiles
        merged_l1_df = merger.merge_profiles(res_df, pev_l1_df, 'L1')
        merged_l2_df = merger.merge_profiles(res_df, pev_l2_df, 'L2')

        # Validate merges
        merger.validate_merges(merged_l1_df, merged_l2_df)

        # Save results
        merger.save_merged_data(merged_l1_df, merged_l2_df)

        logging.info("Data merging completed successfully!")

    except Exception as e:
        logging.error(f"Error in data merging pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()