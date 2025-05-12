"""
demand_processor.py

Processes hourly profiles to create a consolidated demand dataset with:
- Temporal features (day, month, hour, weekday)
- Residential demand (sum)
- EV demand L1 and L2 (sum)
- Total demand with L1 and L2 (sum)

Input files (from /data/processed/):
- Residential-Profiles-By-Hours.csv
- PEV-Profiles-L1-By-Hours.csv
- PEV-Profiles-L2-By-Hours.csv

Output:
- Total-Demand.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DemandProcessor:
    def __init__(self, project_root: Path):
        """Initialize the DemandProcessor."""
        self.project_root = project_root
        self.input_dir = project_root / 'data/processed'
        self.output_dir = project_root / 'data/processed'
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_profiles(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all profile data from processed files.

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
            logging.error(f"Error loading profiles: {str(e)}")
            raise

    def calculate_demands(self, res_df: pd.DataFrame, pev_l1_df: pd.DataFrame,
                         pev_l2_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all demand values and create temporal features.

        Args:
            res_df: Residential profiles DataFrame
            pev_l1_df: PEV L1 profiles DataFrame
            pev_l2_df: PEV L2 profiles DataFrame

        Returns:
            pd.DataFrame: Processed demand DataFrame
        """
        try:
            # Create base DataFrame with temporal features
            demand_df = pd.DataFrame()
            demand_df['Time'] = res_df['Time']
            demand_df['Dia'] = res_df['Time'].dt.day
            demand_df['Mes'] = res_df['Time'].dt.month
            demand_df['Hora'] = res_df['Time'].dt.hour
            
            # Mapeamento: 0-6 (seg-dom) para 1-7 (dom-sab)
            demand_df['Dia_Semana'] = (res_df['Time'].dt.dayofweek + 2) % 7
            demand_df['Dia_Semana'] = demand_df['Dia_Semana'].replace(0, 7)

            # Calculate residential demand (sum of all households)
            household_cols = [col for col in res_df.columns if 'Household' in col]
            demand_df['Demanda_Residencial'] = res_df[household_cols].sum(axis=1)
            logging.info("Calculated residential demand")

            # Calculate PEV L1 demand (sum of all vehicles)
            vehicle_cols_l1 = [col for col in pev_l1_df.columns if 'Vehicle' in col]
            demand_df['Demanda_VE_L1'] = pev_l1_df[vehicle_cols_l1].sum(axis=1)
            logging.info("Calculated PEV L1 demand")

            # Calculate PEV L2 demand (sum of all vehicles)
            vehicle_cols_l2 = [col for col in pev_l2_df.columns if 'Vehicle' in col]
            demand_df['Demanda_VE_L2'] = pev_l2_df[vehicle_cols_l2].sum(axis=1)
            logging.info("Calculated PEV L2 demand")

            # Calculate total demands
            demand_df['Demanda_Total_L1'] = (
                demand_df['Demanda_Residencial'] + demand_df['Demanda_VE_L1']
            )
            demand_df['Demanda_Total_L2'] = (
                demand_df['Demanda_Residencial'] + demand_df['Demanda_VE_L2']
            )
            demand_df['Demanda_Total'] = (
                demand_df['Demanda_Residencial'] +
                demand_df['Demanda_VE_L1'] +
                demand_df['Demanda_VE_L2']
            )
            logging.info("Calculated total demands")

            # Log some statistics for validation
            logging.info("\nDemand Statistics (W):")
            for col in demand_df.columns:
                if 'Demanda' in col:
                    logging.info(f"{col}:")
                    logging.info(f"  Mean: {demand_df[col].mean():,.2f}")
                    logging.info(f"  Max:  {demand_df[col].max():,.2f}")
                    logging.info(f"  Min:  {demand_df[col].min():,.2f}")

            return demand_df

        except Exception as e:
            logging.error(f"Error calculating demands: {str(e)}")
            raise

    def save_demand_data(self, demand_df: pd.DataFrame):
        """
        Save the processed demand data.

        Args:
            demand_df: Processed demand DataFrame
        """
        try:
            output_file = self.output_dir / 'Total-Demand.csv'
            demand_df.to_csv(output_file, index=False)
            logging.info(f"Saved demand data to: {output_file}")

            # Log the first few rows
            logging.info("\nFirst few rows of the output file:")
            logging.info(demand_df.head())

        except Exception as e:
            logging.error(f"Error saving demand data: {str(e)}")
            raise

def main():
    """Main function to execute the demand processing pipeline."""
    try:
        # Get project root directory
        project_root = Path(__file__).parent.parent.parent

        # Create processor instance
        processor = DemandProcessor(project_root)

        # Load profiles
        res_df, pev_l1_df, pev_l2_df = processor.load_profiles()

        # Calculate demands
        demand_df = processor.calculate_demands(res_df, pev_l1_df, pev_l2_df)

        # Save results
        processor.save_demand_data(demand_df)

        logging.info("Demand processing completed successfully!")

    except Exception as e:
        logging.error(f"Error in demand processing pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()