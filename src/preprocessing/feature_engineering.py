"""
feature_engineering.py

Adds additional features to the Total-Demand dataset:
- Seasons (1=Spring, 2=Summer, 3=Fall, 4=Winter) with exact dates
- Day Period (0=Night, 1=Morning, 2=Afternoon, 3=Evening)
- Weekend Flag (0=Weekday, 1=Weekend)
- Number of EVs plugged in (L1 and L2)

Input:
- Total-Demand.csv

Output:
- Total-Demand-Features.csv
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

class FeatureEngineer:
    def __init__(self, project_root: Path):
        """Initialize the FeatureEngineer."""
        self.project_root = project_root
        self.input_dir = project_root / 'data/processed'
        self.output_dir = project_root / 'data/processed'
        self.input_file = self.input_dir / 'Total-Demand.csv'

    def get_season(self, month: int, day: int) -> int:
        """
        Determine season based on exact dates.

        1=Spring (March 21 - June 20)
        2=Summer (June 21 - September 20)
        3=Fall (September 21 - December 20)
        4=Winter (December 21 - March 20)
        """
        if month == 3:
            return 1 if day >= 21 else 4
        elif month == 6:
            return 1 if day < 21 else 2
        elif month == 9:
            return 2 if day < 21 else 3
        elif month == 12:
            return 3 if day < 21 else 4
        elif month in [4, 5]:
            return 1  # Spring
        elif month in [7, 8]:
            return 2  # Summer
        elif month in [10, 11]:
            return 3  # Fall
        else:  # month in [1, 2]
            return 4  # Winter

    def get_day_period(self, hour: int) -> int:
        """
        Determine period of day based on hour.

        0=Night (22-5)
        1=Morning (6-11)
        2=Afternoon (12-17)
        3=Evening (18-21)
        """
        if 6 <= hour <= 11:
            return 1  # Morning
        elif 12 <= hour <= 17:
            return 2  # Afternoon
        elif 18 <= hour <= 21:
            return 3  # Evening
        else:  # 22-5
            return 0  # Night

    def count_plugged_evs(self, row: pd.Series, threshold: float = 0) -> int:
        """Count number of EVs with demand above threshold."""
        return (row > threshold).sum()

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all new features to the dataset."""
        try:
            logging.info("Adding new features...")

            # Convert Time to datetime if not already
            df['Time'] = pd.to_datetime(df['Time'])

            # Add season
            df['Estacao'] = df.apply(
                lambda x: self.get_season(x['Mes'], x['Dia']), axis=1
            )

            # Add day period
            df['Periodo_Dia'] = df['Hora'].apply(self.get_day_period)

            # Add weekend flag (Dia_Semana: 1=Sunday, 7=Saturday)
            df['Fim_Semana'] = df['Dia_Semana'].isin([1, 7]).astype(int)

            # Load original PEV profiles to count plugged vehicles
            pev_l1 = pd.read_csv(self.input_dir / 'PEV-Profiles-L1-By-Hours.csv')
            pev_l2 = pd.read_csv(self.input_dir / 'PEV-Profiles-L2-By-Hours.csv')

            # Get vehicle columns
            veh_cols_l1 = [col for col in pev_l1.columns if 'Vehicle' in col]
            veh_cols_l2 = [col for col in pev_l2.columns if 'Vehicle' in col]

            # Count plugged vehicles (demand > 0)
            df['VEs_Plugados_L1'] = pev_l1[veh_cols_l1].apply(
                lambda row: self.count_plugged_evs(row), axis=1
            )
            df['VEs_Plugados_L2'] = pev_l2[veh_cols_l2].apply(
                lambda row: self.count_plugged_evs(row), axis=1
            )

            # Log feature statistics
            logging.info("\nFeature Statistics:")
            logging.info("\nSeason distribution:")
            logging.info(df['Estacao'].value_counts().sort_index())

            logging.info("\nDay period distribution:")
            logging.info(df['Periodo_Dia'].value_counts().sort_index())

            logging.info("\nWeekend distribution:")
            logging.info(df['Fim_Semana'].value_counts().sort_index())

            logging.info("\nPlugged EVs statistics:")
            logging.info(f"L1 - Mean: {df['VEs_Plugados_L1'].mean():.2f}")
            logging.info(f"L2 - Mean: {df['VEs_Plugados_L2'].mean():.2f}")

            return df

        except Exception as e:
            logging.error(f"Error adding features: {str(e)}")
            raise

    def process_data(self):
        """Main method to load data, add features and save results."""
        try:
            # Load data
            logging.info(f"Loading data from {self.input_file}")
            df = pd.read_csv(self.input_file)

            # Add features
            df_features = self.add_features(df)

            # Save results
            output_file = self.output_dir / 'Total-Demand-Features.csv'
            df_features.to_csv(output_file, index=False)
            logging.info(f"Saved featured data to {output_file}")

            # Log sample of results
            logging.info("\nSample of output data:")
            logging.info(df_features.head())

        except Exception as e:
            logging.error(f"Error processing data: {str(e)}")
            raise

def main():
    """Main function to execute the feature engineering pipeline."""
    try:
        # Get project root directory
        project_root = Path(__file__).parent.parent.parent

        # Create feature engineer instance
        engineer = FeatureEngineer(project_root)

        # Process data
        engineer.process_data()

        logging.info("Feature engineering completed successfully!")

    except Exception as e:
        logging.error(f"Error in feature engineering pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()