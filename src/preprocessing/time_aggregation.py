"""
time_aggregation.py

Aggregates 10-minute resolution data to hourly and monthly sums for:
- Residential-Profiles.csv
- PEV-Profiles-L1.csv
- PEV-Profiles-L2.csv

Outputs:
- Residential-Profiles-By-Hours.csv
- Residential-Profiles-By-Month.csv
- PEV-Profiles-L1-By-Hours.csv
- PEV-Profiles-L2-By-Hours.csv
"""

import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class TimeAggregator:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.raw_dir = project_root / 'data/raw'
        self.proc_dir = project_root / 'data/processed'
        self.proc_dir.mkdir(parents=True, exist_ok=True)

    def aggregate_file_by_hour(self, filename: str, output_name: str):
        """Aggregate a 10-min resolution file to hourly sums."""
        input_path = self.raw_dir / filename
        output_path = self.proc_dir / output_name
        logging.info(f"Aggregating {input_path} to {output_path} (hourly sum)")
        try:
            df = pd.read_csv(input_path)
            if 'Time' not in df.columns:
                raise ValueError("Time column not found in the dataset")
            df['Time'] = pd.to_datetime(df['Time'])
            df = df.set_index('Time')
            hourly_df = df.resample('H').sum().reset_index()
            # Add time columns for convenience
            hourly_df['Hour'] = hourly_df['Time'].dt.hour
            hourly_df['Day'] = hourly_df['Time'].dt.day
            hourly_df['Month'] = hourly_df['Time'].dt.month
            hourly_df['Year'] = hourly_df['Time'].dt.year
            # Reorder columns
            time_cols = ['Time', 'Year', 'Month', 'Day', 'Hour']
            data_cols = [col for col in hourly_df.columns if col not in time_cols]
            hourly_df = hourly_df[time_cols + data_cols]
            hourly_df.to_csv(output_path, index=False)
            logging.info(f"Saved: {output_path}")
            return hourly_df
        except Exception as e:
            logging.error(f"Error aggregating {filename}: {str(e)}")
            return None

    def aggregate_residential_by_month(self, hourly_df: pd.DataFrame):
        """Aggregate hourly residential data to monthly sums."""
        try:
            temporal_cols = ['Time', 'Year', 'Month', 'Day', 'Hour']
            consumption_cols = [col for col in hourly_df.columns if col not in temporal_cols]
            monthly_df = hourly_df.groupby(['Year', 'Month'])[consumption_cols].sum().reset_index()
            monthly_df = monthly_df.sort_values(['Year', 'Month'])
            output_path = self.proc_dir / 'Residential-Profiles-By-Month.csv'
            monthly_df.to_csv(output_path, index=False)
            logging.info(f"Saved: {output_path}")
            return monthly_df
        except Exception as e:
            logging.error(f"Error in monthly aggregation: {str(e)}")
            return None

def main():
    try:
        project_root = Path(__file__).parent.parent.parent
        aggregator = TimeAggregator(project_root)

        # Agregar perfis residenciais por hora e por mês
        res_hourly = aggregator.aggregate_file_by_hour(
            'Residential-Profiles.csv', 'Residential-Profiles-By-Hours.csv'
        )
        if res_hourly is not None:
            aggregator.aggregate_residential_by_month(res_hourly)

        # Agregar PEV L1 por hora
        aggregator.aggregate_file_by_hour(
            'PEV-Profiles-L1.csv', 'PEV-Profiles-L1-By-Hours.csv'
        )

        # Agregar PEV L2 por hora
        aggregator.aggregate_file_by_hour(
            'PEV-Profiles-L2.csv', 'PEV-Profiles-L2-By-Hours.csv'
        )

        logging.info("All aggregations completed successfully!")

    except Exception as e:
        logging.error(f"Error in time aggregation pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()