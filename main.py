"""
Main pipeline for energy demand prediction project.

This script orchestrates the entire data processing and analysis pipeline,
from raw data processing to model training and evaluation.
"""

from pathlib import Path
from src.preprocessing.data_merger import ProfileMerger
from src.preprocessing.feature_engineering import FeatureEngineer
from src.preprocessing.time_aggregation import TimeAggregator
from src.preprocessing.demand_processor import DemandProcessor

def run_preprocessing_pipeline():
    """Execute the complete preprocessing pipeline."""
    data_dir = Path('data')

    # 1. Merge profiles
    merger = ProfileMerger(data_dir)
    merged_data = merger.merge_profiles()
    merger.save_merged_data(merged_data)

    # 2. Process demand
    processor = DemandProcessor(data_dir)
    processor.process_demand()

    # 3. Add features
    engineer = FeatureEngineer(data_dir)
    processed_data = engineer.process_features()
    engineer.save_processed_data(processed_data)

    # 4. Create aggregations
    aggregator = TimeAggregator(data_dir)
    aggregator.process_aggregations()
def main():
    """Main function to execute the complete pipeline."""
    try:
        print("Starting preprocessing pipeline...")
        run_preprocessing_pipeline()

        print("\nPreprocessing completed successfully!")

    except Exception as e:
        print(f"\nError in pipeline execution: {e}")

if __name__ == "main":
    main()