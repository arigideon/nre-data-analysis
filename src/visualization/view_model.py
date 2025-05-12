"""
view_model.py

Visualiza modelos treinados e suas métricas para três cenários,
considerando demandas combinadas em cada cenário.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import logging
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class R2Score(tf.keras.metrics.Metric):
    def __init__(self, name='r2_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.squared_sum = self.add_weight(name='squared_sum', initializer='zeros')
        self.sum = self.add_weight(name='sum', initializer='zeros')
        self.res = self.add_weight(name='residual', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        self.squared_sum.assign_add(tf.reduce_sum(tf.square(y_true)))
        self.sum.assign_add(tf.reduce_sum(y_true))
        self.res.assign_add(tf.reduce_sum(tf.square(y_true - y_pred)))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        mean = self.sum / self.count
        total = self.squared_sum - (self.sum * self.sum / self.count)
        residual = self.res
        return 1 - (residual / total)

    def reset_state(self):
        self.squared_sum.assign(0.)
        self.sum.assign(0.)
        self.res.assign(0.)
        self.count.assign(0.)

class ModelVisualizer:
    def __init__(self, project_root: Path):
        """Initialize the ModelVisualizer."""
        self.project_root = project_root
        self.data_dir = project_root / 'data/processed'
        self.models_dir = project_root / 'outputs/models'
        self.logs_dir = project_root / 'outputs/logs'
        self.figures_dir = project_root / 'outputs/figures'
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        self.scenarios = {
            'L1': {
                'name': 'Residential + EV L1',
                'model_name': 'model_residential_L1',
                'target_col': 'Demanda_Total_L1',
                'demand_cols': ['Demanda_Residencial', 'Demanda_VE_L1'],
                'ev_plugged_col': 'VEs_Plugados_L1'
            },
            'L2': {
                'name': 'Residential + EV L2',
                'model_name': 'model_residential_L2',
                'target_col': 'Demanda_Total_L2',
                'demand_cols': ['Demanda_Residencial', 'Demanda_VE_L2'],
                'ev_plugged_col': 'VEs_Plugados_L2'
            },
            'L1L2': {
                'name': 'Residential + EV L1 + L2',
                'model_name': 'model_residential_L1L2',
                'target_col': 'Demanda_Total',
                'demand_cols': ['Demanda_Residencial', 'Demanda_VE_L1', 'Demanda_VE_L2'],
                'ev_plugged_col': ['VEs_Plugados_L1', 'VEs_Plugados_L2']
            }
        }

    def get_features_for_scenario(self, scenario: str):
        """Return features for scenario including combined demand."""
        base_features = ['Dia', 'Mes', 'Hora', 'Dia_Semana',
                        'Estacao', 'Periodo_Dia', 'Fim_Semana']

        if scenario == 'L1L2':
            return (base_features + ['VEs_Plugados_L1', 'VEs_Plugados_L2', 'Demanda_Combinada'])
        else:
            return (base_features + [self.scenarios[scenario]['ev_plugged_col'], 'Demanda_Combinada'])

    def load_model_and_history(self, scenario: str):
        """Load trained model and training history."""
        try:
            model_path = self.models_dir / f"{self.scenarios[scenario]['model_name']}.keras"
            model = load_model(model_path, custom_objects={'R2Score': R2Score})

            history_path = self.logs_dir / f"{self.scenarios[scenario]['model_name']}_history.csv"
            history = pd.read_csv(history_path)

            return model, history
        except Exception as e:
            logging.error(f"Error loading model and history: {str(e)}")
            raise

    def plot_learning_curves(self, history: pd.DataFrame, scenario: str):
        """Plot learning curves for model training history."""
        plt.figure(figsize=(12, 8))

        # Loss
        plt.subplot(2, 2, 1)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title(f'Model Loss - {self.scenarios[scenario]["name"]}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # MAE
        plt.subplot(2, 2, 2)
        plt.plot(history['mae'], label='Training MAE')
        plt.plot(history['val_mae'], label='Validation MAE')
        plt.title('Mean Absolute Error')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()

        # RMSE
        plt.subplot(2, 2, 3)
        plt.plot(history['root_mean_squared_error'], label='Training RMSE')
        plt.plot(history['val_root_mean_squared_error'], label='Validation RMSE')
        plt.title('Root Mean Squared Error')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.legend()

        # R²
        plt.subplot(2, 2, 4)
        plt.plot(history['r2_score'], label='Training R²')
        plt.plot(history['val_r2_score'], label='Validation R²')
        plt.title('R² Score')
        plt.xlabel('Epoch')
        plt.ylabel('R²')
        plt.legend()

        plt.tight_layout()
        plt.savefig(self.figures_dir / f'learning_curves_{scenario}.png')
        plt.close()

    def plot_scatter(self, scenario: str):
        """Plot scatter plot of actual vs predicted values."""
        try:
            df = pd.read_csv(self.data_dir / 'Total-Demand-Features.csv')
            model, _ = self.load_model_and_history(scenario)

            # Calculate combined demand
            df['Demanda_Combinada'] = df[self.scenarios[scenario]['demand_cols']].sum(axis=1)

            # Get features and prepare data
            features = self.get_features_for_scenario(scenario)
            X = df[features].values
            y = df[self.scenarios[scenario]['target_col']].values

            # Scale data
            x_scaler = MinMaxScaler()
            y_scaler = MinMaxScaler()
            X_scaled = x_scaler.fit_transform(X)
            y_scaled = y_scaler.fit_transform(y.reshape(-1, 1))

            # Make predictions and denormalize
            y_pred_scaled = model.predict(X_scaled)
            y_pred = y_scaler.inverse_transform(y_pred_scaled)

            # Plot
            plt.figure(figsize=(10, 8))
            plt.scatter(y, y_pred, alpha=0.5)
            plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
            plt.xlabel('Actual Demand (W)')
            plt.ylabel('Predicted Demand (W)')
            plt.title(f'Actual vs Predicted Demand - {self.scenarios[scenario]["name"]}')
            plt.tight_layout()
            plt.savefig(self.figures_dir / f'scatter_{scenario}.png')
            plt.close()

        except Exception as e:
            logging.error(f"Error plotting scatter: {str(e)}")
            raise

    def plot_demand_comparison(self, scenario: str):
        """Plot comparison between residential, combined and total demand."""
        try:
            df = pd.read_csv(self.data_dir / 'Total-Demand-Features.csv')

            # Calculate combined demand
            df['Demanda_Combinada'] = df[self.scenarios[scenario]['demand_cols']].sum(axis=1)

            # Get one week of data
            week_data = df.iloc[:168]

            plt.figure(figsize=(15, 8))
            plt.plot(range(len(week_data)), week_data['Demanda_Residencial'],
                    label='Residential Only', alpha=0.7)
            plt.plot(range(len(week_data)), week_data['Demanda_Combinada'],
                    label=f'Combined Demand ({scenario})', alpha=0.7)
            plt.plot(range(len(week_data)), week_data[self.scenarios[scenario]['target_col']],
                    label=f'Total Demand ({scenario})', alpha=0.7)

            plt.title(f'Demand Comparison - {self.scenarios[scenario]["name"]}')
            plt.xlabel('Hours')
            plt.ylabel('Demand (W)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(self.figures_dir / f'demand_comparison_{scenario}.png')
            plt.close()

        except Exception as e:
            logging.error(f"Error plotting demand comparison: {str(e)}")
            raise

    def plot_sensitivity_analysis(self, scenario: str):
        """Plot sensitivity analysis for model features."""
        try:
            model, _ = self.load_model_and_history(scenario)
            df = pd.read_csv(self.data_dir / 'Total-Demand-Features.csv')

            # Calculate combined demand
            df['Demanda_Combinada'] = df[self.scenarios[scenario]['demand_cols']].sum(axis=1)

            # Get features and prepare data
            features = self.get_features_for_scenario(scenario)
            X = df[features].values

            # Scale data
            x_scaler = MinMaxScaler()
            X_scaled = x_scaler.fit_transform(X)

            # Calculate sensitivities
            base_prediction = model.predict(X_scaled)
            sensitivities = []

            for i, feature in enumerate(features):
                X_perturbed = X_scaled.copy()
                X_perturbed[:, i] *= 1.1  # 10% increase
                new_prediction = model.predict(X_perturbed)
                sensitivity = np.mean(np.abs(new_prediction - base_prediction))
                sensitivities.append(sensitivity)

            # Plot
            plt.figure(figsize=(12, 6))
            plt.bar(features, sensitivities)
            plt.title(f'Feature Sensitivity Analysis - {self.scenarios[scenario]["name"]}')
            plt.xticks(rotation=45)
            plt.ylabel('Sensitivity')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(self.figures_dir / f'sensitivity_{scenario}.png')
            plt.close()

        except Exception as e:
            logging.error(f"Error plotting sensitivity analysis: {str(e)}")
            raise

    def show_final_metrics(self, scenario: str):
        """Display final metrics including MAPE."""
        metrics_path = self.logs_dir / f"{self.scenarios[scenario]['model_name']}_metrics.csv"
        if metrics_path.exists():
            metrics = pd.read_csv(metrics_path)
            print("\nFinal Model Metrics:")
            print(metrics.T)
            if 'mape' in metrics.columns:
                print(f"\nFinal MAPE (calculated with denormalized data): {metrics['mape'].values[0]:.2f}%")
        else:
            print("Metrics file not found.")

    def generate_all_plots(self, scenario: str):
        """Generate all visualization plots for a scenario."""
        try:
            logging.info(f"Generating all plots for scenario {scenario}")
            model, history = self.load_model_and_history(scenario)

            self.plot_learning_curves(history, scenario)
            self.plot_scatter(scenario)
            self.plot_demand_comparison(scenario)
            self.plot_sensitivity_analysis(scenario)

            logging.info(f"All plots generated successfully for scenario {scenario}")
        except Exception as e:
            logging.error(f"Error generating all plots: {str(e)}")
            raise

def main():
    """Main function to execute the visualization pipeline."""
    try:
        project_root = Path(__file__).parent.parent.parent
        visualizer = ModelVisualizer(project_root)

        while True:
            print("\nModel Visualization Menu")
            print("1. View L1 Model")
            print("2. View L2 Model")
            print("3. View L1L2 Model")
            print("4. Exit")

            choice = input("\nEnter your choice (1-4): ")

            if choice == '4':
                break

            scenarios = {'1': 'L1', '2': 'L2', '3': 'L1L2'}

            if choice in scenarios:
                scenario = scenarios[choice]
                print(f"\nVisualization Options for {visualizer.scenarios[scenario]['name']}")
                print("1. Learning Curves")
                print("2. Scatter Plot")
                print("3. Demand Comparison")
                print("4. Sensitivity Analysis")
                print("5. Generate All Plots")
                print("6. View Final Metrics")
                print("7. Back to Main Menu")

                viz_choice = input("\nEnter your choice (1-7): ")

                if viz_choice == '1':
                    _, history = visualizer.load_model_and_history(scenario)
                    visualizer.plot_learning_curves(history, scenario)
                elif viz_choice == '2':
                    visualizer.plot_scatter(scenario)
                elif viz_choice == '3':
                    visualizer.plot_demand_comparison(scenario)
                elif viz_choice == '4':
                    visualizer.plot_sensitivity_analysis(scenario)
                elif viz_choice == '5':
                    visualizer.generate_all_plots(scenario)
                elif viz_choice == '6':
                    visualizer.show_final_metrics(scenario)

                if viz_choice not in ['7']:
                    logging.info(f"Plot(s) saved in {visualizer.figures_dir}")

    except Exception as e:
        logging.error(f"Error in visualization pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()