"""
Script for training the MLP model for residential load forecasting.

This script loads the processed dataset, trains a neural network,
saves the trained model and the training history.
"""

import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MLPTrainer:
    def __init__(self, project_root: Path):
        """
        Initialize the MLP trainer.

        Args:
            project_root (Path): Root directory of the project
        """
        self.project_root = project_root
        self.data_dir = project_root / 'data'
        self.output_dir = project_root / 'outputs'

        # Ensure output directories exist
        (self.output_dir / 'models').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'figures').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'logs').mkdir(parents=True, exist_ok=True)

    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Load and prepare data for training.

        Returns:
            tuple: Features (X) and target (y) arrays
        """
        logging.info("Loading and preparing data...")

        df = pd.read_csv(self.data_dir / 'processed' / 'Demanda-Total-Features.csv')

        # Calculate total demand and target
        df['Demanda_Total'] = df['Demanda_Residencial'] + df['Demanda_VEs']
        df['Target'] = df['Demanda_Total'].shift(-1)
        df = df.dropna()

        # Select features
        features = ['Dia', 'Mes', 'Hora', 'Dia_Semana', 'Demanda_Total',
                   'Veiculos_Plugados', 'Estacao', 'Periodo_Dia', 'Fim_Semana']

        X = df[features].values
        y = df['Target'].values

        return X, y

    def create_model(self, input_shape: int) -> keras.Model:
        """
        Create the MLP model architecture.

        Args:
            input_shape (int): Number of input features

        Returns:
            keras.Model: Compiled Keras model
        """
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),

            keras.layers.Dense(32, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),

            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1)
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', keras.metrics.RootMeanSquaredError()]
        )

        return model

    def train_model(self, X: np.ndarray, y: np.ndarray) -> tuple[keras.Model, dict]:
        """
        Train the MLP model.

        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target vector

        Returns:
            tuple: Trained model and training history
        """
        logging.info("Preparing training data...")

        # Scale the data
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

        # Save scalers
        pd.to_pickle(scaler_X, self.output_dir / 'models' / 'scaler_X.pkl')
        pd.to_pickle(scaler_y, self.output_dir / 'models' / 'scaler_y.pkl')

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=0.2, random_state=42
        )

        # Create and train model
        model = self.create_model(X.shape[1])

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.0001,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                self.output_dir / 'models' / 'best_model.keras',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]

        logging.info("Training model...")
        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )

        # Evaluate model
        test_loss = model.evaluate(X_test, y_test, verbose=0)
        logging.info(f"Test loss: {test_loss}")

        return model, history

    def save_training_results(self, model: keras.Model, history: dict):
        """
        Save the trained model, history, and generate plots.

        Args:
            model (keras.Model): Trained model
            history (dict): Training history
        """
        # Save model
        model.save(self.output_dir / 'models' / 'final_model.keras')

        # Save history
        pd.DataFrame(history.history).to_csv(
            self.output_dir / 'logs' / 'training_history.csv',
            index=False
        )

        # Generate and save training plots
        self.plot_training_history(history)

    def plot_training_history(self, history: dict):
        """
        Generate and save training history plots.

        Args:
            history (dict): Training history
        """
        # Loss plot
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True)

        # MAE plot
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training')
        plt.plot(history.history['val_mae'], label='Validation')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' / 'training_history.png')
        plt.close()

def main():
    """Main function to execute the model training pipeline."""
    try:
        # Get project root directory
        project_root = Path(__file__).parent.parent.parent

        # Create trainer instance
        trainer = MLPTrainer(project_root)

        # Load data
        X, y = trainer.load_data()

        # Train model
        model, history = trainer.train_model(X, y)

        # Save results
        trainer.save_training_results(model, history)

        logging.info("Training completed successfully!")

    except Exception as e:
        logging.error(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    main()