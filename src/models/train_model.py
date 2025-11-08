"""
train_model.py

Treina redes neurais para previsão de carga em três cenários:
1. Residencial + VE L1 (demanda combinada)
2. Residencial + VE L2 (demanda combinada)
3. Residencial + VE L1 + L2 (demanda combinada)

O MAPE é calculado apenas na avaliação final, com os dados desnormalizados.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import RootMeanSquaredError
import tensorflow as tf
import random # Adicionar import para a biblioteca random padrão

# --- INÍCIO DA SEÇÃO DE FIXAÇÃO DE SEMENTES ---
SEED_VALUE = 42 # Você pode escolher qualquer número inteiro

# 1. Fixar semente para a biblioteca random do Python
random.seed(SEED_VALUE)

# 2. Fixar semente para NumPy
np.random.seed(SEED_VALUE)

# 3. Fixar semente para TensorFlow/Keras
tf.random.set_seed(SEED_VALUE)

# 4. Configurações adicionais para forçar operações do TensorFlow a serem determinísticas (opcional, mas recomendado para maior controle)
# Pode ter um custo de performance. Remova ou comente se a performance for crítica e a reprodutibilidade já for suficiente com as sementes.
# tf.config.experimental.enable_op_determinism() # Disponível em versões mais recentes do TF
# --- FIM DA SEÇÃO DE FIXAÇÃO DE SEMENTES ---

def setup_logging(logs_dir: Path):
    """Configura o logging."""
    log_file = logs_dir / f"training_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
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

class LoadForecaster:
    def __init__(self, project_root: Path):
        """Initialize the LoadForecaster."""
        self.project_root = project_root
        self.data_dir = project_root / 'data/processed'
        self.models_dir = project_root / 'outputs/models'
        self.logs_dir = project_root / 'outputs/logs'

        self.matrices_dir = project_root / 'outputs/matrices'

        # Create directories if they don't exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        setup_logging(self.logs_dir)

        # Define scenarios with combined demand approach
        self.scenarios = {
            # 'RES': {  # Novo cenário apenas residencial
            #     'input_cols': ['Dia', 'Mes', 'Hora', 'Dia_Semana',
            #                  'Estacao', 'Periodo_Dia', 'Fim_Semana'],
            #     'demand_cols': ['Demanda_Residencial'],
            #     'target_col': 'Demanda_Residencial',
            #     'model_name': 'model_residential'
            # },
            'L1': {
                'input_cols': ['Dia', 'Mes', 'Hora', 'Dia_Semana',
                             'Estacao', 'Periodo_Dia', 'Fim_Semana',
                             'VEs_Plugados_L1'],
                'demand_cols': ['Demanda_Residencial', 'Demanda_VE_L1'],
                'target_col': 'Demanda_Total_L1',
                'model_name': 'model_residential_L1'
            },
            # 'L2': {
            #     'input_cols': ['Dia', 'Mes', 'Hora', 'Dia_Semana',
            #                  'Estacao', 'Periodo_Dia', 'Fim_Semana',
            #                  'VEs_Plugados_L2'],
            #     'demand_cols': ['Demanda_Residencial', 'Demanda_VE_L2'],
            #     'target_col': 'Demanda_Total_L2',
            #     'model_name': 'model_residential_L2'
            # },
            # 'L1L2': {
            #     'input_cols': ['Dia', 'Mes', 'Hora', 'Dia_Semana',
            #                  'Estacao', 'Periodo_Dia', 'Fim_Semana',
            #                  'VEs_Plugados_L1', 'VEs_Plugados_L2'],
            #     'demand_cols': ['Demanda_Residencial', 'Demanda_VE_L1', 'Demanda_VE_L2'],
            #     'target_col': 'Demanda_Total',
            #     'model_name': 'model_residential_L1L2'
            # }
        }

    def prepare_data(self, scenario: str):
        """
        Prepara os dados para treinamento.

        Args:
            scenario: O cenário para preparar os dados ('RES', 'L1', 'L2', ou 'L1L2')

        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        try:
            # Load data
            df = pd.read_csv(self.data_dir / 'Total-Demand-Features.csv')

            # Prepare features based on scenario
            if scenario == 'RES':
                # Para o cenário residencial, usamos a demanda residencial diretamente
                features = self.scenarios[scenario]['input_cols'] + ['Demanda_Residencial']
                X = df[features].values
            else:
                # Para os outros cenários, calculamos a demanda combinada
                df['Demanda_Combinada'] = df[self.scenarios[scenario]['demand_cols']].sum(axis=1)
                features = self.scenarios[scenario]['input_cols'] + ['Demanda_Combinada']
                X = df[features].values

            y = df[self.scenarios[scenario]['target_col']].values

            # Create target for next hour prediction
            y = np.roll(y, -1)
            y = y[:-1]  # Remove last row
            X = X[:-1]  # Remove last row to match y

            # Scale data
            self.x_scaler = MinMaxScaler()
            self.y_scaler = MinMaxScaler()
            X_scaled = self.x_scaler.fit_transform(X)
            y_scaled = self.y_scaler.fit_transform(y.reshape(-1, 1))

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_scaled, test_size=0.2, random_state=42
            )

            # Save feature names for later use
            self.feature_names = features

            logging.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
            logging.info(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

            return X_train, X_test, y_train, y_test

        except Exception as e:
            logging.error(f"Error preparing data: {str(e)}")
            raise

    def create_model(self, input_dim: int) -> Sequential:
        """
        Create neural network model with BatchNormalization.

        Args:
            input_dim: Number of input features

        Returns:
            Sequential: Compiled Keras model
        """
        model = Sequential([
            # Primeira camada
            Dense(9, activation='relu', input_dim=input_dim),
            BatchNormalization(),
            #Dropout(0.1),

            # Segunda camada
            Dense(6, activation='relu'),
            BatchNormalization(),
            #Dropout(0.1),

            # Terceira camada
            Dense(2, activation='relu'),
            BatchNormalization(),

            # Camada de saída
            Dense(1)
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='mse',
            metrics=[
                'mae',
                RootMeanSquaredError(),
                R2Score()
            ]
        )

        return model

    def train_scenario(self, scenario: str):
        """
        Train model for specific scenario.

        Args:
            scenario: The scenario to train ('L1', 'L2', or 'L1L2')
        """
        try:
            logging.info(f"\nTraining scenario: {scenario}")

            # Prepare data
            #X_train, X_test, y_train, y_test = self.prepare_data(scenario)

            # --- Início da Modificação ---

            # 1. Preparar dados (como no código original)
            X_train_full, X_test, y_train_full, y_test = self.prepare_data(scenario)

            # 2. Dividir o conjunto de treino para criar um conjunto de validação explícito
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full, y_train_full, test_size=0.2, random_state=42
            )

            # 3. Salvar todas as matrizes
            logging.info("Saving training, validation, and test matrices...")
            
            # Matrizes de entrada (features)
            np.save(self.matrices_dir / f"{scenario}_X_train.npy", X_train)
            np.save(self.matrices_dir / f"{scenario}_X_val.npy", X_val)
            np.save(self.matrices_dir / f"{scenario}_X_test.npy", X_test)
            
            # Matrizes de saída (target)
            np.save(self.matrices_dir / f"{scenario}_y_train.npy", y_train)
            np.save(self.matrices_dir / f"{scenario}_y_val.npy", y_val)
            np.save(self.matrices_dir / f"{scenario}_y_test.npy", y_test)

            # --- SALVANDO EM .CSV (FORMATO ACESSÍVEL PARA RELATÓRIOS) ---
            logging.info("Saving matrices in CSV format...")
            np.savetxt(self.matrices_dir / f"{scenario}_X_train.csv", X_train, delimiter=",")         # <--- SALVAR EM CSV
            np.savetxt(self.matrices_dir / f"{scenario}_X_val.csv", X_val, delimiter=",")             # <--- SALVAR EM CSV
            np.savetxt(self.matrices_dir / f"{scenario}_X_test.csv", X_test, delimiter=",")           # <--- SALVAR EM CSV
            np.savetxt(self.matrices_dir / f"{scenario}_y_train.csv", y_train, delimiter=",")         # <--- SALVAR EM CSV
            np.savetxt(self.matrices_dir / f"{scenario}_y_val.csv", y_val, delimiter=",")             # <--- SALVAR EM CSV
            np.savetxt(self.matrices_dir / f"{scenario}_y_test.csv", y_test, delimiter=",")           # <--- SALVAR EM CSV
            
            
            logging.info(f"Matrices saved in: {self.matrices_dir}")

            # Create model
            model = self.create_model(X_train.shape[1])

            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                ModelCheckpoint(
                    filepath=str(self.models_dir / f"{self.scenarios[scenario]['model_name']}.keras"),
                    monitor='val_loss',
                    save_best_only=True
                )
            ]

            # Train model
            history = model.fit(
                X_train, y_train,
                epochs=100,
                batch_size=32,
                #validation_split=0.2,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )

            # Save training history
            history_df = pd.DataFrame(history.history)
            history_df.to_csv(
                self.logs_dir / f"{self.scenarios[scenario]['model_name']}_history.csv",
                index=False
            )

            # Save scalers
            np.save(
                str(self.models_dir / f"{self.scenarios[scenario]['model_name']}_x_scaler.npy"),
                self.x_scaler.get_params()
            )
            np.save(
                str(self.models_dir / f"{self.scenarios[scenario]['model_name']}_y_scaler.npy"),
                self.y_scaler.get_params()
            )

            # Save feature names
            pd.Series(self.feature_names).to_csv(
                self.models_dir / f"{self.scenarios[scenario]['model_name']}_features.csv",
                index=False
            )

            # Evaluate model
            scores = model.evaluate(X_test, y_test, verbose=0)
            metrics = dict(zip(model.metrics_names, scores))

            # Calculate MAPE with denormalized data
            y_test_real = self.y_scaler.inverse_transform(y_test)
            y_pred_real = self.y_scaler.inverse_transform(model.predict(X_test))
            mape = np.mean(np.abs((y_test_real - y_pred_real) / y_test_real)) * 100
            metrics['mape'] = float(mape)

            # Save evaluation metrics
            pd.DataFrame([metrics]).to_csv(
                self.logs_dir / f"{self.scenarios[scenario]['model_name']}_metrics.csv",
                index=False
            )

            logging.info("\nModel Evaluation:")
            for metric, value in metrics.items():
                logging.info(f"{metric}: {value:.4f}")

        except Exception as e:
            logging.error(f"Error training scenario {scenario}: {str(e)}")
            raise

def main():
    """Main function to execute the training pipeline."""
    try:
        # Get project root directory
        project_root = Path(__file__).parent.parent.parent

        # Create forecaster instance
        forecaster = LoadForecaster(project_root)

        # Train all scenarios
        for scenario in forecaster.scenarios.keys():
            forecaster.train_scenario(scenario)

        logging.info("Training completed successfully!")

    except Exception as e:
        logging.error(f"Error in training pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()