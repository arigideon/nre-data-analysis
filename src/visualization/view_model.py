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
            'RES': { # Novo cenário residencial
                'name': 'Residential Only',
                'model_name': 'model_residential',
                'target_col': 'Demanda_Residencial',
                'demand_cols': ['Demanda_Residencial'],
                'ev_plugged_col': None  # Não tem VEs
            }, 
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
        if scenario == 'RES':
            return base_features + ['Demanda_Residencial']
        
        ev_col = self.scenarios[scenario]['ev_plugged_col']
        if isinstance(ev_col, list):
            return base_features + ev_col + ['Demanda_Combinada']
        elif ev_col is None:
            return base_features + ['Demanda_Combinada']
        else:
            return base_features + [ev_col, 'Demanda_Combinada']

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
        plt.plot(history['loss'], label='Treino')
        plt.plot(history['val_loss'], label='Validação')
        plt.title(f'Perda do Modelo - {self.scenarios[scenario]["name"]}')
        plt.xlabel('Época')
        plt.ylabel('Perda')
        plt.legend()

        # MAE
        plt.subplot(2, 2, 2)
        plt.plot(history['mae'], label='Treino')
        plt.plot(history['val_mae'], label='Validação')
        plt.title('Erro Absoluto Médio')
        plt.xlabel('Época')
        plt.ylabel('MAE')
        plt.legend()

        # RMSE
        plt.subplot(2, 2, 3)
        plt.plot(history['root_mean_squared_error'], label='Treino')
        plt.plot(history['val_root_mean_squared_error'], label='Validação')
        plt.title('Raiz do Erro Quadrático Médio')
        plt.xlabel('Época')
        plt.ylabel('RMSE')
        plt.legend()

        # R²
        plt.subplot(2, 2, 4)
        plt.plot(history['r2_score'], label='Treino')
        plt.plot(history['val_r2_score'], label='Validação')
        plt.title('Coeficiente de Determinação (R²)')
        plt.xlabel('Época')
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

            df['Demanda_Combinada'] = df[self.scenarios[scenario]['demand_cols']].sum(axis=1)
            features = self.get_features_for_scenario(scenario)
            X = df[features].values
            y = df[self.scenarios[scenario]['target_col']].values

            x_scaler = MinMaxScaler()
            y_scaler = MinMaxScaler()
            X_scaled = x_scaler.fit_transform(X)
            y_scaled = y_scaler.fit_transform(y.reshape(-1, 1))

            y_pred_scaled = model.predict(X_scaled)
            y_pred = y_scaler.inverse_transform(y_pred_scaled)

            plt.figure(figsize=(10, 8))
            plt.scatter(y, y_pred, alpha=0.5)
            plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
            plt.xlabel('Demanda Real (W)')
            plt.ylabel('Demanda Prevista (W)')
            plt.title(f'Demanda Real vs Prevista - {self.scenarios[scenario]["name"]}')
            plt.tight_layout()
            plt.savefig(self.figures_dir / f'scatter_{scenario}.png')
            plt.close()

        except Exception as e:
            logging.error(f"Error plotting scatter: {str(e)}")
            raise

    def plot_demand_comparison(self, scenario: str):
        
        """Plot comparison between residential, combined and total demand, and model prediction."""
        try:
            import numpy as np
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

            df = pd.read_csv(self.data_dir / 'Total-Demand-Features.csv')

            # Adicionar previsão do modelo
            model, _ = self.load_model_and_history(scenario)

            # Calcular demanda combinada
            df['Demanda_Combinada'] = df[self.scenarios[scenario]['demand_cols']].sum(axis=1)

            # Preparar features para o modelo
            features = self.get_features_for_scenario(scenario)
            X = df[features].values
            y = df[self.scenarios[scenario]['target_col']].values

            # Escalar dados
            x_scaler = MinMaxScaler()
            y_scaler = MinMaxScaler()
            X_scaled = x_scaler.fit_transform(X)
            y_scaled = y_scaler.fit_transform(y.reshape(-1, 1))

            # Fazer previsão
            y_pred_scaled = model.predict(X_scaled)
            y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()

            # Adicionar previsão ao dataframe
            df['Demanda_Prevista'] = y_pred

            # Selecionar dados de uma semana para visualização
            week_data = df.iloc[:168]
            x = range(len(week_data))

            # Cores para cada curva
            colors = {
                'Residencial': 'royalblue',
                'Combinada': 'forestgreen',
                'Total': 'firebrick',
                'Prevista': 'darkorange'
            }

            # PLOT 1: Comparação de demandas (original)
            plt.figure(figsize=(15, 8))
            if scenario == 'RES':
                y1 = week_data['Demanda_Residencial']
                plt.fill_between(x, y1, color=colors['Residencial'], alpha=0.4)
                plt.plot(x, y1, label='Demanda Residencial', color=colors['Residencial'], linewidth=2)
            else:
                y1 = week_data['Demanda_Residencial']
                plt.fill_between(x, y1, color=colors['Residencial'], alpha=0.4)
                plt.plot(x, y1, label='Apenas Residencial', color=colors['Residencial'], linewidth=2)
                y3 = week_data[self.scenarios[scenario]['target_col']]
                plt.fill_between(x, y3, color=colors['Total'], alpha=0.4)
                plt.plot(x, y3, label=f'Demanda Total ({scenario})', color=colors['Total'], linewidth=2)
            plt.title(f'Comparação de Demandas - {self.scenarios[scenario]["name"]}', fontsize=14, fontweight='bold')
            plt.xlabel('Horas', fontsize=12)
            plt.ylabel('Demanda (W)', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.figures_dir / f'demand_comparison_{scenario}.png', dpi=300)
            plt.close()

            # PLOT 2: Real vs Previsto (apenas linhas)
            plt.figure(figsize=(15, 8))
            y_real = week_data[self.scenarios[scenario]['target_col']]
            y_pred_week = week_data['Demanda_Prevista']
            plt.plot(x, y_real, label='Demanda Real', color=colors['Total'], linewidth=2)
            plt.plot(x, y_pred_week, label='Demanda Prevista', color=colors['Prevista'], linewidth=2, linestyle='--')

            # Métricas para a semana plotada
            mae_week = mean_absolute_error(y_real, y_pred_week)
            rmse_week = np.sqrt(mean_squared_error(y_real, y_pred_week))
            r2_week = r2_score(y_real, y_pred_week)
            mape_week = np.mean(np.abs((y_real - y_pred_week) / y_real)) * 100

            metrics_text = f'MAE: {mae_week:.2f} W, RMSE: {rmse_week:.2f} W, R²: {r2_week:.4f}, MAPE: {mape_week:.2f}%'
            plt.title(f'Demanda Real vs Prevista - {self.scenarios[scenario]["name"]}\n{metrics_text}',
                    fontsize=14, fontweight='bold')
            plt.xlabel('Horas', fontsize=12)
            plt.ylabel('Demanda (W)', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.figures_dir / f'real_vs_predicted_{scenario}.png', dpi=300)
            plt.close()

            # PLOT 3: Erro de previsão
            plt.figure(figsize=(15, 6))
            error = y_pred_week - y_real
            plt.bar(x, error, color='darkred', alpha=0.7)
            plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
            plt.title(f'Erro de Previsão (Previsto - Real) - {self.scenarios[scenario]["name"]}',
                    fontsize=14, fontweight='bold')
            plt.xlabel('Horas', fontsize=12)
            plt.ylabel('Erro (W)', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.figures_dir / f'prediction_error_{scenario}.png', dpi=300)
            plt.close()

            # Métricas totais do modelo (toda a base)
            mae_total = mean_absolute_error(y, y_pred)
            rmse_total = np.sqrt(mean_squared_error(y, y_pred))
            r2_total = r2_score(y, y_pred)
            mape_total = np.mean(np.abs((y - y_pred) / y)) * 100

            # Salvar métricas em um arquivo txt
            metrics_path = self.figures_dir / f'weekly_metrics_{scenario}.txt'
            with open(metrics_path, 'w', encoding='utf-8') as f:
                f.write(f'Métricas de desempenho para a semana plotada - {self.scenarios[scenario]["name"]}\n')
                f.write(f'MAE: {mae_week:.2f} W\n')
                f.write(f'RMSE: {rmse_week:.2f} W\n')
                f.write(f'R²: {r2_week:.4f}\n')
                f.write(f'MAPE: {mape_week:.2f}%\n')
                f.write('\n')
                f.write(f'Métricas de desempenho para todo o conjunto de dados - {self.scenarios[scenario]["name"]}\n')
                f.write(f'MAE: {mae_total:.2f} W\n')
                f.write(f'RMSE: {rmse_total:.2f} W\n')
                f.write(f'R²: {r2_total:.4f}\n')
                f.write(f'MAPE: {mape_total:.2f}%\n')

        except Exception as e:
            logging.error(f"Error plotting demand comparison: {str(e)}")
            raise

    def plot_sensitivity_analysis(self, scenario: str):
        """Plot sensitivity analysis for model features, with and without demand feature(s)."""
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

            # Plot 1: Todas as features
            plt.figure(figsize=(12, 6))
            plt.bar(features, sensitivities)
            plt.title(f'Análise de Sensibilidade das Features - {self.scenarios[scenario]["name"]}')
            plt.xticks(rotation=45)
            plt.ylabel('Sensibilidade')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(self.figures_dir / f'sensitivity_{scenario}.png')
            plt.close()

            # Plot 2: Sem features de demanda
            # Identifique features de demanda (pode ser mais de uma)
            demand_feats = [f for f in features if 'demanda' in f.lower()]
            # Se você quiser incluir 'Demanda_Combinada', adicione:
            if 'Demanda_Combinada' in features:
                demand_feats.append('Demanda_Combinada')

            features_no_demand = [f for f in features if f not in demand_feats]
            sensitivities_no_demand = [s for f, s in zip(features, sensitivities) if f not in demand_feats]

            plt.figure(figsize=(12, 6))
            plt.bar(features_no_demand, sensitivities_no_demand, color='tab:orange')
            plt.title(f'Análise de Sensibilidade (sem demanda) - {self.scenarios[scenario]["name"]}')
            plt.xticks(rotation=45)
            plt.ylabel('Sensibilidade')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(self.figures_dir / f'sensitivity_{scenario}_sem_demanda.png')
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
        """
        Gera todos os gráficos e salva um arquivo com as métricas finais e arquitetura do modelo.
        """
        try:
            logging.info(f"Generating all plots for scenario {scenario}")
            model, history = self.load_model_and_history(scenario)

            # 1. Plots principais
            self.plot_learning_curves(history, scenario)
            self.plot_scatter(scenario)
            self.plot_demand_comparison(scenario)
            self.plot_sensitivity_analysis(scenario)

            # 2. Sensitivity Analysis (one feature) para todas as features
            features_path = self.models_dir / f"{self.scenarios[scenario]['model_name']}_features.csv"
            features = pd.read_csv(features_path, header=None)[0].tolist()
            if features[0] == '0':
                features = features[1:]

            df = pd.read_csv(self.data_dir / 'Total-Demand-Features.csv')
            df['Demanda_Combinada'] = df[self.scenarios[scenario]['demand_cols']].sum(axis=1)
            X = df[features].values
            x_scaler = MinMaxScaler()
            X_scaled = x_scaler.fit_transform(X)
            base_input = np.mean(X, axis=0)

            y_scaler = MinMaxScaler()
            y = df[self.scenarios[scenario]['target_col']].values
            y_scaler.fit(y.reshape(-1, 1))

            for feature_idx, selected_feature in enumerate(features):
                # Gera valores para a feature selecionada
                if selected_feature in ['Hora', 'Dia_Semana', 'Mes', 'Estacao', 'Periodo_Dia', 'Fim_Semana']:
                    if selected_feature == 'Hora':
                        values = np.arange(0, 24)
                    elif selected_feature == 'Dia_Semana':
                        values = np.arange(1, 8)
                    elif selected_feature == 'Mes':
                        values = np.arange(1, 13)
                    elif selected_feature == 'Estacao':
                        values = np.arange(1, 5)
                    elif selected_feature == 'Periodo_Dia':
                        values = np.arange(0, 4)
                    elif selected_feature == 'Fim_Semana':
                        values = np.array([0, 1])
                else:
                    min_val = df[selected_feature].min()
                    max_val = df[selected_feature].max()
                    values = np.linspace(min_val, max_val, 50)

                predictions = []
                for value in values:
                    input_vec = base_input.copy()
                    input_vec[feature_idx] = value
                    input_scaled = x_scaler.transform([input_vec])
                    pred = model.predict(input_scaled)
                    predictions.append(pred[0, 0])
                predictions = y_scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

                # Plot individual de sensibilidade
                plt.figure(figsize=(12, 6))
                plt.plot(values, predictions, 'b-', linewidth=2)
                plt.scatter(values, predictions, color='blue', alpha=0.5)
                plt.title(f'Análise de Sensibilidade - {selected_feature}\n{self.scenarios[scenario]["name"]}')
                plt.xlabel(selected_feature)
                plt.ylabel('Demanda Prevista (W)')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(self.figures_dir / f'sensitivity_{scenario}_{selected_feature}.png')
                plt.close()

            # 3. Salva arquivo com métricas finais e arquitetura do modelo
            output_txt = self.figures_dir / f"summary_{scenario}.txt"
            with open(output_txt, "w", encoding="utf-8") as f:
                # Métricas finais
                metrics_path = self.logs_dir / f"{self.scenarios[scenario]['model_name']}_metrics.csv"
                if metrics_path.exists():
                    metrics = pd.read_csv(metrics_path)
                    f.write("Final Model Metrics:\n")
                    f.write(metrics.T.to_string())
                    if 'mape' in metrics.columns:
                        f.write(f"\n\nFinal MAPE (calculated with denormalized data): {metrics['mape'].values[0]:.2f}%\n")
                else:
                    f.write("Metrics file not found.\n")

                # Arquitetura do modelo
                f.write("\n\n=== Model Architecture ===\n")
                f.write(f"Scenario: {self.scenarios[scenario]['name']}\n")
                f.write("\nInput Features:\n")
                for i, feature in enumerate(features, 1):
                    f.write(f"{i}. {feature}\n")

                # Camadas do modelo
                f.write("\nLayer Structure:\n")
                # Redireciona o summary para string
                from io import StringIO
                import sys
                old_stdout = sys.stdout
                sys.stdout = mystdout = StringIO()
                model.summary()
                sys.stdout = old_stdout
                f.write(mystdout.getvalue())

                # Parâmetros
                trainable_params = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
                non_trainable_params = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
                f.write("\nParameter Summary:\n")
                f.write(f"Total trainable parameters: {trainable_params:,}\n")
                f.write(f"Total non-trainable parameters: {non_trainable_params:,}\n")
                f.write(f"Total parameters: {trainable_params + non_trainable_params:,}\n")

            logging.info(f"All plots and summary file generated successfully for scenario {scenario}")

        except Exception as e:
            logging.error(f"Error generating all plots: {str(e)}")
            raise
    
    def individual_prediction(self, scenario: str):
        """
        Realiza previsão individual com base nos inputs do usuário.
        Usa exatamente as mesmas features usadas no treino.
        """
        try:
            # Carrega modelo e features do treino
            model, _ = self.load_model_and_history(scenario)
            features_path = self.models_dir / f"{self.scenarios[scenario]['model_name']}_features.csv"
            features = pd.read_csv(features_path).squeeze().tolist()

            # Carrega dados para o scaler
            df = pd.read_csv(self.data_dir / 'Total-Demand-Features.csv')

            # Calcula Demanda_Combinada
            df['Demanda_Combinada'] = df[self.scenarios[scenario]['demand_cols']].sum(axis=1)

            # Prepara dados para os scalers
            X = df[features].values
            y = df[self.scenarios[scenario]['target_col']].values

            # Inicializa e treina os scalers
            x_scaler = MinMaxScaler()
            y_scaler = MinMaxScaler()
            X_scaled = x_scaler.fit_transform(X)
            y_scaled = y_scaler.fit_transform(y.reshape(-1, 1))

            # Coleta inputs do usuário
            print(f"\n=== Previsão Individual - {self.scenarios[scenario]['name']} ===")
            print("Entre com os valores para cada feature:")
            input_data = []

            for feature in features:
                valid_input = False
                while not valid_input:
                    try:
                        if feature == 'Hora':
                            value = float(input(f"{feature} (0-23): "))
                            if 0 <= value <= 23:
                                valid_input = True
                            else:
                                print("Hora deve estar entre 0 e 23")

                        elif feature == 'Dia':
                            value = float(input(f"{feature} (1-31): "))
                            if 1 <= value <= 31:
                                valid_input = True
                            else:
                                print("Dia deve estar entre 1 e 31")

                        elif feature == 'Mes':
                            value = float(input(f"{feature} (1-12): "))
                            if 1 <= value <= 12:
                                valid_input = True
                            else:
                                print("Mês deve estar entre 1 e 12")

                        elif feature == 'Dia_Semana':
                            value = float(input(f"{feature} (1-7, onde 1=Segunda): "))
                            if 1 <= value <= 7:
                                valid_input = True
                            else:
                                print("Dia da semana deve estar entre 1 e 7")

                        elif feature == 'Estacao':
                            print("\nEstações:")
                            print("1 = Primavera")
                            print("2 = Verão")
                            print("3 = Outono")
                            print("4 = Inverno")
                            value = float(input(f"{feature} (1-4): "))
                            if 1 <= value <= 4:
                                valid_input = True
                            else:
                                print("Estação deve estar entre 1 e 4")

                        elif feature == 'Periodo_Dia':
                            print("\nPeríodos do dia:")
                            print("0 = Madrugada (00:00-05:59)")
                            print("1 = Manhã (06:00-11:59)")
                            print("2 = Tarde (12:00-17:59)")
                            print("3 = Noite (18:00-23:59)")
                            value = float(input(f"{feature} (1-4): "))
                            if 0 <= value <= 3:
                                valid_input = True
                            else:
                                print("Período deve estar entre 1 e 4")

                        elif feature == 'Fim_Semana':
                            value = float(input(f"{feature} (0=Dia útil, 1=Fim de semana): "))
                            if value in [0, 1]:
                                valid_input = True
                            else:
                                print("Fim de semana deve ser 0 ou 1")

                        elif feature == 'Demanda_Combinada':
                            value = float(input(f"{feature} (Demanda atual em W): "))
                            if value >= 0:
                                valid_input = True
                            else:
                                print("Demanda deve ser maior ou igual a zero")

                        elif 'VEs_Plugados' in feature:
                            value = float(input(f"{feature} (Número de VEs): "))
                            if value >= 0:
                                valid_input = True
                            else:
                                print("Número de VEs deve ser maior ou igual a zero")

                        else:
                            value = float(input(f"{feature}: "))
                            valid_input = True

                        input_data.append(value)

                    except ValueError:
                        print("Por favor, insira um número válido")

            # Realiza a previsão
            input_scaled = x_scaler.transform([input_data])
            pred_scaled = model.predict(input_scaled)
            prediction = y_scaler.inverse_transform(pred_scaled)

            # Mostra resultados
            print("\n=== Resultado da Previsão ===")
            print(f"Cenário: {self.scenarios[scenario]['name']}")
            print(f"Demanda prevista para próxima hora: {prediction[0][0]:,.2f} W")
            print("===============================")

            # Debug info (opcional - pode remover se quiser)
            logging.info(f"Features utilizadas: {features}")
            logging.info(f"Valores de entrada: {input_data}")
            logging.info(f"Previsão realizada: {prediction[0][0]:.2f}")

        except Exception as e:
            logging.error(f"Erro na previsão individual: {str(e)}")
            raise

    def show_model_architecture(self, scenario: str):
        """Display the model architecture."""
        try:
            model, _ = self.load_model_and_history(scenario)
            features = self.get_features_for_scenario(scenario)

            print("\n=== Model Architecture ===")
            print(f"Scenario: {self.scenarios[scenario]['name']}")
            print("\nInput Features:")
            for i, feature in enumerate(features, 1):
                print(f"{i}. {feature}")

            print("\nLayer Structure:")
            model.summary()

            # Calculate total parameters
            trainable_params = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
            non_trainable_params = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])

            print("\nParameter Summary:")
            print(f"Total trainable parameters: {trainable_params:,}")
            print(f"Total non-trainable parameters: {non_trainable_params:,}")
            print(f"Total parameters: {trainable_params + non_trainable_params:,}")

        except Exception as e:
            logging.error(f"Error showing model architecture: {str(e)}")
            raise

    def sensitivity_analysis_one_feature(self, scenario: str):
        """Perform sensitivity analysis for a single selected feature."""
        try:
            # Carregue as features EXATAMENTE como no treino
            features_path = self.models_dir / f"{self.scenarios[scenario]['model_name']}_features.csv"
            features = pd.read_csv(features_path, header=None)[0].tolist()
            if features[0] == '0': 
                features = features[1:]

            model, _ = self.load_model_and_history(scenario)
            df = pd.read_csv(self.data_dir / 'Total-Demand-Features.csv')
            df['Demanda_Combinada'] = df[self.scenarios[scenario]['demand_cols']].sum(axis=1)

            X = df[features].values
            x_scaler = MinMaxScaler()
            X_scaled = x_scaler.fit_transform(X)

            # Valores médios para cada feature
            base_input = np.mean(X, axis=0)

            # Seleção da feature
            print("\nAvailable features:")
            for i, feature in enumerate(features, 1):
                print(f"{i}. {feature}")

            while True:
                try:
                    feature_idx = int(input("\nSelect feature number: ")) - 1
                    if 0 <= feature_idx < len(features):
                        break
                    print("Invalid selection. Please try again.")
                except ValueError:
                    print("Please enter a valid number")

            selected_feature = features[feature_idx]

            # Gera valores para a feature selecionada
            if selected_feature in ['Hora', 'Dia_Semana', 'Mes', 'Estacao', 'Periodo_Dia', 'Fim_Semana']:
                if selected_feature == 'Hora':
                    values = np.arange(0, 24)
                elif selected_feature == 'Dia_Semana':
                    values = np.arange(1, 8)
                elif selected_feature == 'Mes':
                    values = np.arange(1, 13)
                elif selected_feature == 'Estacao':
                    values = np.arange(1, 5)
                elif selected_feature == 'Periodo_Dia':
                    values = np.arange(0, 4)
                elif selected_feature == 'Fim_Semana':
                    values = np.array([0, 1])
            else:
                min_val = df[selected_feature].min()
                max_val = df[selected_feature].max()
                values = np.linspace(min_val, max_val, 50)

            predictions = []
            for value in values:
                input_vec = base_input.copy()
                input_vec[feature_idx] = value
                input_scaled = x_scaler.transform([input_vec])
                pred = model.predict(input_scaled)
                predictions.append(pred[0, 0])
            
            y_scaler = MinMaxScaler()
            y = df[self.scenarios[scenario]['target_col']].values
            y_scaler.fit(y.reshape(-1, 1))
            predictions = y_scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

            # Plot
            plt.figure(figsize=(12, 6))
            plt.plot(values, predictions, 'b-', linewidth=2)
            plt.scatter(values, predictions, color='blue', alpha=0.5)
            plt.title(f'Análise de Sensibilidade - {selected_feature}\n{self.scenarios[scenario]["name"]}')
            plt.xlabel(selected_feature)
            plt.ylabel('Demanda Prevista (W)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.figures_dir / f'sensitivity_{scenario}_{selected_feature}.png')
            plt.close()

            print(f"\nSensitivity analysis plot saved for {selected_feature}")

        except Exception as e:
            logging.error(f"Error in sensitivity analysis: {str(e)}")
            raise

    def save_sensitivity_table(self, scenario: str):
        """Save a CSV table with feature sensitivities (normalized and in watts)."""
        model, _ = self.load_model_and_history(scenario)
        df = pd.read_csv(self.data_dir / 'Total-Demand-Features.csv')
        df['Demanda_Combinada'] = df[self.scenarios[scenario]['demand_cols']].sum(axis=1)
        features = self.get_features_for_scenario(scenario)
        X = df[features].values

        # Scalers
        x_scaler = MinMaxScaler()
        X_scaled = x_scaler.fit_transform(X)
        y = df[self.scenarios[scenario]['target_col']].values
        y_scaler = MinMaxScaler()
        y_scaler.fit(y.reshape(-1, 1))

        base_prediction = model.predict(X_scaled)
        sensitivities_norm = []
        sensitivities_watt = []

        for i, feature in enumerate(features):
            X_perturbed = X_scaled.copy()
            X_perturbed[:, i] *= 1.1  # 10% increase
            new_prediction = model.predict(X_perturbed)
            # Sensibilidade normalizada
            sens_norm = np.mean(np.abs(new_prediction - base_prediction))
            sensitivities_norm.append(sens_norm)
            # Sensibilidade desnormalizada (em watts)
            base_pred_watt = y_scaler.inverse_transform(base_prediction)
            new_pred_watt = y_scaler.inverse_transform(new_prediction)
            sens_watt = np.mean(np.abs(new_pred_watt - base_pred_watt))
            sensitivities_watt.append(sens_watt)

        # Salva CSV
        sens_df = pd.DataFrame({
            'Feature': features,
            'Sensitivity_Normalized': sensitivities_norm,
            'Sensitivity_Watt': sensitivities_watt
        })
        sens_df = sens_df.sort_values('Sensitivity_Watt', ascending=False)
        sens_df.to_csv(self.figures_dir / f'sensitivity_table_{scenario}.csv', index=False)
        print(f"Sensitivity table saved to {self.figures_dir / f'sensitivity_table_{scenario}.csv'}")

def main():
    """Main function to execute the visualization pipeline."""
    try:
        project_root = Path(__file__).parent.parent.parent
        visualizer = ModelVisualizer(project_root)

        while True:
            print("\nModel Visualization Menu")
            print("1. View Residential Model")  # Nova opção
            print("2. View L1 Model")
            print("3. View L2 Model")
            print("4. View L1L2 Model")
            print("5. Exit")

            choice = input("\nEnter your choice (1-5): ")

            if choice == '5':
                break

            scenarios = {'1': 'RES', '2': 'L1', '3': 'L2', '4': 'L1L2'}
            #scenarios = {'1': 'L1', '2': 'L2', '3': 'L1L2'}
            if choice in scenarios:
                scenario = scenarios[choice]
                print(f"\nVisualization Options for {visualizer.scenarios[scenario]['name']}")
                print("1. Learning Curves")
                print("2. Scatter Plot")
                print("3. Demand Comparison")
                print("4. Sensitivity Analysis (all features)")
                print("5. Generate All Plots")
                print("6. View Final Metrics")
                print("7. Individual Prediction")  # NOVA OPÇÃO
                print("8. View Model Architecture")  # NOVA OPÇÃO
                print("9. Sensitivity Analysis (one feature)")  # NOVA OPÇÃO
                print("10. Sensitivity Table (Summary)")
                print("11. Back to Main Menu")

                viz_choice = input("\nEnter your choice (1-11): ")

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
                elif viz_choice == '7':
                    visualizer.individual_prediction(scenario)  # Função a ser implementada
                elif viz_choice == '8':
                    visualizer.show_model_architecture(scenario)  # Função a ser implementada
                elif viz_choice == '9':
                    visualizer.sensitivity_analysis_one_feature(scenario)  # Função a ser implementada
                elif viz_choice == '10':
                    visualizer.save_sensitivity_table(scenario)

                if viz_choice not in ['11']:
                    logging.info(f"Plot(s) saved in {visualizer.figures_dir}")

    except Exception as e:
        logging.error(f"Error in visualization pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()