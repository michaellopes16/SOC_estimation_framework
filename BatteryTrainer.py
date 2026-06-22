#pip install matplotlib
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class BatteryTrainer:
    def __init__(self, model, X_train, y_train, X_test, y_test, scaler_y):
        """
        Classe responsável por treinar, avaliar e visualizar os resultados do modelo.
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.scaler_y = scaler_y
        self.history = None
        self.training_time = 0
        
        # Atributos para armazenar resultados finais desnormalizados
        self.y_train_real = None
        self.y_pred_train_real = None
        self.y_test_real = None
        self.y_pred_test_real = None

    def train(self, epochs=100, batch_size=32, verbose=1):
        """Executa o loop de treinamento e plota a curva de Loss."""
        print(f"\nIniciando treinamento ({epochs} épocas)...")
        
        start_time = time.time()
        
        self.history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1, # Usa 10% do treino para validação interna
            shuffle=False,      
            verbose=verbose
        )
        
        self.training_time = time.time() - start_time
        print(f"\nTempo total de treinamento: {self.training_time:.2f} segundos")
        
        self._plot_loss()

    def _plot_loss(self):
        """Plota o histórico de perda (Método interno)."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.history.history['loss'], label='Perda (Treinamento)')
        plt.plot(self.history.history['val_loss'], label='Perda (Validação)')
        plt.title('Histórico de Perda do Modelo')
        plt.xlabel('Época')
        plt.ylabel('MSE (Loss)')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def plot_zoom(self, zoom_range=400):
            """
            Plota um recorte detalhado (Zoom) para visualizar ciclos individuais de descarga.
            
            :param zoom_range: Número de amostras iniciais para visualizar (ex: 400 cobre ~1-2 ciclos).
            """
            # offset = self.y_test_real[0] - self.y_pred_test_real[0]
            # self.y_pred_test_real = self.y_pred_test_real + offset
            # Verificação de segurança
            if self.y_test_real is None or self.y_pred_test_real is None:
                print("Erro: É necessário rodar o método .evaluate() primeiro para gerar as previsões.")
                return

            # Garante que não vamos tentar plotar mais dados do que existem
            limit = min(zoom_range, len(self.y_test_real))

            print(f"\nGerando gráfico de Zoom (Primeiras {limit} amostras)...")
            plt.figure(figsize=(15, 7))

            # Plotar dados Reais (usando as variáveis da classe)
            plt.plot(self.y_test_real[:limit], 
                    label='SOC Real (Teste)', color='blue', linewidth=2)

            # Plotar dados Preditos
            plt.plot(self.y_pred_test_real[:limit], 
                    label='SOC Previsto (Modelo)', 
                    color='red', linestyle='--', linewidth=2)

            plt.title(f'Zoom em Ciclos de Descarga: Real vs Predito (Primeiras {limit} amostras)')
            plt.xlabel('Tempo (Amostras)')
            plt.ylabel('SOC (Carga)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
            
    def _calculate_metrics(self, y_true, y_pred, dataset_name, duration=0):
        """Calcula todas as métricas estatísticas solicitadas."""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        # MAPE com epsilon para evitar divisão por zero
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        r2 = r2_score(y_true, y_pred)
        residuals = y_true - y_pred
        std_dev_error = np.std(residuals)
        
        print(f"\n--- Métricas para {dataset_name} ---")
        print(f"MSE:  {mse:.6f} | RMSE: {rmse:.6f} | MAE:  {mae:.6f}")
        print(f"MAPE: {mape:.4f}% | R²:   {r2:.4f}")
        print(f"Tempo: {duration:.4f} s")
        
        return {
            "Dataset": dataset_name, "MSE": mse, "RMSE": rmse, "MAE": mae,
            "MAPE (%)": mape, "R2": r2, "Std_Dev_Error": std_dev_error, "Time (s)": duration
        }

    def evaluate(self):
        """
        Realiza previsões, desnormaliza dados e gera relatório.
        Adaptado para aceitar modelos de Deep Learning (3D/2D output) 
        e Tradicionais (1D output).
        """
        print(f"\n--- Iniciando Avaliação do Modelo ---")
        
        # 1. Previsões no conjunto de Treino
        y_pred_train_scaled = self.model.predict(self.X_train)
        # PLOT DA ESCALA NORMALIZADA (O que você pediu especificamente)
        self.plot_prediction_comparison(
            self.y_train.reshape(-1, 1), 
            y_pred_train_scaled, 
            title="Treino: Escala Normalizada (Scaled)"
        )
        # AJUSTE GENÉRICO: Se a saída for 1D (Tradicionais), transforma em 2D para o scaler
        if len(y_pred_train_scaled.shape) == 1:
            y_pred_train_scaled = y_pred_train_scaled.reshape(-1, 1)
        
        # Desnormalização
        # self.y_pred_test_real = self.scaler_y.inverse_transform(y_pred_test_scaled)
        # self.y_test_real = self.scaler_y.inverse_transform(self.y_test.reshape(-1, 1))
        
        self.y_pred_train_real = self.scaler_y.inverse_transform(y_pred_train_scaled)
        self.y_train_real = self.scaler_y.inverse_transform(self.y_train.reshape(-1, 1))
        
        # Cálculo de métricas de treino
        self._calculate_metrics(self.y_train_real, self.y_pred_train_real, "Treinamento", self.training_time)

        # 2. Previsões no conjunto de Teste
        start_test = time.time()
        y_pred_test_scaled = self.model.predict(self.X_test)
        
              # PLOT DA ESCALA REAL (Para comparação de escala desnormalizada)
        self.plot_prediction_comparison(
            self.y_test.reshape(-1, 1), 
            y_pred_test_scaled, 
            title="Teste: Escala Real (Inverse Transformed)"
        )
        # AJUSTE GENÉRICO: Se a saída for 1D (Tradicionais), transforma em 2D para o scaler
        if len(y_pred_test_scaled.shape) == 1:
            y_pred_test_scaled = y_pred_test_scaled.reshape(-1, 1)
            
        duration_test = time.time() - start_test
        
        # Desnormalização
        self.y_pred_test_real = self.scaler_y.inverse_transform(y_pred_test_scaled)
        self.y_test_real = self.scaler_y.inverse_transform(self.y_test.reshape(-1, 1))
        
        # Cálculo de métricas de teste
        metrics_test = self._calculate_metrics(self.y_test_real, self.y_pred_test_real, "Teste", duration_test)
        
        return metrics_test


    def plot_prediction_comparison(self, y_true, y_pred, title="Comparação: Real vs Predito"):
        """
        Gera um gráfico de linha para comparar os valores reais e as previsões.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label='Real', color='blue', alpha=0.7, linewidth=1.5)
        plt.plot(y_pred, label='Predito', color='red', linestyle='--', alpha=0.8, linewidth=1.5)
        
        plt.title(title, fontsize=14)
        plt.xlabel('Amostras', fontsize=12)
        plt.ylabel('Valor', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.show()
        
    def _plot_results(self):
        """Gera os gráficos de Série Temporal, Zoom e Dispersão."""
        # Gráfico 1: Série Temporal Completa
        plt.figure(figsize=(15, 6))
        plt.plot(self.y_test_real, label='SOC Real', color='blue', linewidth=1.5)
        plt.plot(self.y_pred_test_real, label='SOC Previsto', color='red', linestyle='--', alpha=0.8, linewidth=1.5)
        plt.title('Validação Temporal: SOC Real vs SOC Previsto')
        plt.xlabel('Amostras (Tempo)')
        plt.ylabel('SOC Real')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        # Gráfico 2: Zoom (primeiras 200 amostras)
        plt.figure(figsize=(15, 6))
        limit = min(200, len(self.y_test_real))
        plt.plot(self.y_test_real[:limit], label='SOC Real', color='blue')
        plt.plot(self.y_pred_test_real[:limit], label='SOC Previsto', color='red', linestyle='--')
        plt.title(f'Zoom: Primeiras {limit} amostras (Teste)')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Gráfico 3: Scatter Plot
        plt.figure(figsize=(10, 8))
        plt.scatter(self.y_test_real, self.y_pred_test_real, alpha=0.1)
        plt.plot([0, 1], [0, 1], 'r--', label='Previsão Perfeita (y=x)') # Assumindo SOC 0-1 (ou 0-100%)
        plt.title('Dispersão: SOC Real vs SOC Previsto')
        plt.xlabel('SOC Real')
        plt.ylabel('SOC Previsto')
        plt.legend()
        plt.grid(True)
        plt.show()