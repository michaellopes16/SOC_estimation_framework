import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from BatteryDataPreprocessor import BatteryDataPreprocessor
# from BatteryOptimizerRL import BatteryOptimizerRL
from BatteryModelBuilder import BatteryModelBuilder
from TraditionalModelBuilder import TraditionalModelBuilder
from BatteryTrainer import BatteryTrainer
from BatteryExplainer import BatteryExplainerGRADCAM
from BatteryLimeExplainer import BatteryLimeExplainer
from BatteryShapExplainer import BatteryShapExplainer
from BatteryOptimizerDQN import BatteryOptimizerDQN
# Assumindo as classes já importadas no seu ambiente:
# BatteryDataPreprocessor, BatteryModelBuilder, BatteryTrainer, BatteryOptimizerRL, BatteryExplainer, BatteryLimeExplainer

class BatteryProjectPipeline:
    def __init__(self, df_raw, df_train=None, df_test=None):
        """
        Pipeline Gerenciador do Projeto de Baterias (Deep Learning + Tradicionais).
        """
        self.df = df_raw
        self.df_train = df_train
        self.df_test = df_test
        print(f"instanciando o df: {self.df.size }")
        self.seq_length = 150
        # MUDANÇA AQUI: Trocamos 'Current' por 'Current_Mean'
        self.features = ['Voltage', 'Current_Mean', 'Temperature', 'P_avg', 'R_avg']
        self.target = 'SOC'
        
        # --- ATUALIZAÇÃO: Listas separadas para lógica de shape ---
        self.DL_MODELS = ['lstm', 'bilstm', 'cnn_lstm', 'gru', 'lstm_rnn']
        self.TRAD_MODELS = ['rf', 'xgb', 'dt', 'svr', 'knn']
        self.AVAILABLE_MODELS =  self.TRAD_MODELS +self.DL_MODELS
        
        self.preprocessor = None
        self.trainers = {} 
        self.leaderboard = None 
        self.best_model_name = None 
        
        self.X_train, self.X_test = None, None
        self.y_train, self.y_test = None, None
        
        self.best_params = {
            'filters': 64, 'kernel_size': 3,
            'lstm_units': 50, 'dense_units': 50,
            'learning_rate': 0.001
        }

    def run_step_1_preprocessing(self, df_is_joined=False):
        print("\n--- Passo 1: Pré-processamento e Engenharia de Features ---")
        
        # 1. Cria a feature suave
        self.df['Current_Mean'] = self.df['Current'].rolling(window=60, min_periods=1).mean()
        # self.features = ['Voltage', 'Current_Mean', 'Temperature']
        # Ou, se quiser ser ainda mais seguro, remova colunas inúteis antes:
        cols_to_check = self.features + [self.target]
        self.df = self.df.dropna(subset=cols_to_check)
        # print(self.df.head()) 
        self.preprocessor = BatteryDataPreprocessor(
            seq_length=self.seq_length,
            features=self.features,
            target=self.target
        )
        print("Pré-processamento iniciado...")
        print("São os dados de treino e teste separados? ", df_is_joined)
        if df_is_joined:
            self.X_train, self.X_test, self.y_train, self.y_test = self.preprocessor.process_separated(self.df_train, self.df_test)
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = self.preprocessor.process(self.df)

        print("Pré-processamento concluído. Printando X")
        print("Shape de X_train:", self.X_train.shape)
        print(self.X_train) 
        # --- AQUI ESTÁ A CORREÇÃO ---
        # Pegamos o número de features diretamente do X_train processado
        # Se X_train é (amostras, seq_length, n_features), pegamos o index 2
        self.n_features = self.X_train.shape[2] 
        print(f"Número de features detectado: {self.n_features}")

    def run_step_2_optimization(self, episodes=10, force_run=False):
        print("\n=== PASSO 2: Otimização de Hiperparâmetros (HPO) via RL ===")
        
        if not force_run:
            print("Modo Rápido: Usando parâmetros padrão.")
            return

        # 1. Definição dos Grids com base nas suas classes Build
        hpo_configs = [
                        {
                'name': 'Modelos Tradicionais',
                'builder': TraditionalModelBuilder,
                'grid': {
                    'model_type': ['rf', 'xgb', 'svr', 'knn'],
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'learning_rate': [0.01, 0.1],
                    'n_neighbors': [3, 5, 10],
                    'kernel': ['rbf']
                }
            },
            {
                'name': 'Modelos de Deep Learning',
                'builder': BatteryModelBuilder,
                'grid': {
                    'model_type': ['cnn_lstm', 'bilstm', 'cnn_gru', 'lstm_rnn','cnn_attention'],
                    'filters': [32, 128],
                    'kernel_size': [3, 5],
                    'lstm_units': [64, 128],
                    'dense_units': [32, 128],
                    'learning_rate': [0.001]
                }
            }
        ]

        self.optimized_results = {}

        for config in hpo_configs:
            print(f"\nIniciando DQN para: {config['name']}...")
            
            # optimizer = BatteryOptimizerRL(
            #     model_builder_class=config['builder'],
            #     X_train=self.X_train, 
            #     y_train=self.y_train,
            #     scaler_y=self.preprocessor.scaler_y,
            #     param_grid=config['grid']
            # )
            optimizer = BatteryOptimizerDQN(
                model_builder_class=config['builder'],
                X_train=self.X_train, 
                y_train=self.y_train,
                scaler_y=self.preprocessor.scaler_y,
                param_grid=config['grid']
            )
            
            # O RL vai testar diferentes model_types dentro de cada builder!
            best_found, best_rmse = optimizer.optimize(episodes=episodes)
            self.optimized_results[config['name']] = {'params': best_found, 'rmse': best_rmse}

        print("\n=== Resultados de HPO via RL ===")
        for name, res in self.optimized_results.items():
            print(f"{name}: Melhor RMSE = {res['rmse']:.4f} com parâmetros {res['params']}")
        
        print("\n=== Otimização Concluída ===")

    def run_step_3_training(self, models_to_run='all', epochs=50):
        """
        Treina modelos de Deep Learning e Tradicionais.
        """
        print("\n=== PASSO 3: Treinamento de Modelos ===")
        
        if models_to_run == 'all':
            target_models = self.AVAILABLE_MODELS
        else:
            target_models = [models_to_run] if isinstance(models_to_run, str) else models_to_run
            
        for model_name in target_models:
            print(f"\nTraining >>> {model_name.upper()} <<<")
            
            # --- LÓGICA DE SELEÇÃO DE BUILDER E SHAPE ---
            if model_name in self.TRAD_MODELS:
                # 1. Preparar dados 2D (Achatados)
                X_train_run = self.X_train.reshape(self.X_train.shape[0], -1)
                X_test_run = self.X_test.reshape(self.X_test.shape[0], -1)
                
                # 2. Builder Tradicional
                builder = TraditionalModelBuilder(model_type=model_name)
                model = builder.build()
                
                # 3. Trainer (Nota: SKLearn fit não usa epochs/verbose da mesma forma)
                # Adaptamos o trainer ou chamamos o fit diretamente se necessário
                model.fit(X_train_run, self.y_train)
                
                # Para manter compatibilidade com seu BatteryTrainer:
                trainer = BatteryTrainer(
                    model=model,
                    X_train=X_train_run, y_train=self.y_train,
                    X_test=X_test_run, y_test=self.y_test,
                    scaler_y=self.preprocessor.scaler_y
                )
                # trainer.training_time = 0 # Defina um timer aqui se desejar
                
            else:
                # Builder de Deep Learning (TensorFlow)
                X_train_run, X_test_run = self.X_train, self.X_test
                builder = BatteryModelBuilder(
                    seq_length=self.seq_length,
                    n_features=self.n_features,
                    model_type=model_name,
                    **self.best_params
                )
                model = builder.build()
                trainer = BatteryTrainer(
                    model=model,
                    X_train=X_train_run, y_train=self.y_train,
                    X_test=X_test_run, y_test=self.y_test,
                    scaler_y=self.preprocessor.scaler_y
                )
                trainer.train(epochs=epochs, verbose=0)

            self.trainers[model_name] = trainer
            print(f"{model_name.upper()} finalizado.")

    def run_step_4_evaluation(self):
        """
        Avalia todos os modelos treinados, gera o Leaderboard e SALVA em CSV.
        """
        print("\n=== PASSO 4: Avaliação Comparativa ===")
        
        if not self.trainers:
            print("Nenhum modelo treinado encontrado.")
            return

        results_list = []

        # Itera sobre todos os treinadores armazenados
        for name, trainer in self.trainers.items():
            print(f"Avaliando {name}...")
            
            # Roda a avaliação visual (gráficos)
            trainer.evaluate() 
            
            # Coleta métricas
            y_real = trainer.y_test_real
            y_pred = trainer.y_pred_test_real
            time_taken = trainer.training_time
            
            # Verificação de Sanidade (Sanity Check)
            print(f"   -> Check de Escala [{name}]: Real({y_real.min():.2f} a {y_real.max():.2f}) | Pred({y_pred.min():.2f} a {y_pred.max():.2f})")

            # Se o range real for > 10 e o predito for < 1, há um erro de desnormalização
            if (y_real.max() - y_real.min()) > 5 and (y_pred.max() - y_pred.min()) < 2:
                print(f"   ⚠️ ERRO CRÍTICO: {name} parece estar com y_pred ainda normalizado!")
                
            mse = mean_squared_error(y_real, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_real, y_pred)
            r2 = r2_score(y_real, y_pred)
            
            results_list.append({
                "Modelo": name.upper(),
                "RMSE": rmse,
                "MAE": mae,
                "R2 Score": r2,
                "MSE": mse,
                "Tempo Treino (s)": round(time_taken, 2)
            })
            trainer.plot_zoom(1000)
        # Cria DataFrame e ordena pelo menor erro (RMSE)
        self.leaderboard = pd.DataFrame(results_list).sort_values(by="RMSE", ascending=True)
        
        
        # --- SALVAR O CSV ---
        nome_arquivo = "resultado_comparativo_modelos.csv"
        self.leaderboard.to_csv(nome_arquivo, index=False)
        print(f"Tabela consolidada salva em '{nome_arquivo}'")
        
        # Define o melhor
        self.best_model_name = self.leaderboard.iloc[0]["Modelo"].lower()
        
        print("\n--- LEADERBOARD DE DESEMPENHO ---")
        print(self.leaderboard.to_string(index=False))
        print(f"\nMelhor modelo identificado: {self.best_model_name.upper()}")

    def run_step_5_explainability(self, model_name=None, num_samples=1, methods=['lime', 'grad_cam','shap']):
        print("\n" + "="*60)
        print("=== PASSO 5: EXPLICABILIDADE (XAI) MULTI-MODELO ===")
        print("="*60)
        
        # Itera sobre todos os modelos treinados
        for model_name, trainer in self.trainers.items():
            print(f"\n>>> Analisando: {model_name.upper()}")
            
            # Identifica o tipo de dado necessário
            is_traditional = model_name in self.TRAD_MODELS
            
            if is_traditional:
                # Dado 2D para: RF, XGB, KNN, etc.
                X_train_xai = self.preprocessor.flatten_for_traditional(self.X_train)
                X_test_xai = self.preprocessor.flatten_for_traditional(self.X_test)
            else:
                # Dado 3D para: LSTM, CNN, GRU, etc.
                X_train_xai = self.X_train
                X_test_xai = self.X_test

            # Escolhe índices aleatórios do teste para esta rodada
            indices = np.random.randint(0, len(X_test_xai), num_samples)

            # --- A. GRAD-CAM (Apenas para modelos com Conv1D) ---
            if 'grad_cam' in methods:
                if is_traditional:
                    print(f"   [Grad-CAM] Ignorado: {model_name} não é um modelo Convolucional.")
                else:
                    try:
                        explainer_gc = BatteryExplainerGRADCAM(
                            model=trainer.model, 
                            scaler_y=self.preprocessor.scaler_y, 
                            feature_names=self.features
                        )
                        for idx in indices:
                            # Nota: Grad-CAM precisa do y_test original para mostrar SOC real vs predito
                            explainer_gc.explain_index(self.X_test, self.y_test, idx)
                    except Exception as e:
                        print(f"   [Grad-CAM] Não suportado para {model_name}: {e}")

            # --- B. LIME (Agnóstico) ---
            if 'lime' in methods:
                try:
                    explainer_lime = BatteryLimeExplainer(
                        model=trainer.model,
                        X_train=X_train_xai,
                        feature_names=self.features
                    )
                    for idx in indices:
                        explainer_lime.explain_index(X_test_xai, idx)
                except Exception as e:
                    print(f"   [LIME] Erro no modelo {model_name}: {e}")

            # --- C. SHAP (Agnóstico) ---
            if 'shap' in methods:
                try:
                    explainer_shap = BatteryShapExplainer(
                        model=trainer.model,
                        X_train=X_train_xai,
                        feature_names=self.features,
                        n_background=20 # Valor seguro para não estourar memória
                    )
                    for idx in indices:
                        explainer_shap.explain_heatmap(X_test_xai, idx)
                except Exception as e:
                    print(f"   [SHAP] Erro no modelo {model_name}: {e}")

        print("\n--- Pipeline de Explicabilidade Finalizado ---")

    def run_full_pipeline(self, models='all', optimize=False, epochs=50, df_is_joined=False):
        """Executa todo o fluxo de uma vez."""
        self.run_step_1_preprocessing(df_is_joined=df_is_joined)
        self.run_step_2_optimization(force_run=optimize)
        self.run_step_3_training(models_to_run=models, epochs=epochs)
        self.run_step_4_evaluation()
        self.run_step_5_explainability() # Explica o melhor modelo automaticamente