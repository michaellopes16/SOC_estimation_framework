#%pip install shap
import shap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class BatteryShapExplainer:
    def __init__(self, model, X_train, feature_names, n_background=50):
        self.model = model
        self.feature_names = feature_names
        self.is_3d = len(X_train.shape) == 3
        self.input_shape = X_train.shape[1:] # Guarda (timesteps, features) se for 3D
        
        # Seleção de background
        idx_bg = np.random.choice(X_train.shape[0], min(n_background, len(X_train)), replace=False)
        self.background = X_train[idx_bg]
        
        # Função de predição unificada com suporte a reconstrução de shape para KernelExplainer
        def p_fn(d):
            # O KernelExplainer muitas vezes achata os dados para (n_samples, total_features)
            # Se o modelo for BiLSTM (3D), precisamos reconstruir o shape (n, 150, n_vars)
            if self.is_3d and len(d.shape) == 2:
                d = d.reshape((d.shape[0],) + self.input_shape)
            
            if hasattr(self.model, 'predict'):
                try: 
                    preds = self.model.predict(d, verbose=0)
                except: 
                    preds = self.model.predict(d)
            else:
                preds = self.model(d).numpy()
            
            return preds.flatten()

        # Usamos KernelExplainer que é mais genérico para Ensembles e modelos mistos
        self.explainer = shap.KernelExplainer(p_fn, self.background.reshape(self.background.shape[0], -1) if self.is_3d else self.background)

    def explain_index(self, X_data, idx):
        # Para o KernelExplainer no explain_index, passamos o dado achatado
        sample = X_data[idx:idx+1]
        sample_flat = sample.reshape(1, -1) if self.is_3d else sample
        
        shap_values = self.explainer.shap_values(sample_flat)

        # Trata o retorno que pode ser lista (regressão/classificação)
        curr_shap = shap_values[0] if isinstance(shap_values, list) else shap_values

        plt.figure()
        if self.is_3d:
            # Reconstrói para (1, 150, n_vars) para fazer a média temporal correta
            reshaped_shap = curr_shap.reshape(self.input_shape)
            avg_shap = np.mean(reshaped_shap, axis=0).reshape(1, -1) # Média dos timesteps
            avg_sample = np.mean(sample[0], axis=0).reshape(1, -1)
            
            shap.summary_plot(avg_shap, avg_sample, feature_names=self.feature_names, plot_type="bar", show=False)
        else:
            feat_names = self.feature_names
            if len(feat_names) < sample.shape[1]:
                feat_names = [f"f_{i}" for i in range(sample.shape[1])]
            shap.summary_plot(curr_shap, sample, feature_names=feat_names, plot_type="bar", show=False)
        
        plt.title(f"SHAP - Índice {idx}")
        plt.show()

    def explain_heatmap(self, X_data, idx):
        """
        Visualiza a importância temporal de variáveis como Temperatura, Corrente e Carga.
        Adaptado para modelos KNN (2D) e BiLSTM (3D).
        """
        # 1. Preparação da Amostra
        sample = X_data[idx:idx+1]
        # O KernelExplainer espera a entrada no mesmo formato do background definido no __init__
        sample_for_shap = sample.reshape(1, -1) if self.is_3d else sample
        
        try:
            # Calcula valores SHAP
            shap_values = self.explainer.shap_values(sample_for_shap)
        except Exception as e:
            print(f"Erro ao calcular SHAP: {e}")
            return

        # 2. Normalização do formato SHAP
        curr_shap = shap_values[0] if isinstance(shap_values, list) else shap_values

        # Garante que curr_shap seja um array 1D com todas as features (achatado)
        if isinstance(curr_shap, np.ndarray):
            curr_shap = curr_shap.flatten()
        
        # 3. Lógica de Reshape (150 pontos por Variável: Temperatura, Corrente, Carga...)
        num_pontos = 150 
        total_features = len(curr_shap)
        
        if total_features % num_pontos == 0:
            num_vars = total_features // num_pontos
            # Para BiLSTM, o reshape precisa respeitar a ordem (Variáveis, TimeSteps)
            # Se os dados originais forem (150, n_vars), após o flatten e reshape:
            shap_matrix = curr_shap.reshape(num_pontos, num_vars).T
            
            if self.feature_names and len(self.feature_names) == num_vars:
                yticklabels = self.feature_names
            else:
                yticklabels = [f"Variável {i}" for i in range(num_vars)]
        else:
            shap_matrix = curr_shap.reshape(1, -1)
            num_vars = 1
            yticklabels = ["Geral"]

        # 4. Plotagem dos Gráficos de Área por Variável
        fig, axes = plt.subplots(num_vars, 1, figsize=(12, 3 * num_vars), sharex=True)
        
        if num_vars == 1:
            axes = [axes]

        for i in range(num_vars):
            ax = axes[i]
            data = shap_matrix[i]
            x = np.arange(len(data))
            
            # Preenchimento estético (Vermelho: aumenta valor previsto, Azul: diminui)
            ax.fill_between(x, 0, data, where=(data > 0), color='#ff0051', alpha=0.6)
            ax.fill_between(x, 0, data, where=(data < 0), color='#008bfb', alpha=0.6)
            
            ax.plot(x, data, color='black', linewidth=0.8, alpha=0.4)
            ax.axhline(0, color='black', lw=1.2)
            
            ax.set_ylabel("Impacto SHAP")
            ax.set_title(f"Contribuição Temporal: {yticklabels[i]} (Índice {idx})")
            ax.grid(True, linestyle='--', alpha=0.3)

        axes[-1].set_xlabel("Janela de Tempo (Passos)")
        plt.tight_layout()
        plt.show()