#%pip install lime
import lime
from lime import lime_tabular
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, HTML

class BatteryLimeExplainer:
    def __init__(self, model, X_train, feature_names, mode='regression'):
        self.model = model
        self.mode = mode
        self.is_3d = len(X_train.shape) == 3
        
        # AJUSTE DE NOMES DE FEATURES PARA MODELOS TRADICIONAIS (FLATTEN)
        # Se o dado é 2D mas a lista de nomes é curta, precisamos expandir os nomes
        n_cols_data = X_train.shape[-1]
        if not self.is_3d and len(feature_names) < n_cols_data:
            # Cria nomes genéricos para o dado achatado (ex: f0, f1...)
            # ou você pode expandir os nomes reais se preferir.
            self.feature_names = [f"feat_{i}" for i in range(n_cols_data)]
        else:
            self.feature_names = feature_names

        if self.is_3d:
            self.explainer = lime.lime_tabular.RecurrentTabularExplainer(
                X_train, feature_names=self.feature_names, mode=mode)
        else:
            self.explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train, feature_names=self.feature_names, mode=mode, discretize_continuous=True)

    def _predict_fn(self, data):
        # Garante que funcione com sklearn (KNN) e Keras
        if hasattr(self.model, 'predict'):
            preds = self.model.predict(data)
            return preds.flatten()
        return self.model(data).flatten()

    def _predict_fn(self, data):
        # Suporta Keras (com verbose=0) e Scikit-Learn (sem verbose)
        if hasattr(self.model, 'predict'):
            try:
                return self.model.predict(data, verbose=0).flatten()
            except:
                return self.model.predict(data).flatten()
        return self.model(data).numpy().flatten()

    def explain_index(self, X_data, idx, num_features=5):
        sample = X_data[idx]
        exp = self.explainer.explain_instance(sample, self._predict_fn, num_features=num_features)
        exp.as_pyplot_figure()
        plt.title(f"LIME - Índice {idx}")
        plt.show()