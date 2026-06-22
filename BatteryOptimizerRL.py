import numpy as np
import random
import itertools
import tensorflow as tf
from sklearn.metrics import mean_squared_error

class BatteryOptimizerRL:
    def __init__(self, model_builder_class, X_train, y_train, scaler_y, param_grid):
        self.ModelBuilder = model_builder_class
        self.scaler_y = scaler_y
        self.param_grid = param_grid
        
        # Dimensões para DL
        self.seq_length = X_train.shape[1] if len(X_train.shape) == 3 else None
        self.n_features = X_train.shape[2] if len(X_train.shape) == 3 else X_train.shape[1]
        
        # Divisão interna para HPO
        from sklearn.model_selection import train_test_split
        self.X_train_rl, self.X_val_rl, self.y_train_rl, self.y_val_rl = train_test_split(
            X_train, y_train, test_size=0.2, shuffle=False, random_state=42
        )
        self.y_val_rl_original = scaler_y.inverse_transform(self.y_val_rl.reshape(-1, 1))

        # Detectar o tipo de builder pelo nome da classe
        self.is_deep_builder = "BatteryModelBuilder" in str(model_builder_class)
        
        self.hyperparam_combinations = self._generate_combinations()
        self.n_actions = len(self.hyperparam_combinations)
        self.q_table = None
        self.best_params = None
        self.best_rmse = float('inf')

    def _generate_combinations(self):
        keys, values = zip(*self.param_grid.items())
        return [dict(zip(keys, v)) for v in itertools.product(*values)]

    def _prepare_data(self, X):
        """Ajusta o shape: DL precisa de 3D, Tradicional precisa de 2D."""
        if not self.is_deep_builder and len(X.shape) == 3:
            # Achata 3D (samples, seq, feat) para 2D (samples, seq*feat)
            return X.reshape(X.shape[0], -1)
        return X

    def _setup_environment(self):
        self.rmse_bins = np.linspace(0, 0.5, 6) 
        self.complexity_bins = np.linspace(0, 500000, 6) 
        self.n_states = (len(self.rmse_bins)-1) * (len(self.complexity_bins)-1)
        self.q_table = np.zeros((self.n_states, self.n_actions))

    def optimize(self, episodes=10):
        if self.q_table is None: self._setup_environment()
        
        epsilon = 1.0
        current_state_idx = 0 
        
        # Prepara os dados uma única vez fora do loop de episódios
        X_train_prep = self._prepare_data(self.X_train_rl)
        X_val_prep = self._prepare_data(self.X_val_rl)

        for ep in range(episodes):
            action_idx = random.randint(0, self.n_actions-1) if random.random() < epsilon else np.argmax(self.q_table[current_state_idx])
            params = self.hyperparam_combinations[action_idx]

            # --- LOGICA DE INSTANCIAÇÃO  ---
            if self.is_deep_builder:
                # DL precisa das dimensões
                builder = self.ModelBuilder(
                    seq_length=self.seq_length, 
                    n_features=self.n_features, 
                    **params
                )
            else:
                # Tradicional não aceita seq_length/n_features
                builder = self.ModelBuilder(**params)
            
            try:
                model = builder.build()
                
                if self.is_deep_builder:
                    model.fit(X_train_prep, self.y_train_rl, epochs=50, batch_size=64, verbose=0)
                else:
                    model.fit(X_train_prep, self.y_train_rl)

                y_pred_scaled = model.predict(X_val_prep, verbose=0) if self.is_deep_builder else model.predict(X_val_prep)
                y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
                
                rmse = np.sqrt(mean_squared_error(self.y_val_rl_original, y_pred))
                
                # Atualização do Estado e Q-Table
                next_state_idx = np.clip(np.digitize(rmse, self.rmse_bins) - 1, 0, self.n_states-1)
                reward = -rmse
                
                # Bellman
                self.q_table[current_state_idx, action_idx] += 0.1 * (reward + 0.9 * np.max(self.q_table[next_state_idx]) - self.q_table[current_state_idx, action_idx])
                
                if rmse < self.best_rmse:
                    self.best_rmse = rmse
                    self.best_params = params
                    print(f"Ep {ep+1}: Novo Recorde! RMSE: {rmse:.4f}")

                current_state_idx = next_state_idx
            except Exception as e:
                print(f"Erro ao testar combinação {params}: {e}")
            
            epsilon = max(0.01, epsilon * 0.9)
            
        return self.best_params