import numpy as np
import random
import itertools

from collections import deque

import tensorflow as tf
from tensorflow.keras import models, layers, optimizers

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


class BatteryOptimizerDQN:

    def __init__(
        self,
        model_builder_class,
        X_train,
        y_train,
        scaler_y,
        param_grid
    ):

        self.ModelBuilder = model_builder_class
        self.scaler_y = scaler_y
        self.param_grid = param_grid

        # --------------------------------------------------
        # Dados
        # --------------------------------------------------

        self.seq_length = (
            X_train.shape[1]
            if len(X_train.shape) == 3
            else None
        )

        self.n_features = (
            X_train.shape[2]
            if len(X_train.shape) == 3
            else X_train.shape[1]
        )

        (
            self.X_train_rl,
            self.X_val_rl,
            self.y_train_rl,
            self.y_val_rl
        ) = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            shuffle=False,
            random_state=42
        )

        self.y_val_rl_original = scaler_y.inverse_transform(
            self.y_val_rl.reshape(-1, 1)
        )

        # --------------------------------------------------
        # Tipo do modelo
        # --------------------------------------------------

        self.is_deep_builder = (
            "BatteryModelBuilder"
            in str(model_builder_class)
        )

        self.hyperparam_combinations = (
            self._generate_combinations()
        )

        self.n_actions = len(
            self.hyperparam_combinations
        )

        # --------------------------------------------------
        # DQN
        # --------------------------------------------------

        self.state_dim = 5

        self.memory = deque(maxlen=5000)

        self.gamma = 0.95

        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99

        self.learning_rate = 0.001

        self.batch_size = 32

        self.model = self._build_compile_model()
        self.target_model = self._build_compile_model()

        self.update_target_model()

        self.best_rmse = np.inf
        self.best_params = None

    # ======================================================
    # Hiperparâmetros
    # ======================================================

    def _generate_combinations(self):

        keys, values = zip(*self.param_grid.items())

        return [
            dict(zip(keys, v))
            for v in itertools.product(*values)
        ]

    # ======================================================
    # Rede DQN
    # ======================================================

    def _build_compile_model(self):

        model = models.Sequential([
            layers.Input(shape=(self.state_dim,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(
                self.n_actions,
                activation='linear'
            )
        ])

        model.compile(
            optimizer=optimizers.Adam(
                learning_rate=self.learning_rate
            ),
            loss='mse'
        )

        return model

    # ======================================================
    # Target Network
    # ======================================================

    def update_target_model(self):

        self.target_model.set_weights(
            self.model.get_weights()
        )

    # ======================================================
    # Replay Memory
    # ======================================================

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        self.memory.append(
            (
                state,
                action,
                reward,
                next_state,
                done
            )
        )

    # ======================================================
    # Escolha de ação
    # ======================================================

    def act(self, state):

        if np.random.rand() <= self.epsilon:
            return random.randrange(
                self.n_actions
            )

        q_values = self.model.predict(
            state,
            verbose=0
        )

        return np.argmax(q_values[0])

    # ======================================================
    # Replay
    # ======================================================

    def replay(self):

        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(
            self.memory,
            self.batch_size
        )

        for (
            state,
            action,
            reward,
            next_state,
            done
        ) in minibatch:

            target = reward

            if not done:

                next_q = np.max(
                    self.target_model.predict(
                        next_state,
                        verbose=0
                    )[0]
                )

                target = (
                    reward +
                    self.gamma * next_q
                )

            target_f = self.model.predict(
                state,
                verbose=0
            )

            target_f[0][action] = target

            self.model.fit(
                state,
                target_f,
                epochs=1,
                verbose=0
            )

        if self.epsilon > self.epsilon_min:

            self.epsilon *= self.epsilon_decay

            self.epsilon = max(
                self.epsilon,
                self.epsilon_min
            )

    # ======================================================
    # Estado
    # ======================================================

    def build_state(
        self,
        rmse,
        reward,
        best_rmse,
        epsilon,
        progress
    ):

        rmse_norm = rmse / 100.0
        best_rmse_norm = best_rmse / 100.0

        return np.array([[
            rmse_norm,
            reward,
            best_rmse_norm,
            epsilon,
            progress
        ]])

    # ======================================================
    # Otimização
    # ======================================================

    def optimize(self, episodes=100):

        if not self.is_deep_builder:

            X_train_p = self.X_train_rl.reshape(
                self.X_train_rl.shape[0],
                -1
            )

            X_val_p = self.X_val_rl.reshape(
                self.X_val_rl.shape[0],
                -1
            )

        else:

            X_train_p = self.X_train_rl
            X_val_p = self.X_val_rl

        current_state = self.build_state(
            rmse=1.0,
            reward=0.0,
            best_rmse=1.0,
            epsilon=self.epsilon,
            progress=0.0
        )

        for ep in range(episodes):

            action_idx = self.act(
                current_state
            )

            params = (
                self.hyperparam_combinations[
                    action_idx
                ]
            )

            try:

                # ------------------------------
                # Criação do modelo
                # ------------------------------

                if self.is_deep_builder:

                    builder = self.ModelBuilder(
                        seq_length=self.seq_length,
                        n_features=self.n_features,
                        **params
                    )

                else:

                    builder = self.ModelBuilder(
                        **params
                    )

                model = builder.build()

                # ------------------------------
                # Treinamento
                # ------------------------------

                if self.is_deep_builder:

                    model.fit(
                        X_train_p,
                        self.y_train_rl,
                        epochs=10,
                        batch_size=64,
                        verbose=0
                    )

                    y_pred_scaled = model.predict(
                        X_val_p,
                        verbose=0
                    )

                else:

                    model.fit(
                        X_train_p,
                        self.y_train_rl
                    )

                    y_pred_scaled = model.predict(
                        X_val_p
                    )

                # ------------------------------
                # Avaliação
                # ------------------------------

                y_pred = (
                    self.scaler_y.inverse_transform(
                        y_pred_scaled.reshape(-1, 1)
                    )
                )

                rmse = np.sqrt(
                    mean_squared_error(
                        self.y_val_rl_original,
                        y_pred
                    )
                )

                reward = 1.0 / (
                    1.0 + rmse
                )

                if rmse < self.best_rmse:

                    self.best_rmse = rmse
                    self.best_params = params

                    print(
                        f"[EP {ep+1}] "
                        f"Novo melhor RMSE: "
                        f"{rmse:.4f}"
                    )

                next_state = self.build_state(
                    rmse=rmse,
                    reward=reward,
                    best_rmse=self.best_rmse,
                    epsilon=self.epsilon,
                    progress=(ep + 1) / episodes
                )

                self.remember(
                    current_state,
                    action_idx,
                    reward,
                    next_state,
                    False
                )

                current_state = next_state

            except Exception as e:

                print(
                    f"Erro episódio "
                    f"{ep+1}: {e}"
                )

                self.remember(
                    current_state,
                    action_idx,
                    -1.0,
                    current_state,
                    True
                )

            self.replay()

            if ep % 10 == 0:
                self.update_target_model()

        return (
            self.best_params,
            self.best_rmse
        )