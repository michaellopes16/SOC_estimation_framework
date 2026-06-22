from pyexpat import features

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class BatteryDataPreprocessor:
    def __init__(self, seq_length=50, test_size=0.2, features=None, target='SOC'):
        """
        Inicializa o pré-processador.
        
        :param seq_length: Tamanho da janela de sequência (timesteps anteriores).
        :param test_size: Porcentagem de dados para teste.
        :param features: Lista de colunas a serem usadas como input (X).
        :param target: Coluna alvo a ser prevista (y).
        """
        self.seq_length = seq_length
        self.test_size = test_size
        self.features = features if features else ['Voltage', 'Current_Mean', 'Temperature', 'P_avg', 'R_avg']        
        self.target = target
        
        # Inicializa os scalers
        self.scaler_X = MinMaxScaler(feature_range=(0, 1))
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))
    # Dentro de BatteryDataPreprocessor.py

    def add_engineered_features(self, df):
        """
        Implementa o aumento de dados: V, I, T + P_avg e R_avg.
        """
        df = df.copy()
        window = 50 

        # --- ADICIONE ESTE BLOCO PARA DEFINIR A VARIÁVEL ---
        if 'Current' in df.columns:
            col_corrente = 'Current'
        elif 'Current_Mean' in df.columns:
            col_corrente = 'Current_Mean'
        else:
            # Caso não encontre nenhum dos nomes conhecidos
            raise KeyError(f"Coluna de corrente não encontrada. Colunas: {list(df.columns)}")
        # --------------------------------------------------

        # 1. Suavização das bases (V e I)
        v_avg = df['Voltage'].rolling(window=window, min_periods=1).mean()
        # Agora 'col_corrente' existe e pode ser usada abaixo:
        i_avg = df[col_corrente].rolling(window=window, min_periods=1).mean()

        # 2. Cálculo da Potência Média (P_avg = V_avg * I_avg)
        df['P_avg'] = v_avg * i_avg

        # 3. Cálculo da Resistência Média (R_avg = ΔV_avg / ΔI_avg)
        delta_v = v_avg.diff().fillna(0)
        delta_i = i_avg.diff().replace(0, 1e-6) 
        df['R_avg'] = (delta_v / delta_i).abs()

        # Limpeza final
        df['R_avg'] = df['R_avg'].replace([np.inf, -np.inf], 0).fillna(0)
        
        # Opcional: Para manter compatibilidade com o Pipeline que pede 'Current_Mean'
        if col_corrente == 'Current':
            df = df.rename(columns={'Current': 'Current_Mean'})

        return df

    def _create_sequences(self, X, y):
        """Método auxiliar interno para criar sequências de janelas deslizantes."""
        X_seq, y_seq = [], []
        for i in range(len(X) - self.seq_length):
            X_seq.append(X[i:(i + self.seq_length)])
            y_seq.append(y[i + self.seq_length]) # Prever o SOC no próximo timestep
        return np.array(X_seq), np.array(y_seq)
    
    def process_separated(self, df_train, df_test):
        """
        Versão robusta para bases separadas (ex: Treino em FUDS, Teste em US06).
        Garante que o scaler NUNCA veja os dados de teste durante o ajuste (fit).
        """
        print("Iniciando pré-processamento de bases separadas...")
        
        if df_train is None or df_test is None:
            raise ValueError("Erro: Um ou ambos os DataFrames (treino/teste) estão vazios.")
        
        # 1. Feature Engineering em ambos (P_avg, R_avg, etc.)
        # Usamos .copy() para evitar o SettingWithCopyWarning do Pandas
        df_tr = self.add_engineered_features(df_train).copy()
        df_ts = self.add_engineered_features(df_test).copy()

        print("Feature Engineering concluída. Verificando os dados...")
        print(self.features)
        # print(df_tr.head())  # Debug: Verificar as primeiras linhas do treino
        # 2. Escalonamento Rigoroso (Fit no TREINO, Transform em AMBOS)
        # Isso garante que a escala de 0 a 1 do modelo seja baseada apenas no que ele "viu" no treino.
        df_tr[self.features] = self.scaler_X.fit_transform(df_tr[self.features])
        df_tr[self.target] = self.scaler_y.fit_transform(df_tr[[self.target]])
        
        # O teste apenas sofre a transformação (sem fit) para evitar Data Leakage
        df_ts[self.features] = self.scaler_X.transform(df_ts[self.features])
        df_ts[self.target] = self.scaler_y.transform(df_ts[[self.target]])
        # 3. Função auxiliar para criação de sequências respeitando os ciclos
        def build_cycle_sequences(df_in):
            all_X_seq, all_y_seq = [], []
            
            # Agrupa por Cycle_Number para que a janela temporal não pule de um ciclo para outro
            for _, group in df_in.groupby('Cycle_Number'):
                if len(group) > self.seq_length:
                    x, y = self._create_sequences(
                        group[self.features].values, 
                        group[self.target].values
                    )
                    all_X_seq.append(x)
                    all_y_seq.append(y)
            
            if not all_X_seq:
                return np.array([]), np.array([])
                
            return np.concatenate(all_X_seq), np.concatenate(all_y_seq)

        print("Gerando sequências para o conjunto de Treino...")
        # print(df_tr.head())  # Debug: Verificar as primeiras linhas do treino após escalonamento
        X_train, y_train = build_cycle_sequences(df_tr)
        
        print("Gerando sequências para o conjunto de Teste...")
        X_test, y_test = build_cycle_sequences(df_ts)

        print(f"Concluído! Shape Treino: {X_train.shape} | Shape Teste: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def process(self, df):
        """
        Executa o pipeline completo: Engenharia -> Split -> Scaling -> Sequencing.
        Esta versão corrige o Data Leakage e o Bias de escala.
        """
        print("Iniciando pré-processamento com bases juntas...")
        
        if df is None or df.empty:
            raise ValueError("Erro: O DataFrame fornecido está vazio.")

        # 1. Feature Engineering (Adiciona P_avg, R_avg, etc. antes do split)
        # Importante: usamos o DF completo aqui pois cálculos de rolling 
        # precisam de continuidade temporal.
        df_processed = self.add_engineered_features(df)

        # 2. Divisão Treino/Teste ANTES do Escalonamento
        # shuffle=False é obrigatório para séries temporais.
        train_df, test_df = train_test_split(
            df_processed, 
            test_size=self.test_size, 
            shuffle=False 
        )

        # 3. Escalonamento Rigoroso (Fit apenas no treino)
        # Evita que o modelo conheça os limites (min/max) do conjunto de teste.
        train_df = train_df.copy()
        test_df = test_df.copy()

        train_df[self.features] = self.scaler_X.fit_transform(train_df[self.features])
        train_df[self.target] = self.scaler_y.fit_transform(train_df[[self.target]])
        
        test_df[self.features] = self.scaler_X.transform(test_df[self.features])
        test_df[self.target] = self.scaler_y.transform(test_df[[self.target]])

        # 4. Criação de Sequências por Ciclo
        # Criamos uma função auxiliar interna para evitar repetição de código
        def build_cycle_sequences(df_split):
            all_X_seq = []
            all_y_seq = []
            
            # Agrupa por ciclo para evitar que o final de um ciclo 
            # se misture com o início de outro no janelamento
            grouped = df_split.groupby('Cycle_Number')
            
            for _, df_cycle in grouped:
                X_values = df_cycle[self.features].values
                y_values = df_cycle[self.target].values
                
                if len(X_values) > self.seq_length:
                    X_s, y_s = self._create_sequences(X_values, y_values)
                    all_X_seq.append(X_s)
                    all_y_seq.append(y_s)
            
            if not all_X_seq:
                return np.array([]), np.array([])
                
            return np.concatenate(all_X_seq, axis=0), np.concatenate(all_y_seq, axis=0)

        print("Gerando sequências de treino...")
        X_train, y_train = build_cycle_sequences(train_df)
        
        print("Gerando sequências de teste...")
        X_test, y_test = build_cycle_sequences(test_df)

        print(f"Processamento concluído. Treino: {X_train.shape} | Teste: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    # def process_separated(self, df_train, df_test):
    #     """
    #     NOVO MÉTODO: Recebe bases já separadas. 
    #     Útil para: FUDS+DST (Treino) e US06 (Teste).
    #     """
    #     # 1. Feature Engineering em ambos
    #     df_tr = self.add_engineered_features(df_train)
    #     df_ts = self.add_engineered_features(df_test)

    #     # 2. Escalonamento (Fit apenas no treino para evitar leakage)
    #     df_tr[self.features] = self.scaler_X.fit_transform(df_tr[self.features])
    #     df_tr[self.target] = self.scaler_y.fit_transform(df_tr[[self.target]])
        
    #     # Apenas transform no teste
    #     df_ts[self.features] = self.scaler_X.transform(df_ts[self.features])
    #     df_ts[self.target] = self.scaler_y.transform(df_ts[[self.target]])

    #     # 3. Criação de sequências respeitando os ciclos de cada base
    #     def build_arrays(df_in):
    #         all_X, all_y = [], []
    #         for _, group in df_in.groupby('Cycle_Number'):
    #             if len(group) > self.seq_length:
    #                 x, y = self._create_sequences(group[self.features].values, group[self.target].values)
    #                 all_X.append(x); all_y.append(y)
    #         return np.concatenate(all_X), np.concatenate(all_y)

    #     X_train, y_train = build_arrays(df_tr)
    #     X_test, y_test = build_arrays(df_ts)

    #     return X_train, X_test, y_train, y_test
    
    def inverse_transform_target(self, y_pred):
        """
        Converte as previsões (0 a 1) de volta para o valor real (ex: SOC %).
        Útil para plotar os gráficos finais.
        """
        return self.scaler_y.inverse_transform(y_pred)
    
    def flatten_for_traditional(self, X):
        """
        Converte (amostras, seq_length, features) para (amostras, seq_length * features).
        Necessário para RF, XGBoost e SVR.
        """
        n_samples = X.shape[0]
        return X.reshape(n_samples, -1)