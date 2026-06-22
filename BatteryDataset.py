import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
import os
from scipy.integrate import cumulative_trapezoid

class BatteryDataset:
    def __init__(self, file_path):
        self.file_path = file_path
        
    def add_physics_features(self, df, window_size=50):
        """
        Calcula P_avg e R_avg conforme a técnica do artigo[cite: 11, 96].
        """
        df = df.copy()
        
        # 1. Cálculo da Potência Instantânea e Média (P = V * I) [cite: 221, 243]
        df['Power'] = df['Voltage'] * df['Current']
        df['P_avg'] = df['Power'].rolling(window=window_size, min_periods=1).mean()
        
        # 2. Cálculo da Resistência Média (R = ΔV_avg / ΔI_avg) [cite: 179, 241]
        # Usamos a variação da média móvel para evitar divisão por zero e ruído
        v_avg = df['Voltage'].rolling(window=window_size, min_periods=1).mean()
        i_avg = df['Current'].rolling(window=window_size, min_periods=1).mean()
        
        delta_v = v_avg.diff().fillna(0)
        delta_i = i_avg.diff().replace(0, 1e-6) # Evita divisão por zero
        
        df['R_avg'] = (delta_v / delta_i).abs()
        df['R_avg'] = df['R_avg'].clip(lower=0, upper=2.0)
        # Limpeza de possíveis infinitos gerados pela divisão
        df['R_avg'] = df['R_avg'].replace([np.inf, -np.inf], 0).fillna(0)
        
        return df
        
    def clean_battery_data(self, df, is_dynamic=False, is_test=False):
        # threshold_start = 0.01 
        # active_indices = df.index[df['Current'].abs() > threshold_start].tolist()

        # if not active_indices:
        #     return None
        
        # 1. IDENTIFICAR O INÍCIO
        if is_dynamic:
            # Busca oscilações (Original para Pulse)
            diff_current = df['Current'].diff().abs().fillna(0)
            threshold = 0.05 
            dinamico = df.index[diff_current > threshold].tolist()
            start_idx = dinamico[0] if dinamico else 0
        else:
            # Para NASA/CALCE: Busca o momento que a corrente sai de ~0
            # (Início da descarga constante)
            # threshold_start = 0.01 
            # active_indices = df.index[df['Current'].abs() > threshold_start].tolist()
            # start_idx = active_indices[0] if active_indices else 0
            if is_test:
                start_idx = 2330
            else:
                start_idx = 3000
            #if pd.isna(start_idx): start_idx = 2330
            
        # start_idx = active_indices[0]
        
        


        # Ajuste dinâmico do limite de voltagem
        # Se a voltagem média for alta (>10), assume bateria 12V, senão assume Li-ion
        v_mean = df['Voltage'].mean()
        v_limit = 10.5 if v_mean > 8 else 3.0 
        
        under_v = df.index[df['Voltage'] <= v_limit].tolist()
        
        # Filtra apenas índices que ocorrem após o início da corrente
        end_candidates = [i for i in under_v if i > start_idx]
        
        if end_candidates:
            end_idx = end_candidates[0]
        else:
            # Se nunca atingir o limite de tensão, pega até o último ponto de corrente ativa
            threshold_start = 0.01 
            active_indices = df.index[df['Current'].abs() > threshold_start].tolist()
            end_idx = active_indices[-1]

        df_useful = df.loc[start_idx : end_idx].copy()
        
        # Validação mínima: Se o corte resultar em menos de 10 pontos, algo está errado
        if len(df_useful) < 10:
            print(f"Aviso: Ciclo muito curto detectado ({len(df_useful)} pontos). Verifique v_limit.")
            # Opcional: retornar o df original se o corte for muito agressivo
            return df.loc[start_idx:].copy() 

        df_useful['Time'] = df_useful['Time'] - df_useful['Time'].iloc[0]
        return df_useful
    
    def apply_smoothing(self, df, window_size=10):
        """
        Aplica média móvel na Corrente e Voltagem para reduzir ruído.
        """
        df_smooth = df.copy()
        # Aplicamos a média móvel. O parâmetro 'window' define a suavização.
        # center=True ajuda a não deslocar o sinal no tempo.
        df_smooth['Current'] = df['Current'].rolling(window=window_size, center=True).mean()
        df_smooth['Voltage'] = df['Voltage'].rolling(window=window_size, center=True).mean()
        
        # O rolling deixa NaNs nas extremidades, precisamos removê-los
        return df_smooth.dropna()

    def load_and_process_nasa_data(self, is_test=False):
        """
        Carrega dados da NASA (.mat) e aplica a limpeza e padronização.
        """
        print(f"Carregando NASA: {self.file_path}...")
        try:
            mat = scipy.io.loadmat(self.file_path)
        except Exception as e:
            print(f"Erro: {e}")
            return None, None

        battery_key = os.path.splitext(os.path.basename(self.file_path))[0]
        all_cycles = mat[battery_key][0, 0]['cycle'][0]

        all_discharge_dfs = []
        capacity_data = []
        cycle_counter = 0

        for cycle in all_cycles:
            if cycle['type'][0] == 'discharge':
                cycle_counter += 1
                data = cycle['data'][0, 0]
                
                # Criar DF temporário para usar o clean_battery_data
                temp_df = pd.DataFrame({
                    'Time': data['Time'].flatten(),
                    'Voltage': data['Voltage_measured'].flatten(),
                    'Current': data['Current_measured'].flatten(),
                    'Temperature': data['Temperature_measured'].flatten()
                })
                
                # APLICA LIMPEZA (Remove Início/Fim constantes)
                cleaned_df = self.clean_battery_data(temp_df)
                
                if cleaned_df is not None and len(cleaned_df) > 10:
                    # Cálculo do SOC no dado limpo
                    capacity_Ah = data['Capacity'][0, 0]
                    capacity_As = capacity_Ah * 3600.0
                    
                    # Corrente na NASA é negativa na descarga
                    discharged_As = cumulative_trapezoid(-cleaned_df['Current'], cleaned_df['Time'], initial=0)
                    cleaned_df['SOC'] = np.clip(1.0 - (discharged_As / capacity_As), 0.0, 1.0)
                    cleaned_df['Cycle_Number'] = cycle_counter
                    
                    cleaned_df = self.add_physics_features(cleaned_df)
                    
                    all_discharge_dfs.append(cleaned_df)
                    capacity_data.append({'Cycle': cycle_counter, 'Capacity': capacity_Ah})

        df_timeseries = pd.concat(all_discharge_dfs, ignore_index=True)
        df_capacity = pd.DataFrame(capacity_data)
        return df_timeseries, df_capacity

    def load_and_process_calce_xlsx(self, is_test=False):
        """
        Lê ficheiro Excel (.xlsx) da CALCE, aplica limpeza e padroniza para o modelo de IA.
        """
        print(f"Carregando CALCE (.xlsx): {self.file_path}...")
        try:
            # Tenta ler a aba de dados (geralmente a segunda)
            df_raw = pd.read_excel(self.file_path, sheet_name=1)
        except Exception:
            df_raw = pd.read_excel(self.file_path)

        # Mapeamento de Colunas
        column_map = {
            'Test_Time(s)': 'Time',
            'Voltage(V)': 'Voltage',
            'Current(A)': 'Current',
            'Cycle_Index': 'Cycle_Number',
            'Temperature(C)': 'Temperature'
        }
        df = df_raw.rename(columns=column_map)

        all_cycles = []
        capacity_trend = []

        # Processamento por Ciclo
        for cycle_id, group in df.groupby('Cycle_Number'):
            group = group.sort_values('Time').reset_index(drop=True)
            
            # APLICA LIMPEZA (Remove Início/Fim constantes)
            group_cleaned = self.clean_battery_data(group, is_test=is_test)
            
            if group_cleaned is not None and len(group_cleaned) > 20:
                group_cleaned = self.apply_smoothing(group_cleaned, window_size=15)
                t = group_cleaned['Time'].values
                current = group_cleaned['Current'].values
                
                # Cálculo da Capacidade e SOC (Coulomb Counting)
                dt = np.diff(t, prepend=0) / 3600.0
                # No CALCE descarga é negativa, usamos abs para integrar carga gasta
                discharged_ah = np.cumsum(np.abs(current) * dt)
                total_cap_ah = discharged_ah[-1]
                
                if total_cap_ah < 0.1: continue

                group_cleaned['SOC'] = np.clip(1.0 - (discharged_ah / total_cap_ah), 0.0, 1.0)
                group_cleaned['Cycle_Number'] = cycle_id
                
                group_cleaned = self.add_physics_features(group_cleaned)
                # Garante que Temperatura existe
                if 'Temperature' not in group_cleaned.columns:
                    group_cleaned['Temperature'] = 25.0

                all_cycles.append(group_cleaned)
                capacity_trend.append({'Cycle': cycle_id, 'Capacity': total_cap_ah})

        df_timeseries = pd.concat(all_cycles, ignore_index=True)
        df_capacity = pd.DataFrame(capacity_trend)
        
        print("Processamento CALCE concluído.")
        return df_timeseries, df_capacity

    def load_and_process_pulse_data(self, profile_name='FTP75', is_test=False):
        """
        Carrega dados do Fiat Pulse a partir de arquivos Excel/CSV, 
        permitindo especificar o perfil de carga (FTP75, UDDS, NYDC).
        """
        print(f"Carregando dados Pulse - Perfil: {profile_name}...")
        
        try:
            # Se self.file_path for o .xlsx original, usamos o sheet_name
            # Se você estiver lendo os CSVs individuais, a lógica de path mudaria levemente
            df_raw = pd.read_excel(self.file_path, sheet_name=profile_name)
        except Exception as e:
            print(f"Erro ao carregar o perfil {profile_name}: {e}")
            return None, None

        # 1. MAPEAMENTO DE COLUNAS (Ajuste conforme os headers do seu arquivo)
        # Geralmente esses arquivos usam 'Tempo [s]', 'Tensao [V]', etc.
        column_map = {
            'Tempo (s)': 'Time',
            'Tensão (V)': 'Voltage',
            'Corrente (A)': 'Current',
            'Temp_Bateria [°C]': 'Temperature'
        }
        
        # Renomeia apenas as colunas que existirem no arquivo
        df = df_raw.rename(columns=column_map)

        # 2. SELEÇÃO DE COLUNAS ESSENCIAIS
        required_cols = ['Time', 'Voltage', 'Current']
        for col in required_cols:
            if col not in df.columns:
                print(f"Aviso: Coluna essencial '{col}' não encontrada no perfil {profile_name}.")
                # Se não houver corrente/tensão, o processamento é impossível
                return None, None

        # 3. LIMPEZA E PADRONIZAÇÃO
        # Remove linhas vazias e garante que os dados são numéricos
        df = df.dropna(subset=required_cols).reset_index(drop=True)
        df = self.add_physics_features(df)
        # Se não houver temperatura, assume ambiente (25°C) para não quebrar o framework
        if 'Temperature' not in df.columns:
            df['Temperature'] = 25.0

        # 4. TRATAMENTO DE CICLO
        # Para esses arquivos compilados, geralmente tratamos o arquivo todo como um "Ciclo 1"
        # ou usamos uma coluna de 'Ciclo' se ela existir.
        if 'Cycle_Number' not in df.columns:
            df['Cycle_Number'] = 1

        # 5. CÁLCULO DE SOC E CAPACIDADE (Coulomb Counting)
        # Calculamos o delta tempo entre as amostras
        dt = df['Time'].diff().fillna(0) / 3600.0  # s para h
        
        # No Pulse, precisamos checar se a corrente de descarga é positiva ou negativa
        # Se for positiva para descarga: discharged_ah = np.cumsum(df['Current'] * dt)
        # Se for negativa (padrão NASA/CALCE):
        discharged_ah = np.cumsum(np.abs(df['Current']) * dt)
        
        total_cap_ah = discharged_ah.max()
        
        # Evita divisão por zero
        if total_cap_ah > 0:
            df['SOC'] = np.clip(1.0 - (discharged_ah / total_cap_ah), 0.0, 1.0)
        else:
            df['SOC'] = 1.0

        # 6. LIMPEZA ADICIONAL E FEATURES FÍSICAS
        df_cleaned = self.clean_battery_data(df, is_dynamic=True, is_test=is_test)
        
        if df_cleaned is not None:
            # Adiciona P_avg, R_avg, etc.
            # df_cleaned = self.add_physics_features(df_cleaned)
            
            # Cria o DF de tendência de capacidade (um valor por ciclo)
            df_capacity = pd.DataFrame([{
                'Cycle': 1, 
                'Capacity': total_cap_ah,
                'Profile': profile_name
            }])
            
            print(f"Processamento {profile_name} concluído com sucesso.")
            
        return df_cleaned, df_capacity

    def do_visual_analisys(self, df_timeseries, df_capacity):
        """
        Gera gráficos para validar a limpeza e o cálculo do SOC.
        """
        if df_timeseries is None or df_timeseries.empty:
            print("Sem dados para visualizar.")
            return

        # Amostra do Ciclo 1
        df_ciclo = df_timeseries[df_timeseries['Cycle_Number'] == df_timeseries['Cycle_Number'].iloc[0]]

        plt.figure(figsize=(18, 10))
        plt.suptitle(f'Análise de Dados Processados - {os.path.basename(self.file_path)}', fontsize=16)

        plt.subplot(2, 2, 1)
        plt.plot(df_ciclo['Time'], df_ciclo['Voltage'], color='blue')
        plt.title('Voltagem (Limpa)')
        plt.ylabel('V')

        plt.subplot(2, 2, 2)
        plt.plot(df_ciclo['Time'], df_ciclo['Current'], color='red')
        plt.title('Corrente (Limpa)')
        plt.ylabel('A')

        plt.subplot(2, 2, 3)
        plt.plot(df_ciclo['Time'], df_ciclo['SOC'], color='purple')
        plt.title('SOC Estimado (Inicia em 1.0)')
        plt.ylabel('SOC')
        plt.ylim(-0.1, 1.1)

        plt.subplot(2, 2, 4)
        plt.plot(df_capacity['Cycle'], df_capacity['Capacity'], marker='o')
        plt.title('Capacidade por Ciclo')
        plt.ylabel('Ah')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        
    def plot_augmented_features(self, df, cycle_num=1):
        """
        Gera o gráfico comparativo de Potência e Resistência (Similar à Fig 2 do artigo).
        """
        df_cycle = df[df['Cycle_Number'] == cycle_num]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot Potência Média (Reflete transferência de energia [cite: 268])
        ax1.plot(df_cycle['Time'], df_cycle['P_avg'], color='green')
        ax1.set_title(f'Potência Média (P_avg) - Ciclo {cycle_num}')
        ax1.set_ylabel('Watts')
        
        # Plot Resistência Média (Reflete impedância interna [cite: 265])
        ax2.plot(df_cycle['Time'], df_cycle['R_avg'], color='red')
        ax2.set_title(f'Resistência Média (R_avg) - Ciclo {cycle_num}')
        ax2.set_ylabel('Ohms')
        
        plt.tight_layout()
        plt.show()