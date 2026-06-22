import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Bidirectional, GRU, Dropout, Input
from tensorflow.keras.optimizers import Adam

class BatteryModelBuilder:
    def __init__(self, seq_length, n_features, model_type='cnn_lstm', 
                 filters=128, kernel_size=3, lstm_units=50, 
                 dense_units=64, learning_rate=0.001):
        """
        Classe flexível para construir diferentes arquiteturas de Deep Learning para estimativa de bateria.
        
        :param model_type: Tipo de arquitetura a construir:
                           - 'lstm': LSTM Pura (Vanilla)
                           - 'bilstm': Bidirectional LSTM
                           - 'cnn_lstm': Híbrida CNN + LSTM
                           - 'gru': Gated Recurrent Unit
                           - 'lstm_rnn': Stacked LSTM (Dupla camada LSTM - LSTM profunda)
        """
        self.seq_length = seq_length
        self.n_features = n_features
        self.model_type = model_type.lower().strip()
        
        # Hiperparâmetros
        self.filters = filters
        self.kernel_size = kernel_size
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.learning_rate = learning_rate
        self.model = None

    def build(self):
        print(f"A construir a arquitetura: {self.model_type.upper()}...")
        
        self.model = Sequential()
                
        # Se for um modelo híbrido com CNN, começamos com camadas convolucionais
        if 'cnn' in self.model_type:
            self.model.add(Conv1D(filters=self.filters, kernel_size=self.kernel_size, 
                                padding='same', input_shape=(self.seq_length, self.n_features)))
            self.model.add(tf.keras.layers.BatchNormalization()) # Estabiliza o aprendizado
            self.model.add(tf.keras.layers.Activation('relu'))
            self.model.add(MaxPooling1D(pool_size=2))
        else:
            # Caso contrário, definimos a entrada explicitamente para as recorrentes
            self.model.add(Input(shape=(self.seq_length, self.n_features)))
        
        if self.model_type == 'lstm' or self.model_type == 'cnn_lstm':
            # LSTM Padrão
            # self.model.add(LSTM(self.lstm_units, activation='tanh', return_sequences=False))
            self.model.add(LSTM(self.lstm_units, return_sequences=True))
            self.model.add(tf.keras.layers.SpatialDropout1D(0.2)) 
            # self.model.add(tf.keras.layers.Dropout(0.2))
            self.model.add(LSTM(self.lstm_units, return_sequences=False))  
        elif self.model_type == 'bilstm':
            # Bidirectional: Lê a sequência do passado para o futuro e vice-versa
            self.model.add(Bidirectional(LSTM(self.lstm_units, activation='tanh', return_sequences=False)))
            
        elif self.model_type == 'gru' or self.model_type == 'cnn_gru':
            # GRU: Similar à LSTM mas computacionalmente mais leve
            self.model.add(GRU(self.lstm_units, activation='tanh', return_sequences=False))
            
        elif self.model_type == 'lstm_rnn':
            # LSTM-RNN (Stacked LSTM): Arquitetura mais profunda
            # 1ª Camada: return_sequences=True (passa a sequência temporal completa para a próxima camada)
            self.model.add(LSTM(self.lstm_units, activation='tanh', return_sequences=True))
            self.model.add(Dropout(0.2)) 
            # 2ª Camada: return_sequences=False (condensa para um vetor único para a camada Dense)
            self.model.add(LSTM(self.lstm_units, activation='tanh', return_sequences=False))
        elif self.model_type == 'cnn_attention':
            # Extração de features temporal
            self.model.add(Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation='relu'))
            self.model.add(MaxPooling1D(pool_size=2))
            
            # Camada recorrente que mantém a sequência
            self.model.add(LSTM(self.lstm_units, return_sequences=True))
            
            # Camada de Atenção Simples (Global)
            # Isso faz uma média ponderada da sequência baseada na importância de cada ponto
            self.model.add(tf.keras.layers.Attention()) 
            self.model.add(tf.keras.layers.Flatten())
        else:
            raise ValueError(f"Tipo de modelo '{self.model_type}' não reconhecido.")
        
        self.model.add(Dense(self.dense_units, activation='relu'))
        self.model.add(Dropout(0.2)) 
        
        # Saída (Sigmoid para SOC normalizado entre 0 e 1)
        self.model.add(Dense(1, activation='sigmoid'))
        
        optimizer = Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(), metrics=['mse', 'mae'])
        
        return self.model

    def get_summary(self):
        if self.model:
            self.model.summary()
        else:
            print("O modelo ainda não foi construído.")