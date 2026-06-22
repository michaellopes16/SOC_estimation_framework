#%pip install opencv-python-headless
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

class BatteryExplainerGRADCAM:
    def __init__(self, model, scaler_y, feature_names):
        self.model = model
        self.scaler_y = scaler_y
        self.feature_names = feature_names
        self.last_conv_layer_name = self._find_last_conv_layer()

    def _find_last_conv_layer(self):
        for layer in reversed(self.model.layers):
            if 'conv1d' in layer.name.lower():
                return layer.name
        # Em vez de raise ValueError, retorne None para o loop tratar com elegância
        return None

    def _make_gradcam_heatmap(self, img_array):
        """
        Gera o Heatmap reconstruindo o grafo para compatibilidade com Keras 3.
        """
        # 1. Identificar o shape de entrada corretamente
        # Se img_array é (1, 50, 3), o input_shape é (50, 3)
        input_shape = img_array.shape[1:] 
        
        # 2. Criar um grafo funcional temporário (Functional Clone)
        # Isso 'conserta' o erro de "layer never called" conectando tudo manualmente
        inputs = tf.keras.Input(shape=input_shape)
        x = inputs
        conv_output = None
        
        # Reconstrói a passagem dos dados camada por camada
        for layer in self.model.layers:
            x = layer(x)
            if layer.name == self.last_conv_layer_name:
                conv_output = x
        
        # O modelo 'grad_model' agora é garantidamente funcional
        grad_model = tf.keras.models.Model(inputs, [conv_output, x])

        # 3. Calcular gradientes (igual ao anterior)
        with tf.GradientTape() as tape:
            conv_outputs_val, predictions = grad_model(img_array)
            loss = predictions[:, 0]

        grads = tape.gradient(loss, conv_outputs_val)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

        conv_outputs_val = conv_outputs_val[0]
        heatmap = conv_outputs_val @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        return heatmap.numpy()

    def explain_index(self, X_data, y_data, idx, feature_to_plot='Voltage'):
        if self.last_conv_layer_name is None:
            print("Pulo: Modelo não possui camadas Conv1D.")
            return
        # Prepara dados
        sample_input = X_data[idx:idx+1]
        
        # Previsão e Valores Reais
        # Nota: Usamos sample_input direto no modelo original para previsão numérica
        pred_scaled = self.model.predict(sample_input, verbose=0)
        
        # Recupera valor real e predito
        target_scaled = y_data[idx]
        real_soc = self.scaler_y.inverse_transform(target_scaled.reshape(-1, 1)).flatten()[0]
        pred_soc = self.scaler_y.inverse_transform(pred_scaled).flatten()[0]
        
        print(f"\n--- Análise do Índice {idx} ---")
        print(f"Camada Convolucional: {self.last_conv_layer_name}")
        
        # Gera Heatmap (agora usando a versão corrigida)
        heatmap = self._make_gradcam_heatmap(sample_input)
        
        # Redimensionar (Zoom)
        seq_length = sample_input.shape[1]
        zoom_factor = seq_length / len(heatmap)
        heatmap_resized = zoom(heatmap, zoom_factor)
        
        self._plot_explanation(sample_input[0], heatmap_resized, real_soc, pred_soc, feature_to_plot)

    def _plot_explanation(self, input_sequence, heatmap, real_soc, pred_soc, feature_name):
        try:
            feat_idx = self.feature_names.index(feature_name)
        except ValueError:
            feat_idx = 0
            feature_name = self.feature_names[0]

        signal = input_sequence[:, feat_idx]
        seq_len = len(signal)

        plt.figure(figsize=(14, 6))
        plt.scatter(range(seq_len), signal, c=heatmap, cmap='jet', s=60, alpha=0.8)
        plt.plot(range(seq_len), signal, 'k-', alpha=0.2)
        plt.colorbar(label='Importância')
        plt.title(f"Grad-CAM: {feature_name}\nSOC Real: {real_soc:.2f} | Pred: {pred_soc:.2f}")
        plt.xlabel("Tempo")
        plt.ylabel("Valor Normalizado")
        plt.grid(True, alpha=0.3)
        plt.show()