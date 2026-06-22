# Deep Learning and XAI Framework for Battery SOC Estimation

This repository contains the complete infrastructure built in Python for processing, modeling, optimizing, and explaining time series geared towards battery State of Charge (SOC) estimation. The project was structured under the organization of **ISI-TICs**, focusing on methodological rigor, predictive accuracy, and transparency of *Deep Learning* models through eXplainable Artificial Intelligence (XAI).

---

## 📌 What does the application do?

The application ingests raw operational battery data (Voltage, Current, and Temperature) and dynamically builds complex neural networks (CNNs, LSTMs, and Attention mechanisms) to predict the SOC. 

The main differentiators of this tool are:
1. **Physics-Based Feature Engineering:** Autonomous inclusion of derived variables, such as Average Power ($P_{avg}$) and Average Resistance ($R_{avg}$).
2. **Intelligent Optimization:** Use of Reinforcement Learning agents (Q-Learning and DQN) to autonomously optimize the neural network's hyperparameters.
3. **Transparency:** The processing culminates in the generation of the **Quantitative XAI Ensemble Results**, outputting visual reports based on gradients (Grad-CAM) and local approximation (LIME) to justify the decisions of the "black-box" model.

---

## ⚙️ Prerequisites (Requirements)

To run the framework, you will need **Python 3.8+**. The use of a virtual environment (`venv` or `conda`) is recommended. 

Create a file named `requirements.txt` in the root of your project with the content below and install the dependencies via `pip install -r requirements.txt`:

```text
numpy
pandas
scikit-learn
scipy
tensorflow
matplotlib
opencv-python-headless
lime
shap
```

---

## 🏗️ System Architecture

The framework follows a modular design pattern (Facade, Builder, Pipeline). Below is the system architecture detailed in JSON format for easy visualization of each class's responsibilities:

```json
{
  "name": "Machine Learning Framework for Batteries",
  "version": "1.0",
  "research_group": "ISI-TICs",
  "layers": [
    {
      "name": "1. Orchestration and Control",
      "components": [
        {"file": "Main.ipynb", "description": "User entry point and interactive experimentation."},
        {"file": "BatteryProjectPipeline.py", "description": "Facade that coordinates the entire data, model, and XAI pipeline."}
      ]
    },
    {
      "name": "2. Data and Preprocessing",
      "components": [
        {"file": "BatteryDataset.py", "description": "Data loading and physics-based feature engineering (P_avg, R_avg)."},
        {"file": "BatteryDataPreprocessor.py", "description": "Normalization and 3D sequential windowing for time series."}
      ]
    },
    {
      "name": "3. Modeling and Optimization",
      "components": [
        {"file": "BatteryModelBuilder.py", "description": "Dynamic neural network factory (CNN, LSTM, Attention)."},
        {"file": "BatteryOptimizerRL.py", "description": "Hyperparameter optimization using Reinforcement Learning (RL)."},
        {"file": "BatteryOptimizerDQN.py", "description": "Advanced hyperparameter optimization using Deep Q-Networks (DQN)."}
      ]
    },
    {
      "name": "4. Training and Execution",
      "components": [
        {"file": "BatteryTrainer.py", "description": "Training loop, evaluation, and plotting of convergence metrics (RMSE, MAE)."}
      ]
    },
    {
      "name": "5. Explainability (XAI)",
      "components": [
        {"file": "BatteryExplainer.py", "description": "Convolution-focused heatmaps using Grad-CAM."},
        {"file": "BatteryLimeExplainer.py", "description": "Local and agnostic feature weight explanation using LIME."}
      ]
    }
  ]
}
```

### Visual Flow
<img width="2816" height="1536" alt="Main Flow" src="https://github.com/user-attachments/assets/1631d344-d8e4-4a95-b04a-945e810bc343" />

---

## 💻 How to Use

The simplest way to run experiments is by using the interactive environment provided in the `Main.ipynb` file. However, if you want to instantiate the code in your own scripts and heavy routines, usage follows this pattern with just a few lines:

```python
import pandas as pd
from BatteryProjectPipeline import BatteryProjectPipeline

# 1. Load your battery time series dataset
df_raw = pd.read_csv("example_battery_dataset.csv")

# 2. Initialize the orchestrator Pipeline
pipeline = BatteryProjectPipeline(df_raw=df_raw)

# 3. Execute the full flow
# Encompasses: Extraction -> Windowing -> RL Optimization -> Training -> XAI Visualization
pipeline.run_full_pipeline(
    models=['all'],      # Options: lstm, bilstm, gru, cnn_attention, or all...
    optimize=True,       # Enables the RL/DQN Agent to find the best parameters
    epochs=100           # Total epochs for the final training stage
)
```
