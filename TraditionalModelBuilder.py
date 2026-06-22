#%pip install XGBoost
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor

class TraditionalModelBuilder:
    def __init__(self, model_type='rf', n_estimators=100, max_depth=None, 
                 learning_rate=0.1, kernel='rbf', n_neighbors=5):
        """
        Classe flexível para construir modelos de Machine Learning tradicionais 
        para estimativa de bateria.
        
        :param model_type: Tipo de algoritmo a construir:
                           - 'rf': Random Forest Regressor
                           - 'xgb': XGBoost Regressor
                           - 'dt': Decision Tree Regressor
                           - 'svr': Support Vector Regression
                           - 'knn': K-Nearest Neighbors
        """
        self.model_type = model_type.lower().strip()
        
        # Hiperparâmetros Gerais
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.kernel = kernel
        self.n_neighbors = n_neighbors
        
        self.model = None

    def build(self):
        print(f"A instanciar o modelo tradicional: {self.model_type.upper()}...")
        
        if self.model_type == 'rf':
            # Random Forest: Ensemble de árvores de decisão com Bagging
            self.model = RandomForestRegressor(
                n_estimators=self.n_estimators, 
                max_depth=self.max_depth, 
                random_state=42,
                n_jobs=-1
            )
            
        elif self.model_type == 'xgb':
            # XGBoost: Gradient Boosting de alta performance
            self.model = XGBRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth if self.max_depth else 6,
                random_state=42,
                n_jobs=-1
            )
            
        elif self.model_type == 'dt':
            # Decision Tree: Árvore de decisão simples (Baseline)
            self.model = DecisionTreeRegressor(
                max_depth=self.max_depth, 
                random_state=42
            )
            
        elif self.model_type == 'svr':
            # SVR: Support Vector Regression (Tradicional para problemas não lineares)
            self.model = SVR(
                kernel=self.kernel, 
                C=1.0, 
                epsilon=0.1
            )
            
        elif self.model_type == 'knn':
            # KNN: Baseado em proximidade espacial
            self.model = KNeighborsRegressor(
                n_neighbors=self.n_neighbors,
                weights='distance'
            )
            
        else:
            raise ValueError(f"Tipo de modelo '{self.model_type}' não reconhecido.")
            
        return self.model

    def get_params(self):
        if self.model:
            return self.model.get_params()
        else:
            print("O modelo ainda não foi construído.")
            return None