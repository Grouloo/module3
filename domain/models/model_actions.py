from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import pandas as pd

def split(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def preprocessing(df):
    """
    Fonction pour effectuer le prétraitement des données :
    - Imputation des valeurs manquantes.
    - Standardisation des variables numériques.
    - Encodage des variables catégorielles.
    """

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    numerical_cols = ["age", "imc", "historique_credits", "revenu_estime_mois", "risque_personnel", "score_credit", "loyer_mensuel", "date_creation_compte"]
    categorical_cols = ["sport_licence", "niveau_etude", "smoker", "situation_familiale"]


    # Prétraitement
    X = df.drop(columns=["id", "montant_pret"])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, numerical_cols),
        ("cat", cat_pipeline, categorical_cols)
    ])
   
    X_processed = preprocessor.fit_transform(X)

    y = df["montant_pret"]
    return X_processed, y, preprocessor

def create_nn_model(input_dim):
    """
    Fonction pour créer et compiler un modèle de réseau de neurones simple.
    """
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=input_dim))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def train_model(model, X, y, X_val=None, y_val=None, epochs=50, batch_size=32, verbose=0 ):
    hist = model.fit(X, y, 
                validation_data=(X_val, y_val) if X_val is not None and y_val is not None else None,
                epochs=epochs, batch_size=batch_size, verbose=verbose)

    y_pred = model_predict(model, X_val)
    metrics = evaluate_performance(y_val, y_pred) 

    return model , hist, metrics

def model_predict(model, X):
    y_pred = model.predict(X).flatten()
    return y_pred


def evaluate_performance(y_true: list, y_pred: list):
    """
    Fonction pour mesurer les performances du modèle avec MSE, MAE et R².
    Entrées :
        - y_true : Sorties attendues (list)
        - y_pred : Sorties prédites par le modèle (list)
    Sortie :
        - MSE : Erreur quadratique moyenne (int)
        - MAE : Erreur absolue moyenne, écart entre les prédictions et les résultats attendus (int)
        - R² : Coefficient de détermination, mesure de la qualité des prédictions du modèle entre 0 et 1 (float)
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'MSE': mse, 'MAE': mae, 'R²': r2} 

def draw_loss(history):
    """
    Affiche les courbes de loss et val_loss de l'historique d'entraînement.
    """
    fig = plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Loss (Entraînement)')
    plt.plot(history.history['val_loss'], label='Val Loss (Validation)', linestyle='--')
    plt.title('Courbes de Loss et Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    return fig