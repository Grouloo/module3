from domain.loans.loan import Loan
from server import session
from domain.models.model_actions import preprocessing, split, create_nn_model, train_model, evaluate_performance, draw_loss, transfer_weights
import pandas as pd
import joblib
from os.path import join as join
import mlflow
import mlflow.data
from datetime import datetime
import tensorflow as tf

def find_outliers(col):
        Q1 = col.quantile(0.25)
        Q3 = col.quantile(0.75)
        IQR = Q3 - Q1
        outliers = cleaned_dataset[(col < Q1 - threshold * IQR) | (col > Q3 + threshold * IQR)]
        return outliers

mlflow.set_experiment("Entraînement du " + datetime.now().strftime("%d/%m/%Y, %H:%M:%S"))

with mlflow.start_run(run_name = "Entraînement du premier modèle"):
    epochs = 50
    mlflow.log_param("epochs", epochs)

    first_df = pd.read_sql(session.query(Loan).where(Loan.id < 10000).statement,session.bind)
    first_df.reset_index(drop=True, inplace=True)

     # Nettoyage
    cleaned_dataset = first_df.drop_duplicates(inplace=False)
    cleaned_dataset = cleaned_dataset.drop(columns=["nb_enfants", "quotient_caf"], inplace=False)

    num_columns=["age", "historique_credits", "revenu_estime_mois", "risque_personnel", "score_credit", "loyer_mensuel", "date_creation_compte", "montant_pret"]
    cleaned_dataset[num_columns] = cleaned_dataset[num_columns].fillna(value=cleaned_dataset[num_columns].median(), inplace=False) 

    cleaned_dataset = cleaned_dataset.dropna(subset=['situation_familiale'], inplace=False) 

    cleaned_dataset = cleaned_dataset.drop(cleaned_dataset.loc[cleaned_dataset["loyer_mensuel"] < 0.0].index, inplace=False)

    threshold = 1.5

    imc_outliers = find_outliers(cleaned_dataset["imc"])
    cleaned_dataset = cleaned_dataset.drop(imc_outliers.index)

    revenu_outliers = find_outliers(cleaned_dataset["revenu_estime_mois"])
    cleaned_dataset = cleaned_dataset.drop(revenu_outliers.index)

    pret_outliers = find_outliers(cleaned_dataset["montant_pret"])
    cleaned_dataset = cleaned_dataset.drop(pret_outliers.index)


    new_dataset = mlflow.data.from_pandas(
        cleaned_dataset, name="Premier Dataset", targets="montant_pret"
    )

    numerical_cols = ["age", "imc", "historique_credits", "revenu_estime_mois", "risque_personnel", "score_credit", "loyer_mensuel", "date_creation_compte"]
    categorical_cols = ["sport_licence", "niveau_etude", "smoker", "situation_familiale"]

    # preprocesser les data
    new_X, new_y, preprocessor = preprocessing(cleaned_dataset, numerical_cols, categorical_cols)

    # split data in train and test dataset
    new_X_train, new_X_test, new_y_train, new_y_test = split(new_X, new_y)

    mlflow.log_input(new_dataset)


    new_model = create_nn_model(new_X_train.shape[1])

    new_model, hist, metrics = train_model(
        new_model, 
        new_X_train, 
        new_y_train, 
        X_val=new_X_test, 
        y_val=new_y_test, 
        epochs=epochs
    )
    plot = draw_loss(hist)

    mlflow.log_metric("MSE", metrics["MSE"])
    mlflow.log_metric("MAE", metrics["MAE"])
    mlflow.log_metric("R²", metrics["R²"])
    mlflow.log_figure(plot, "loss.png")
    mlflow.sklearn.log_model(new_model, "Premier modèle")

    # sauvegarder le nouveau modèle
    joblib.dump(preprocessor, join('models','first_preprocessor.pkl'))
    joblib.dump(new_model, join('models','first_model.pkl'))
    new_model.save(join('models', "first_model.keras"))



with mlflow.start_run(run_name = "Entraînement du deuxième modèle"):
    epochs = 50
    mlflow.log_param("epochs", epochs)

    second_df = pd.read_sql(session.query(Loan).statement,session.bind)
    second_df.reset_index(drop=True, inplace=True)

    cleaned_dataset = second_df.drop_duplicates(inplace=False)
    num_columns=["nb_enfants", "quotient_caf", "age", "historique_credits", "revenu_estime_mois", "risque_personnel", "score_credit", "loyer_mensuel", "date_creation_compte", "montant_pret"]
    cleaned_dataset[num_columns] = cleaned_dataset[num_columns].fillna(value=cleaned_dataset[num_columns].median(), inplace=False) 

    cleaned_dataset = cleaned_dataset.dropna(subset=['situation_familiale'], inplace=False) 

    cleaned_dataset = cleaned_dataset.drop(cleaned_dataset.loc[cleaned_dataset["loyer_mensuel"] < 0.0].index, inplace=False)

    threshold = 1.5

    imc_outliers = find_outliers(cleaned_dataset["imc"])
    cleaned_dataset = cleaned_dataset.drop(imc_outliers.index)

    revenu_outliers = find_outliers(cleaned_dataset["revenu_estime_mois"])
    cleaned_dataset = cleaned_dataset.drop(revenu_outliers.index)

    pret_outliers = find_outliers(cleaned_dataset["montant_pret"])
    cleaned_dataset = cleaned_dataset.drop(pret_outliers.index)

    new_dataset = mlflow.data.from_pandas(
        cleaned_dataset, name="Second Dataset", targets="montant_pret"
    )

    numerical_cols = ["age", "imc", "historique_credits", "revenu_estime_mois", "risque_personnel", "score_credit", "loyer_mensuel", "date_creation_compte", "nb_enfants", "quotient_caf"]
    categorical_cols = ["sport_licence", "niveau_etude", "smoker", "situation_familiale"]

    # preprocesser les data
    new_X, new_y, preprocessor = preprocessing(cleaned_dataset, numerical_cols, categorical_cols)

    # split data in train and test dataset
    new_X_train, new_X_test, new_y_train, new_y_test = split(new_X, new_y)

    mlflow.log_input(new_dataset)

    first_model = tf.keras.models.load_model(join("models", "first_model.keras"))


    new_model = create_nn_model(new_X_train.shape[1])

    new_model = transfer_weights(first_model, new_model)

    new_model, hist, metrics = train_model(
        new_model, 
        new_X_train, 
        new_y_train, 
        X_val=new_X_test, 
        y_val=new_y_test, 
        epochs=epochs
    )
    plot = draw_loss(hist)

    mlflow.log_metric("MSE", metrics["MSE"])
    mlflow.log_metric("MAE", metrics["MAE"])
    mlflow.log_metric("R²", metrics["R²"])
    mlflow.log_figure(plot, "loss.png")
    mlflow.sklearn.log_model(new_model, "Premier modèle")

    # sauvegarder le nouveau modèle
    joblib.dump(preprocessor, join('models','second_preprocessor.pkl'))
    joblib.dump(new_model, join('models','second_model.pkl'))