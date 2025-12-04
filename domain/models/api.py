from sqlalchemy import select
from domain.loans.loan import Loan
from server import server, session
from fastapi.templating import Jinja2Templates
from fastapi import Request, status, Form
from pydantic import BaseModel, Field
from fastapi.responses import RedirectResponse
from typing import Annotated, Optional
from domain.models.model_actions import preprocessing, split, create_nn_model, train_model, evaluate_performance, draw_loss, model_predict
import pandas as pd
import joblib
from os.path import join as join
import mlflow
import mlflow.data
from datetime import datetime

templates = Jinja2Templates(directory="domain/models/templates")


class TrainModelForm(BaseModel):
    epochs: int

@server.get("/models/train")
async def train_model_form(request: Request):
    return templates.TemplateResponse("list_loans.html", request=request, context={"request": request, "loans": loans})

@server.post("/models/train")
async def train_model_action(request: Request, input: Annotated[TrainModelForm, Form()]):
    mlflow.set_experiment("Entraînement du " + datetime.now().strftime("%d/%m/%Y, %H:%M:%S"))

    df = pd.read_sql(session.query(Loan).statement,session.bind) 

    new_dataset = mlflow.data.from_pandas(
        df, name="Dataset", targets="montant_pret"
    )

    # preprocesser les data
    new_X, new_y, _ = preprocessing(df)

    # split data in train and test dataset
    new_X_train, new_X_test, new_y_train, new_y_test = split(new_X, new_y)

    with mlflow.start_run(run_name = "Entraînement du nouveau modèle"):
        mlflow.log_param("epochs", input.epochs)
        mlflow.log_input(new_dataset)

        new_model = create_nn_model(new_X_train.shape[1])
        new_model, hist, metrics = train_model(
            new_model, 
            new_X_train, 
            new_y_train, 
            X_val=new_X_test, 
            y_val=new_y_test, 
            epochs=input.epochs
        )
        plot = draw_loss(hist)

        mlflow.log_metric("MSE", metrics["MSE"])
        mlflow.log_metric("MAE", metrics["MAE"])
        mlflow.log_metric("R²", metrics["R²"])
        mlflow.log_figure(plot, "loss.png")
        mlflow.sklearn.log_model(new_model, "Nouveau modèle")

        # sauvegarder le nouveau modèle
        joblib.dump(new_model, join('models','model_2025_11.pkl'))

    return RedirectResponse("/models/train", status_code=status.HTTP_303_SEE_OTHER)    

