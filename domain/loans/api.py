from sqlalchemy import select
from domain.loans.loan import Loan
from server import server, session
from fastapi.templating import Jinja2Templates
from fastapi import Request, status, Form
from pydantic import BaseModel, Field
from fastapi.responses import RedirectResponse
from typing import Annotated, Optional
import datetime
import joblib
from os.path import join as join
import pandas as pd

templates = Jinja2Templates(directory="domain/loans/templates")

preprocessor = joblib.load(join('models','second_preprocessor.pkl'))
model = joblib.load(join('models','second_model.pkl'))


@server.get("/loans")
async def list_loans(request: Request):
    loans = session.execute(select(Loan)).scalars().all()
    return templates.TemplateResponse("list_loans.html", request=request, context={"request": request, "loans": loans})


class AddLoanForm(BaseModel):
    age: int
    niveau_etude: int
    situation_familiale: str
    revenu_estime_mois: float
    loyer_mensuel: float
    imc: float
    smoker: Optional[bool] = Field(None)
    sport_licence: Optional[bool] = Field(None)
    date_creation_compte: str
    historique_credits: float
    risque_personnel: float
    score_credit: float
    montant_pret: float
    nb_enfants: int
    quotient_caf: float

@server.get("/loans/add")
async def add_loan_form(request: Request):
    return templates.TemplateResponse("add_loan.html", request=request, context={"request": request})

@server.post("/loans/add")
async def add_loan_action(request: Request, input: Annotated[AddLoanForm, Form()]):
    if input.smoker:
        smoker = "oui"
    else:
        smoker = "non"

    if input.sport_licence:
        sport_licence = "oui"
    else:
        sport_licence = "non"

    new_loan = Loan(
        age = input.age,
        niveau_etude = input.niveau_etude,
        situation_familiale = input.situation_familiale,
        revenu_estime_mois = input.revenu_estime_mois,
        loyer_mensuel = input.loyer_mensuel,
        imc = input.imc,
        smoker = smoker,
        sport_licence = sport_licence,
        date_creation_compte = datetime.datetime.fromisoformat( input.date_creation_compte).timestamp(),
        historique_credits = input.historique_credits,
        risque_personnel = input.risque_personnel,
        score_credit = input.score_credit,
        montant_pret = input.montant_pret,
         nb_enfants = input.nb_enfants,
        quotient_caf = input.quotient_caf
    )
    session.add(new_loan)
    session.commit()
    return RedirectResponse("/loans", status_code=status.HTTP_303_SEE_OTHER)    


@server.delete("/loans/{loan_id}")
async def remove_a_loan(request: Request, loan_id: int):
    loan = session.execute(select(Loan).where(Loan.id == loan_id)).scalar_one()
    session.delete(loan)
    session.commit()
    loans = session.execute(select(Loan)).scalars().all()
    return templates.TemplateResponse("list_loans.html", request=request, context={"request": request, "loans": loans})


@server.get("/loans/predict")
async def predict_loan_form(request: Request):
    return templates.TemplateResponse("predict_loan.html", request=request, context={"request": request})

class PredictLoanForm(BaseModel):
    age: int
    niveau_etude: str
    situation_familiale: str
    revenu_estime_mois: float
    loyer_mensuel: float
    imc: float
    smoker: Optional[bool] = Field(None)
    sport_licence: Optional[bool] = Field(None)
    date_creation_compte: str
    historique_credits: float
    risque_personnel: float
    score_credit: float
    nb_enfants: int
    quotient_caf: float

@server.post("/loans/predict")
async def predict_loan_action(request: Request, input: Annotated[PredictLoanForm, Form()]):
    if input.smoker:
        smoker = "oui"
    else:
        smoker = "non"

    if input.sport_licence:
        sport_licence = "oui"
    else:
        sport_licence = "non"

    loan = Loan(
        age = input.age,
        niveau_etude = input.niveau_etude,
        situation_familiale = input.situation_familiale,
        revenu_estime_mois = input.revenu_estime_mois,
        loyer_mensuel = input.loyer_mensuel,
        imc = input.imc,
        smoker = smoker,
        sport_licence = sport_licence,
        date_creation_compte = datetime.datetime.fromisoformat(input.date_creation_compte).timestamp(),
        historique_credits = input.historique_credits,
        risque_personnel = input.risque_personnel,
        score_credit = input.score_credit,
        nb_enfants = input.nb_enfants,
        quotient_caf = input.quotient_caf
    )
    df = pd.DataFrame({
        "age": loan.age,
        "sport_licence": loan.sport_licence,
        "niveau_etude": loan.niveau_etude,
        "smoker": loan.smoker,
        "revenu_estime_mois": loan.revenu_estime_mois,
        "situation_familiale": loan.situation_familiale,
        "historique_credits": loan.historique_credits,
        "risque_personnel": loan.risque_personnel,
        "date_creation_compte": loan.date_creation_compte,
        "score_credit": loan.score_credit,
        "loyer_mensuel": loan.loyer_mensuel,
        "imc": loan.imc,
        "nb_enfants": loan.nb_enfants,
        "quotient_caf": loan.quotient_caf
    }, index=[0])
    # print(preprocessor)
    data_transformed = preprocessor.transform(df)
    print(data_transformed)
    prediction_pipeline = model.predict(data_transformed)
    return f"""<p>Montant du prêt : {prediction_pipeline[0]} €</p>""" 