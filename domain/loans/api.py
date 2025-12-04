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


templates = Jinja2Templates(directory="domain/loans/templates")

model = joblib.load(join('models','model.pkl'))
preprocessor_loaded = joblib.load(join('models','preprocessor.pkl'))


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

@server.get("/loans/add")
async def add_loan_form(request: Request):
    return templates.TemplateResponse("add_loan.html", request=request, context={"request": request})

@server.post("/loans/add")
async def add_loan_action(request: Request, input: Annotated[AddLoanForm, Form()]):
    new_loan = Loan(
        age = input.age,
        niveau_etude = input.niveau_etude,
        situation_familiale_célibataire = input.situation_familiale == "célibataire",
        situation_familiale_marié = input.situation_familiale == "marié",
        situation_familiale_divorcé = input.situation_familiale == "divorcé",
        situation_familiale_veuf = input.situation_familiale == "veuf",
        revenu_estime_mois = input.revenu_estime_mois,
        loyer_mensuel = input.loyer_mensuel,
        imc = input.imc,
        smoker_oui = input.smoker,
        smoker_non = not input.smoker,
        sport_licence_oui = input.sport_licence,
        sport_licence_non = not input.sport_licence,
        date_creation_compte = datetime.datetime.fromisoformat( input.date_creation_compte),
        historique_credits = input.historique_credits,
        risque_personnel = input.risque_personnel,
        score_credit = input.score_credit,
        montant_pret = input.montant_pret,
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

class PredictLoanForm(BaseModel):
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

@server.get("/loans/predict")
async def predict_loan_form(request: Request):
    return templates.TemplateResponse("add_loan.html", request=request, context={"request": request})

@server.post("/loans/predict")
async def predict_loan_action(request: Request, input: Annotated[AddLoanForm, Form()]):
    loan = Loan(
        age = input.age,
        niveau_etude = input.niveau_etude,
        situation_familiale_célibataire = input.situation_familiale == "célibataire",
        situation_familiale_marié = input.situation_familiale == "marié",
        situation_familiale_divorcé = input.situation_familiale == "divorcé",
        situation_familiale_veuf = input.situation_familiale == "veuf",
        revenu_estime_mois = input.revenu_estime_mois,
        loyer_mensuel = input.loyer_mensuel,
        imc = input.imc,
        smoker_oui = input.smoker,
        smoker_non = not input.smoker,
        sport_licence_oui = input.sport_licence,
        sport_licence_non = not input.sport_licence,
        date_creation_compte = datetime.datetime.fromisoformat( input.date_creation_compte),
        historique_credits = input.historique_credits,
        risque_personnel = input.risque_personnel,
        score_credit = input.score_credit,
        montant_pret = input.montant_pret,
    )
    prediction_pipeline = model.predict([
        [loan.age, loan.niveau_etude, loan.situation_familiale_célibataire, situation_familiale_marié, ]
    ])
    return f"<p>Montant du prêt : {prediction_pipeline[0]}</p>" 