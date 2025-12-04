from sqlalchemy import Column, Integer, Float,  String
from domain.abstract import Base

class Loan(Base):
    __tablename__ = "loans"
    id = Column(Integer, primary_key=True, autoincrement=True)
    age = Column(Integer)
    imc = Column(Float)
    niveau_etude = Column(Integer)
    revenu_estime_mois = Column(Float)
    historique_credits = Column(Float)
    risque_personnel = Column(Float)
    date_creation_compte = Column(Integer)
    score_credit = Column(Float)
    loyer_mensuel = Column(Float)
    sport_licence = Column(String)
    smoker = Column(String)
    situation_familiale = Column(String)
    montant_pret = Column(Float)