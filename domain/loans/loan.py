from sqlalchemy import Column, Integer, Float, DateTime
from domain.abstract import Base

class Loan(Base):
    __tablename__ = "loans"
    id = Column(Integer, primary_key=True)
    age = Column(Integer)
    imc = Column(Float)
    # taille = Column(Float)
    # poids = Column(Float)
    #nationalité_francaise_oui = Column(Integer)
    #nationalité_francaise_non = Column(Integer)
    niveau_etude = Column(Integer)
    revenu_estime_mois = Column(Float)
    historique_credits = Column(Float)
    risque_personnel = Column(Float)
    date_creation_compte = Column(DateTime)
    score_credit = Column(Float)
    loyer_mensuel = Column(Float)
    sport_licence_oui = Column(Integer)
    sport_licence_non = Column(Integer)
    smoker_oui = Column(Integer)
    smoker_non = Column(Integer)
    situation_familiale_célibataire = Column(Integer)
    situation_familiale_marié = Column(Integer)
    situation_familiale_divorcé = Column(Integer)
    situation_familiale_veuf = Column(Integer)
    montant_pret = Column(Float)