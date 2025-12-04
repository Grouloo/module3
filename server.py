from fastapi import FastAPI
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

server = FastAPI(docs_url=None, redoc_url="/documentation")
engine = create_engine("sqlite:///database.db")
session = Session(engine)