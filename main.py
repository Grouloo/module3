from fastapi import FastAPI
# from loguru import logger
# from sys import stderr
from sqlalchemy import create_engine
from server import server
import domain.loans.api
from server import engine


# logger.add(stderr, format="{time} {level} {message}", filter="my_module", level="INFO")
# logger.add("logs/api.log")

# logger.debug("L'API est en cours de démarrage...")


@server.get("/")
async def homepage():
    return {"message": "Hello World"}

