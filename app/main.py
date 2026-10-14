import tomllib
from pathlib import Path

from fastapi import FastAPI

from app.db import lifespan
from app.routes import api_router, router

# щас мув по копированию pyproject.toml чтобы попасть в docker

project_file = Path(__file__).resolve().parents[1] / "pyproject.toml"
# resolve - преобраз. в абсолютный путь и убирает ../

with project_file.open("rb") as file:
    project = tomllib.load(file)


app = FastAPI(
    title="Fake News API",
    version=project["project"]["version"],
    lifespan=lifespan,
)


app.include_router(router)
app.include_router(api_router)
