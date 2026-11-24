"""Standalone dev server for the summarization module (does not touch service/app.py).

    uvicorn summarization_app:app --host 127.0.0.1 --port 8100 --reload
    # docs: http://localhost:8100/docs

CORS: the browser frontend (e.g. http://localhost:5500) calls this server directly.
Allowed origins come from SUMMARY_CORS_ORIGINS (comma-separated). Default "*" is fine
for local development; set explicit origins before exposing the service.
"""
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from summarization.api import router

app = FastAPI(title="Connect Interview AI - Summarization (standalone)")

_origins = [o.strip() for o in os.getenv("SUMMARY_CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
