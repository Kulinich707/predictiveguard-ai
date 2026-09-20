from fastapi import FastAPI

from predictiveguard.api.routes import router

app = FastAPI(title="PredictiveGuard AI")
app.include_router(router)
