"""The web API.  Run locally with:   uvicorn app.main:app --reload
Then open http://127.0.0.1:8000/docs to try it in the browser."""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.predict import DefaultModel
from app.schemas import PredictRequest, PredictResponse

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = DefaultModel()          # load the model ONCE when the server starts, not per request
    yield
    state.clear()


app = FastAPI(
    title="AmEx default prediction API",
    description="Send a customer's raw monthly statements, get the probability of default.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    """Is the service up, and which model is it serving?"""
    model = state["model"]
    return {"status": "ok", "model_version": model.version, "n_features": len(model.feature_names)}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    model = state["model"]
    statements = [s.model_dump(mode="json") for s in request.statements]   # back to raw format (dates as text)
    [result] = model.predict(statements)
    return PredictResponse(**result, model_version=model.version)


from fastapi.responses import RedirectResponse


@app.get("/", include_in_schema=False)
def root():
    """Opening the bare address shows the interactive docs instead of a 404."""
    return RedirectResponse(url="/docs")