from app.routers import segment, churn, trend, pricing, inventory, basket, demand, sentiment, cv, retention, ecom_churn, customerpick, salaryprediction, evaluate
from app.middleware.auth import verify_api_key, rate_limit_middleware
from app.core.config import settings

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

import logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    logger.info(f"Starting FastAPI in {settings.ENVIRONMENT} environment")
    logger.info(f"Allowed origins: {settings.ALLOWED_ORIGINS}")
    yield
    # Shutdown logic
    logger.info("Shutting down application")
    # Additional cleanup: close connections, release resources, etc.

app = FastAPI(
    title="ML Predictions API",
    description="API for machine learning predictions",
    version="1.0.0",
    lifespan=lifespan)

app.middleware("http")(rate_limit_middleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,  # Allow requests from React
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*", "X-API-Key"],  # Allow all headers
    # allow_headers=["Authorization", "Content-Type"],
)

@app.get("/health")
def health_check():
    return {"status": "healthy", 
        "environment": settings.ENVIRONMENT}


app.include_router(segment.router, dependencies=[Depends(verify_api_key)])
app.include_router(churn.router, dependencies=[Depends(verify_api_key)])
app.include_router(trend.router, dependencies=[Depends(verify_api_key)])
app.include_router(pricing.router, dependencies=[Depends(verify_api_key)])
app.include_router(inventory.router, dependencies=[Depends(verify_api_key)])
app.include_router(basket.router, dependencies=[Depends(verify_api_key)])
app.include_router(demand.router, dependencies=[Depends(verify_api_key)])
app.include_router(sentiment.router, dependencies=[Depends(verify_api_key)])
app.include_router(cv.router, dependencies=[Depends(verify_api_key)])
app.include_router(retention.router, dependencies=[Depends(verify_api_key)])
app.include_router(ecom_churn.router, dependencies=[Depends(verify_api_key)])
app.include_router(customerpick.router, dependencies=[Depends(verify_api_key)])
app.include_router(salaryprediction.router, dependencies=[Depends(verify_api_key)])
app.include_router(evaluate.router, dependencies=[Depends(verify_api_key)])
