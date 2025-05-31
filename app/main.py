import logging
import os
from contextlib import asynccontextmanager

# Set up logging first
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import FastAPI first
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

logger.info("Starting ML Service...")

# Try importing core dependencies first
try:
    from server.core.config import settings
    logger.info("Successfully imported settings")
except ImportError as e:
    logger.error(f"Failed to import settings: {e}")
    # Create fallback settings
    class FallbackSettings:
        LOG_LEVEL = "INFO"
        ENVIRONMENT = "production"
        ALLOWED_ORIGINS = ["*"]
    settings = FallbackSettings()

try:
    from server.middleware.auth import verify_api_key, rate_limit_middleware
    logger.info("Successfully imported auth middleware")
except ImportError as e:
    logger.error(f"Failed to import auth middleware: {e}")
    # Create dummy middleware functions
    async def verify_api_key():
        return True
    async def rate_limit_middleware(request, call_next):
        return await call_next(request)

# Import routers with error handling
routers_to_import = [
    ("segment", "server.routers.segment"),
    ("churn", "server.routers.churn"),
    ("trend", "server.routers.trend"),
    ("pricing", "server.routers.pricing"),
    ("inventory", "server.routers.inventory"),
    ("basket", "server.routers.basket"),
    ("demand", "server.routers.demand"),
    ("sentiment", "server.routers.sentiment"),
    ("cv", "server.routers.cv"),
    ("retention", "server.routers.retention"),
    ("ecom_churn", "server.routers.ecom_churn"),
    ("customerpick", "server.routers.customerpick"),
    ("salaryprediction", "server.routers.salaryprediction"),
    ("evaluate", "server.routers.evaluate"),
    ("imagetoxl", "server.routers.imagetoxl"),
    ("dbconnect", "server.routers.dbconnect"),
]

imported_routers = {}
for router_name, module_path in routers_to_import:
    try:
        module = __import__(module_path, fromlist=[router_name])
        imported_routers[router_name] = getattr(module, 'router')
        logger.info(f"Successfully imported {router_name} router")
    except ImportError as e:
        logger.error(f"Failed to import {router_name} router: {e}")
    except AttributeError as e:
        logger.error(f"Router not found in {router_name} module: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    logger.info(f"Starting FastAPI in {settings.ENVIRONMENT} environment")
    logger.info(f"Allowed origins: {settings.ALLOWED_ORIGINS}")
    logger.info(f"Successfully imported {len(imported_routers)} routers")
    yield
    # Shutdown logic
    logger.info("Shutting down application")

app = FastAPI(
    title="ML Predictions API",
    description="API for machine learning predictions",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    return {
        "message": "ML Predictions API is running",
        "available_routers": list(imported_routers.keys()),
        "environment": settings.ENVIRONMENT
    }

# Add middleware
try:
    app.middleware("http")(rate_limit_middleware)
    logger.info("Rate limiting middleware added")
except Exception as e:
    logger.error(f"Failed to add rate limiting middleware: {e}")

try:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*", "X-API-Key"],
    )
    logger.info("CORS middleware added")
except Exception as e:
    logger.error(f"Failed to add CORS middleware: {e}")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "environment": settings.ENVIRONMENT,
        "imported_routers": list(imported_routers.keys()),
        "port": os.environ.get("PORT", "8080")
    }

# Include routers that were successfully imported
for router_name, router in imported_routers.items():
    try:
        if router_name == "imagetoxl":
            # Special case for imagetoxl without auth
            app.include_router(router)
            logger.info(f"Included {router_name} router (no auth)")
        else:
            app.include_router(router, dependencies=[Depends(verify_api_key)])
            logger.info(f"Included {router_name} router (with auth)")
    except Exception as e:
        logger.error(f"Failed to include {router_name} router: {e}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"Starting server on 0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")