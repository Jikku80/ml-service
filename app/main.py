import logging
import os
import sys
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

# Set up logging first
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add current directory to Python path (app folder)
current_dir = Path(__file__).parent  # This is the app folder
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Also add parent directory in case we need it
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

logger.info(f"Python path: {sys.path}")
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"Script location: {__file__}")
logger.info(f"App directory: {current_dir}")

# List contents of app directory
if current_dir.exists():
    app_contents = [f.name for f in current_dir.iterdir()]
    logger.info(f"App directory contents: {app_contents}")

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
    logger.error(f"Traceback: {traceback.format_exc()}")
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
    logger.error(f"Traceback: {traceback.format_exc()}")
    # Create dummy middleware functions
    async def verify_api_key():
        return True
    async def rate_limit_middleware(request, call_next):
        return await call_next(request)

# Check if server directory exists (should be in app folder)
server_path = current_dir / "server"
routers_path = server_path / "routers"
core_path = server_path / "core"
middleware_path = server_path / "middleware"

logger.info(f"Server directory exists: {server_path.exists()}")
logger.info(f"Routers directory exists: {routers_path.exists()}")
logger.info(f"Core directory exists: {core_path.exists()}")
logger.info(f"Middleware directory exists: {middleware_path.exists()}")

if routers_path.exists():
    router_files = list(routers_path.glob("*.py"))
    logger.info(f"Found router files: {[f.name for f in router_files]}")

# Check for __init__.py files
init_files_status = {
    "server/__init__.py": (server_path / "__init__.py").exists(),
    "server/routers/__init__.py": (routers_path / "__init__.py").exists(),
    "server/core/__init__.py": (core_path / "__init__.py").exists(),
    "server/middleware/__init__.py": (middleware_path / "__init__.py").exists(),
}
logger.info(f"__init__.py files status: {init_files_status}")

missing_init_files = [path for path, exists in init_files_status.items() if not exists]
if missing_init_files:
    logger.warning(f"Missing __init__.py files: {missing_init_files}")
    logger.warning("This will cause import failures. Please create these files.")

# Simple import test function
def test_import(module_path):
    try:
        __import__(module_path)
        return True, None
    except Exception as e:
        return False, str(e)

# Test basic server imports
logger.info("Testing basic server imports...")
for module in ["server", "server.routers", "server.core", "server.middleware"]:
    success, error = test_import(module)
    logger.info(f"Import test {module}: {'SUCCESS' if success else f'FAILED - {error}'}")

# Import routers with enhanced error handling
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
    ("supply", "server.routers.supply"),
    ("imagetoxl", "server.routers.imagetoxl"),
    ("dbconnect", "server.routers.dbconnect"),
    ("anamoly", "server.routers.anamoly"),
]

imported_routers = {}
failed_routers = {}

for router_name, module_path in routers_to_import:
    try:
        logger.info(f"Attempting to import {router_name} from {module_path}")
        module = __import__(module_path, fromlist=[router_name])
        
        # Check if router attribute exists
        if hasattr(module, 'router'):
            imported_routers[router_name] = getattr(module, 'router')
            logger.info(f"✓ Successfully imported {router_name} router")
        else:
            available_attrs = [attr for attr in dir(module) if not attr.startswith('_')]
            logger.error(f"✗ Router attribute not found in {router_name} module. Available attributes: {available_attrs}")
            failed_routers[router_name] = f"No 'router' attribute found. Available: {available_attrs}"
            
    except ImportError as e:
        logger.error(f"✗ Import error for {router_name}: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        failed_routers[router_name] = f"ImportError: {e}"
    except Exception as e:
        logger.error(f"✗ Unexpected error importing {router_name}: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        failed_routers[router_name] = f"Exception: {e}"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    logger.info(f"Starting FastAPI in {settings.ENVIRONMENT} environment")
    logger.info(f"Allowed origins: {settings.ALLOWED_ORIGINS}")
    logger.info(f"Successfully imported {len(imported_routers)} routers: {list(imported_routers.keys())}")
    if failed_routers:
        logger.warning(f"Failed to import {len(failed_routers)} routers: {list(failed_routers.keys())}")
        for name, error in failed_routers.items():
            logger.warning(f"  {name}: {error}")
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
        "failed_routers": list(failed_routers.keys()),
        "environment": settings.ENVIRONMENT,
        "total_attempted": len(routers_to_import),
        "successful_imports": len(imported_routers)
    }

@app.get("/debug")
async def debug_info():
    """Debug endpoint to help diagnose import issues"""
    return {
        "python_version": sys.version,
        "python_path": sys.path,
        "current_directory": os.getcwd(),
        "environment_variables": dict(os.environ),
        "server_directory_exists": (Path(__file__).parent / "server").exists(),
        "routers_directory_exists": (Path(__file__).parent / "server" / "routers").exists(),
        "imported_routers": list(imported_routers.keys()),
        "failed_routers": failed_routers,
        "available_files": {
            "app_root": [f for f in os.listdir(".") if not f.startswith(".")],
            "server": [f for f in os.listdir("server")] if os.path.exists("server") else [],
            "routers": [f for f in os.listdir("server/routers")] if os.path.exists("server/routers") else [],
            "core": [f for f in os.listdir("server/core")] if os.path.exists("server/core") else [],
            "middleware": [f for f in os.listdir("server/middleware")] if os.path.exists("server/middleware") else []
        },
        "init_files_status": {
            "server/__init__.py": os.path.exists("server/__init__.py"),
            "server/routers/__init__.py": os.path.exists("server/routers/__init__.py"),
            "server/core/__init__.py": os.path.exists("server/core/__init__.py"),
            "server/middleware/__init__.py": os.path.exists("server/middleware/__init__.py")
        }
    }

# Add middleware with better error handling
try:
    app.middleware("http")(rate_limit_middleware)
    logger.info("✓ Rate limiting middleware added")
except Exception as e:
    logger.error(f"✗ Failed to add rate limiting middleware: {e}")

try:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*", "X-API-Key"],
    )
    logger.info("✓ CORS middleware added")
except Exception as e:
    logger.error(f"✗ Failed to add CORS middleware: {e}")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "environment": settings.ENVIRONMENT,
        "imported_routers": list(imported_routers.keys()),
        "failed_routers": list(failed_routers.keys()),
        "port": os.environ.get("PORT", "8080"),
        "success_rate": f"{len(imported_routers)}/{len(routers_to_import)}"
    }

# Include routers that were successfully imported
for router_name, router in imported_routers.items():
    try:
        if router_name == "imagetoxl":
            # Special case for imagetoxl without auth
            app.include_router(router)
            logger.info(f"✓ Included {router_name} router (no auth)")
        else:
            app.include_router(router, dependencies=[Depends(verify_api_key)])
            logger.info(f"✓ Included {router_name} router (with auth)")
    except Exception as e:
        logger.error(f"✗ Failed to include {router_name} router: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")

# Graceful degradation message
if not imported_routers:
    logger.warning("⚠️  No routers were successfully imported! The API will run but with limited functionality.")
elif len(imported_routers) < len(routers_to_import):
    logger.warning(f"⚠️  Only {len(imported_routers)} out of {len(routers_to_import)} routers imported successfully.")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"Starting server on 0.0.0.0:{port}")
    logger.info(f"Final status: {len(imported_routers)} routers imported successfully")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")