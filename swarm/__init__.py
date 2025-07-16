"""
SWARM CORE PACKAGE INITIALIZATION
Version: 2.0.0
Created: 2025-07-17
Author: Muhammad-Fauzan22
"""

# Package metadata
__version__ = "2.0.0"
__author__ = "Muhammad-Fauzan22"
__license__ = "MIT"
__status__ = "Production"

# Core imports for easy access
from .core import Orchestrator
from .agents import (
    MetaPlanner,
    Researcher,
    Scraper,
    DataCleaner,
    Healer,
    ModelTrainer,
    ModelEvaluator
)
from .storage import MongoDBManager, GoogleDriveManager, HFCacheManager
from .cloud import AzureDeployer, ServerlessTrigger, AutoScaler
from .utils import Logger, RetryHandler, SecretManager, CostOptimizer

# Initialize global components
logger = Logger(name="SwarmCore")
secret_manager = SecretManager()

# Package initialization message
logger.info(f"Initializing Swarm Core v{__version__}")

# Environment check
def check_environment():
    """Verify required environment variables"""
    required_envs = [
        "MONGO_URI",
        "GDRIVE_FOLDER_ID",
        "HF_TOKEN"
    ]
    
    missing = [env for env in required_envs if env not in os.environ]
    
    if missing:
        logger.critical(f"Missing environment variables: {', '.join(missing)}")
        raise EnvironmentError("Critical environment variables not set")
    
    logger.info("Environment verification passed")

# Initialize only when package is actually imported
if "SWARM_INITIALIZED" not in globals():
    import os
    import sys
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Add package to path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # Perform environment check
    try:
        check_environment()
        SWARM_INITIALIZED = True
    except Exception as e:
        logger.error(f"Initialization failed: {str(e)}")
        SWARM_INITIALIZED = False

# Package-level functions
def get_version():
    return __version__

def get_components():
    """Return available components in the package"""
    return {
        "agents": ["MetaPlanner", "Researcher", "Scraper", "DataCleaner", "Healer", "ModelTrainer"],
        "storage": ["MongoDBManager", "GoogleDriveManager", "HFCacheManager"],
        "cloud": ["AzureDeployer", "ServerlessTrigger", "AutoScaler"],
        "utils": ["Logger", "RetryHandler", "SecretManager", "CostOptimizer"]
    }

# Shortcut for common operations
def create_orchestrator():
    """Factory method for creating orchestrator instance"""
    return Orchestrator(
        planner=MetaPlanner(),
        researcher=Researcher(),
        scraper=Scraper(),
        cleaner=DataCleaner(),
        healer=Healer(),
        trainer=ModelTrainer(),
        db=MongoDBManager(),
        storage=GoogleDriveManager()
    )

# Clean exit handler
def shutdown_handler(signal, frame):
    logger.warning("Shutdown signal received. Terminating swarm...")
    # Add cleanup operations here
    sys.exit(0)

# Register signal handlers
if SWARM_INITIALIZED:
    import signal
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    logger.debug("Signal handlers registered")

# Package initialization complete
if SWARM_INITIALIZED:
    logger.info("Swarm package initialized successfully")
else:
    logger.error("Swarm package initialization incomplete")
