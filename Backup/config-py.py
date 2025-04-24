"""
Configuration settings for the ML Platform.
"""

import os
from pathlib import Path

# Application settings
APP_NAME = "Unified ML Platform"
VERSION = "2.0.0"
AUTHOR = "ML Platform Team"

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data_files"
MODELS_DIR = BASE_DIR / "saved_models"
PLUGINS_DIR = BASE_DIR / "plugins" / "installed"

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLUGINS_DIR, exist_ok=True)

# Data settings
MAX_UPLOAD_SIZE_MB = 200
ALLOWED_EXTENSIONS = ['.csv', '.xlsx', '.parquet', '.json', '.pkl', '.txt']
SAMPLE_DATA_ROWS = 5

# Model settings
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
CV_FOLDS = 5
MAX_TUNING_ITERATIONS = 50

# Plugin settings
PLUGIN_ENABLED = True
PLUGIN_AUTO_DISCOVERY = True

# UI settings
UI_THEME = "light"  # Options: light, dark
SIDEBAR_WIDTH = 300
MAX_PLOT_HEIGHT = 600
