"""Shared constants for local model service."""

from __future__ import annotations

import os
from pathlib import Path

from helix.constants import HELIX_HOME, SERVICES_ROOT

# -- Service identity --------------------------------------------------------

SERVICE_NAME = "local-model-service"

# -- Disk layout -------------------------------------------------------------
#
#   ~/.helix/services/local-model-service/   ← SERVICES_ROOT / SERVICE_NAME
#     models/                                ← MODELS_SUBDIR
#       <repo_id (/ → --)>/                  ← downloaded model weights
#     venvs/                                 ← VENVS_SUBDIR
#       <backend>/                           ← per-backend Python venv
#     sources/                               ← SOURCES_SUBDIR
#       <skill-name>/<commit>/               ← downloaded runtime source files
#     service.json                           ← SERVICE_STATE_FILE
#

MODELS_SUBDIR = "models"
VENVS_SUBDIR = "venvs"
SOURCES_SUBDIR = "sources"
SERVICE_STATE_FILE = "state.json"
SERVICE_ROOT = SERVICES_ROOT / SERVICE_NAME


def sources_path(skill_name: str, commit: str) -> Path:
    """Return the path to a skill's downloaded source files."""
    return SERVICE_ROOT / SOURCES_SUBDIR / skill_name / commit

# -- Backend modes -----------------------------------------------------------

FAKE_BACKEND = "fake"
DEFAULT_BACKEND_MODE = os.environ.get(
    "HELIX_LOCAL_MODEL_SERVICE_BACKEND", "real"
).strip().lower() or "real"

# -- Task types --------------------------------------------------------------

TASK_TEXT_TO_IMAGE = "text_to_image"
TASK_TEXT_TO_VIDEO = "text_to_video"
TASK_TEXT_IMAGE_TO_VIDEO = "text_image_to_video"
TASK_TEXT_TO_AUDIO = "text_to_audio"

# -- Timeouts ----------------------------------------------------------------

DEFAULT_IDLE_SECONDS = 600
HTTP_TIMEOUT_SECONDS = int(os.environ.get("HELIX_LOCAL_MODEL_SERVICE_HTTP_TIMEOUT", "30"))
STARTUP_TIMEOUT_SECONDS = int(os.environ.get("HELIX_LOCAL_MODEL_SERVICE_STARTUP_TIMEOUT", "20"))
WORKER_REQUEST_TIMEOUT_SECONDS = int(
    os.environ.get("HELIX_LOCAL_MODEL_SERVICE_WORKER_TIMEOUT", "1200")
)

# -- Misc --------------------------------------------------------------------

COORDINATOR_HEALTH_PATH = "/health"
DEFAULT_AUDIO_SAMPLE_RATE = 24000
