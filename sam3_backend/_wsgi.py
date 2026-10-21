import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# label-studio-ml 1.0.9: a single shared model instance (needed so /setup and /predict agree)
os.environ.setdefault("LABEL_STUDIO_ML_BACKEND_V2", "true")
logging.basicConfig(level=logging.INFO)

from label_studio_ml.api import init_app  # noqa: E402
from sam3_model import SAM3Backend, load_model  # noqa: E402

load_model()  # load weights once at startup (fail fast if gated / no token)
# model_dir must be set: LS posts annotation events to /webhook and the SDK's job manager writes under it
MODEL_DIR = os.getenv("MODEL_DIR") or str(Path(__file__).resolve().parent / "model_dir")
app = init_app(model_class=SAM3Backend, model_dir=MODEL_DIR)



def _noop_webhook(event, data):
    """LS posts annotation events (ANNOTATION_CREATED/UPDATED, START_TRAINING ...). SAM3 is not trained, so
    acknowledge them without forking a job (forking a process that holds MPS/torch state is risky)."""
    logging.getLogger("sam3").info("LS event %s received: nothing to train, ignored", event)
    return {"job_id": None}


import label_studio_ml.api as _ls_api  # noqa: E402
_ls_api._manager.webhook = _noop_webhook

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("ML_BACKEND_PORT", "9090")))
