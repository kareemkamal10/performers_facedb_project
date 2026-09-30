"""
Thin wrapper around huggingface_hub for what this project needs from the
private HF dataset:
  - downloading performers_data.json (if not already dropped in project root)
  - downloading the two face-embedding models, if not already placed locally
  - mirroring the checkpoint after every batch (cheap resume insurance)
  - uploading the final face_db_output/ folder at the end of the run
"""
from __future__ import annotations

import logging
import os

from huggingface_hub import hf_hub_download, upload_file, upload_folder

import config

logger = logging.getLogger(__name__)


def _require_creds() -> None:
    if not config.HF_DATASET_REPO_ID or not config.HF_TOKEN:
        raise RuntimeError(
            "HF_DATASET_REPO_ID / HF_TOKEN are not set - fill them in the Kaggle cell."
        )


def download_input_json(local_path: str = None) -> str:
    _require_creds()
    local_path = local_path or config.INPUT_JSON
    path = hf_hub_download(
        repo_id=config.HF_DATASET_REPO_ID,
        repo_type="dataset",
        filename="performers_data.json",
        token=config.HF_TOKEN,
        local_dir=os.path.dirname(local_path),
    )
    return path


def download_face_models_if_needed() -> None:
    """Fetches w600k_mbf.onnx and face_landmarker.task from the dataset root,
    unless they're already present locally (e.g. placed manually)."""
    _require_creds()
    os.makedirs(config.FACE_EMBED_DIR, exist_ok=True)
    if not os.path.exists(config.EMBED_MODEL_PATH):
        hf_hub_download(
            repo_id=config.HF_DATASET_REPO_ID,
            repo_type="dataset",
            filename="w600k_mbf.onnx",
            token=config.HF_TOKEN,
            local_dir=config.FACE_EMBED_DIR,
        )
        logger.info("Fetched w600k_mbf.onnx from the HF dataset")
    if not os.path.exists(config.FACE_LANDMARKER_TASK_PATH):
        hf_hub_download(
            repo_id=config.HF_DATASET_REPO_ID,
            repo_type="dataset",
            filename="face_landmarker.task",
            token=config.HF_TOKEN,
            local_dir=config.FACE_EMBED_DIR,
        )
        logger.info("Fetched face_landmarker.task from the HF dataset")


def download_result_file(filename: str) -> str:
    """Fetches a single file that's already sitting under the result folder
    on the dataset (e.g. face_db.json, excluded.json from a finished run)."""
    _require_creds()
    return hf_hub_download(
        repo_id=config.HF_DATASET_REPO_ID,
        repo_type="dataset",
        filename=f"{config.HF_RESULT_FOLDER_NAME}/{filename}",
        token=config.HF_TOKEN,
    )


def download_checkpoint_if_exists():
    _require_creds()
    try:
        return hf_hub_download(
            repo_id=config.HF_DATASET_REPO_ID,
            repo_type="dataset",
            filename="checkpoint.json",
            token=config.HF_TOKEN,
        )
    except Exception:  # noqa: BLE001 - no checkpoint on the dataset yet
        return None


def upload_checkpoint() -> None:
    _require_creds()
    if not os.path.exists(config.CHECKPOINT_PATH):
        return
    upload_file(
        path_or_fileobj=config.CHECKPOINT_PATH,
        path_in_repo="checkpoint.json",
        repo_id=config.HF_DATASET_REPO_ID,
        repo_type="dataset",
        token=config.HF_TOKEN,
        commit_message="Update pipeline checkpoint",
    )
    logger.info("Checkpoint uploaded to HF dataset")


def upload_result_output() -> None:
    """Uploads face_db_output/ to its OWN folder on the dataset, separate
    from the older dedup pipeline's result_output/ folder."""
    _require_creds()
    upload_folder(
        folder_path=config.RESULT_OUTPUT_DIR,
        path_in_repo=config.HF_RESULT_FOLDER_NAME,
        repo_id=config.HF_DATASET_REPO_ID,
        repo_type="dataset",
        token=config.HF_TOKEN,
        commit_message="Upload final face_db_output",
    )
    logger.info("face_db_output uploaded to HF dataset")


def upload_folder_generic(local_dir: str, path_in_repo: str) -> None:
    """Generic whole-folder uploader for anything that doesn't fit
    upload_result_output()'s fixed face_db_output/ shape (e.g. the LanceDB
    output, which is its own separate folder on the dataset)."""
    _require_creds()
    upload_folder(
        folder_path=local_dir,
        path_in_repo=path_in_repo,
        repo_id=config.HF_DATASET_REPO_ID,
        repo_type="dataset",
        token=config.HF_TOKEN,
        commit_message=f"Upload {path_in_repo}",
    )
    logger.info("%s uploaded to HF dataset", path_in_repo)


def upload_file_generic(local_path: str, path_in_repo: str) -> None:
    """Generic single-file uploader for anything that doesn't fit the
    whole-folder upload_result_output() shape."""
    _require_creds()
    if not os.path.exists(local_path):
        return
    upload_file(
        path_or_fileobj=local_path,
        path_in_repo=path_in_repo,
        repo_id=config.HF_DATASET_REPO_ID,
        repo_type="dataset",
        token=config.HF_TOKEN,
        commit_message=f"Upload {path_in_repo}",
    )
    logger.info("%s uploaded to HF dataset", path_in_repo)
