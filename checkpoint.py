"""
Batch-level checkpoint so a killed/restarted Kaggle session can resume
without re-downloading or re-processing batches that already finished.
Saved locally after every batch, and (when HF creds are set) mirrored to the
HF dataset so a fresh Kaggle session can pick it back up too.
"""
from __future__ import annotations

import json
import logging
import os

import config

logger = logging.getLogger(__name__)


def load_checkpoint() -> dict:
    if os.path.exists(config.CHECKPOINT_PATH):
        with open(config.CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "completed_batches": [],
        "results": [],           # [{"id":..., "embedding":[...512 floats], "num_images_used": n}]
        "excluded": [],          # [{"id":..., "reason": "..."}]
        "failed_downloads": [],  # [{"id":..., "url":..., "error":...}]
        "stats": {},
    }


def save_checkpoint(state: dict) -> None:
    os.makedirs(os.path.dirname(config.CHECKPOINT_PATH), exist_ok=True)
    tmp_path = config.CHECKPOINT_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)
    os.replace(tmp_path, config.CHECKPOINT_PATH)
    logger.info("Checkpoint saved (%d batches completed)", len(state["completed_batches"]))
