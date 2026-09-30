"""
Converts face_db.json (produced by main.py, topped up by recovery.py) into a
LanceDB table - a disk-based, memory-mapped vector index that doesn't need
the whole embedding matrix resident in RAM to search, unlike a plain
brute-force-in-memory search server.

Fetches face_db.json fresh from the HF dataset, builds the LanceDB table
locally, creates a cosine-distance vector index on it, and uploads the whole
resulting LanceDB directory to its OWN folder on the dataset
(config.HF_LANCEDB_FOLDER_NAME) - kept separate from face_db_output/.

This is a standalone, one-off conversion: pure Python, no image downloading,
no Node/Puppeteer/Chrome needed. Run it any time after main.py / recovery.py
to refresh the LanceDB copy from the latest face_db.json.

Usage: paste kaggle_cell_lancedb.py into a Kaggle cell, or run
`python build_lancedb.py` locally with HF_DATASET_REPO_ID / HF_TOKEN set.
"""
from __future__ import annotations

import json
import logging
import os
import shutil

import lancedb
from lancedb.index import IvfPq

import config
import hf_sync

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("build_lancedb")


def main() -> None:
    use_hf = bool(config.HF_DATASET_REPO_ID and config.HF_TOKEN)
    if not use_hf:
        logger.error("HF_DATASET_REPO_ID / HF_TOKEN must be set.")
        raise SystemExit(1)

    logger.info("Fetching face_db.json from the HF dataset...")
    face_db_path = hf_sync.download_result_file("face_db.json")
    with open(face_db_path, "r", encoding="utf-8") as f:
        face_db = json.load(f)
    logger.info("Loaded %d embeddings", len(face_db))

    rows = [
        {
            "id": str(item["id"]),
            "vector": item["embedding"],
            "num_images_used": int(item.get("num_images_used", 0)),
        }
        for item in face_db
    ]

    if os.path.exists(config.LANCEDB_LOCAL_DIR):
        shutil.rmtree(config.LANCEDB_LOCAL_DIR)
    os.makedirs(os.path.dirname(config.LANCEDB_LOCAL_DIR), exist_ok=True)

    logger.info("Building LanceDB table at %s ...", config.LANCEDB_LOCAL_DIR)
    db = lancedb.connect(config.LANCEDB_LOCAL_DIR)
    table = db.create_table(config.LANCEDB_TABLE_NAME, data=rows, mode="overwrite")

    logger.info("Building the cosine-distance vector index (can take a bit)...")
    table.create_index("vector", config=IvfPq(distance_type="cosine"))

    row_count = table.count_rows()
    logger.info("LanceDB table built: %d rows", row_count)

    hf_sync.upload_folder_generic(config.LANCEDB_LOCAL_DIR, config.HF_LANCEDB_FOLDER_NAME)
    logger.info("Uploaded LanceDB output to the HF dataset under %s/", config.HF_LANCEDB_FOLDER_NAME)


if __name__ == "__main__":
    main()
