"""
Recovery pass: re-attempts face-embedding for elements sitting in
excluded.json with reason "no_successful_face_detection" (images downloaded
fine, but the detector found no face on the first pass - typically because
the face is small relative to the whole frame, e.g. full-body shots).

Re-downloads just those elements' images (they were deleted after the main
run, per the pipeline's cleanup step), retries detection with a lower
confidence threshold PLUS progressive top-crops (see build_page.html's
processImageRecovery), and merges any newly successful embeddings into
face_db.json - fetched fresh from the HF dataset, updated, and pushed back.

This is a separate, smaller entry point from main.py / pipeline.py (which is
for the full first-pass build over all ~118k elements) - run it any time
after a full run to top up the DB without re-processing everything that
already succeeded.

Usage: paste kaggle_cell_recovery.py into a Kaggle cell (same repo, same HF
dataset), or run `python recovery.py` locally with HF_DATASET_REPO_ID /
HF_TOKEN set and performers_data.json present.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess

import config
import hf_sync
from concurrency import get_download_workers, get_embed_concurrency
from downloader import download_batch
from merge_dedupe import merge_and_filter

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("recovery")

# Only this reason is targeted by default - "all_downloads_failed" elements
# would almost certainly fail again for the same reason (dead links), so
# retrying them isn't worth the time unless explicitly asked for.
TARGET_REASONS = {"no_successful_face_detection"}


def _run_embedding_recovery(downloaded_map: dict) -> dict:
    items = [
        {"id": eid, "paths": [p for _, p in pairs]}
        for eid, pairs in downloaded_map.items()
        if pairs
    ]
    if not items:
        return {}

    manifest = {
        "baseDir": config.FACE_EMBED_DIR,
        "concurrency": get_embed_concurrency(),
        "recoveryMode": True,
        "minFaceDetectionConfidence": config.RECOVERY_MIN_FACE_CONFIDENCE,
        "items": items,
    }
    manifest_path = os.path.join(config.KAGGLE_WORKING_DIR, "_recovery_manifest.json")
    output_path = os.path.join(config.KAGGLE_WORKING_DIR, "_recovery_output.json")
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    logger.info(
        "Recovery face-embedding worker: %d elements, %d images total",
        len(items),
        sum(len(it["paths"]) for it in items),
    )
    subprocess.run(
        ["node", config.NODE_SCRIPT_PATH, manifest_path, output_path],
        check=True,
        cwd=config.FACE_EMBED_DIR,
    )
    with open(output_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    use_hf = bool(config.HF_DATASET_REPO_ID and config.HF_TOKEN)
    if not use_hf:
        logger.error("HF_DATASET_REPO_ID / HF_TOKEN must be set for the recovery pass.")
        raise SystemExit(1)

    if not os.path.exists(config.EMBED_MODEL_PATH) or not os.path.exists(
        config.FACE_LANDMARKER_TASK_PATH
    ):
        logger.info("Face-embedding models not found locally - fetching from HF dataset...")
        hf_sync.download_face_models_if_needed()

    if not os.path.exists(config.INPUT_JSON):
        logger.info("performers_data.json not found locally - fetching from HF dataset...")
        hf_sync.download_input_json(config.INPUT_JSON)

    logger.info("Fetching current face_db.json and excluded.json from the HF dataset...")
    face_db_path = hf_sync.download_result_file("face_db.json")
    excluded_path = hf_sync.download_result_file("excluded.json")

    with open(face_db_path, "r", encoding="utf-8") as f:
        face_db = json.load(f)
    with open(excluded_path, "r", encoding="utf-8") as f:
        excluded = json.load(f)

    target_ids = {e["id"] for e in excluded if e.get("reason") in TARGET_REASONS}
    logger.info("Targeting %d excluded elements (reasons: %s)", len(target_ids), TARGET_REASONS)
    if not target_ids:
        logger.info("Nothing to recover - done.")
        return

    with open(config.INPUT_JSON, "r", encoding="utf-8") as f:
        raw_elements = json.load(f)
    kept_elements, _ = merge_and_filter(raw_elements)
    targets = [el for el in kept_elements if el["id"] in target_ids]
    logger.info("%d of those elements still have URLs to retry", len(targets))

    # This is a standalone, one-off pass - nothing else is running at the
    # same time, so it uses the fast "first batch" download pace throughout.
    workers = get_download_workers(first_batch=True)
    downloaded_map, failed = download_batch(targets, workers)
    logger.info(
        "Recovery download: %d elements got at least one image, %d urls failed",
        sum(1 for v in downloaded_map.values() if v),
        len(failed),
    )

    embed_results = _run_embedding_recovery(downloaded_map)

    excluded_by_id = {e["id"]: e for e in excluded}
    recovered = 0

    for el in targets:
        eid = el["id"]
        r = embed_results.get(eid)
        if r and r.get("ok"):
            face_db.append(
                {
                    "id": eid,
                    "embedding": r["embedding"],
                    "num_images_used": r["num_images_used"],
                }
            )
            excluded_by_id.pop(eid, None)
            recovered += 1
        # else: stays excluded exactly as it already was - nothing to change.

    remaining_excluded = list(excluded_by_id.values())
    logger.info("Recovered %d / %d elements", recovered, len(targets))

    os.makedirs(config.RESULT_OUTPUT_DIR, exist_ok=True)
    face_db_out = os.path.join(config.RESULT_OUTPUT_DIR, "face_db.json")
    excluded_out = os.path.join(config.RESULT_OUTPUT_DIR, "excluded.json")
    with open(face_db_out, "w", encoding="utf-8") as f:
        json.dump(face_db, f, ensure_ascii=False)
    with open(excluded_out, "w", encoding="utf-8") as f:
        json.dump(remaining_excluded, f, ensure_ascii=False, indent=2)

    # Clean up this small set's downloaded images now that we're done with them.
    for el in targets:
        shutil.rmtree(os.path.join(config.KAGGLE_TEMP_DIR, str(el["id"])), ignore_errors=True)

    hf_sync.upload_file_generic(face_db_out, f"{config.HF_RESULT_FOLDER_NAME}/face_db.json")
    hf_sync.upload_file_generic(excluded_out, f"{config.HF_RESULT_FOLDER_NAME}/excluded.json")
    logger.info("Updated face_db.json and excluded.json uploaded to the HF dataset.")


if __name__ == "__main__":
    main()
