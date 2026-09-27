"""
Orchestrates the whole run:
  - splits kept elements into fixed-size batches
  - while batch i is being face-embedded (a Node/Puppeteer subprocess), batch
    i+1 is already downloading in the background, so it's ready the moment
    batch i finishes
  - deletes each batch's images from /kaggle/temp the moment everything
    needed from them (the embeddings) has been safely recorded
  - checkpoints (and optionally uploads the checkpoint to HF) after every batch
"""
from __future__ import annotations

import gc
import json
import logging
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import config
import hf_sync
from checkpoint import load_checkpoint, save_checkpoint
from concurrency import get_embed_concurrency
from downloader import download_batch
from merge_dedupe import merge_and_filter

logger = logging.getLogger(__name__)


def _make_batches(elements: list, size: int):
    for i in range(0, len(elements), size):
        yield elements[i : i + size]


def _run_embedding_batch(downloaded_map: dict) -> dict:
    """
    downloaded_map: {id: [(url, local_path), ...]}
    Returns: {id: {"ok": bool, "embedding": [...512 floats], "num_images_used": n}}

    Runs the Node/Puppeteer worker ONCE for the whole batch - it launches a
    single headless Chromium and fans out across several tabs internally
    (config.MAX_EMBED_CONCURRENCY_CAP), rather than paying browser-launch
    cost per image.
    """
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
        "items": items,
    }
    manifest_path = os.path.join(config.KAGGLE_WORKING_DIR, "_embed_manifest.json")
    output_path = os.path.join(config.KAGGLE_WORKING_DIR, "_embed_output.json")
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    logger.info(
        "Face-embedding worker: %d elements, %d images total",
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


def run(input_json_path: str, use_hf: bool = True):
    started_at = time.time()
    os.makedirs(config.RESULT_OUTPUT_DIR, exist_ok=True)

    with open(input_json_path, "r", encoding="utf-8") as f:
        raw_elements = json.load(f)

    kept_elements, merge_stats = merge_and_filter(raw_elements)
    batches = list(_make_batches(kept_elements, config.BATCH_SIZE))
    logger.info("Split into %d batches of up to %d elements", len(batches), config.BATCH_SIZE)

    state = load_checkpoint()
    completed = set(state["completed_batches"])
    results_by_id = {r["id"]: r for r in state["results"]}
    excluded_by_id = {r["id"]: r for r in state.get("excluded", [])}
    failed_downloads = list(state["failed_downloads"])

    agg = {
        "total_urls_attempted": 0,
        "total_urls_downloaded": 0,
        "total_urls_failed": len(failed_downloads),
        "elements_with_embedding": 0,
        "elements_excluded_no_face": 0,
        "elements_excluded_all_downloads_failed": 0,
    }
    agg.update(state.get("stats", {}))

    # A single-worker pool just for kicking off/awaiting whole-batch downloads,
    # so "download batch i+1" can run in the background while we face-embed
    # batch i.
    downloader_pool = ThreadPoolExecutor(max_workers=1)

    def kick_off_download(batch_idx):
        if batch_idx >= len(batches):
            return None
        return downloader_pool.submit(download_batch, batches[batch_idx])

    # Find the first not-yet-completed batch so we don't re-download batches
    # that a previous, interrupted run already finished.
    first_pending = next((i for i in range(len(batches)) if i not in completed), len(batches))
    prefetch_future = kick_off_download(first_pending)

    for i, batch in enumerate(batches):
        if i in completed:
            continue

        logger.info("=== Batch %d/%d (%d elements) ===", i + 1, len(batches), len(batch))

        downloaded_map, batch_failed = prefetch_future.result()
        failed_downloads.extend(batch_failed)
        agg["total_urls_failed"] += len(batch_failed)

        # Kick off the NEXT batch's download now, so it overlaps this batch's
        # (CPU-bound) face-embedding step below.
        prefetch_future = kick_off_download(i + 1)

        batch_urls_attempted = sum(len(el["urls"]) for el in batch)
        agg["total_urls_attempted"] += batch_urls_attempted
        agg["total_urls_downloaded"] += sum(len(v) for v in downloaded_map.values())

        embed_results = _run_embedding_batch(downloaded_map)

        for el in batch:
            eid = el["id"]
            had_download = bool(downloaded_map.get(eid))
            r = embed_results.get(eid)
            if r and r.get("ok"):
                results_by_id[eid] = {
                    "id": eid,
                    "embedding": r["embedding"],
                    "num_images_used": r["num_images_used"],
                }
                agg["elements_with_embedding"] += 1
            elif had_download:
                excluded_by_id[eid] = {"id": eid, "reason": "no_successful_face_detection"}
                agg["elements_excluded_no_face"] += 1
            else:
                excluded_by_id[eid] = {"id": eid, "reason": "all_downloads_failed"}
                agg["elements_excluded_all_downloads_failed"] += 1

        completed.add(i)

        # Everything needed from this batch's files (the embeddings) is now
        # safely recorded -> delete the images to free space.
        for el in batch:
            shutil.rmtree(
                os.path.join(config.KAGGLE_TEMP_DIR, str(el["id"])), ignore_errors=True
            )
        gc.collect()

        state = {
            "completed_batches": sorted(completed),
            "results": list(results_by_id.values()),
            "excluded": list(excluded_by_id.values()),
            "failed_downloads": failed_downloads,
            "stats": agg,
        }
        save_checkpoint(state)
        if use_hf:
            try:
                hf_sync.upload_checkpoint()
            except Exception as e:  # noqa: BLE001 - don't kill the run over this
                logger.warning("Checkpoint upload failed (will retry next batch): %s", e)

    downloader_pool.shutdown(wait=True)

    return results_by_id, excluded_by_id, failed_downloads, agg, merge_stats, started_at
