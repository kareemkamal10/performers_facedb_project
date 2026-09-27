"""
Stage 2: concurrent, retrying downloader. Downloads every url for every
element in a batch into KAGGLE_TEMP_DIR/<id>/<index><ext>, keeping the
(url, local_path) pairing so later stages can map a kept file back to its
original URL.

A small random delay is added before every attempt (config.DOWNLOAD_STAGGER_*)
so a large worker pool doesn't fire hundreds of requests in the same instant -
this is what keeps the download pace steady rather than bursty.
"""
from __future__ import annotations

import logging
import mimetypes
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

import config

logger = logging.getLogger(__name__)


def _guess_ext(url: str, content_type) -> str:
    ext = os.path.splitext(url.split("?")[0])[1]
    if ext and len(ext) <= 5:
        return ext
    if content_type:
        guessed = mimetypes.guess_extension(content_type.split(";")[0].strip())
        if guessed:
            return guessed
    return ".jpg"


def _download_one(element_id, url: str, dest_dir: str, index: int) -> dict:
    time.sleep(
        random.uniform(
            config.DOWNLOAD_STAGGER_MIN_SECONDS, config.DOWNLOAD_STAGGER_MAX_SECONDS
        )
    )
    last_err = None
    for attempt in range(1, config.DOWNLOAD_MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=config.DOWNLOAD_TIMEOUT_SECONDS, stream=True)
            resp.raise_for_status()
            ext = _guess_ext(url, resp.headers.get("Content-Type"))
            path = os.path.join(dest_dir, f"{index:04d}{ext}")
            with open(path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
            return {"ok": True, "id": element_id, "url": url, "path": path}
        except Exception as e:  # noqa: BLE001 - any network/IO error should retry
            last_err = str(e)
            if attempt < config.DOWNLOAD_MAX_RETRIES:
                time.sleep(config.DOWNLOAD_RETRY_BACKOFF_BASE * (2 ** (attempt - 1)))
    return {"ok": False, "id": element_id, "url": url, "error": last_err}


def download_batch(batch_elements: list, workers: int) -> tuple[dict, list]:
    """
    batch_elements: list of {"id":..., "urls":[...]}
    workers: concurrency level to use for this call - the caller (pipeline.py)
      decides this via concurrency.get_download_workers(first_batch=...),
      since it's the one that knows whether anything else is running at the
      same time.
    Returns:
      downloaded_map: {id: [(url, local_path), ...]}  (only successful ones)
      failed_list:    [{"id":..., "url":..., "error":...}, ...]
    """
    temp_root = config.KAGGLE_TEMP_DIR
    os.makedirs(temp_root, exist_ok=True)

    tasks = []
    for el in batch_elements:
        dest_dir = os.path.join(temp_root, str(el["id"]))
        os.makedirs(dest_dir, exist_ok=True)
        for idx, url in enumerate(el["urls"]):
            tasks.append((el["id"], url, dest_dir, idx))

    downloaded_map = {el["id"]: [] for el in batch_elements}
    failed_list = []

    logger.info("Downloading %d urls with %d workers", len(tasks), workers)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_download_one, *t) for t in tasks]
        for fut in as_completed(futures):
            result = fut.result()
            if result["ok"]:
                downloaded_map[result["id"]].append((result["url"], result["path"]))
            else:
                failed_list.append(
                    {"id": result["id"], "url": result["url"], "error": result["error"]}
                )

    return downloaded_map, failed_list
