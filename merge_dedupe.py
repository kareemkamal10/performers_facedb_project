"""
Stage 1: merge `image` + `source_images` into one deduped URL list per
element. Unlike the original dedup pipeline, elements left with only a
single unique URL are KEPT - there's no near-duplicate comparison happening
anymore, so one image is perfectly usable for building its embedding.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def collect_urls(element: dict) -> list:
    urls = []
    single = element.get("image")
    if single:
        urls.append(single)
    extra = element.get("source_images") or []
    if isinstance(extra, str):
        extra = [extra]
    urls.extend(u for u in extra if u)
    return urls


def merge_and_filter(elements: list) -> tuple[list, dict]:
    """
    Returns (kept_elements, stats).
    Each kept element is: {"id": ..., "urls": [unique urls, order preserved]}.
    Only elements with ZERO urls after merging are dropped (nothing to
    download at all); single-url elements are kept and go through the same
    pipeline as everyone else.
    """
    kept = []
    excluded_empty = 0

    for el in elements:
        raw_urls = collect_urls(el)
        seen = set()
        unique_urls = []
        for u in raw_urls:
            if u not in seen:
                seen.add(u)
                unique_urls.append(u)

        if len(unique_urls) == 0:
            excluded_empty += 1
            continue

        kept.append({"id": el.get("id"), "urls": unique_urls})

    stats = {
        "total_input_elements": len(elements),
        "excluded_no_url": excluded_empty,
        "kept_elements": len(kept),
    }
    logger.info("Merge/dedupe stage: %s", stats)
    return kept, stats
