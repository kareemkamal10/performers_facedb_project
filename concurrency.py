"""
Auto-detects safe worker counts based on the machine's CPU, so the pipeline
adapts to whatever Kaggle session it happens to run on without manual tuning,
while never exceeding the hard caps in config.py (which protect against
starving the Kaggle session or overloading it).
"""
import os

import config


def get_download_workers() -> int:
    cpu = os.cpu_count() or 4
    # I/O-bound work: scale past core count, but capped lower than a
    # pure-download pipeline would use, since downloading now shares the
    # machine with the face-embedding step running at the same time.
    return min(cpu * 3, config.MAX_DOWNLOAD_WORKERS_CAP)


def get_embed_concurrency() -> int:
    cpu = os.cpu_count() or 4
    # Each unit of concurrency here is a full headless-Chrome tab doing face
    # detection + embedding - much heavier than a plain CPU worker, so this
    # stays deliberately low regardless of core count.
    return max(1, min(cpu - 1, config.MAX_EMBED_CONCURRENCY_CAP))
