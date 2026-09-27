"""
Auto-detects safe worker counts based on the machine's CPU, so the pipeline
adapts to whatever Kaggle session it happens to run on without manual tuning,
while never exceeding the hard caps in config.py (which protect against
starving the Kaggle session or overloading it).
"""
import os

import config


def get_download_workers(first_batch: bool = False) -> int:
    # I/O-bound work isn't limited by CPU core count - threads mostly sleep
    # waiting on the network. The first batch has nothing else running yet,
    # so it uses many more connections than every later batch, which
    # downloads WHILE the previous batch's CPU-heavy face-embedding step is
    # running.
    return (
        config.DOWNLOAD_WORKERS_FIRST_BATCH
        if first_batch
        else config.DOWNLOAD_WORKERS_OVERLAPPED
    )


def get_embed_concurrency() -> int:
    cpu = os.cpu_count() or 4
    # Each unit of concurrency here is a full headless-Chrome tab doing face
    # detection + embedding - much heavier than a plain CPU worker, so this
    # stays deliberately low regardless of core count.
    return max(1, min(cpu - 1, config.MAX_EMBED_CONCURRENCY_CAP))
