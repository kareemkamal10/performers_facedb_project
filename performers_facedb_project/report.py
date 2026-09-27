"""
Builds the final human-readable .txt report and the failed-downloads .csv.
"""
from __future__ import annotations

import csv
import os
import time


def write_failed_csv(failed_downloads: list, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "url", "error"])
        writer.writeheader()
        for row in failed_downloads:
            writer.writerow(row)


def write_txt_report(stats: dict, path: str, started_at: float) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    duration_min = (time.time() - started_at) / 60

    lines = [
        "=" * 62,
        "Performers Face-DB Build Pipeline - Final Report",
        "=" * 62,
        f"Run duration:                        {duration_min:.1f} minutes",
        "",
        "-- Merge stage (before download) --",
        f"Total input elements:                {stats['total_input_elements']}",
        f"Excluded (no urls at all):           {stats['excluded_no_url']}",
        f"Elements kept for download:          {stats['kept_elements']}",
        "",
        "-- Download stage --",
        f"Total URLs attempted:                {stats['total_urls_attempted']}",
        f"Successful downloads:                {stats['total_urls_downloaded']}",
        f"Failed downloads (after 3 retries):  {stats['total_urls_failed']}",
        "",
        "-- Face-embedding stage --",
        f"Elements with a final embedding:     {stats['elements_with_embedding']}",
        f"Excluded (no face detected):         {stats['elements_excluded_no_face']}",
        f"Excluded (all downloads failed):     {stats['elements_excluded_all_downloads_failed']}",
        f"Elements present in final DB:        {stats['final_elements']}",
        "",
        "=" * 62,
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
