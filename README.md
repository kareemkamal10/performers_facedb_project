# Performers Face-DB Build Pipeline

Merges each performer's `image` + `source_images` URLs, downloads them, runs
each surviving image through the **exact same** face detection + alignment +
embedding code used by the browser search page (via a real headless Chrome
tab, not a Python re-implementation), and produces a final `face_db.json`
containing one embedding per performer — built to run on Kaggle against
~118k elements without blowing up the session or the disk.

This is a fork of the original dedup pipeline, with the near-duplicate
removal stage replaced by the face-embedding stage. See "What changed from
the dedup pipeline" below if you're comparing the two.

## Why a real browser instead of Python + GPU

Face detection/alignment/embedding done in Python (even matching the same
models) does not produce byte-identical results to the same models running
in a browser via `onnxruntime-web` + MediaPipe's JS build — different image
decoders, different canvas/affine-warp implementations, different floating
point paths. That mismatch previously showed up as ~90% similarity for an
image that should score ~100% against itself.

The fix: build the DB using a real (headless) Chromium tab running the exact
same JS the search page runs, via Puppeteer. This is why GPU/WebGPU is
deliberately **not** used anywhere in this pipeline — WebGPU results vary by
GPU/driver, which would reintroduce the same mismatch. Everything runs on
CPU/WASM, in both the build step and the search page.

## What it does, stage by stage

1. **Merge** (`merge_dedupe.py`) — combines `image` + `source_images` into
   one deduped URL list per element. Elements with **zero** URLs are
   dropped; single-URL elements are **kept** (there's no duplicate
   comparison anymore, so one image is enough to build an embedding).

2. **Batching** — kept elements are split into fixed batches of **10,000**
   (`config.BATCH_SIZE`).

3. **Pipelined download + embedding** (`pipeline.py`) — while batch *N* is
   being face-embedded, batch *N+1* is already downloading in the
   background, so it's ready the moment batch *N* finishes. Images download
   straight to `/kaggle/temp` (never `/kaggle/working`).

4. **Downloading** (`downloader.py`) — each URL gets up to 3 attempts with
   backoff, plus a small random delay before every attempt so a large
   worker pool doesn't fire requests in a single burst (kept steady, not
   too fast / not too slow). Concurrency auto-scales to CPU cores with a
   conservative cap (`config.MAX_DOWNLOAD_WORKERS_CAP`).

5. **Face-embedding** (`face_embed/embed_batch.js`, invoked once per batch
   by `pipeline.py`) — launches one headless Chromium, opens several tabs in
   parallel (`config.MAX_EMBED_CONCURRENCY_CAP`, also auto-scaled but kept
   low since each tab is heavy), and runs every downloaded image through
   `build_page.html`'s `processImage()` — the same MediaPipe FaceLandmarker
   + w600k_mbf.onnx alignment/embedding code the browser search page uses.
   - An element with **multiple** successful images gets the **average** of
     their (unit-normalized) embeddings, re-normalized.
   - An element with **one** successful image just uses that one.
   - An element where **no** image produced a usable face is excluded.
   - Every embedding component is stored as **float16**-precision.

6. **Cleanup** — once a batch's embeddings are safely recorded, that batch's
   images are deleted from `/kaggle/temp` immediately.

7. **Checkpointing** (`checkpoint.py` + `hf_sync.py`) — after every batch, a
   checkpoint (completed batches, embeddings so far, excluded elements,
   failed downloads, stats) is saved locally and, if HF credentials are
   set, mirrored to the HF dataset. A killed/restarted Kaggle session
   resumes from there — no re-downloading, no re-embedding.

8. **Final outputs**, written to `face_db_output/` and (if HF credentials
   are set) uploaded to the HF dataset **under its own `face_db_output/`
   folder** — kept separate from the older dedup pipeline's `result_output/`:
   - `face_db.json` — `[{"id": ..., "embedding": [...512 floats], "num_images_used": n}, ...]`
   - `excluded.json` — `[{"id": ..., "reason": "no_successful_face_detection" | "all_downloads_failed"}, ...]`
   - `report.txt` — full run stats.
   - `failed_downloads.csv` — every URL that failed after 3 retries.

## Recovery pass (topping up excluded.json)

Some elements land in `excluded.json` with reason
`no_successful_face_detection` even though the image is fine — usually one
of two patterns: the face is small relative to the whole frame (full-body
shots, or just a torso-up shot where the face isn't very large), or the
face is at a steep angle/profile rather than roughly frontal.

`recovery.py` targets exactly those elements: it re-downloads just their
images (deleted after the main run), retries detection with a lower
confidence threshold (`config.RECOVERY_MIN_FACE_CONFIDENCE` /
`RECOVERY_MIN_FACE_PRESENCE_CONFIDENCE`) against several re-rendered
versions of each image (see `build_page.html`'s `processImageRecovery`) —
the image upscaled 1.5x/2x (helps a small-but-centered face), then
progressively tighter top-crops (helps full-body shots specifically) —
stopping at the first one that finds a face. Any newly successful
embeddings are merged into `face_db.json`, fetched fresh from the dataset,
updated, and pushed back. Elements with reason `all_downloads_failed` are
left alone (dead links almost always fail again).

**Known limitation**: a steep side-profile face may still not be recovered.
The underlying detector and the ArcFace-style embedding model are both
trained on roughly-frontal faces — even when a profile face IS detected,
the 5-point alignment this project uses (eyes/nose/mouth) becomes unstable
when the eyes aren't both clearly visible, which can produce a low-quality
embedding rather than a clean failure. Lowering the confidence threshold
recovers borderline/angled cases, but isn't a fix for the model's frontal-
face assumption — some profile-only elements are expected to stay excluded.

Run it with `kaggle_cell_recovery.py` (same shape as `kaggle_cell.py`, just
calls `recovery.py` instead of `main.py`) any time after a full run — it's
a small, one-off pass, safe to run repeatedly.

## LanceDB conversion (for serving search without loading everything into RAM)

`build_lancedb.py` converts `face_db.json` into a **LanceDB** table — a
disk-based, memory-mapped vector index (same lineage as Parquet/Arrow) that
lets a search touch only a small relevant slice of the data instead of
holding the whole ~118k × 512 embedding matrix in RAM. It:

1. Fetches the latest `face_db.json` from the HF dataset.
2. Builds a LanceDB table (`id`, `vector`, `num_images_used` columns).
3. Builds a cosine-distance IVF-PQ index on it.
4. Uploads the whole resulting LanceDB directory to its own folder on the
   dataset (`config.HF_LANCEDB_FOLDER_NAME`, separate from `face_db_output/`).

This is a standalone, pure-Python step — no image downloading, no
Node/Puppeteer/Chrome involved. Run it with `kaggle_cell_lancedb.py` any
time after `main.py` or `recovery.py` updates `face_db.json`, to refresh the
LanceDB copy.

## What changed from the dedup pipeline

- **Removed entirely**: `visual_dedup.py`, `quality.py`, `webp_analysis.py`,
  `single_image_check.py`, `kaggle_cell_single.py` — none of that applies
  once the goal is "one embedding per element" instead of "one best photo
  per element". Single-URL elements now go through the normal pipeline
  instead of a separate pass.
- **Added**: `face_embed/` (the Node/Puppeteer worker + `build_page.html` +
  its own `package.json`), plus the face-embedding stage in `pipeline.py`.
- **`merge_dedupe.py`**: no longer drops single-URL elements.
- **`config.py`**: dropped the pHash/quality-weight settings; added the
  face-embedding model paths, `MAX_EMBED_CONCURRENCY_CAP`, download-stagger
  settings, and `EMBEDDING_DTYPE`.
- **Everything else** (batching, checkpointing, HF sync pattern, the
  download-then-process-overlap in `pipeline.py`, the Kaggle cell shape) is
  the same architecture, just re-pointed at the new stage.

## One-time setup before your first run

Upload these two files to the **root** of the HF dataset
(`abdelwahabnabil500/datafile`) once — `main.py` fetches them automatically
on every run after that, the same way it already fetches
`performers_data.json`:

- `w600k_mbf.onnx`
- `face_landmarker.task`

(If you'd rather not rely on that, you can instead just drop both files
directly into `face_embed/` before running — `main.py` skips the fetch if
they're already there.)

## Running it

### On Kaggle (recommended)

Open `kaggle_cell.py`, copy its **entire contents** into a single Kaggle
notebook cell, fill in the 3 placeholders at the top, make sure the
notebook's **Internet** toggle is on, and run the cell. It clones this
repo, installs the Python deps, installs Node.js if it isn't already
present, runs `npm install` inside `face_embed/` (this also downloads
Chromium for Puppeteer — can take a couple of minutes the first time), then
runs the whole pipeline.

### Manually / locally

```
# once:
pip install -r requirements.txt
cd face_embed && npm install && cd ..
# place performers_data.json, face_embed/w600k_mbf.onnx,
# face_embed/face_landmarker.task manually

python main.py
```

HF credentials are optional in this mode — without them, the input file and
both models must already be present locally, and results are only written
locally under `face_db_output/` (no auto-upload, no remote checkpoint
resume).

## Tuning knobs (all in `config.py`)

| Setting                          | What it controls                                              |
| --------------------------------- | --------------------------------------------------------------- |
| `BATCH_SIZE`                      | Elements per batch (fixed at 10,000 per spec)                  |
| `DOWNLOAD_MAX_RETRIES`            | Attempts per URL before marking it failed                      |
| `DOWNLOAD_STAGGER_MIN/MAX_SECONDS`| Random delay added before each download attempt                |
| `MAX_DOWNLOAD_WORKERS_CAP`        | Hard ceiling for auto-detected download concurrency             |
| `MAX_EMBED_CONCURRENCY_CAP`       | Hard ceiling for parallel browser tabs during face-embedding    |
| `EMBEDDING_DTYPE`                 | Storage precision for embedding components (float16)           |

## Project layout

```
config.py             - all tunable settings in one place
concurrency.py         - auto-detects safe worker/tab counts
merge_dedupe.py         - stage 1: merge + dedupe URLs (keeps single-url elements)
downloader.py           - stage 2: concurrent, retrying, paced downloader
checkpoint.py           - batch-level local checkpoint read/write
hf_sync.py              - HuggingFace dataset download/upload helpers
report.py               - builds report.txt and failed_downloads.csv
pipeline.py             - orchestrates batching + download/embed overlap
main.py                 - entry point
kaggle_cell.py           - the single Kaggle notebook cell to paste and run
face_embed/
  package.json           - Node deps (express, puppeteer)
  build_page.html         - the actual detection/alignment/embedding code,
                            identical to the real browser search page
  embed_batch.js          - Node/Puppeteer worker, invoked once per batch
                            by pipeline.py
```
