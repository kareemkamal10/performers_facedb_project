"""
Central configuration for the performers face-DB build pipeline.
Edit values here to tune behavior; nothing here should require code changes
in the other modules.
"""
import os

# ---- Paths ----
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
INPUT_JSON = os.path.join(PROJECT_ROOT, "performers_data.json")

# Kaggle has two writable areas: /kaggle/working (persists as notebook output,
# small quota) and /kaggle/temp (much larger, wiped when the session ends).
# All downloaded images MUST live under KAGGLE_TEMP_DIR per project spec.
KAGGLE_TEMP_DIR = "/kaggle/temp/performers_images"
KAGGLE_WORKING_DIR = "/kaggle/working"
RESULT_OUTPUT_DIR = os.path.join(KAGGLE_WORKING_DIR, "face_db_output")
CHECKPOINT_PATH = os.path.join(KAGGLE_WORKING_DIR, "checkpoint.json")

# Face-embedding models. Must exist here before the run starts - either
# placed manually, or auto-fetched from the HF dataset (see hf_sync.py /
# README). Kept OUTSIDE git (see .gitignore) - same pattern this project
# already used for performers_data.json.
FACE_EMBED_DIR = os.path.join(PROJECT_ROOT, "face_embed")
FACE_LANDMARKER_TASK_PATH = os.path.join(FACE_EMBED_DIR, "face_landmarker.task")
EMBED_MODEL_PATH = os.path.join(FACE_EMBED_DIR, "w600k_mbf.onnx")
NODE_SCRIPT_PATH = os.path.join(FACE_EMBED_DIR, "embed_batch.js")

# ---- Batching ----
BATCH_SIZE = 5_000

# ---- Downloading ----
DOWNLOAD_MAX_RETRIES = 3
DOWNLOAD_TIMEOUT_SECONDS = 20
DOWNLOAD_RETRY_BACKOFF_BASE = 1.5  # seconds; backoff = base * 2**(attempt-1)
# A small random delay is added before every download attempt so hundreds of
# workers don't all hit the network in the same instant. This - not the
# worker count alone - is what keeps the download rate steady: not bursty
# enough to fail requests, not so slow it wastes the session's time.
DOWNLOAD_STAGGER_MIN_SECONDS = 0.05
DOWNLOAD_STAGGER_MAX_SECONDS = 0.25

# ---- Concurrency ----
# Downloading is I/O-bound (threads mostly sleep waiting on the network), so
# it should NOT be scaled by CPU core count - a 4-core Kaggle session can
# still hold hundreds of sockets open at once. Two fixed levels instead:
#   - FIRST_BATCH: nothing else is running yet, so go as hard as is safe.
#   - OVERLAPPED: every batch after the first downloads WHILE the previous
#     batch is being face-embedded (a CPU-heavy headless-Chrome step), so
#     it steps back to leave that room.
DOWNLOAD_WORKERS_FIRST_BATCH = 128
DOWNLOAD_WORKERS_OVERLAPPED = 48

# Face detection + alignment + embedding runs inside a real headless Chrome
# tab per image, via Puppeteer, so results match the browser search page
# exactly (GPU/WebGPU is deliberately NOT used here - see README). Multiple
# tabs run in parallel inside ONE shared browser instance; this cap IS tied
# to CPU count (unlike downloading above) since each tab is a real CPU-bound
# Chrome renderer process, kept conservative since each one is far heavier
# than a plain CPU worker.
MAX_EMBED_CONCURRENCY_CAP = 4

# ---- Embedding storage precision ----
# Every embedding component is snapped to the nearest float16-representable
# value before being written out (half the precision -> shorter numbers in
# the JSON, and matches what the browser search side will eventually read).
EMBEDDING_DTYPE = "float16"

# ---- HuggingFace dataset (private) ----
# Filled in from the Kaggle cell placeholders at runtime via env vars.
HF_DATASET_REPO_ID = os.environ.get("HF_DATASET_REPO_ID", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Separate folder on the dataset from the older dedup pipeline's result_output/
HF_RESULT_FOLDER_NAME = "face_db_output"

# ---- LanceDB conversion (build_lancedb.py) ----
# Converts the flat face_db.json into a disk-based, memory-mapped LanceDB
# table + cosine vector index - the "serve this without holding the whole
# embedding matrix in RAM" option, for whenever the search backend is built.
LANCEDB_LOCAL_DIR = os.path.join(KAGGLE_WORKING_DIR, "face_lancedb")
LANCEDB_TABLE_NAME = "performers"
HF_LANCEDB_FOLDER_NAME = "face_lancedb_output"

# ---- Recovery pass (excluded.json follow-up) ----
# Used only by recovery.py, for elements that got no face on the first pass.
# Two different failure modes get addressed (see build_page.html's
# processImageRecovery): a face that's small relative to the frame (helped
# by upscaling and top-crops), and a face near-profile or at an angle
# (helped somewhat by a lower confidence bar, though a steep profile shot
# may genuinely be undetectable/unalignable regardless - see README).
RECOVERY_MIN_FACE_CONFIDENCE = 0.2
RECOVERY_MIN_FACE_PRESENCE_CONFIDENCE = 0.2

# ---- Logging ----
LOG_LEVEL = "INFO"
