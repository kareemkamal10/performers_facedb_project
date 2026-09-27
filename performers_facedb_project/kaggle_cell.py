# ============================================================================
# PASTE THIS ENTIRE CELL INTO A KAGGLE NOTEBOOK (as ONE cell) AND RUN IT.
# Fill in the 3 placeholders below first - everything else is automatic.
# Make sure the notebook has "Internet" turned ON (needed for pip, npm,
# downloading Chromium, downloading images, and talking to HF).
# ============================================================================

GITHUB_REPO_URL    = ""                                   # e.g. "https://github.com/kareemkamal10/YOUR-REPO.git"
HF_DATASET_REPO_ID = "abdelwahabnabil500/datafile"         # already filled in for you
HF_TOKEN           = ""                                    # your HuggingFace access token (dataset is private)

# ----------------------------------------------------------------------------
import os
import subprocess
import sys

assert GITHUB_REPO_URL, "Set GITHUB_REPO_URL above before running this cell."
assert HF_TOKEN, "Set HF_TOKEN above before running this cell."

PROJECT_DIR = "/kaggle/working/project"

if not os.path.exists(PROJECT_DIR):
    subprocess.run(["git", "clone", GITHUB_REPO_URL, PROJECT_DIR], check=True)

# ---- Python deps ----
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-r",
     os.path.join(PROJECT_DIR, "requirements.txt")],
    check=True,
)

# ---- Node.js (most Kaggle images already have it; only installs if missing) ----
node_missing = subprocess.run(["bash", "-lc", "command -v node"], capture_output=True).returncode != 0
if node_missing:
    subprocess.run(
        ["bash", "-lc",
         "curl -fsSL https://deb.nodesource.com/setup_20.x | bash - "
         "&& apt-get update -qq && apt-get install -y -qq nodejs"],
        check=True,
    )

# ---- Chromium's shared-library dependencies. Kaggle's base image is
# missing most of these (headless Chrome fails with "libatk-1.0.so.0: cannot
# open shared object file" etc. without them), so install unconditionally -
# apt skips anything already present, this is quick either way.
subprocess.run(
    ["bash", "-lc",
     "apt-get update -qq && apt-get install -y -qq "
     "libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 libxkbcommon0 "
     "libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libasound2 "
     "libpangocairo-1.0-0 libpango-1.0-0 libcairo2 libnspr4 libnss3 "
     "libxss1 libxtst6 fonts-liberation libgtk-3-0"],
    check=True,
)

# ---- Node deps for the face-embedding worker (this also downloads Chromium
# for Puppeteer - can take a couple of minutes the first time) ----
subprocess.run(
    ["npm", "install", "--no-fund", "--no-audit"],
    cwd=os.path.join(PROJECT_DIR, "face_embed"),
    check=True,
)

# These env vars are read by config.py inside the project.
os.environ["HF_DATASET_REPO_ID"] = HF_DATASET_REPO_ID
os.environ["HF_TOKEN"] = HF_TOKEN

os.chdir(PROJECT_DIR)
subprocess.run([sys.executable, "main.py"], check=True)
