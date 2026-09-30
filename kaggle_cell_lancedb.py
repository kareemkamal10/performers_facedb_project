# ============================================================================
# PASTE THIS ENTIRE CELL INTO A KAGGLE NOTEBOOK (as ONE cell) AND RUN IT.
# Converts face_db.json into a LanceDB table (build_lancedb.py) - NOT the
# full build, NOT the recovery pass. Pure Python, no Node/Chrome needed here.
# Fill in the 3 placeholders below first. Make sure "Internet" is turned ON.
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
else:
    subprocess.run(["git", "-C", PROJECT_DIR, "pull"], check=True)

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-r",
     os.path.join(PROJECT_DIR, "requirements.txt")],
    check=True,
)

os.environ["HF_DATASET_REPO_ID"] = HF_DATASET_REPO_ID
os.environ["HF_TOKEN"] = HF_TOKEN

os.chdir(PROJECT_DIR)
subprocess.run([sys.executable, "build_lancedb.py"], check=True)
