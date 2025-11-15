#!/usr/bin/env python3
"""
indexer.py

Incremental indexer for Gemini File Search.

Behavior:
- On first run:
  - If .store_name does NOT exist:
      * Creates a new File Search store with a display name.
      * Writes the store resource name to .store_name.
  - If .indexed_files.json does NOT exist:
      * Starts with an empty index state.

- On every run:
  - Scans ./data for allowed file types (.txt, .pdf, .docx, .doc).
  - Computes SHA-256 hash of each file.
  - Compares hashes against .indexed_files.json:
      * New file  -> upload & index.
      * Changed file -> re-upload & index.
      * Unchanged file -> skip.
  - Updates .indexed_files.json after successful indexing.
"""

import os
import json
import time
import hashlib
from typing import Dict

from dotenv import load_dotenv
from google import genai

load_dotenv()

STORE_NAME_FILE = ".store_name"
INDEX_STATE_FILE = ".indexed_files.json"
DATA_DIR = "data"
ALLOWED_EXT = (".txt", ".pdf", ".docx", ".doc")


def make_client() -> genai.Client:
    """Create a Gemini Developer API client (NOT Vertex)."""
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY in environment/.env")
    # vertexai=False forces Developer API mode
    return genai.Client(api_key=api_key, vertexai=False)


def load_index_state() -> Dict[str, str]:
    """Load the local map of filename -> sha256 hash."""
    if not os.path.exists(INDEX_STATE_FILE):
        return {}
    with open(INDEX_STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_index_state(state: Dict[str, str]) -> None:
    """Persist the local map of filename -> sha256 hash."""
    with open(INDEX_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)


def file_hash(path: str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_or_create_store(client: genai.Client, display_name: str = "my_docs_store") -> str:
    """
    Load an existing File Search store from .store_name if present.
    If missing, create a new store and persist the name.
    """
    # 1) If .store_name exists, use it.
    if os.path.exists(STORE_NAME_FILE):
        with open(STORE_NAME_FILE, "r", encoding="utf-8") as f:
            name = f.read().strip()
            if name:
                print(f"Using existing File Search store: {name}")
                return name

    # 2) Otherwise create a new store.
    print("No .store_name found. Creating a new File Search store ...")
    store = client.file_search_stores.create(config={"display_name": display_name})
    store_name = store.name
    print(f"Created File Search store: {store_name}")

    # Persist the store name
    with open(STORE_NAME_FILE, "w", encoding="utf-8") as f:
        f.write(store_name)

    return store_name


def incremental_index(display_name: str = "my_docs_store") -> None:
    """
    Incrementally index files from ./data into the File Search store.

    - Creates a File Search store on first run (if .store_name is missing).
    - Uses .indexed_files.json to detect new/changed files.
    """
    client = make_client()
    store_name = load_or_create_store(client, display_name=display_name)
    state = load_index_state()

    data_dir = os.path.join(os.getcwd(), DATA_DIR)
    if not os.path.isdir(data_dir):
        raise RuntimeError(f"data directory not found: {data_dir}")

    new_or_changed = []

    # 1) Detect new/changed files
    for fname in os.listdir(data_dir):
        if not fname.lower().endswith(ALLOWED_EXT):
            continue
        path = os.path.join(data_dir, fname)
        if not os.path.isfile(path):
            continue

        h = file_hash(path)
        old_hash = state.get(fname)

        if old_hash is None:
            print(f"[NEW] {fname}")
            new_or_changed.append((fname, path, h))
        elif old_hash != h:
            print(f"[CHANGED] {fname}")
            new_or_changed.append((fname, path, h))
        else:
            print(f"[SKIP] {fname} (already indexed and unchanged)")

    if not new_or_changed:
        print("No new or changed documents. Index is up-to-date.")
        return

    # 2) Upload only new/changed files
    for fname, path, h in new_or_changed:
        print(f"Uploading {fname} to store {store_name} ...")
        op = client.file_search_stores.upload_to_file_search_store(
            file=path,
            file_search_store_name=store_name,
            config={"display_name": fname},
        )
        # Wait for indexing to finish
        while not getattr(op, "done", False):
            time.sleep(2)
            op = client.operations.get(op)

        print(f"Indexed {fname}")
        state[fname] = h

    # 3) Persist updated state
    save_index_state(state)
    print("Indexing complete. State updated.")


if __name__ == "__main__":
    # Optional: allow overriding the store display name via env or later via CLI.
    display_name = os.getenv("FILE_SEARCH_DISPLAY_NAME", "my_docs_store")
    incremental_index(display_name=display_name)
