#!/usr/bin/env python3
"""
gcs_indexer.py

Incremental indexer for Gemini File Search, using Google Cloud Storage as the
source of documents AND as the place to store indexing metadata.

Behavior:
- On first run:
  - If config/store_name.txt does NOT exist in the bucket:
      * Creates a new File Search store with a display name.
      * Writes the store resource name to config/store_name.txt.
  - If config/indexed_files.json does NOT exist:
      * Starts with an empty index state.

- On every run:
  - Scans gs://<BUCKET>/<DOCS_PREFIX> for allowed file types (.txt, .pdf, .docx, .doc).
  - Uses each blob's md5_hash as a fingerprint.
  - Compares against config/indexed_files.json:
      * New blob      -> upload & index.
      * Changed blob  -> re-upload & index.
      * Unchanged     -> skip.
  - Updates config/indexed_files.json after successful indexing.
"""

import os
import json
import time
from typing import Dict, Tuple, List

from dotenv import load_dotenv
from google import genai
from google.cloud import storage  # GCS client


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()

# GCS bucket and prefixes
GCS_BUCKET = os.getenv("DOCS_BUCKET", "adk-test-kd")
DOCS_PREFIX = os.getenv("DOCS_PREFIX", "PdfDocuments/")  # documents live here
CONFIG_PREFIX = os.getenv("CONFIG_PREFIX", "config/")    # config files live here

STORE_NAME_BLOB = CONFIG_PREFIX + "store_name.txt"
INDEX_STATE_BLOB = CONFIG_PREFIX + "indexed_files.json"

ALLOWED_EXT = (".txt", ".pdf", ".docx", ".doc")


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

def make_genai_client() -> genai.Client:
    """Create a Gemini Developer API client (NOT Vertex)."""
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY in environment/.env")
    return genai.Client(api_key=api_key, vertexai=False)


def make_storage_client() -> storage.Client:
    """Create a Google Cloud Storage client (uses Application Default Credentials)."""
    return storage.Client()


# ---------------------------------------------------------------------------
# GCS helpers for config/state
# ---------------------------------------------------------------------------

def get_bucket(storage_client: storage.Client):
    bucket = storage_client.bucket(GCS_BUCKET)
    if not bucket.exists():
        raise RuntimeError(f"GCS bucket '{GCS_BUCKET}' does not exist.")
    return bucket


def gcs_blob_exists(bucket, name: str) -> bool:
    blob = bucket.blob(name)
    return blob.exists()


def load_store_name_from_gcs(bucket) -> str:
    """
    Load File Search store name from GCS (config/store_name.txt).
    If missing, returns empty string.
    """
    blob = bucket.blob(STORE_NAME_BLOB)
    if not blob.exists():
        return ""
    return blob.download_as_text().strip()


def save_store_name_to_gcs(bucket, store_name: str) -> None:
    blob = bucket.blob(STORE_NAME_BLOB)
    blob.upload_from_string(store_name)


def load_index_state_from_gcs(bucket) -> Dict[str, str]:
    """
    Load local map of blob_name -> md5_hash from GCS (config/indexed_files.json).
    If missing, return {}.
    """
    blob = bucket.blob(INDEX_STATE_BLOB)
    if not blob.exists():
        return {}
    data = blob.download_as_text()
    return json.loads(data)


def save_index_state_to_gcs(bucket, state: Dict[str, str]) -> None:
    blob = bucket.blob(INDEX_STATE_BLOB)
    blob.upload_from_string(json.dumps(state, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# File Search store
# ---------------------------------------------------------------------------

def load_or_create_store(client: genai.Client, bucket) -> str:
    """
    Load an existing File Search store name from GCS.
    If missing, create a new store and persist the name in GCS.
    """
    # 1) Try to load store name from GCS
    existing_name = load_store_name_from_gcs(bucket)
    if existing_name:
        print(f"Using existing File Search store: {existing_name}")
        return existing_name

    # 2) Otherwise create new store
    print("No store_name found in GCS. Creating a new File Search store ...")
    store = client.file_search_stores.create(config={"display_name": "my_docs_store"})
    store_name = store.name
    print(f"Created File Search store: {store_name}")

    save_store_name_to_gcs(bucket, store_name)
    return store_name


# ---------------------------------------------------------------------------
# Indexing logic
# ---------------------------------------------------------------------------

def list_candidate_blobs(bucket) -> List[Tuple[str, str]]:
    """
    Return list of (blob_name, md5_hash) for blobs under DOCS_PREFIX
    that match allowed extensions.
    """
    blobs = bucket.list_blobs(prefix=DOCS_PREFIX)  # lists all objects with this prefix 
    candidates = []
    for blob in blobs:
        name = blob.name
        # Skip "folders" (prefix objects) and non-matching ext
        if not any(name.lower().endswith(ext) for ext in ALLOWED_EXT):
            continue
        if not blob.md5_hash:
            # In practice this should be set; if not, you could fallback to generation or CRC32C.
            print(f"[WARN] Blob {name} has no md5_hash; treating as new/changed every time.")
            candidates.append((name, None))
        else:
            candidates.append((name, blob.md5_hash))
    return candidates


def incremental_index() -> None:
    """
    Incrementally index documents from GCS into the File Search store.

    Source docs:    gs://GCS_BUCKET/DOCS_PREFIX/*
    Store name:     stored in gs://GCS_BUCKET/config/store_name.txt
    Index state:    stored in gs://GCS_BUCKET/config/indexed_files.json
    """
    genai_client = make_genai_client()
    storage_client = make_storage_client()
    bucket = get_bucket(storage_client)

    store_name = load_or_create_store(genai_client, bucket)
    state = load_index_state_from_gcs(bucket)  # { blob_name: md5_hash }

    candidates = list_candidate_blobs(bucket)

    new_or_changed: List[Tuple[str, str]] = []

    # Detect new/changed GCS objects
    for blob_name, md5 in candidates:
        old_hash = state.get(blob_name)

        if old_hash is None:
            print(f"[NEW] {blob_name}")
            new_or_changed.append((blob_name, md5))
        elif md5 is None or old_hash != md5:
            print(f"[CHANGED] {blob_name}")
            new_or_changed.append((blob_name, md5))
        else:
            print(f"[SKIP] {blob_name} (already indexed and unchanged)")

    if not new_or_changed:
        print("No new or changed documents. Index is up-to-date.")
        return

    # Upload only new/changed files
    for blob_name, md5 in new_or_changed:
        blob = bucket.blob(blob_name)
        local_filename = os.path.basename(blob_name)
        tmp_path = os.path.join("/tmp", local_filename)

        print(f"Downloading gs://{GCS_BUCKET}/{blob_name} to {tmp_path} ...")
        blob.download_to_filename(tmp_path)  # download file from GCS 

        print(f"Uploading {blob_name} to File Search store {store_name} ...")
        operation = genai_client.file_search_stores.upload_to_file_search_store(
            file=tmp_path,   # path to the downloaded file 
            file_search_store_name=store_name,
            config={"display_name": local_filename},
        )

        # Wait for indexing to finish
        while not getattr(operation, "done", False):
            time.sleep(2)
            operation = genai_client.operations.get(operation)

        print(f"Indexed {blob_name}")
        if md5 is not None:
            state[blob_name] = md5

        # Clean up temp file
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    # Persist updated state
    save_index_state_to_gcs(bucket, state)
    print("Indexing complete. State updated in GCS.")


if __name__ == "__main__":
    incremental_index()