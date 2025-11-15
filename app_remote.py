# app.py

import os
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from google import genai
from google.genai import types
from google.cloud import storage  # GCS client


# ---------------------------------------------------------------------------
# Configuration & client setup
# ---------------------------------------------------------------------------

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY in your environment or .env file.")

# Gemini client (Developer API, not Vertex)
client = genai.Client(api_key=API_KEY, vertexai=False)

# GCS settings
GCS_BUCKET = os.getenv("DOCS_BUCKET", "adk-test-kd")
CONFIG_PREFIX = os.getenv("CONFIG_PREFIX", "config/")
STORE_NAME_BLOB = os.getenv("STORE_NAME_BLOB", CONFIG_PREFIX + "store_name.txt")


def make_storage_client() -> storage.Client:
    """
    Create a GCS client using Application Default Credentials.

    In Cloud Run / GCE / GKE this uses the attached service account.
    Locally, this uses `gcloud auth application-default login` or a key file
    if GOOGLE_APPLICATION_CREDENTIALS is set.
    """
    return storage.Client()


def load_store_name_from_gcs() -> str:
    """
    Load the File Search store name from GCS:
        gs://<GCS_BUCKET>/<STORE_NAME_BLOB>

    This object should be created by your indexing script.
    """
    storage_client = make_storage_client()
    bucket = storage_client.bucket(GCS_BUCKET)

    if not bucket.exists():
        raise RuntimeError(f"GCS bucket '{GCS_BUCKET}' does not exist or is not accessible.")

    blob = bucket.blob(STORE_NAME_BLOB)
    if not blob.exists():
        raise RuntimeError(
            f"GCS object '{STORE_NAME_BLOB}' not found in bucket '{GCS_BUCKET}'. "
            "Run your GCS indexer first so it writes the File Search store name."
        )

    store_name = blob.download_as_text().strip()
    if not store_name:
        raise RuntimeError(
            f"GCS object '{STORE_NAME_BLOB}' is empty. "
            "Ensure the indexer wrote the store resource name correctly."
        )

    return store_name


STORE_NAME = load_store_name_from_gcs()


# ---------------------------------------------------------------------------
# FastAPI models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    citations: List[str]


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Gemini File Search Query API (GCS-backed)",
    description=(
        "Query an existing Gemini File Search store. "
        "Indexing and store-name management are handled separately in GCS."
    ),
    version="1.0.0",
)


@app.get("/status")
def status():
    """
    Simple health/status endpoint.

    Returns the currently configured File Search store name.
    """
    return {
        "store_name": STORE_NAME,
        "bucket": GCS_BUCKET,
        "store_name_blob": STORE_NAME_BLOB,
        "ready": bool(STORE_NAME),
    }


@app.post("/ask", response_model=QueryResponse)
def ask_question(req: QueryRequest):
    """
    Ask a natural-language question against the existing File Search store.

    The store is created / updated by the external GCS indexer script.
    This endpoint only:
      - Calls Gemini with File Search as a tool
      - Returns the answer + citations
    """
    if not STORE_NAME:
        raise HTTPException(
            status_code=500,
            detail="File Search store not loaded. Ensure store_name.txt is created in GCS by the indexer.",
        )

    # Configure File Search as a tool
    file_search_tool = types.Tool(
        file_search=types.FileSearch(file_search_store_names=[STORE_NAME])
    )

    # Call the Gemini model with File Search as a tool
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=req.query,
        config=types.GenerateContentConfig(
            tools=[file_search_tool],
        ),
    )

    # Main answer text
    text = getattr(response, "text", "") or ""

    # Collect citations from grounding metadata
    citations: List[str] = []
    candidates = getattr(response, "candidates", None) or []

    # Expected structure:
    # response.candidates[0].grounding_metadata.grounding_chunks[*].retrieved_context
    for candidate in candidates:
        gm = getattr(candidate, "grounding_metadata", None)
        if not gm:
            continue

        chunks = getattr(gm, "grounding_chunks", []) or []
        for chunk in chunks:
            rc = getattr(chunk, "retrieved_context", None)
            if not rc:
                continue
            source_title = getattr(rc, "title", "") or "Unknown source"
            snippet = getattr(rc, "text", "") or ""
            citations.append(f'{source_title}: "{snippet}"')

    return QueryResponse(answer=text, citations=citations)
