# app.py

import os
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Configuration & client setup
# ---------------------------------------------------------------------------

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY in your environment or .env file.")

# Force Gemini Developer API (not Vertex) by passing vertexai=False
client = genai.Client(api_key=API_KEY, vertexai=False)

STORE_NAME_FILE = ".store_name"


def load_store_name() -> str:
    """
    Load the File Search store name from .store_name.

    This file should be created by the indexing script (indexer.py).
    """
    if not os.path.exists(STORE_NAME_FILE):
        raise RuntimeError(
            f"{STORE_NAME_FILE} not found. "
            "Run the indexing script first to create a File Search store."
        )
    with open(STORE_NAME_FILE, "r", encoding="utf-8") as f:
        name = f.read().strip()
        if not name:
            raise RuntimeError(
                f"{STORE_NAME_FILE} is empty. "
                "Ensure the indexing script wrote the store resource name correctly."
            )
        return name


STORE_NAME = load_store_name()

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
    title="Gemini File Search Query API",
    description="Query an existing Gemini File Search store. Indexing is handled separately.",
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
        "ready": bool(STORE_NAME),
    }


@app.post("/ask", response_model=QueryResponse)
def ask_question(req: QueryRequest):
    """
    Ask a natural-language question against the existing File Search store.

    The store is created / updated by the external indexer script.
    This endpoint only:
      - Calls Gemini with File Search as a tool
      - Returns the answer + citations
    """
    if not STORE_NAME:
        raise HTTPException(
            status_code=500,
            detail="File Search store not loaded. Ensure .store_name is created by the indexer.",
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
    # retrieved_context has fields like title/text for File Search
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