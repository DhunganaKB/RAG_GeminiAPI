# File Search Tool in Gemini API

This folder contains two variants of an incremental indexer and a small FastAPI query service:

- `index_create_local.py` — index documents from the local `./data` directory into a Gemini File Search store.
- `index_create_gcp_cloud.py` — index documents stored in a Google Cloud Storage bucket into a Gemini File Search store; state and store_name are persisted in the GCS bucket under `config/`.
- `app.py` — FastAPI server that queries an existing File Search store and returns answers + citations.

Overview
--------
Workflows supported:

1. Local flow (running entirely locally)
   - Use `index_create_local.py` to create (if needed) and incrementally index files from `./data` into a Gemini File Search store.
   - The script maintains a local `.
   - After indexing, `app.py` reads `.store_name` and exposes a `/ask` endpoint that answers queries using the indexed documents.

2. GCS flow (source files and state live in Google Cloud Storage)
   - Use `index_create_gcp_cloud.py` to index documents stored in a GCS bucket.
   - The script stores `config/store_name.txt` and `config/indexed_files.json` in the bucket to track state and the store resource.
   - `app.py` still reads a local `.store_name` file — you can either download `config/store_name.txt` to `.store_name` or modify `app.py` to read from GCS directly.

Key properties
--------------
- Incremental: both index scripts keep a record of which files were already indexed (local: `.indexed_files.json`; GCS: `config/indexed_files.json` in the bucket). On subsequent runs, only new or changed files are uploaded.
- Idempotent: unchanged files are skipped to save API calls and indexing costs.
- Citations: when the query endpoint returns an answer it also returns citations — snippets and titles from the retrieved contexts used to construct the answer.

Requirements
------------
See `requirements.txt` in this folder. Minimum:

- Python 3.10+
- python-dotenv
- google-genai (Gemini Developer client)
- google-cloud-storage (for the GCS variant)
- fastapi, uvicorn, pydantic (for `app.py`)

Setup
-----
1. Create and activate a virtualenv (macOS / zsh):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Provide credentials:

- Add a `.env` file in this folder with your Gemini API key and any other variables used by the scripts, for example:

```ini
GOOGLE_API_KEY=your_api_key_here
DOCS_BUCKET=your-gcs-bucket-name
DOCS_PREFIX=PdfDocuments/
CONFIG_PREFIX=config/
```

- For the GCS workflow, ensure your environment has Google Cloud credentials available (ADC). This usually means setting `GOOGLE_APPLICATION_CREDENTIALS` to a service account JSON key:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
```

Local indexing flow
-------------------
1. Put the documents you want to index into the `data/` directory (supported: .txt, .pdf, .docx, .doc).

2. Run the local indexer:

```bash
python3 index_create_local.py
```

Behavior:
- If `.store_name` does not exist, the script will attempt to create a new File Search store and write its resource name to `.store_name`.
- The script computes a SHA-256 hash for each file and stores a map of filename -> hash in `.indexed_files.json`.
- On subsequent runs, only new or changed files will be uploaded and indexed.

GCS indexing flow
-----------------
1. Upload your documents to the configured GCS bucket under the `DOCS_PREFIX` (default `PdfDocuments/`).

2. Run the GCS indexer:

```bash
python3 index_create_gcp_cloud.py
```

Behavior:
- The script looks for `config/store_name.txt` in the bucket. If missing, it tries to create a store and writes the name to that blob.
- It keeps `config/indexed_files.json` in GCS containing a map of blob_name -> md5_hash and only uploads new/changed files.
- The script downloads new/changed blobs to `/tmp` before uploading them to the File Search store, then cleans up the temp files.

Querying with FastAPI
---------------------
1. Ensure `.store_name` exists locally (you can copy from GCS `config/store_name.txt` if using the GCS flow):

```bash
# download store name from GCS (example)
python3 - <<'PY'
from google.cloud import storage
b = storage.Client().bucket('your-bucket-name')
print(b.blob('config/store_name.txt').download_as_text())
PY
```

or simply copy the file:

```bash
gsutil cp gs://your-bucket-name/config/store_name.txt .store_name
```

2. Start the FastAPI app:

```bash
uvicorn app:app --reload
```

3. POST a question to `/ask`:

```bash
curl -X POST "http://127.0.0.1:8000/ask" -H "Content-Type: application/json" -d '{"query":"What is the main topic of lostinmiddle.pdf?"}'
```

The response will contain the generated answer and a list of citations (document title + snippet) that the model grounded on.

Notes & troubleshooting
-----------------------
- The indexing scripts use the Gemini Developer API (vertexai=False). If your installed `google-genai` client doesn't support store creation, create the store manually and write the resource name into `.store_name` (or into the GCS config location).
- If you encounter authentication errors, check these:
  - `GOOGLE_API_KEY` in `.env` (for genai Developer client authentication).
  - `GOOGLE_APPLICATION_CREDENTIALS` for GCS access (service account JSON).
- Large files may take time to index; the scripts poll operations until indexing completes.

Next steps / improvements
-------------------------
- Add a small wrapper script to sync `config/store_name.txt` from GCS to local `.store_name` automatically.
- Add unit tests that mock `genai.Client` and `google.cloud.storage`.
- Add retry/backoff logic for transient API failures.

---

If you want, I can also add a `Makefile` or small helper scripts to automate the full flow (index -> copy store_name -> run app -> ask).
