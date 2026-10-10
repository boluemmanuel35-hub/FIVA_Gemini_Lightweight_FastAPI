# FIVA Academic Adviser — Lightweight Gemini FastAPI

This backend is built for the uploaded:

`fiva_adviser_knowledge_gemini.json`

It intentionally does **not** use:

- sentence-transformers
- PyTorch
- Transformers
- scikit-learn
- NumPy

That makes it much smaller and more suitable for Render Free and Vercel.

## Architecture

Student question
→ FastAPI
→ Gemini `gemini-embedding-2` question embedding (768 dimensions)
→ pure-Python cosine similarity against the 142 precomputed handbook vectors
→ top handbook chunks
→ Gemini generation model
→ grounded Academic Adviser answer + sources

## Files

```text
main.py
fiva_adviser_knowledge_gemini.json
requirements.txt
.env.example
.python-version
.gitignore
```

## 1. Local setup

Windows:

```cmd
py -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your Gemini API key.

```env
GEMINI_API_KEY=your_key
```

Run:

```cmd
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Open:

```text
http://127.0.0.1:8000/docs
```

## 2. Test health

`GET /health`

Expected:

```json
{
  "status": "ok",
  "knowledge_area": "ACADEMIC_ADVISER",
  "knowledge_records": 142,
  "embedding_model": "gemini-embedding-2",
  "embedding_dimensions": 768,
  "generation_model": "gemini-3.8-flash",
  "gemini_api_configured": true
}
```

## 3. Test retrieval

Use `POST /retrieve`:

```json
{
  "question": "What are the requirements to proceed from 100 level to 200 level?",
  "top_k": 4
}
```

This calls Gemini only for the question embedding, then performs retrieval locally.

## 4. Test full RAG

Use `POST /ask`:

```json
{
  "question": "What are the requirements to proceed from 100 level to 200 level?",
  "top_k": 4
}
```

Other good tests:

```text
What are the admission requirements for Computer Engineering?
How long is the Computer Engineering programme?
Explain SIWES and SWEP.
What are the graduation requirements?
How are results computed?
What is the grading system?
```

## 5. Render deployment

Push these files to GitHub.

Render Web Service settings:

```text
Language:
Python 3

Build Command:
pip install -r requirements.txt

Start Command:
uvicorn main:app --host 0.0.0.0 --port $PORT
```

Environment variables:

```text
GEMINI_API_KEY = your key
GEMINI_EMBEDDING_MODEL = gemini-embedding-2
GEMINI_EMBEDDING_DIMENSIONS = 768
GEMINI_GENERATION_MODEL = gemini-3.8-flash
FIVA_TOP_K = 4
```

Do NOT upload `.env` to GitHub.

## 6. Vercel deployment

Import the GitHub repository into Vercel.

Vercel should detect `main.py` as a FastAPI application automatically.

Add Environment Variables:

```text
GEMINI_API_KEY = your key
GEMINI_EMBEDDING_MODEL = gemini-embedding-2
GEMINI_EMBEDDING_DIMENSIONS = 768
GEMINI_GENERATION_MODEL = gemini-3.8-flash
FIVA_TOP_K = 4
```

No Render-style Uvicorn start command is required in Vercel's project UI.

## 7. Android

After deployment, your Retrofit base URL becomes your cloud URL, for example:

```kotlin
private const val BASE_URL =
    "https://your-fiva-service.onrender.com/"
```

or:

```kotlin
private const val BASE_URL =
    "https://your-fiva-project.vercel.app/"
```

At that point you no longer need `adb reverse`.
