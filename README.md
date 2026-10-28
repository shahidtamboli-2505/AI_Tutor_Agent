# AI Tutor Agent

A Streamlit interface for the supplied AI Tutor Agent Python backend. Upload a text-based PDF or TXT document, extract topics, ask document-grounded questions, and generate structured lessons.

## Features
- PDF/TXT text extraction and source passages
- AI topic extraction and course title generation
- Question answering grounded in retrieved source passages
- Full, short, misconception-focused, and prerequisite-bridge lesson variants
- Basic Groq API usage metrics

## Requirements
- Python 3.10+
- A Groq API key
- Internet access for model calls

## Run locally

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
```

Set your API key in the shell (do not commit it):

```powershell
$env:GROQ_API_KEY="your-key-here"
streamlit run app.py
```

On macOS/Linux:

```bash
export GROQ_API_KEY="your-key-here"
streamlit run app.py
```

## Deploy to Streamlit Community Cloud
1. Push this project folder to a GitHub repository.
2. Sign in at https://share.streamlit.io/ and connect GitHub.
3. Select the repository and branch, and set the main file path to `app.py`.
4. In the app's **Settings → Secrets**, add:

```toml
GROQ_API_KEY = "your-real-groq-api-key"
```

5. Deploy and inspect the app logs if startup or model calls fail.

Never commit `.streamlit/secrets.toml`, `.env`, or API keys. Streamlit Community Cloud local files may not be durable; JSON storage in this prototype is not a production database.

## Project structure

```text
app.py
config.py
requirements.txt
backend/
  documents.py
  topics.py
  profile.py
  pretest.py
  questions.py
  lessons.py
  tutor.py
  retrieval.py
  progress.py
  support.py
  review.py
  storage.py
  usage.py
  schemas.py
```

## Notes
- The backend model list is configured in `config.py`; confirm current model IDs available to your Groq account if the default choices stop working.
- Scanned PDFs without selectable text are not supported unless OCR is added.
- This is a prototype. Validate model outputs and test with sample documents before using it for important learning decisions.
