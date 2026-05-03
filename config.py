import os

# ── Groq (transcription) ───────────────────────────────────────────────────
GROQ_API_KEY     = os.getenv("GROQ_API_KEY",     "YOUR_GROQ_API_KEY_HERE")
MODEL_NAME       = os.getenv("MODEL_NAME",       "whisper-large-v3-turbo")
API_URL          = os.getenv("API_URL",          "")   # empty = Groq default
FALLBACK_MODEL   = os.getenv("FALLBACK_MODEL",   "whisper-large-v3")
FALLBACK_API_KEY = os.getenv("FALLBACK_API_KEY", "")   # empty = same as GROQ_API_KEY
FALLBACK_API_URL = os.getenv("FALLBACK_API_URL", "")   # empty = same as API_URL

# ── Gemini (AI reorganizer) ────────────────────────────────────────────────
GEMINI_API_KEY          = os.getenv("GEMINI_API_KEY",          "YOUR_GEMINI_API_KEY_HERE")
GEMINI_FALLBACK_API_KEY = os.getenv("GEMINI_FALLBACK_API_KEY", "")

# ── File discovery ─────────────────────────────────────────────────────────
FILE_DEPTH = int(os.getenv("FILE_DEPTH", "1"))
