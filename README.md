# AR-class-transcriber

A command-line tool for batch transcription of Arabic audio lectures and classes. It records your remaining API quota before processing, trims content to fit within that quota, transcribes using Groq Whisper, optionally reorganizes the output with Google Gemini, and maintains a per-folder log so interrupted runs resume cleanly.

---

## Features

- **Batch transcription** of `.m4a` files from one or more folders or individual files
- **Live rate-limit awareness** — probes Groq before processing to use your actual remaining quota, not a hardcoded value
- **Smart overflow handling** — if total audio exceeds your quota, the largest file is split into `part1` (transcribed) and `part2` (saved for the next run), so no audio is lost
- **Automatic model fallback** — switches to `whisper-large-v3` (or any fallback you specify) mid-run when the primary model hits its rate limit
- **Arabic-only output** — strips timestamps, annotations, non-Arabic characters, and Whisper repetition hallucinations from every transcript
- **Per-folder JSONL log** — tracks every transcription so re-runs skip already-completed files
- **Merged transcript** — combines all transcripts in a folder into a single `merged.txt` after a clean run; skipped if any file failed (so you never get a partial merge)
- **AI reorganizer** (optional) — sends each transcript to Google Gemini to add structural titles and markers; two modes available
- **Merged reorganized output** (optional) — combines all reorganized files with clear dividers into `merged_reorganized.txt`
- **Flexible configuration** — script defaults → `config.py` overrides → CLI overrides, in that priority order

---

## Requirements

### System
- Python 3.10+
- `ffmpeg` and `ffprobe`

```bash
apt install ffmpeg
```

### Python packages

```bash
pip install groq google-genai
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/AR-class-transcriber.git
cd AR-class-transcriber
```

### 2. Configure API keys

Copy or create `config.py` in the project directory. The file is **not required** — all values can be passed via CLI — but it is the recommended way to store keys.

```python
# config.py

# ── Groq (transcription) ──────────────────────────────────────────────────
GROQ_API_KEY     = "gsk_your_key_here"
MODEL_NAME       = "whisper-large-v3-turbo"   # optional, this is the default
API_URL          = ""                          # optional, leave empty for Groq default
FALLBACK_MODEL   = "whisper-large-v3"         # optional, this is the default
FALLBACK_API_KEY = ""                          # optional, defaults to GROQ_API_KEY
FALLBACK_API_URL = ""                          # optional, defaults to API_URL

# ── Gemini (AI reorganizer) ───────────────────────────────────────────────
GEMINI_API_KEY          = "AIza_your_key_here"   # required only if using reorganizer
GEMINI_FALLBACK_API_KEY = ""                      # optional, defaults to GEMINI_API_KEY

# ── File discovery ────────────────────────────────────────────────────────
FILE_DEPTH = 1   # optional: 1 = top-level only, 2 = one subfolder deep, 0 = unlimited
```

Get your keys at:
- Groq: https://console.groq.com/keys
- Gemini: https://aistudio.google.com/app/apikey

---

## Usage

```
python3 transcribe.py <path> [<path> ...] [key=value ...]
```

`<path>` can be a folder or an individual `.m4a` file. Multiple paths are accepted.

### Basic examples

```bash
# Transcribe all .m4a files in a folder
python3 transcribe.py /path/to/lectures

# Transcribe specific files
python3 transcribe.py lecture1.m4a lecture2.m4a

# Mix of folders and files
python3 transcribe.py /path/to/lectures /path/to/extra/session.m4a
```

### With reorganization

```bash
# Simple reorganization (titles and subtitles only)
python3 transcribe.py /path/to/lectures reorgmode=1

# Advanced reorganization + produce a single merged reorganized file
python3 transcribe.py /path/to/lectures reorgmode=2 reorgmerge=1
```

### Without config.py (passing all keys via CLI)

```bash
python3 transcribe.py /path/to/lectures \
  groq_api_key=gsk_xxx \
  gemini_api_key=AIza_yyy \
  reorgmode=2 \
  reorgmerge=1
```

---

## CLI Reference

All `key=value` arguments support flexible spacing: `key=value`, `key= value`, and `key = value` all work.

### Groq (transcription)

| Argument | Default | Description |
|---|---|---|
| `groq_api_key` | — | Primary Groq API key |
| `model_name` | `whisper-large-v3-turbo` | Primary transcription model |
| `api_url` | Groq default | Primary API base URL |
| `fallbackmodel` | `whisper-large-v3` | Model used when primary hits rate limit |
| `fallbackapikey` | same as primary | Fallback API key |
| `fallbackapiurl` | same as primary | Fallback API base URL |
| `filedepth` | `1` | Folder depth to search for `.m4a` files (`0` = unlimited) |

### Gemini (reorganizer)

| Argument | Default | Description |
|---|---|---|
| `gemini_api_key` | — | Gemini API key |
| `geminimodel` | `gemini-2.5-flash-lite` | Primary reorganization model |
| `geminiapiurl` | Google default | Gemini API base URL |
| `geminifallbackmodel` | `gemini-2.5-flash` | Fallback reorganization model |
| `geminifallbackapikey` | same as primary | Fallback Gemini API key |
| `geminifallbackapiurl` | same as primary | Fallback Gemini API base URL |
| `reorgmode` | `0` | `0` = none, `1` = simple, `2` = advanced |
| `reorgmerge` | `0` | `1` = produce `merged_reorganized.txt` |

---

## How it works

### Transcription pipeline

```
1. Collect .m4a files from all given paths
2. Read audio durations with ffprobe
3. Probe Groq API → get remaining quota (seconds)
4. If total audio > quota:
     - Find the largest file
     - Split it: {name}part1.m4a (fits in quota) + {name}part2.m4a (overflow, saved for later)
5. For each file (in order):
     - Check JSONL log + disk: skip if already transcribed
     - If file > 20 MB or > 10 min: split into 10-min chunks, transcribe each, stitch
     - Send to Groq → whisper-large-v3-turbo
     - On rate-limit error: switch all remaining files to fallback model
     - Clean output (remove non-Arabic characters, timestamps, hallucinations)
     - Save {name}.txt alongside the audio file
     - Append entry to transcription_log.jsonl
6. If no failures: merge all logged .txt files in the folder → merged.txt
   If any failure: skip merge, report which files failed
```

### Reorganization pipeline (when reorgmode ≠ 0)

```
For each successfully transcribed .txt file:
  - Estimate token count (~3 chars per token for Arabic)
  - If > 250,000 tokens: skip with warning
  - Send to Gemini with structured system prompt
  - On error: retry with fallback model
  - Save {name}_reorganized.txt alongside the original
If reorgmerge=1 and multiple files reorganized:
  - Concatenate all reorganized files
  - Separate with: ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                   original_filename.m4a
  - Save as merged_reorganized.txt
```

### Reorganization modes

**Mode 1 — Simple**
Adds `# Title` and `## Subtitle` markers at topic and sub-topic boundaries, and blank lines between paragraphs. Original text is preserved exactly.

**Mode 2 — Advanced**
Adds `# Title`, `## Header`, and `### Subtitle` markers. Additionally, passages that are off-topic or unrelated to the main subject of the recording are wrapped as:
```
[تعليق: original off-topic text here]
```
Original text is preserved exactly.

---

## Output files

For a folder containing `lecture1.m4a`, `lecture2.m4a`, `lecture3.m4a`:

```
lecture1.m4a
lecture1.txt                    ← raw Arabic transcript
lecture1_reorganized.txt        ← reorganized (if reorgmode ≥ 1)
lecture2.m4a
lecture2.txt
lecture2_reorganized.txt
lecture3.m4a
lecture3.txt
lecture3_reorganized.txt
merged.txt                      ← all transcripts concatenated
merged_reorganized.txt          ← all reorganized files concatenated (if reorgmerge=1)
transcription_log.jsonl         ← processing log
```

If overflow splitting occurred (`lecture1.m4a` was the largest file and got trimmed):
```
lecture1.m4a                    ← original, untouched
lecture1part1.m4a               ← trimmed version that fits in quota (transcribed)
lecture1part2.m4a               ← overflow (not transcribed yet)
lecture1part1.txt
...
```

### Log format (`transcription_log.jsonl`)

One JSON object per line:

```json
{
  "stem": "lecture1",
  "audio_file": "lecture1.m4a",
  "txt_file": "lecture1.txt",
  "model": "whisper-large-v3-turbo",
  "duration_s": 3600.0,
  "chars": 42500,
  "timestamp": "2026-05-03T10:30:00+00:00",
  "status": "done"
}
```

---

## Handling quota and resuming

The Groq free tier allows **7,200 audio-seconds (2 hours) per day** for `whisper-large-v3-turbo`. The script reads your actual remaining quota at runtime — not a hardcoded number — so if you have already used some quota today it adjusts automatically.

**If your folder exceeds today's quota:**
1. The script splits the largest file and transcribes everything that fits
2. It reports the overflow file and its duration
3. Run the script again tomorrow — previously transcribed files are skipped via the log, and the overflow file is processed with the new day's quota

**If the primary model hits its limit mid-run:**
- The script automatically switches to `whisper-large-v3` (or your configured fallback) for all remaining files in that run without interruption

---

## Configuration priority

```
script defaults  →  config.py  →  CLI arguments
(lowest)                           (highest)
```

Values set in the script are always available as a baseline. `config.py` overrides them per-deployment. CLI arguments override everything, including `config.py`, useful for one-off runs with a different key or model.

---

## Free tier limits (as of 2026)

| Service | Model | Free limit |
|---|---|---|
| Groq | `whisper-large-v3-turbo` | 7,200 sec/day |
| Groq | `whisper-large-v3` | 7,200 sec/day |
| Gemini | `gemini-2.5-flash-lite` | 15 RPM / 1,500 RPD |
| Gemini | `gemini-2.5-flash` | 10 RPM / 500 RPD |

---

## Project structure

```
AR-class-transcriber/
├── transcribe.py       # main script
├── config.py           # API keys and overrides (not committed to git)
└── README.md
```

Add `config.py` to `.gitignore` to avoid accidentally committing API keys:

```
echo "config.py" >> .gitignore
```

---

## License

Beginning with the revision that merges this license into the default branch, binjamshid licenses the original project software in that revision and subsequent versions carrying this notice under the [Binjamshid Public Fork License, Version 1.0](LICENSE). Separately licensed third-party material retains its own terms and notices.

The [initial public revision (`7bf0712`)](https://github.com/binjamshid/AR-class-transcriber/blob/7bf07125fb5165f36a6f7f2376638d3b6fbe809a/README.md#license) was marked MIT. Rights already granted for that revision continue under their original terms, including permission under MIT for commercial use and private modifications.

The new license prohibits commercial use by or on behalf of for-profit organizations. If you edit covered software, you must make the corresponding source of every modified version, including private and abandoned edits, publicly available under the same license within 30 calendar days of the edit or before first distribution or use by others as a service, whichever comes first. Section 4 sets out a limited exception for coordinated security fixes.
