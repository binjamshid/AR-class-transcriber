#!/usr/bin/env python3
"""
transcribe.py — Batch Arabic transcription + AI reorganization of .m4a files.

GENERAL FUNCTION:
  Accepts one or more .m4a files and/or folders as CLI arguments plus optional
  key=value configuration overrides.  Reads all audio durations, queries Groq
  for the caller's remaining rate-limit budget, trims the largest file if the
  total exceeds that budget, then transcribes each file with the primary Groq
  model.  On a rate-limit error it switches to the fallback model/key/URL.
  Optionally runs a Gemini-powered reorganizer on each transcript (mode 1 or
  2) and can merge all reorganized outputs into a single file with dividers.

  Configuration priority: script defaults → config.py overrides → CLI overrides.

USAGE:
  python3 transcribe.py <path> [<path> ...] [key=value ...]

  Supported key=value overrides (spaces/underscores around = are optional):
  ── Groq (transcription) ──────────────────────────────────────────────────
    groq_api_key=<key>          Primary Groq API key
    model_name=<model>          Primary Whisper model
    api_url=<url>               Primary Groq API base URL
    fallbackmodel=<model>       Fallback Whisper model
    fallbackapikey=<key>        Fallback API key
    fallbackapiurl=<url>        Fallback API base URL
    filedepth=<n>               Folder search depth (1=top only, 0=unlimited)
  ── Gemini (reorganizer) ──────────────────────────────────────────────────
    gemini_api_key=<key>        Gemini API key (also accepted from config.py)
    geminimodel=<model>         Gemini primary model
    geminiapiurl=<url>          Gemini API base URL
    geminifallbackmodel=<model> Gemini fallback model
    geminifallbackapikey=<key>  Gemini fallback API key
    geminifallbackapiurl=<url>  Gemini fallback API base URL
    reorgmode=<0|1|2>           0=none (default), 1=simple, 2=advanced
    reorgmerge=<0|1>            1=produce merged_reorganized.txt (default 0)

FUNCTIONS:
  parse_args(argv)
    Splits raw sys.argv[1:] into path strings and canonical key→value pairs.
    Handles key=value, key= value, and key = value spacing variants.  Keys are
    normalised (lowercase, no underscores/dashes/spaces) then matched against
    a canonical alias table.  Returns (paths, kwargs).

  make_groq_client(api_key, base_url)
    Creates and returns a Groq client with GROQ_TIMEOUT applied.

  make_gemini_client(api_key, base_url)
    Creates and returns a google.genai.Client.  Passes a custom base_url only
    when non-empty, otherwise uses the Gemini default endpoint.  Raises
    ImportError if google-genai is not installed.

  get_duration(path)
    Calls ffprobe to read the duration (in seconds) of an audio file.

  ffmpeg_trim(src, dst, start, duration)
    Extracts a segment of src via ffmpeg stream copy into dst.

  chunk_file(path, tmpdir, known_dur)
    Splits path into CHUNK_SECONDS-length pieces.  Uses known_dur when
    provided to avoid reading unreliable container metadata on stream-copied
    files.  Returns list of chunk Paths.

  needs_chunking(path, known_dur)
    Returns True if path exceeds MAX_FILE_MB or CHUNK_SECONDS duration.

  probe_rate_limit(client, model)
    Sends a 0.5 s silent clip to Groq and reads x-ratelimit-* headers.
    Returns (limit_seconds, remaining_seconds); falls back to (7200, 7200).

  clean(text)
    Strips non-Arabic artefacts from a raw Whisper transcript: SRT/VTT
    timestamps, bracket annotations, non-Arabic characters, consecutive
    duplicate lines, and repeated sentences.

  transcribe_one(client, path, model)
    Uploads one audio file to Groq transcription. Returns raw text.

  transcribe(client, path, model, known_dur)
    Transcribes path, chunking first if needed.  Returns full raw text.

  estimate_tokens(text)
    Returns a conservative token-count estimate for Arabic text (len // 3).

  reorganize_text(text, mode, client, model, fallback_client, fallback_model)
    Sends text to Gemini with a structured system prompt.  Mode 1 extracts
    titles and subtitles only; mode 2 adds headers and marks off-topic
    passages as [تعليق: ...].  Falls back to fallback_client/model on error.
    Returns the reorganized text.

  fmt(secs)
    Formats seconds as "1h 4m 20s" or "34m 0s".

  collect_files(args, depth)
    Expands path strings into deduplicated .m4a Paths.  Searches folders up
    to depth levels (0 = unlimited rglob).

  load_log(folder)
    Reads transcription_log.jsonl from folder.  Returns {stem: entry} for all
    "done" entries.

  append_log(folder, entry)
    Appends one JSON line to folder's JSONL log.

  is_processed(stem, txt_path, log)
    Returns True if stem is in the log or txt_path exists on disk.

  merge_transcripts(folder, log)
    Reads ALL "done" entries for folder from log, collects their .txt files
    in alphabetical order, and writes merged.txt into folder.  Called only
    when no failures occurred in the current run.

  merge_reorganized(entries, out_path)
    Concatenates reorganized .txt files separated by a divider line and the
    original audio filename.  Writes result to out_path.

  main()
    Entry point.  Full pipeline: collect → durations → probe rate limit →
    overflow split → transcribe (Groq, with fallback) → merge transcripts
    (log-based, skipped if any failure) → reorganize (Gemini, optional) →
    merge reorganized (optional) → report overflow.

APP DEPENDENCIES:
  groq          — Groq Python SDK            (pip install groq)
  google-genai  — Google Gemini SDK          (pip install google-genai)
  ffmpeg        — audio trimming/chunking    (apt install ffmpeg)
  ffprobe       — duration detection, ships with ffmpeg

FILE DEPENDENCIES:
  config.py     — optional; defines GROQ_API_KEY, MODEL_NAME, API_URL,
                  FALLBACK_MODEL, FALLBACK_API_KEY, FALLBACK_API_URL,
                  GEMINI_API_KEY, GEMINI_FALLBACK_API_KEY, FILE_DEPTH.
                  All values can be supplied via CLI instead.
"""

import json
import re
import sys
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import groq as groq_sdk
from groq import Groq

# ── Defaults (script-level; overridden by config.py, then CLI) ────────────
_GROQ_API_KEY            = ""
_MODEL_NAME              = "whisper-large-v3-turbo"
_API_URL                 = ""
_FALLBACK_MODEL          = "whisper-large-v3"
_FALLBACK_API_KEY        = ""
_FALLBACK_API_URL        = ""
_FILE_DEPTH              = 1

_GEMINI_API_KEY          = ""
_GEMINI_MODEL            = "gemini-2.5-flash-lite"
_GEMINI_API_URL          = "https://generativelanguage.googleapis.com/v1beta/"
_GEMINI_FALLBACK_MODEL   = "gemini-2.5-flash"
_GEMINI_FALLBACK_API_KEY = ""
_GEMINI_FALLBACK_API_URL = "https://generativelanguage.googleapis.com/v1beta/"
_REORG_MODE              = 0
_REORG_MERGE             = 0

try:
    import config as _cfg
    _GROQ_API_KEY            = getattr(_cfg, "GROQ_API_KEY",            _GROQ_API_KEY)
    _MODEL_NAME              = getattr(_cfg, "MODEL_NAME",              _MODEL_NAME)
    _API_URL                 = getattr(_cfg, "API_URL",                 _API_URL)
    _FALLBACK_MODEL          = getattr(_cfg, "FALLBACK_MODEL",          _FALLBACK_MODEL)
    _FALLBACK_API_KEY        = getattr(_cfg, "FALLBACK_API_KEY",        _FALLBACK_API_KEY)
    _FALLBACK_API_URL        = getattr(_cfg, "FALLBACK_API_URL",        _FALLBACK_API_URL)
    _GEMINI_API_KEY          = getattr(_cfg, "GEMINI_API_KEY",          _GEMINI_API_KEY)
    _GEMINI_FALLBACK_API_KEY = getattr(_cfg, "GEMINI_FALLBACK_API_KEY", _GEMINI_FALLBACK_API_KEY)
    _FILE_DEPTH              = getattr(_cfg, "FILE_DEPTH",              _FILE_DEPTH)
except ImportError:
    pass

LOG_FILENAME    = "transcription_log.jsonl"
CHUNK_SECONDS   = 600    # 10-min chunks — safe margin under Groq's 25 MB limit
MAX_FILE_MB     = 20
GROQ_TIMEOUT    = 600.0  # seconds
MAX_REORG_TOKENS = 250_000

# ── Reorganizer system prompts ─────────────────────────────────────────────

REORG_PROMPT_SIMPLE = """\
You are an Arabic academic transcript organizer.
Your only task is to add structural markers to the provided Arabic transcript.

Rules:
- Add a main title (# Title) at the start of each new major topic
- Add a subtitle (## Subtitle) at the start of each sub-topic or point
- Add a blank line between paragraphs
- Preserve ALL original Arabic text exactly — do not add, remove, or alter any word
- Output ONLY the reorganized text with added markers and blank lines, nothing else
"""

REORG_PROMPT_ADVANCED = """\
You are an Arabic academic transcript organizer.
Your task is to add full structural markers to the provided Arabic transcript.

Rules:
- Add a main title (# Title) for each major topic
- Add a header (## Header) for major sections within a topic
- Add a subtitle (### Subtitle) for sub-topics and specific points
- Wrap any passage that is unrelated to the main subject of the recording \
in a comment tag exactly like this: [تعليق: <original text>]
  (off-topic = digressions, side conversations, unrelated anecdotes)
- Add a blank line between paragraphs
- Preserve ALL original Arabic text exactly — do not add, remove, or alter any word
- Output ONLY the reorganized text with added markers and comment tags, nothing else
"""

REORG_DIVIDER = "—" * 44   # divider line used in merged_reorganized.txt


# ── CLI argument parsing ───────────────────────────────────────────────────

_KEY_ALIASES: dict[str, str] = {
    # Groq
    "groqapikey":           "api_key",
    "apikey":               "api_key",
    "modelname":            "model",
    "model":                "model",
    "apiurl":               "api_url",
    "fallbackmodel":        "fallback_model",
    "fallbackapikey":       "fallback_api_key",
    "fallbackkey":          "fallback_api_key",
    "fallbackapiurl":       "fallback_api_url",
    "fallbackurl":          "fallback_api_url",
    "filedepth":            "file_depth",
    "depth":                "file_depth",
    # Gemini
    "geminiapikey":         "gemini_api_key",
    "geminimodel":          "gemini_model",
    "geminiapiurl":         "gemini_api_url",
    "geminifallbackmodel":  "gemini_fallback_model",
    "geminifallbackapikey": "gemini_fallback_api_key",
    "geminifallbackkey":    "gemini_fallback_api_key",
    "geminifallbackapiurl": "gemini_fallback_api_url",
    "geminifallbackurl":    "gemini_fallback_api_url",
    "reorgmode":            "reorg_mode",
    "reorganizemode":       "reorg_mode",
    "reorganizationmode":   "reorg_mode",
    "reorgmerge":           "reorg_merge",
    "mergeorganized":       "reorg_merge",
    "mergeranized":         "reorg_merge",
}


def _norm(key: str) -> str:
    return key.strip().lower().replace("_", "").replace("-", "").replace(" ", "")


def parse_args(argv: list[str]) -> tuple[list[str], dict[str, str]]:
    """
    Split argv into path strings and canonical key→value pairs.
    Handles key=value, key= value, and key = value spacing variants.
    """
    paths:  list[str]      = []
    kwargs: dict[str, str] = {}
    i = 0
    while i < len(argv):
        token = argv[i].strip()
        if "=" in token:
            raw_key, _, value = token.partition("=")
            value = value.strip()
            if not value and i + 1 < len(argv):
                nxt = argv[i + 1].strip()
                if "=" not in nxt:
                    value = nxt
                    i += 1
            canonical = _KEY_ALIASES.get(_norm(raw_key))
            if canonical:
                kwargs[canonical] = value
        elif token == "=":
            pass
        else:
            if i + 1 < len(argv):
                nxt = argv[i + 1].strip()
                if nxt == "=" and i + 2 < len(argv):
                    canonical = _KEY_ALIASES.get(_norm(token))
                    if canonical:
                        kwargs[canonical] = argv[i + 2].strip()
                        i += 3
                        continue
                elif nxt.startswith("=") and len(nxt) > 1:
                    canonical = _KEY_ALIASES.get(_norm(token))
                    if canonical:
                        kwargs[canonical] = nxt[1:].strip()
                        i += 2
                        continue
            paths.append(token)
        i += 1
    return paths, kwargs


# ── Client factories ───────────────────────────────────────────────────────

def make_groq_client(api_key: str, base_url: str = "") -> Groq:
    kw: dict = {"api_key": api_key, "timeout": GROQ_TIMEOUT}
    if base_url:
        kw["base_url"] = base_url
    return Groq(**kw)


def make_gemini_client(api_key: str, base_url: str = ""):
    try:
        from google import genai as _genai
    except ImportError:
        raise ImportError("google-genai is required for reorganization. Run: pip install google-genai")
    if base_url:
        try:
            from google.genai.types import HttpOptions
            return _genai.Client(api_key=api_key, http_options=HttpOptions(base_url=base_url))
        except ImportError:
            pass
    return _genai.Client(api_key=api_key)


# ── ffmpeg / ffprobe helpers ───────────────────────────────────────────────

def get_duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error",
         "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1",
         str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(r.stdout.strip())


def ffmpeg_trim(src: Path, dst: Path, start: float, duration: float) -> None:
    cmd = ["ffmpeg", "-y", "-i", str(src), "-c", "copy"]
    if start:
        cmd += ["-ss", str(start)]
    cmd += ["-t", str(duration), str(dst)]
    subprocess.run(cmd, capture_output=True, check=True)


def chunk_file(path: Path, tmpdir: str, known_dur: float | None = None) -> list[Path]:
    """Split into CHUNK_SECONDS-length pieces for the Groq 25 MB limit."""
    total = known_dur if known_dur is not None else get_duration(path)
    chunks, start, idx = [], 0.0, 0
    while start < total:
        out = Path(tmpdir) / f"{path.stem}_ck{idx:03d}.m4a"
        ffmpeg_trim(path, out, start, CHUNK_SECONDS)
        chunks.append(out)
        start += CHUNK_SECONDS
        idx   += 1
    return chunks


def needs_chunking(path: Path, known_dur: float | None = None) -> bool:
    if path.stat().st_size / 1_048_576 >= MAX_FILE_MB:
        return True
    dur = known_dur if known_dur is not None else get_duration(path)
    return dur > CHUNK_SECONDS


# ── Groq rate-limit probe ──────────────────────────────────────────────────

def probe_rate_limit(client: Groq, model: str) -> tuple[int, int]:
    """
    Send a tiny silent clip to Groq and read rate-limit headers.
    Returns (limit_seconds, remaining_seconds); falls back to (7200, 7200).
    """
    print("Probing Groq rate limit...")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            probe = Path(tmpdir) / "probe.m4a"
            subprocess.run(
                ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
                 "-t", "0.5", "-c:a", "aac", str(probe)],
                capture_output=True, check=True,
            )
            with open(probe, "rb") as f:
                raw = client.audio.transcriptions.with_raw_response.create(
                    model=model, file=f, language="ar", response_format="text",
                )
            h         = raw.headers
            limit     = int(float(h.get("x-ratelimit-limit-audio-seconds",     7200)))
            remaining = int(float(h.get("x-ratelimit-remaining-audio-seconds", limit)))
            print(f"  Limit    : {fmt(limit)}")
            print(f"  Remaining: {fmt(remaining)}")
            return limit, remaining
    except Exception as exc:
        print(f"  Probe failed ({exc}), using default 2-hour budget.")
        return 7200, 7200


# ── Transcript cleaning ────────────────────────────────────────────────────

_STRIP = re.compile(
    r"[^"
    r"؀-ۿ"   # Basic Arabic
    r"ݐ-ݿ"   # Arabic Supplement
    r"ࢠ-ࣿ"   # Arabic Extended-A
    r"ﭐ-﷿"   # Arabic Presentation Forms-A
    r"ﹰ-﻿"   # Arabic Presentation Forms-B
    r"٠-٩"   # Arabic-Indic digits
    r"0-9\s"
    r"،؛؟ـ"
    r'.,;:!?()\-\"\']+'
)
_TIMESTAMPS = re.compile(r"\b\d{1,2}:\d{2}(:\d{2})?(\.\d+)?\b")


def clean(text: str) -> str:
    text = re.sub(r"\d{2}:\d{2}:\d{2}[,\.]\d+\s*-->\s*\d{2}:\d{2}:\d{2}[,\.]\d+", "", text)
    text = _TIMESTAMPS.sub("", text)
    text = re.sub(r"\[.*?\]|\(.*?\)", " ", text, flags=re.DOTALL)
    text = _STRIP.sub(" ", text)
    deduped: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            deduped.append("")
        elif not deduped or deduped[-1].strip() != s:
            deduped.append(line)
    sentences = re.split(r"(?<=[.،؟!])\s+", "\n".join(deduped))
    seen: dict[str, int] = {}
    out: list[str] = []
    for s in sentences:
        key = s.strip()
        if not key:
            continue
        seen[key] = seen.get(key, 0) + 1
        if seen[key] == 1:
            out.append(s)
    text = " ".join(out)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Groq transcription ─────────────────────────────────────────────────────

def transcribe_one(client: Groq, path: Path, model: str) -> str:
    with open(path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model=model, file=f, language="ar", response_format="text",
        )
    return resp if isinstance(resp, str) else resp.text


def transcribe(client: Groq, path: Path, model: str, known_dur: float | None = None) -> str:
    if not needs_chunking(path, known_dur):
        return transcribe_one(client, path, model)
    parts: list[str] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for chunk in chunk_file(path, tmpdir, known_dur):
            parts.append(transcribe_one(client, chunk, model))
    return "\n".join(parts)


# ── Gemini reorganizer ─────────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """Conservative token estimate for Arabic text (3 chars per token)."""
    return len(text) // 3


def reorganize_text(
    text: str,
    mode: int,
    client,
    model: str,
    fallback_client=None,
    fallback_model: str = "",
) -> str:
    """Call Gemini to reorganize transcript text according to mode."""
    from google.genai import types as _types

    prompt = REORG_PROMPT_SIMPLE if mode == 1 else REORG_PROMPT_ADVANCED

    def _call(c, m: str) -> str:
        resp = c.models.generate_content(
            model=m,
            contents=text,
            config=_types.GenerateContentConfig(
                system_instruction=prompt,
                temperature=0.2,
            ),
        )
        return resp.text

    try:
        return _call(client, model)
    except Exception as exc:
        if fallback_client and fallback_model:
            print(f"  Gemini primary error ({exc}), trying fallback {fallback_model}...")
            return _call(fallback_client, fallback_model)
        raise


# ── Helpers ────────────────────────────────────────────────────────────────

def fmt(secs: float) -> str:
    m, s = divmod(int(secs), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s" if h else f"{m}m {s}s"


def collect_files(args: list[str], depth: int = 1) -> list[Path]:
    """Expand path strings into .m4a Paths, searching folders up to depth levels."""
    files: list[Path] = []
    for arg in args:
        p = Path(arg).resolve()
        if p.is_dir():
            if depth == 0:
                files.extend(sorted(p.rglob("*.m4a")))
            else:
                for level in range(depth):
                    prefix  = "/".join(["*"] * level)
                    pattern = f"{prefix}/*.m4a" if prefix else "*.m4a"
                    files.extend(sorted(p.glob(pattern)))
        elif p.is_file() and p.suffix.lower() == ".m4a":
            files.append(p)
        else:
            print(f"  Warning: '{arg}' is not a .m4a file or folder — skipped.")
    seen:   set[Path]  = set()
    unique: list[Path] = []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return unique


# ── JSONL log ─────────────────────────────────────────────────────────────

def load_log(folder: Path) -> dict[str, dict]:
    """Return {stem: entry} for all 'done' entries in folder's JSONL log."""
    log_path = folder / LOG_FILENAME
    if not log_path.exists():
        return {}
    entries: dict[str, dict] = {}
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            if entry.get("status") == "done":
                entries[entry["stem"]] = entry
        except (json.JSONDecodeError, KeyError):
            pass
    return entries


def append_log(folder: Path, entry: dict) -> None:
    """Append one JSON line to folder's JSONL log."""
    log_path = folder / LOG_FILENAME
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def is_processed(stem: str, txt_path: Path, log: dict[str, dict]) -> bool:
    """True if already transcribed: checks log first, then .txt on disk."""
    if stem in log:
        return True
    return txt_path.exists()


# ── Merge helpers ──────────────────────────────────────────────────────────

def merge_transcripts(folder: Path, log: dict[str, dict]) -> Path | None:
    """
    Merge ALL 'done' .txt files for folder (from log, sorted alphabetically)
    into merged.txt.  Returns the merged path, or None if nothing to merge.
    """
    entries = sorted(log.values(), key=lambda e: e["audio_file"])
    txt_files = [folder / e["txt_file"] for e in entries if (folder / e["txt_file"]).exists()]
    if len(txt_files) < 2:
        return None
    merged_path = folder / "merged.txt"
    parts = [p.read_text(encoding="utf-8") for p in txt_files]
    merged_path.write_text("\n\n".join(parts), encoding="utf-8")
    return merged_path


def merge_reorganized(entries: list[tuple[Path, str]], out_path: Path) -> None:
    """
    Concatenate reorganized .txt files with a divider + original audio name.
    entries: list of (reorganized_txt_path, original_audio_filename)
    """
    parts: list[str] = []
    for reorg_path, audio_name in entries:
        if reorg_path.exists():
            content = reorg_path.read_text(encoding="utf-8")
            parts.append(f"{REORG_DIVIDER}\n{audio_name}\n{content}")
    if parts:
        out_path.write_text("\n\n".join(parts), encoding="utf-8")


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Usage: python3 transcribe.py <path> [<path> ...] [key=value ...]\n"
            "\n"
            "Groq (transcription):\n"
            "  groq_api_key=   model_name=   api_url=\n"
            "  fallbackmodel=  fallbackapikey=  fallbackapiurl=  filedepth=\n"
            "\n"
            "Gemini (reorganizer):\n"
            "  gemini_api_key=  geminimodel=  geminiapiurl=\n"
            "  geminifallbackmodel=  geminifallbackapikey=  geminifallbackapiurl=\n"
            "  reorgmode=<0|1|2>  reorgmerge=<0|1>\n"
        )
        sys.exit(1)

    path_args, cli = parse_args(sys.argv[1:])

    # ── Resolve effective configuration ───────────────────────────────────
    api_key        = cli.get("api_key")          or _GROQ_API_KEY
    primary_model  = cli.get("model")            or _MODEL_NAME
    api_url        = cli.get("api_url")          or _API_URL
    fallback_model = cli.get("fallback_model")   or _FALLBACK_MODEL
    fb_api_key     = cli.get("fallback_api_key") or _FALLBACK_API_KEY or api_key
    fb_api_url     = cli.get("fallback_api_url") or _FALLBACK_API_URL or api_url
    file_depth     = int(cli.get("file_depth")   or _FILE_DEPTH)

    gemini_api_key  = cli.get("gemini_api_key")          or _GEMINI_API_KEY
    gemini_model    = cli.get("gemini_model")             or _GEMINI_MODEL
    gemini_api_url  = cli.get("gemini_api_url")           or _GEMINI_API_URL
    gfb_model       = cli.get("gemini_fallback_model")    or _GEMINI_FALLBACK_MODEL
    gfb_api_key     = cli.get("gemini_fallback_api_key")  or _GEMINI_FALLBACK_API_KEY or gemini_api_key
    gfb_api_url     = cli.get("gemini_fallback_api_url")  or _GEMINI_FALLBACK_API_URL or gemini_api_url
    reorg_mode      = int(cli.get("reorg_mode")  or _REORG_MODE)
    reorg_merge     = int(cli.get("reorg_merge") or _REORG_MERGE)

    if not api_key:
        print("ERROR: no Groq API key. Pass groq_api_key=<key> or set GROQ_API_KEY in config.py")
        sys.exit(1)

    if reorg_mode not in (0, 1, 2):
        print(f"ERROR: reorgmode must be 0, 1, or 2 (got {reorg_mode})")
        sys.exit(1)

    # ── Step 1: collect files ──────────────────────────────────────────────
    m4a_files = collect_files(path_args, file_depth)
    if not m4a_files:
        print("No .m4a files found.")
        sys.exit(0)

    # ── Step 2: read durations ─────────────────────────────────────────────
    print("Reading durations...")
    file_dur: list[tuple[Path, float]] = []
    for f in m4a_files:
        try:
            d = get_duration(f)
            file_dur.append((f, d))
            print(f"  {f.name:<50} {fmt(d)}")
        except Exception as exc:
            print(f"  {f.name}: skipped — {exc}")

    total = sum(d for _, d in file_dur)
    print(f"\n  Total: {fmt(total)}\n")

    # ── Step 3: probe Groq for rate-limit budget ───────────────────────────
    primary_client = make_groq_client(api_key, api_url)
    _, budget = probe_rate_limit(primary_client, primary_model)
    print()

    # ── Step 4: resolve overflow ───────────────────────────────────────────
    overflow_path: Path | None = None
    overflow_secs: float       = 0.0

    # to_transcribe: (audio_path, output_stem, known_dur | None)
    if total <= budget:
        to_transcribe = [(p, p.stem, None) for p, _ in file_dur]
    else:
        overflow_secs = total - budget
        print(f"  Overflow: {fmt(overflow_secs)} — trimming the largest file.")

        largest_path, largest_dur = max(file_dur, key=lambda x: x[1])

        if overflow_secs >= largest_dur:
            print(
                f"\nERROR: overflow ({fmt(overflow_secs)}) ≥ largest file "
                f"({largest_path.name}, {fmt(largest_dur)}).\n"
                f"       Cannot absorb with a single split. Reduce input."
            )
            sys.exit(1)

        keep_secs = largest_dur - overflow_secs
        part1 = largest_path.parent / f"{largest_path.stem}part1.m4a"
        part2 = largest_path.parent / f"{largest_path.stem}part2.m4a"

        print(f"\n  Splitting {largest_path.name}:")
        print(f"    → {part1.name}  ({fmt(keep_secs)})  [transcribed]")
        print(f"    → {part2.name}  ({fmt(overflow_secs)})  [overflow — skipped]")

        ffmpeg_trim(largest_path, part1, 0.0, keep_secs)
        ffmpeg_trim(largest_path, part2, keep_secs, overflow_secs)
        overflow_path = part2

        to_transcribe = []
        for p, _ in file_dur:
            if p == largest_path:
                to_transcribe.append((part1, part1.stem, keep_secs))
            else:
                to_transcribe.append((p, p.stem, None))

    # ── Step 5: transcribe ─────────────────────────────────────────────────
    print()
    current_model        = primary_model
    current_client       = primary_client
    fallback_groq_client: Groq | None = None
    saved_txts:  list[Path]             = []
    has_failures                        = False

    logs: dict[Path, dict[str, dict]] = {}

    def get_log(folder: Path) -> dict[str, dict]:
        if folder not in logs:
            logs[folder] = load_log(folder)
        return logs[folder]

    def save_result(audio_path: Path, stem: str, out_txt: Path,
                    model: str, known_dur: float | None, text: str) -> None:
        out_txt.write_text(text, encoding="utf-8")
        saved_txts.append(out_txt)
        dur = known_dur if known_dur is not None else get_duration(audio_path)
        entry = {
            "stem":       stem,
            "audio_file": audio_path.name,
            "txt_file":   out_txt.name,
            "model":      model,
            "duration_s": round(dur, 2),
            "chars":      len(text),
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "status":     "done",
        }
        append_log(audio_path.parent, entry)
        logs.setdefault(audio_path.parent, {})[stem] = entry
        print(f"  ✓ saved  {out_txt.name}  ({len(text):,} chars)")

    for audio_path, stem, known_dur in to_transcribe:
        out_txt = audio_path.parent / f"{stem}.txt"
        log     = get_log(audio_path.parent)

        if is_processed(stem, out_txt, log):
            print(f"Skipping {audio_path.name} (already transcribed)")
            if out_txt.exists():
                saved_txts.append(out_txt)
            continue

        print(f"Transcribing  {audio_path.name}  [{current_model}] ...")
        try:
            raw  = transcribe(current_client, audio_path, current_model, known_dur)
            text = clean(raw)
            save_result(audio_path, stem, out_txt, current_model, known_dur, text)

        except groq_sdk.RateLimitError:
            if current_client is primary_client:
                print(f"  Rate limit hit on {current_model}. Switching to fallback: {fallback_model} ...")
                current_model = fallback_model
                if fallback_groq_client is None:
                    fallback_groq_client = make_groq_client(fb_api_key, fb_api_url)
                current_client = fallback_groq_client
                try:
                    raw  = transcribe(current_client, audio_path, current_model, known_dur)
                    text = clean(raw)
                    save_result(audio_path, stem, out_txt, current_model, known_dur, text)
                except Exception as exc2:
                    print(f"  ✗ ERROR on fallback: {exc2}")
                    has_failures = True
            else:
                print(f"  ✗ Rate limit also hit on fallback {current_model}: {audio_path.name} skipped.")
                has_failures = True

        except Exception as exc:
            print(f"  ✗ ERROR: {exc}")
            has_failures = True

    # ── Step 6: merge transcripts (log-based, all runs) ────────────────────
    if has_failures:
        print("\n  Merge skipped — transcription failures in this run. Retry to merge.")
    else:
        involved_folders = {audio_path.parent for audio_path, _, _ in to_transcribe}
        for folder in involved_folders:
            log = logs.get(folder) or load_log(folder)
            merged = merge_transcripts(folder, log)
            if merged:
                n = len([e for e in log.values() if (folder / e["txt_file"]).exists()])
                print(f"\n  Merged {n} transcripts → {merged}")

    # ── Step 7: reorganize (Gemini) ────────────────────────────────────────
    if reorg_mode == 0:
        pass
    elif not gemini_api_key:
        print("\n  Reorganization skipped — no Gemini API key provided.")
        print("  Set GEMINI_API_KEY in config.py or pass gemini_api_key=<key>")
    else:
        print(f"\nReorganizing transcripts (mode {reorg_mode}) ...")
        try:
            gemini_client = make_gemini_client(gemini_api_key, gemini_api_url)
            gfb_client    = (make_gemini_client(gfb_api_key, gfb_api_url)
                             if gfb_api_key != gemini_api_key or gfb_model != gemini_model
                             else gemini_client)
        except ImportError as exc:
            print(f"  Skipped: {exc}")
            gemini_client = None

        if gemini_client:
            reorg_entries: list[tuple[Path, str]] = []   # (reorg_txt, audio_name)

            for audio_path, stem, known_dur in to_transcribe:
                src_txt = audio_path.parent / f"{stem}.txt"
                if not src_txt.exists():
                    continue

                out_reorg = audio_path.parent / f"{stem}_reorganized.txt"
                if out_reorg.exists():
                    print(f"  Skipping {src_txt.name} (reorganized file exists)")
                    reorg_entries.append((out_reorg, audio_path.name))
                    continue

                text = src_txt.read_text(encoding="utf-8")
                tok  = estimate_tokens(text)
                if tok > MAX_REORG_TOKENS:
                    print(f"  Skipping {src_txt.name} (~{tok:,} tokens > {MAX_REORG_TOKENS:,} limit)")
                    continue

                print(f"  Reorganizing  {src_txt.name}  (~{tok:,} tokens) ...")
                try:
                    reorganized = reorganize_text(
                        text, reorg_mode,
                        gemini_client, gemini_model,
                        gfb_client,    gfb_model,
                    )
                    out_reorg.write_text(reorganized, encoding="utf-8")
                    reorg_entries.append((out_reorg, audio_path.name))
                    print(f"    ✓ saved  {out_reorg.name}  ({len(reorganized):,} chars)")
                except Exception as exc:
                    print(f"    ✗ ERROR: {exc}")

            if reorg_merge and len(reorg_entries) > 1:
                parents    = {p.parent for p, _ in reorg_entries}
                merge_dir  = reorg_entries[0][0].parent if len(parents) == 1 else Path.cwd()
                out_path   = merge_dir / "merged_reorganized.txt"
                merge_reorganized(reorg_entries, out_path)
                print(f"\n  Merged {len(reorg_entries)} reorganized transcripts → {out_path}")

    # ── Step 8: report overflow ────────────────────────────────────────────
    if overflow_path:
        print(
            f"\n⚠  Overflow file NOT transcribed: {overflow_path.name} "
            f"({fmt(overflow_secs)})\n"
            f"   Run again (or with a fresh API key) to transcribe the rest."
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
