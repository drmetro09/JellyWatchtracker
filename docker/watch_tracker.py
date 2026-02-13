from flask import Flask, request, jsonify, send_file, Response
from datetime import datetime, timedelta, timezone
import json
import os
from collections import Counter, defaultdict
import io
import requests
import traceback
from concurrent.futures import ThreadPoolExecutor
import threading
from werkzeug.utils import secure_filename
import csv
import time
import re
import unicodedata
from difflib import SequenceMatcher
import requests

app = Flask(__name__)

@app.after_request
def add_no_cache_headers(response):
    """
    Prevent stale HTML/JS/API responses in browsers and service-worker caches.
    This avoids running old frontend code after backend updates.
    """
    try:
        p = request.path or ""
        if p == "/" or p == "/sw.js" or p == "/manifest.json" or p.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
    except Exception:
        pass
    return response

DATA_FILE = "/data/watch_history.json"
POSTER_DIR = "/data/custom_posters"
JELLYFIN_URL = os.getenv("JELLYFIN_URL", "").rstrip("/")
JELLYFIN_API_KEY = os.getenv("JELLYFIN_API_KEY", "")
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
SONARR_URL = os.getenv("SONARR_URL", "").rstrip("/")
SONARR_API_KEY = os.getenv("SONARR_API_KEY", "")
RADARR_URL = os.getenv("RADARR_URL", "").rstrip("/")
RADARR_API_KEY = os.getenv("RADARR_API_KEY", "")
# Used for "Open in Jellyfin" links in the UI. If Jellyfin is behind a reverse proxy,
# set this to the public base URL (e.g. https://media.example.com/jellyfin).
JELLYFIN_PUBLIC_URL = os.getenv("JELLYFIN_PUBLIC_URL", "").rstrip("/")

# Jellyfin polling fallback:
# Some clients/autoplay flows don't reliably emit a "stop/ended" webhook.
# This lightweight poller periodically checks Jellyfin for newly played items
# and appends them to the local history.
def _env_bool(name, default=False):
    v = os.getenv(name)
    if v is None:
        return default
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y", "on")

def _env_int(name, default):
    try:
        v = os.getenv(name, None)
        if v is None:
            return default
        return int(str(v).strip())
    except Exception:
        return default

JELLYFIN_POLL_ENABLED = _env_bool("JELLYFIN_POLL_ENABLED", default=True)
JELLYFIN_POLL_INTERVAL_S = _env_int("JELLYFIN_POLL_INTERVAL_S", 60)
JELLYFIN_POLL_LIMIT = _env_int("JELLYFIN_POLL_LIMIT", 80)
JELLYFIN_POLL_LOOKBACK_S = _env_int("JELLYFIN_POLL_LOOKBACK_S", 7 * 24 * 3600)
JELLYFIN_POLL_STATE_FILE = "/data/jellyfin_poll_state.json"

cache = {"data": None, "time": None}
poster_cache = {}
series_cache = {}
# Track one-time TMDB retries for shows that previously failed
tmdb_retry = set()
custom_posters = {}
manual_complete = {}
season_complete = {}
genre_insights_cache = {"sig": None, "data": None}
history_sig_fast = None
genre_recs_cache = {"sig": None, "top_genre": None, "ts": 0, "movies": [], "shows": []}
_prewarm_lock = threading.Lock()
_prewarm_busy = False
_tmdb_prewarm_lock = threading.Lock()
_tmdb_prewarm_busy = False
realtime_seq = 0
realtime_cv = threading.Condition()
webhook_recent = {}
webhook_recent_lock = threading.Lock()
WEBHOOK_DEDUPE_SECONDS = 20
TMDB_TOUCH_FILE = "/data/tmdb_touch.txt"
_tmdb_touch_lock = threading.Lock()
_tmdb_touch_timer = None
_tmdb_pending_lock = threading.Lock()
_tmdb_pending_posters = set()
_tmdb_pending_series = set()
_tmdb_executor = ThreadPoolExecutor(max_workers=4)

def _strip_year_suffix(name):
    s = str(name or "").strip()
    m = re.search(r"\s*\((\d{4})\)\s*$", s)
    return (s[:m.start()].strip() if m else s)

def _update_show_totals_from_tmdb(series_name, expected_year, series_info):
    """
    Persist TMDB total-episode counts so /api/history can stay cache-only (fast)
    while still showing correct watched/total for shows.
    """
    try:
        if not isinstance(series_info, dict):
            return
        total_all = _coerce_int(series_info.get("total_episodes_in_series"), 0) or 0
        if total_all <= 0:
            return

        seasons_meta = series_info.get("seasons", {}) or {}
        season_map = {}
        if isinstance(seasons_meta, dict):
            for snum, meta in seasons_meta.items():
                try:
                    sn = _coerce_int(snum, None)
                except Exception:
                    sn = None
                if sn is None or sn <= 0:
                    continue
                if isinstance(meta, dict):
                    st = _coerce_int(meta.get("total_episodes"), 0) or 0
                else:
                    st = _coerce_int(meta, 0) or 0
                if st > 0:
                    season_map[str(sn)] = st

        # Store under a few safe aliases to survive webhook naming differences.
        raw = str(series_name or "").strip()
        base = _strip_year_suffix(raw)
        aliases = []
        for n in (raw, base):
            if not n:
                continue
            aliases.extend([n, n.lower(), _norm_title(n)])
        # Also store under TMDB-canonical name to survive manual-entry formatting differences.
        try:
            tmdbn = str((series_info or {}).get("tmdb_name") or "").strip()
        except Exception:
            tmdbn = ""
        if tmdbn:
            tmdb_base = _strip_year_suffix(tmdbn)
            for n in (tmdbn, tmdb_base):
                if not n:
                    continue
                aliases.extend([n, n.lower(), _norm_title(n)])

        payload = load_json_file(SHOW_TOTALS_FILE)
        if not isinstance(payload, dict):
            payload = {}

        changed = False
        for key in aliases:
            if not key:
                continue
            cur = payload.get(key)
            if not isinstance(cur, dict):
                cur = {"total": 0, "seasons": {}}
            cur_total = _coerce_int(cur.get("total"), 0) or 0
            if total_all > cur_total:
                cur["total"] = total_all
                changed = True
            cur_seasons = cur.get("seasons")
            if not isinstance(cur_seasons, dict):
                cur_seasons = {}
                cur["seasons"] = cur_seasons
            for sn, st in season_map.items():
                prev = _coerce_int(cur_seasons.get(sn), 0) or 0
                if st > prev:
                    cur_seasons[sn] = st
                    changed = True
            payload[key] = cur

        if changed:
            save_json_file(SHOW_TOTALS_FILE, payload)
    except Exception:
        pass

def _file_sig(path):
    try:
        st = os.stat(path)
        return f"{st.st_mtime_ns}:{st.st_size}"
    except Exception:
        return "0"

def _data_signature():
    return "|".join([
        _file_sig(DATA_FILE),
        _file_sig(RATINGS_FILE),
        _file_sig("/data/manual_complete.json"),
        _file_sig("/data/season_complete.json"),
        _file_sig("/data/custom_posters.json"),
        _file_sig(SHOW_TOTALS_FILE),
        _file_sig(TMDB_TOUCH_FILE),
    ])

def _touch_tmdb_cache_and_notify():
    """
    Touch a small marker file so the UI can re-render after poster/series caches
    are rebuilt, without including the full cache files in the data signature.
    """
    try:
        os.makedirs("/data", exist_ok=True)
        with open(TMDB_TOUCH_FILE, "w") as f:
            f.write(str(time.time_ns()))
    except Exception:
        pass
    # Invalidate the organized cache and notify clients.
    try:
        cache["time"] = None
        cache["sig"] = None
    except Exception:
        pass
    refresh_history_sig_fast()

def _schedule_tmdb_touch(delay_s=1.5):
    global _tmdb_touch_timer
    try:
        with _tmdb_touch_lock:
            if _tmdb_touch_timer and _tmdb_touch_timer.is_alive():
                return
            t = threading.Timer(delay_s, _touch_tmdb_cache_and_notify)
            t.daemon = True
            _tmdb_touch_timer = t
            t.start()
    except Exception:
        pass

def _queue_tmdb_poster_fetch(title, year=None, media_type="movie"):
    if not TMDB_API_KEY:
        return
    try:
        key = f"{media_type}_{title}_{year}".lower()
        cached = poster_cache.get(key)
        if isinstance(cached, str) and cached:
            return
        if isinstance(cached, dict) and cached.get("url"):
            return
        with _tmdb_pending_lock:
            if key in _tmdb_pending_posters:
                return
            _tmdb_pending_posters.add(key)

        def _run():
            try:
                get_tmdb_poster(title, year, media_type, allow_network=True)
            finally:
                with _tmdb_pending_lock:
                    _tmdb_pending_posters.discard(key)
                _schedule_tmdb_touch()

        _tmdb_executor.submit(_run)
    except Exception:
        return

def _queue_tmdb_series_fetch(series_name, expected_year=None):
    if not TMDB_API_KEY:
        return
    try:
        skey = f"{str(series_name or '').strip().lower()}|{str(expected_year or '')}"
        with _tmdb_pending_lock:
            if skey in _tmdb_pending_series:
                return
            _tmdb_pending_series.add(skey)

        def _run():
            try:
                # Webhooks often provide episode-year values, and transient TMDB failures
                # shouldn't pin a show to "fetching..." for hours. Try multiple lookups:
                # 1) year-hinted (if provided)
                # 2) yearless (title-only)
                # 3) forced refresh yearless to bypass stale negative caches
                info = get_tmdb_series_info(series_name, expected_year=expected_year, allow_network=True)
                if not info and expected_year is not None:
                    info = get_tmdb_series_info(series_name, expected_year=None, allow_network=True)
                if not info:
                    info = get_tmdb_series_info(series_name, expected_year=None, force_refresh=True, allow_network=True)
                if info:
                    _update_show_totals_from_tmdb(series_name, expected_year, info)
            finally:
                with _tmdb_pending_lock:
                    _tmdb_pending_series.discard(skey)
                _schedule_tmdb_touch()

        _tmdb_executor.submit(_run)
    except Exception:
        return

def prewarm_tmdb_cache_async(limit_posters=200, limit_series=150, force=False):
    global _tmdb_prewarm_busy
    if not TMDB_API_KEY:
        return
    with _tmdb_prewarm_lock:
        if _tmdb_prewarm_busy and not force:
            return
        _tmdb_prewarm_busy = True

    def _run():
        global _tmdb_prewarm_busy
        try:
            org = {}
            try:
                # organize_data() is fast now (cache-only TMDB reads).
                org = organize_data() or {}
            except Exception:
                org = {}

            posters = 0
            series = 0

            for m in (org.get("movies") or []):
                if posters >= limit_posters:
                    break
                if not m:
                    continue
                if not m.get("poster"):
                    _queue_tmdb_poster_fetch(m.get("name"), m.get("year"), "movie")
                    posters += 1

            for s in (org.get("shows") or []):
                if posters < limit_posters and not s.get("poster"):
                    _queue_tmdb_poster_fetch(s.get("series_name"), s.get("year"), "tv")
                    posters += 1
                if series >= limit_series:
                    continue
                # When forced (e.g. after clearing TMDB cache), always rebuild series info.
                # Otherwise, rebuild only when we don't have reliable totals yet.
                total_watched = _coerce_int(s.get("total_episodes"), 0) or 0
                total_possible = _coerce_int(s.get("total_episodes_possible"), 0) or 0
                looks_inferred = (
                    (total_watched > 0 and total_possible <= 0) or
                    (total_possible > 0 and total_watched > 0 and total_possible <= total_watched)
                )
                if force or (not s.get("has_tmdb_data")) or looks_inferred:
                    # Stored "year" for episode records is often the episode year, not the series start year.
                    # Passing it here can cause mismatches and prolonged negative caching.
                    _queue_tmdb_series_fetch(s.get("series_name"), expected_year=None)
                    series += 1
        finally:
            with _tmdb_prewarm_lock:
                _tmdb_prewarm_busy = False

    threading.Thread(target=_run, daemon=True).start()

# === New files and constants for advanced features ===
# Ratings storage file. Each entry keyed by unique item identifier (e.g., "movie|Inception|2010").
RATINGS_FILE = "/data/ratings.json"
# Action history file for undo functionality. Stores the last 20 actions performed that can be undone.
ACTION_HISTORY_FILE = "/data/action_history.json"
# User preferences file used to persist theme/layout selections across devices.
PREFERENCES_FILE = "/data/user_preferences.json"
SHOW_TOTALS_FILE = "/data/show_totals_cache.json"

def load_json_file(path):
    """
    Helper to load a JSON file. Returns an empty dict if the file does not exist
    or cannot be parsed.
    """
    try:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def save_json_file(path, data):
    """
    Helper to save a dictionary to a JSON file. Fails silently on error.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)
        refresh_history_sig_fast()
    except Exception:
        pass

def load_ratings():
    """
    Load ratings from RATINGS_FILE. Returns a dict mapping item keys to rating info.
    """
    return load_json_file(RATINGS_FILE)

def save_ratings(ratings):
    """
    Save ratings dictionary to RATINGS_FILE.
    """
    save_json_file(RATINGS_FILE, ratings)

def load_action_history():
    """
    Load the action history list from ACTION_HISTORY_FILE. The file stores a dict
    with key 'history' mapping to a list of actions.
    """
    data = load_json_file(ACTION_HISTORY_FILE)
    return data.get("history", []) if isinstance(data, dict) else []

def save_action_history(history):
    """
    Save the provided history list to ACTION_HISTORY_FILE under key 'history'.
    """
    save_json_file(ACTION_HISTORY_FILE, {"history": history})

def record_action(action_type, payload):
    """
    Append a new action to the history. Keeps only the last 20 actions.

    :param action_type: string identifier for the action (e.g. 'delete_movie')
    :param payload: dict containing the information needed to undo the action
    """
    try:
        history = load_action_history()
        history.append({"type": action_type, "payload": payload, "timestamp": datetime.now().isoformat()})
        # keep only last 20 actions
        history = history[-20:]
        save_action_history(history)
    except Exception as e:
        print(f"Failed to record action: {e}")


def load_caches():
    global poster_cache, series_cache, custom_posters, manual_complete, season_complete
    try:
        if os.path.exists("/data/poster_cache.json"):
            with open("/data/poster_cache.json") as f:
                poster_cache = json.load(f)
        if os.path.exists("/data/series_cache.json"):
            with open("/data/series_cache.json") as f:
                series_cache = json.load(f)
        if os.path.exists("/data/custom_posters.json"):
            with open("/data/custom_posters.json") as f:
                custom_posters = json.load(f)
        if os.path.exists("/data/manual_complete.json"):
            with open("/data/manual_complete.json") as f:
                manual_complete = json.load(f)
        if os.path.exists("/data/season_complete.json"):
            with open("/data/season_complete.json") as f:
                season_complete = json.load(f)
    except Exception as e:
        print(f"Cache load error: {e}")


def save_cache(cache_type):
    try:
        if cache_type == "poster":
            with open("/data/poster_cache.json", "w") as f:
                json.dump(poster_cache, f)
        elif cache_type == "series":
            with open("/data/series_cache.json", "w") as f:
                json.dump(series_cache, f)
        elif cache_type == "custom":
            with open("/data/custom_posters.json", "w") as f:
                json.dump(custom_posters, f)
        elif cache_type == "complete":
            with open("/data/manual_complete.json", "w") as f:
                json.dump(manual_complete, f)
        elif cache_type == "season_complete":
            with open("/data/season_complete.json", "w") as f:
                json.dump(season_complete, f)
        # Poster/series TMDB caches are derived metadata and should not trigger
        # full history refresh churn in the UI.
        if cache_type in ("custom", "complete", "season_complete"):
            refresh_history_sig_fast()
    except Exception as e:
        print(f"Cache save error: {e}")


def _norm_title(s: str) -> str:
    s = (s or "").lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))  # remove accents
    s = s.replace("&", "and").replace("’", "'")
    s = re.sub(r"[\'\"]", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _score_title(qn: str, cn: str) -> int:
    if not cn:
        return 0
    if cn == qn:
        return 1000
    q = set(qn.split())
    c = set(cn.split())
    hit = sum(1 for t in q if t in c)
    bonus = 30 if (qn and qn in cn) else 0
    return hit * 10 + bonus


def _coerce_int(value, default=None):
    """
    Best-effort integer parser for values that may be numeric strings like
    "5", "5.0", "E5" or "S1E5". Returns `default` when parsing fails.
    """
    try:
        if isinstance(value, bool):
            return default
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        s = str(value).strip()
        if not s:
            return default
        if re.fullmatch(r"-?\d+", s):
            return int(s)
        if re.fullmatch(r"-?\d+\.\d+", s):
            return int(float(s))
        m = re.search(r"-?\d+", s)
        if m:
            return int(m.group(0))
    except Exception:
        pass
    return default


def _normalize_genres(value):
    """
    Normalize genre inputs to a de-duplicated list of non-empty strings.
    Supports list/tuple/set, comma/pipe separated strings, and JSON-like arrays.
    """
    items = []
    if isinstance(value, (list, tuple, set)):
        items = list(value)
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        if raw.startswith("[") and raw.endswith("]"):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    items = parsed
                else:
                    items = [raw]
            except Exception:
                items = [raw]
        else:
            items = re.split(r"[|,;/]", raw)
    elif value is None:
        return []
    else:
        items = [value]

    out = []
    seen = set()
    for g in items:
        try:
            genre = str(g or "").strip()
        except Exception:
            genre = ""
        if not genre:
            continue
        key = genre.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(genre)
    return out


def _is_duplicate_webhook_event(event_key):
    try:
        now = time.time()
        with webhook_recent_lock:
            last = webhook_recent.get(event_key)
            webhook_recent[event_key] = now
            # Keep the dedupe map bounded.
            if len(webhook_recent) > 1024:
                cutoff = now - (WEBHOOK_DEDUPE_SECONDS * 3)
                stale = [k for k, ts in webhook_recent.items() if ts < cutoff]
                for k in stale:
                    webhook_recent.pop(k, None)
        return bool(last and (now - float(last)) < WEBHOOK_DEDUPE_SECONDS)
    except Exception:
        return False

def notify_realtime_change():
    global realtime_seq
    try:
        with realtime_cv:
            realtime_seq += 1
            realtime_cv.notify_all()
    except Exception:
        pass

def refresh_history_sig_fast():
    """
    Keep a cheap in-memory signature for /api/history_sig so frequent polls do
    not need to stat multiple files each time.
    """
    global history_sig_fast
    try:
        history_sig_fast = _data_signature()
    except Exception:
        history_sig_fast = str(time.time_ns())
    notify_realtime_change()


def get_tmdb_poster(title, year=None, media_type="movie", allow_network=True):
    """
    FIXED version with year stripping and better TV show support
    """
    if not TMDB_API_KEY or not title:
        return None

    # ---- Strip year from title if present (e.g., "Bodies (2023)" -> "Bodies") ----
    title_clean = title
    year_match = re.search(r'\s*\((\d{4})\)\s*$', title)
    if year_match:
        title_clean = title.replace(year_match.group(0), '').strip()
        if not year:
            year = year_match.group(1)

    # ---- custom posters override ----
    custom_key = f"{media_type}_{title}_{year}"
    if custom_key in custom_posters:
        return custom_posters[custom_key]

    key = f"{media_type}_{title}_{year}".lower()
    now = int(time.time())
    NEG_TTL = 7 * 24 * 3600

    # ---- cache read (supports old and new formats) ----
    if key in poster_cache:
        cached = poster_cache.get(key)

        if isinstance(cached, str) and cached:
            return cached

        if isinstance(cached, dict):
            url = cached.get("url")
            ts = int(cached.get("ts") or 0)
            if url:
                return url
            if (now - ts) < NEG_TTL:
                return None

        if cached is None:
            poster_cache[key] = {"url": None, "ts": now}
            save_cache("poster")
            return None

    if not allow_network:
        # Fast-path for UI responsiveness: don't block history rendering on TMDB.
        return None

    # ---- TMDB helper ----
    def tmdb_get(url, params):
        try:
            r = requests.get(url, params=params, timeout=8)
            if r.status_code != 200:
                return []
            j = r.json()
            return j.get("results") or []
        except Exception:
            return []

    # Use smaller images (w342 instead of w780) - faster for all devices
    TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w342"
    base = "https://api.themoviedb.org/3/search"
    qnorm = _norm_title(title_clean)

    def pick_best_with_poster(items):
        best = None
        best_score = -1
        for it in (items or [])[:20]:
            if not it.get("poster_path"):
                continue

            cand_title = (it.get("name") or it.get("original_name") or "") if media_type == "tv" \
                else (it.get("title") or it.get("original_title") or "")

            cnorm = _norm_title(cand_title)
            score = _score_title(qnorm, cnorm)

            if year:
                date_key = "first_air_date" if media_type == "tv" else "release_date"
                d = (it.get(date_key) or "")[:4]
                if d.isdigit():
                    dy = int(d)
                    if abs(dy - int(year)) <= 1:
                        score += 20

            if score > best_score:
                best_score = score
                best = it
        return best

    # 1) primary search (with year only for movies, not TV shows)
    params = {"api_key": TMDB_API_KEY, "query": title_clean, "language": "en-US"}
    if year and media_type == "movie":
        params["year"] = int(year)

    results = tmdb_get(f"{base}/{media_type}", params)
    best = pick_best_with_poster(results)

    # 2) fallback: remove year filter
    if not best and year and media_type == "movie":
        params2 = {"api_key": TMDB_API_KEY, "query": title_clean, "language": "en-US"}
        results2 = tmdb_get(f"{base}/{media_type}", params2)
        best = pick_best_with_poster(results2)

    # 3) fallback: multi search
    if not best:
        params3 = {"api_key": TMDB_API_KEY, "query": title_clean, "language": "en-US"}
        multi = tmdb_get(f"{base}/multi", params3)
        filtered = [it for it in (multi or []) if it.get("media_type") == media_type and it.get("poster_path")]
        best = pick_best_with_poster(filtered)

    # 4) TV show specific fallback: try without special characters
    if not best and media_type == "tv":
        clean_title = title_clean.replace(":", "").replace("-", " ").replace("'", "").replace("  ", " ").strip()
        if clean_title != title_clean:
            params4 = {"api_key": TMDB_API_KEY, "query": clean_title, "language": "en-US"}
            results4 = tmdb_get(f"{base}/tv", params4)
            best = pick_best_with_poster(results4)

    poster_path = best.get("poster_path") if best else None
    poster_url = (TMDB_IMAGE_BASE + poster_path) if poster_path else None

    # cache and SAVE TO DISK
    poster_cache[key] = {"url": poster_url, "ts": now}
    save_cache("poster")

    return poster_url


def get_tmdb_movie_info(title, expected_year=None, allow_network=True):
    """
    Best-effort TMDB movie match for canonical title/year formatting.
    Returns dict: {"tmdb_name": str, "tmdb_year": int|None, "tmdb_id": int|None}
    """
    if not TMDB_API_KEY or not title or not allow_network:
        return None

    raw_title = str(title or "").strip()
    if not raw_title:
        return None

    # Strip "(YYYY)" suffix if present.
    title_clean = raw_title
    year_match = re.search(r"\s*\((\d{4})\)\s*$", raw_title)
    if year_match:
        title_clean = raw_title.replace(year_match.group(0), "").strip()
        if expected_year is None:
            expected_year = _coerce_int(year_match.group(1), None)

    try:
        expected_year = _coerce_int(expected_year, None)
    except Exception:
        expected_year = None

    qnorm = _norm_title(title_clean)

    def tmdb_get_results(url, params):
        try:
            r = requests.get(
                url,
                params=params,
                timeout=8,
            )
            if r.status_code != 200:
                return []
            j = r.json() or {}
            return j.get("results") or []
        except Exception:
            return []

    def score_item(it):
        cand_title = (it.get("title") or it.get("original_title") or "").strip()
        cnorm = _norm_title(cand_title)
        score = _score_title(qnorm, cnorm)
        # Prefer results that have posters (helps ensure the next poster fetch succeeds).
        if it.get("poster_path"):
            score += 15
        if expected_year:
            d = (it.get("release_date") or "")[:4]
            if d.isdigit():
                dy = int(d)
                if abs(dy - int(expected_year)) <= 1:
                    score += 20
        return score

    params = {"api_key": TMDB_API_KEY, "query": title_clean, "language": "en-US"}
    if expected_year:
        params["year"] = int(expected_year)
    results = tmdb_get_results("https://api.themoviedb.org/3/search/movie", params)

    # Fallback: drop year filter
    if not results and expected_year:
        params2 = {"api_key": TMDB_API_KEY, "query": title_clean, "language": "en-US"}
        results = tmdb_get_results("https://api.themoviedb.org/3/search/movie", params2)

    # Fallback: multi search (matches get_tmdb_poster behavior)
    if not results:
        params3 = {"api_key": TMDB_API_KEY, "query": title_clean, "language": "en-US"}
        multi = tmdb_get_results("https://api.themoviedb.org/3/search/multi", params3)
        results = [it for it in (multi or []) if it.get("media_type") == "movie"]

    if not results:
        return None

    best = None
    best_score = -1
    for it in (results or [])[:25]:
        try:
            s = score_item(it)
            if s > best_score:
                best_score = s
                best = it
        except Exception:
            continue

    if not best:
        return None

    tmdb_name = (best.get("title") or best.get("original_title") or "").strip()
    ry = None
    try:
        d = (best.get("release_date") or "")[:4]
        if d.isdigit():
            ry = int(d)
    except Exception:
        ry = None

    return {"tmdb_name": tmdb_name, "tmdb_year": ry, "tmdb_id": best.get("id")}

    
def get_tmdb_series_info(series_name, expected_year=None, force_refresh=False, allow_network=True):
    if not TMDB_API_KEY:
        return None

    raw_name = str(series_name or "").strip()
    search_name = raw_name
    try:
        expected_year = _coerce_int(expected_year, None)
    except Exception:
        expected_year = None
    if raw_name:
        try:
            m = re.search(r"\((\d{4})\)\s*$", raw_name)
            if m:
                if expected_year is None:
                    expected_year = int(m.group(1))
                search_name = raw_name[:m.start()].strip() or raw_name
        except Exception:
            search_name = raw_name

    now = int(time.time())
    # Negative cache TTL: keep relatively short so background rebuilds can recover
    # from mismatches (especially when webhooks provide episode-year values).
    NEG_TTL = 2 * 3600

    # Versioned key to invalidate older TMDB series-cache structures.
    cache_key = f"v3|{search_name}|{expected_year}" if expected_year else f"v3|{search_name}"

    def _is_neg(v):
        return isinstance(v, dict) and v.get("_neg") is True

    # If we don't have an expected year, and we have exactly one cached year-specific
    # match for this title, use it to avoid drifting between similarly-named shows.
    if not force_refresh and expected_year is None and cache_key not in series_cache:
        try:
            prefix = f"v3|{search_name}|"
            candidates = []
            for k, v in series_cache.items():
                if not isinstance(k, str) or not k.startswith(prefix):
                    continue
                if v is None:
                    continue
                if _is_neg(v):
                    ts = int(v.get("ts") or 0)
                    if ts and (now - ts) < NEG_TTL:
                        continue
                if isinstance(v, dict) and v.get("tmdb_id"):
                    candidates.append(v)
            if len(candidates) == 1:
                # Store an alias for future yearless lookups
                series_cache[cache_key] = candidates[0]
                save_cache("series")
                return candidates[0]
        except Exception:
            pass

    if not force_refresh and cache_key in series_cache:
        cached = series_cache.get(cache_key)
        if cached and not _is_neg(cached):
            return cached
        elif cached is None:
            # Legacy negative cache entry (no timestamp). Treat as expired so
            # we can retry in the future instead of permanently resetting totals.
            pass
        elif _is_neg(cached):
            try:
                ts = int(cached.get("ts") or 0)
                if ts and (now - ts) < NEG_TTL:
                    # A year-specific negative should not block a valid yearless alias.
                    # This is important when webhooks supply episode-year values.
                    if expected_year is not None:
                        try:
                            alias_key = f"v3|{search_name}"
                            cached_alias = series_cache.get(alias_key)
                            if cached_alias and not _is_neg(cached_alias):
                                return cached_alias
                        except Exception:
                            pass
                    return None
            except Exception:
                return None
    # If we have an expected year but only a yearless alias is cached, use it.
    if not force_refresh and expected_year is not None:
        try:
            alias_key = f"v3|{search_name}"
            cached_alias = series_cache.get(alias_key)
            if cached_alias and not _is_neg(cached_alias):
                return cached_alias
        except Exception:
            pass

    def _norm(title):
        if not title:
            return ""
        t = unicodedata.normalize("NFKD", str(title))
        t = "".join(ch for ch in t if not unicodedata.combining(ch))
        t = re.sub(r"[^a-zA-Z0-9]+", " ", t).strip().lower()
        return re.sub(r"\s+", " ", t)

    try:
        base_name = _norm(search_name)
        search_queries = [
            search_name,
            search_name.replace(":", ""),
            search_name.replace("-", " "),
            search_name.replace("'", ""),
            search_name.split(":")[0].strip() if ":" in search_name else search_name,
        ]

        best_result = None
        best_score = -1.0

        if not allow_network:
            # Fast-path for UI responsiveness: don't block history rendering on TMDB.
            # Cache reads are handled above; if we get here we have no usable cached hit.
            return None

        for query in search_queries:
            q = (query or "").strip()
            if not q:
                continue

            r = requests.get(
                "https://api.themoviedb.org/3/search/tv",
                params={"api_key": TMDB_API_KEY, "query": q, "language": "en-US"},
                timeout=10,
            )

            if r.status_code != 200:
                continue

            data = r.json() or {}
            results = data.get("results", [])

            for result in results[:20]:
                result_name_raw = result.get("name", "")
                result_name = _norm(result_name_raw)
                if not result_name:
                    continue

                original_name = _norm(result.get("original_name", ""))
                score_name = SequenceMatcher(None, base_name, result_name).ratio() * 100
                if original_name:
                    score_name = max(score_name, SequenceMatcher(None, base_name, original_name).ratio() * 100)
                score = score_name

                if result_name == base_name or original_name == base_name:
                    score += 40
                elif result_name.startswith(base_name) or base_name.startswith(result_name) or (
                    original_name and (original_name.startswith(base_name) or base_name.startswith(original_name))
                ):
                    score += 20
                elif base_name and (base_name in result_name or result_name in base_name):
                    score += 10

                first_air_date = result.get("first_air_date") or ""
                result_year = None
                if isinstance(first_air_date, str) and len(first_air_date) >= 4:
                    try:
                        result_year = int(first_air_date[:4])
                    except Exception:
                        result_year = None
                if expected_year and result_year:
                    diff = abs(expected_year - result_year)
                    if diff == 0:
                        score += 35
                    elif diff == 1:
                        score += 24
                    elif diff <= 3:
                        score += 12 - (diff * 2)
                    else:
                        # Avoid huge penalties; webhooks sometimes provide episode-year,
                        # which would otherwise cause correct matches to be rejected.
                        score -= min(10, diff * 2)
                elif expected_year and not result_year:
                    score -= 5

                popularity = result.get("popularity", 0) or 0
                try:
                    score += min(float(popularity) / 10.0, 20.0)
                except Exception:
                    pass

                vote_count = result.get("vote_count", 0) or 0
                try:
                    score += min(float(vote_count) / 200.0, 15.0)
                except Exception:
                    pass

                if score > best_score:
                    best_score = score
                    best_result = result

        if not best_result or best_score < 30:
            series_cache[cache_key] = {"_neg": True, "ts": now}
            save_cache("series")
            return None

        tv_id = best_result.get("id")

        r2 = requests.get(
            f"https://api.themoviedb.org/3/tv/{tv_id}",
            params={"api_key": TMDB_API_KEY, "language": "en-US"},
            timeout=10,
        )

        if r2.status_code != 200:
            series_cache[cache_key] = None
            save_cache("series")
            return None

        detail = r2.json() or {}
        seasons_info = {}
        total_all_episodes = _coerce_int(detail.get("number_of_episodes"), 0) or 0

        seasons = detail.get("seasons", [])
        summed_regular_seasons = 0
        for season in seasons:
            snum = season.get("season_number")
            ep_count = season.get("episode_count", 0)

            if snum is not None and snum > 0:
                seasons_info[snum] = {
                    "total_episodes": ep_count,
                    "name": season.get("name", f"Season {snum}"),
                }
                try:
                    summed_regular_seasons += int(ep_count or 0)
                except Exception:
                    pass

        # Fallback if TMDB does not provide number_of_episodes reliably.
        if total_all_episodes <= 0:
            total_all_episodes = summed_regular_seasons

        info = {
            "seasons": seasons_info,
            "total_episodes_in_series": total_all_episodes,
            "tmdb_name": best_result.get("name", ""),
            "tmdb_id": tv_id,
        }

        series_cache[cache_key] = info

        # If this lookup was year-specific, also store a safe yearless alias when
        # it doesn't conflict with an existing different TMDB id.
        if expected_year is not None:
            alias_key = f"v3|{search_name}"
            existing = series_cache.get(alias_key)
            if existing is None or _is_neg(existing) or (isinstance(existing, dict) and existing.get("tmdb_id") == tv_id):
                series_cache[alias_key] = info
        save_cache("series")
        return info

    except Exception:
        series_cache[cache_key] = {"_neg": True, "ts": now}
        save_cache("series")
        return None

def get_tmdb_episode_name(series_name, season_num, episode_num):
    if not TMDB_API_KEY:
        return None
    
    try:
        series_info = get_tmdb_series_info(series_name)
        if not series_info or not series_info.get("tmdb_id"):
            return None
        
        tv_id = series_info["tmdb_id"]
        
        r = requests.get(
            f"https://api.themoviedb.org/3/tv/{tv_id}/season/{season_num}",
            params={"api_key": TMDB_API_KEY, "language": "en-US"},
            timeout=10
        )
        
        if r.status_code != 200:
            return None
        
        season_data = r.json()
        episodes = season_data.get("episodes", [])
        
        for ep in episodes:
            if ep.get("episode_number") == episode_num:
                return ep.get("name", f"Episode {episode_num}")
        
        return None
        
    except Exception as e:
        return None


def get_tmdb_season_episodes(series_name, season_num):
    if not TMDB_API_KEY:
        return []
    
    try:
        series_info = get_tmdb_series_info(series_name)
        if not series_info or not series_info.get("tmdb_id"):
            return []
        
        tv_id = series_info["tmdb_id"]
        
        r = requests.get(
            f"https://api.themoviedb.org/3/tv/{tv_id}/season/{season_num}",
            params={"api_key": TMDB_API_KEY, "language": "en-US"},
            timeout=10
        )
        
        if r.status_code != 200:
            return []
        
        season_data = r.json()
        episodes = season_data.get("episodes", [])
        
        result = []
        for ep in episodes:
            result.append({
                "episode_number": ep.get("episode_number"),
                "name": ep.get("name", f"Episode {ep.get('episode_number')}")
            })
        
        return result
        
    except Exception as e:
        return []


def get_all_records():
    if not os.path.exists(DATA_FILE):
        return []
    records = []
    with open(DATA_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except:
                    pass
    return records


def save_all_records(records):
    os.makedirs("/data", exist_ok=True)
    with open(DATA_FILE, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())
    # Keep the previous organized payload in memory so organize_data() can reuse
    # totals (e.g. TMDB total episodes) during transient TMDB/cache failures.
    # The signature change will still force a recompute.
    cache["time"] = None
    cache["sig"] = None
    refresh_history_sig_fast()


def append_record(record):
    os.makedirs("/data", exist_ok=True)
    with open(DATA_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())
    # Keep the previous organized payload in memory so organize_data() can reuse
    # totals (e.g. TMDB total episodes) during transient TMDB/cache failures.
    # The signature change will still force a recompute.
    cache["time"] = None
    cache["sig"] = None
    refresh_history_sig_fast()

def _tail_json_records(path, max_lines=400, chunk_size=8192):
    """
    Read up to the last `max_lines` JSONL records from `path` efficiently.
    Returns a list of parsed dicts in file order (oldest -> newest for the tail).
    """
    if max_lines <= 0:
        return []
    try:
        if not os.path.exists(path):
            return []
        size = os.path.getsize(path)
        if size <= 0:
            return []
        buf = b""
        lines = []
        with open(path, "rb") as f:
            pos = size
            while pos > 0 and len(lines) < (max_lines + 5):
                read_size = chunk_size if pos >= chunk_size else pos
                pos -= read_size
                f.seek(pos)
                chunk = f.read(read_size)
                buf = chunk + buf
                parts = buf.split(b"\n")
                buf = parts[0]  # partial first line
                for ln in parts[1:][::-1]:
                    if ln.strip():
                        lines.append(ln)
                        if len(lines) >= (max_lines + 5):
                            break
            lines = lines[: max_lines]
            lines.reverse()
        out = []
        for ln in lines:
            try:
                out.append(json.loads(ln.decode("utf-8", errors="replace")))
            except Exception:
                continue
        return out
    except Exception:
        return []

def _record_dedupe_key(r):
    """
    Best-effort dedupe key across webhook/poll/import sources.
    Uses second-level timestamp resolution to avoid duplicates on repeated sync.
    """
    try:
        ts = str(r.get("timestamp") or "")[:19]
        t = str(r.get("type") or "")
        user = str(r.get("user") or "")
        if t.lower() == "episode":
            return ("Episode", str(r.get("series_name") or ""), _coerce_int(r.get("season"), 0) or 0, _coerce_int(r.get("episode"), 0) or 0, user, ts)
        if t.lower() == "movie":
            return ("Movie", str(r.get("name") or ""), _coerce_int(r.get("year"), 0) or 0, user, ts)
        return (t, str(r.get("name") or ""), user, ts)
    except Exception:
        return None

def _record_dedupe_key_loose(r):
    """
    Dedupe key that ignores timestamp. Used to avoid poll/webhook duplicates when
    timestamps differ (webhook uses receive time; poll uses Jellyfin LastPlayedDate).
    """
    try:
        t = str(r.get("type") or "")
        user = str(r.get("user") or "")
        if t.lower() == "episode":
            return ("Episode", str(r.get("series_name") or ""), _coerce_int(r.get("season"), 0) or 0, _coerce_int(r.get("episode"), 0) or 0, user)
        if t.lower() == "movie":
            return ("Movie", str(r.get("name") or ""), _coerce_int(r.get("year"), 0) or 0, user)
        return (t, str(r.get("name") or ""), user)
    except Exception:
        return None

def _parse_iso_to_epoch_seconds(value):
    """
    Parse ISO-ish timestamps like:
    - 2026-02-13T10:20:30Z
    - 2026-02-13T10:20:30.1234567Z
    - 2026-02-13T10:20:30.123+00:00
    Returns int epoch seconds (UTC) or 0 on failure.
    """
    try:
        s = str(value or "").strip()
        if not s:
            return 0
        # Normalize Zulu timestamps to offset format
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        # Trim fractional seconds to microseconds if present
        if "." in s:
            # split on timezone separator if any
            tz_pos = max(s.rfind("+"), s.rfind("-"))
            if tz_pos > s.find("T"):
                main = s[:tz_pos]
                tz = s[tz_pos:]
            else:
                main = s
                tz = ""
            if "." in main:
                left, frac = main.split(".", 1)
                frac_digits = "".join(ch for ch in frac if ch.isdigit())
                frac_digits = (frac_digits + "000000")[:6]
                main = left + "." + frac_digits
            s = main + tz
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            # Assume UTC if no tz present
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except Exception:
        return 0

def _append_records_batch(records):
    if not records:
        return
    os.makedirs("/data", exist_ok=True)
    with open(DATA_FILE, "a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
        f.flush()
        os.fsync(f.fileno())
    cache["time"] = None
    cache["sig"] = None
    refresh_history_sig_fast()

def jellyfin_poll_sync_once():
    """
    Poll Jellyfin for newly played items since the last poll state.
    This is a fallback when playback-stop webhooks are missed (e.g. "Next episode" flows).
    """
    if not (JELLYFIN_POLL_ENABLED and JELLYFIN_URL and JELLYFIN_API_KEY):
        return 0
    try:
        interval = int(JELLYFIN_POLL_INTERVAL_S or 60)
        if interval < 15:
            interval = 15
    except Exception:
        interval = 60

    try:
        headers = {"X-Emby-Token": JELLYFIN_API_KEY}
        state = load_json_file(JELLYFIN_POLL_STATE_FILE)
        if not isinstance(state, dict):
            state = {}

        user_id = state.get("user_id")
        user_name = state.get("user_name")
        last_ts = _coerce_int(state.get("last_ts"), 0) or 0
        last_ids = state.get("last_ids") or []
        if not isinstance(last_ids, list):
            last_ids = []
        last_ids = [str(x) for x in last_ids if x]
        seen_map = state.get("seen") or {}
        if not isinstance(seen_map, dict):
            seen_map = {}

        now_epoch = int(time.time())
        # If the saved timestamp is in the future (clock skew or bad parsing),
        # clamp it so we don't skip real new plays.
        if last_ts > (now_epoch + 60):
            last_ts = max(0, now_epoch - 3600)

        lookback = _coerce_int(JELLYFIN_POLL_LOOKBACK_S, 0) or 0
        if lookback < 300:
            lookback = 300
        lookback_cutoff = max(0, now_epoch - lookback)

        # First-run: derive a safe baseline from our own history tail so we don't import everything.
        if last_ts <= 0:
            tail = _tail_json_records(DATA_FILE, max_lines=200)
            mx = 0
            for r in tail:
                ts = _parse_iso_to_epoch_seconds(r.get("timestamp"))
                if ts > mx:
                    mx = ts
            if mx > 0:
                # Go slightly back to avoid missing same-second plays.
                last_ts = max(0, int(mx) - 5)
            else:
                # No local history yet: only look back a little so we don't import
                # the user's entire played history on first boot.
                last_ts = max(0, int(time.time()) - 3600)
            last_ids = []

        if not user_id or not user_name:
            try:
                r = requests.get(f"{JELLYFIN_URL}/Users", headers=headers, timeout=15)
                if r.status_code == 200:
                    users = r.json() or []
                    if users:
                        user_id = users[0].get("Id")
                        user_name = users[0].get("Name")
            except Exception:
                pass

        if not user_id:
            return 0
        if not user_name:
            user_name = "Unknown"

        params = {
            "Filters": "IsPlayed",
            "Recursive": "true",
            "Fields": "Genres,UserData",
            "IncludeItemTypes": "Movie,Episode",
            "StartIndex": 0,
            "Limit": int(JELLYFIN_POLL_LIMIT or 80),
            "SortBy": "DatePlayed",
            "SortOrder": "Descending",
        }
        r2 = requests.get(f"{JELLYFIN_URL}/Users/{user_id}/Items", headers=headers, params=params, timeout=20)
        if r2.status_code != 200:
            return 0
        page = r2.json() or {}
        items = page.get("Items", []) or []

        # Dedupe against the tail of our local history (covers webhook+poll overlaps).
        tail_recent = _tail_json_records(DATA_FILE, max_lines=400)
        recent_keys = set()
        recent_loose = set()
        for rr in tail_recent:
            k = _record_dedupe_key(rr)
            if k:
                recent_keys.add(k)
            lk = _record_dedupe_key_loose(rr)
            if lk:
                recent_loose.add(lk)

        new_records = []
        seen_ts = last_ts
        seen_ids = set(last_ids)
        for it in items:
            try:
                it_id = str(it.get("Id") or "")
                last_played = (it.get("UserData") or {}).get("LastPlayedDate")
                ts = _parse_iso_to_epoch_seconds(last_played)
                if ts <= 0:
                    continue

                # Only consider recent plays; prevents the poller from backfilling
                # your entire Jellyfin "played" history.
                if ts < lookback_cutoff:
                    continue

                # Per-item guard: if we've already seen this Jellyfin item played at
                # this timestamp, skip it even if global last_ts is wrong.
                prev_seen = _coerce_int(seen_map.get(it_id), 0) or 0
                if it_id and ts <= prev_seen:
                    continue

                rec = {
                    "timestamp": last_played,
                    "type": it.get("Type"),
                    "name": it.get("Name"),
                    "year": it.get("ProductionYear"),
                    "series_name": it.get("SeriesName"),
                    "season": it.get("ParentIndexNumber"),
                    "episode": it.get("IndexNumber"),
                    "user": user_name,
                    "genres": it.get("Genres", []) or [],
                    "source": "poll",
                    "jellyfin_id": it_id,
                }
                k = _record_dedupe_key(rec)
                lk = _record_dedupe_key_loose(rec)
                if k and k in recent_keys:
                    # Already in local history (likely via webhook/import)
                    pass
                elif lk and lk in recent_loose:
                    # Same item already present but timestamp differs (common when webhook vs poll)
                    pass
                else:
                    new_records.append((ts, it_id, rec))
                if it_id:
                    seen_map[it_id] = int(ts)
            except Exception:
                continue

        if not new_records:
            # Persist updated user info / baseline if needed
            state["user_id"] = user_id
            state["user_name"] = user_name
            state["last_ts"] = last_ts
            state["last_ids"] = last_ids
            save_json_file(JELLYFIN_POLL_STATE_FILE, state)
            return 0

        # Append in chronological order (oldest first)
        new_records.sort(key=lambda x: x[0])
        to_append = []
        for ts, it_id, rec in new_records:
            to_append.append(rec)
            if ts > seen_ts:
                seen_ts = ts
                seen_ids = set([it_id] if it_id else [])
            elif ts == seen_ts and it_id:
                seen_ids.add(it_id)

            # Queue TMDB work for new items (non-blocking)
            try:
                rtype = str(rec.get("type") or "").strip().lower()
                if rtype == "movie":
                    _queue_tmdb_poster_fetch(rec.get("name"), rec.get("year"), "movie")
                elif rtype == "episode":
                    _queue_tmdb_poster_fetch(rec.get("series_name") or rec.get("name"), rec.get("year"), "tv")
                    _queue_tmdb_series_fetch(rec.get("series_name") or rec.get("name"), expected_year=None)
            except Exception:
                pass

        _append_records_batch(to_append)
        prewarm_organized_cache_async()
        try:
            print(f"✓ Jellyfin poll sync added {len(to_append)} new record(s)")
        except Exception:
            pass

        # Trim seen_map if it grows too large (keep most recent plays).
        try:
            if len(seen_map) > 2000:
                keep = 1200
                items_seen = []
                for k, v in seen_map.items():
                    vv = _coerce_int(v, 0) or 0
                    if k:
                        items_seen.append((vv, k))
                items_seen.sort(reverse=True)
                keep_keys = set(k for _, k in items_seen[:keep])
                seen_map = {k: seen_map[k] for k in keep_keys if k in seen_map}
        except Exception:
            pass

        state["user_id"] = user_id
        state["user_name"] = user_name
        state["last_ts"] = int(seen_ts)
        state["last_ids"] = sorted([x for x in seen_ids if x])[:200]
        state["seen"] = seen_map
        save_json_file(JELLYFIN_POLL_STATE_FILE, state)
        return len(to_append)
    except Exception:
        return 0

def start_jellyfin_poll_thread():
    if not (JELLYFIN_POLL_ENABLED and JELLYFIN_URL and JELLYFIN_API_KEY):
        return
    try:
        interval = int(JELLYFIN_POLL_INTERVAL_S or 60)
        if interval < 15:
            interval = 15
    except Exception:
        interval = 60
    try:
        print(f"✓ Jellyfin poll enabled: interval={interval}s limit={int(JELLYFIN_POLL_LIMIT or 80)}")
    except Exception:
        pass

    def _run():
        # small startup delay so the app is ready first
        time.sleep(5)
        while True:
            try:
                jellyfin_poll_sync_once()
            except Exception:
                pass
            time.sleep(interval)

    threading.Thread(target=_run, daemon=True).start()

def prewarm_organized_cache_async():
    global _prewarm_busy
    with _prewarm_lock:
        if _prewarm_busy:
            return
        _prewarm_busy = True

    def _run():
        global _prewarm_busy
        try:
            organize_data()
        except Exception:
            pass
        finally:
            with _prewarm_lock:
                _prewarm_busy = False

    threading.Thread(target=_run, daemon=True).start()


def calculate_watch_streak(records):
    dates = set()
    for r in records:
        try:
            dt = datetime.fromisoformat(r["timestamp"].replace("Z", "").split("+")[0])
            dates.add(dt.date())
        except:
            pass
    
    if not dates:
        return {"current": 0, "longest": 0}
    
    sorted_dates = sorted(dates, reverse=True)
    current_streak = 0
    longest_streak = 0
    temp_streak = 1
    
    today = datetime.now().date()
    if sorted_dates[0] == today or sorted_dates[0] == today - timedelta(days=1):
        current_streak = 1
        for i in range(1, len(sorted_dates)):
            if sorted_dates[i] == sorted_dates[i-1] - timedelta(days=1):
                current_streak += 1
            else:
                break
    
    for i in range(1, len(sorted_dates)):
        if sorted_dates[i] == sorted_dates[i-1] - timedelta(days=1):
            temp_streak += 1
            longest_streak = max(longest_streak, temp_streak)
        else:
            temp_streak = 1
    
    longest_streak = max(longest_streak, temp_streak, current_streak)
    
    return {"current": current_streak, "longest": longest_streak}


def get_quick_stats(movies, shows, records):
    stats = {}
    
    most_rewatched = max(movies, key=lambda m: m.get("watch_count", 0), default=None)
    stats["most_rewatched"] = {
        "name": most_rewatched.get("name") if most_rewatched else "N/A",
        "count": most_rewatched.get("watch_count", 0) if most_rewatched else 0
    }
    
    most_binged = max(shows, key=lambda s: s.get("total_episodes", 0), default=None)
    stats["most_binged"] = {
        "name": most_binged.get("series_name") if most_binged else "N/A",
        "count": most_binged.get("total_episodes", 0) if most_binged else 0
    }
    
    completed_shows = [s for s in shows if s.get("completion_percentage", 0) == 100 and s.get("has_tmdb_data")]
    if completed_shows:
        fastest = None
        fastest_days = float('inf')
        
        for show in completed_shows:
            timestamps = []
            for season in show.get("seasons", []):
                for ep in season.get("episodes", []):
                    for watch in ep.get("watches", []):
                        timestamps.append(watch.get("timestamp"))
            
            if len(timestamps) >= 2:
                timestamps.sort()
                try:
                    first = datetime.fromisoformat(timestamps[0].replace("Z", "").split("+")[0])
                    last = datetime.fromisoformat(timestamps[-1].replace("Z", "").split("+")[0])
                    days = (last - first).days
                    if days < fastest_days:
                        fastest_days = days
                        fastest = show
                except:
                    pass
        
        stats["fastest_completion"] = {
            "name": fastest.get("series_name") if fastest else "N/A",
            "days": fastest_days if fastest_days != float('inf') else 0
        }
    else:
        stats["fastest_completion"] = {"name": "N/A", "days": 0}
    
    episode_records = [r for r in records if r.get("type") == "Episode"]
    sessions = defaultdict(list)
    for r in episode_records:
        try:
            dt = datetime.fromisoformat(r["timestamp"].replace("Z", "").split("+")[0])
            date = dt.date()
            sessions[date].append(r)
        except:
            pass
    
    if sessions:
        avg_per_session = sum(len(eps) for eps in sessions.values()) / len(sessions)
        stats["avg_episodes_per_day"] = round(avg_per_session, 1)
    else:
        stats["avg_episodes_per_day"] = 0
    
    return stats


def organize_data():
    try:
        sig = _data_signature()
        if cache.get("data") and cache.get("sig") == sig:
            return cache["data"]
    except Exception:
        sig = None

    prev_show_totals = {}
    prev_season_totals = {}
    # Persisted totals survive webhook-triggered cache invalidation and prevent
    # temporary TMDB lookup failures from collapsing totals to watched-only.
    try:
        persisted_totals = load_json_file(SHOW_TOTALS_FILE)
        if isinstance(persisted_totals, dict):
            for series_name, meta in persisted_totals.items():
                if not isinstance(meta, dict):
                    continue
                total_val = _coerce_int(meta.get("total"), 0) or 0
                if total_val > 0:
                    prev_show_totals[series_name] = max(prev_show_totals.get(series_name, 0), total_val)
                seasons_meta = meta.get("seasons", {})
                if not isinstance(seasons_meta, dict):
                    continue
                season_map = prev_season_totals.get(series_name, {})
                for skey, sval in seasons_meta.items():
                    snum = _coerce_int(skey, None)
                    stotal = _coerce_int(sval, 0) or 0
                    if snum and snum > 0 and stotal > 0:
                        season_map[snum] = max(season_map.get(snum, 0), stotal)
                if season_map:
                    prev_season_totals[series_name] = season_map
    except Exception:
        pass

    def _add_prev_show_aliases():
        # Create tolerant lookup aliases for series names so minor webhook/title
        # formatting differences don't reset totals (case/punct/year suffix).
        try:
            def _strip_year_suffix(n):
                s = str(n or "").strip()
                m = re.search(r"\s*\((\d{4})\)\s*$", s)
                return (s[:m.start()].strip() if m else s)

            show_items = list(prev_show_totals.items())
            for raw_name, total in show_items:
                raw = str(raw_name or "").strip()
                if not raw:
                    continue
                stripped = _strip_year_suffix(raw)
                norm_raw = _norm_title(raw)
                norm_stripped = _norm_title(stripped)
                for alias in (raw.lower(), stripped, stripped.lower(), norm_raw, norm_stripped):
                    if alias and alias not in prev_show_totals:
                        prev_show_totals[alias] = total
                    if alias and raw_name in prev_season_totals and alias not in prev_season_totals:
                        prev_season_totals[alias] = prev_season_totals.get(raw_name) or {}
        except Exception:
            pass

    _add_prev_show_aliases()
    try:
        if cache.get("data") and isinstance(cache["data"], dict):
            for s in cache["data"].get("shows", []) or []:
                name = s.get("series_name")
                if not name:
                    continue
                # Never treat watched-only counts as the series total. This was the
                # source of the "1/1, 2/2" regression for brand-new shows.
                if s.get("tmdb_pending"):
                    continue
                tp = _coerce_int(s.get("total_episodes_possible"), 0) or 0
                tw = _coerce_int(s.get("total_episodes"), 0) or 0
                if tp > 0 and (tp > tw or s.get("has_tmdb_data") or s.get("manually_completed")):
                    prev_show_totals[name] = max(prev_show_totals.get(name, 0), tp)
                st = prev_season_totals.get(name, {})
                for season in s.get("seasons", []) or []:
                    snum = season.get("season_number")
                    if snum is None:
                        continue
                    stotal = _coerce_int(season.get("total_episodes"), 0) or 0
                    if stotal > 0 and (season.get("has_tmdb_data") or s.get("has_tmdb_data") or s.get("manually_completed")):
                        st[snum] = max(st.get(snum, 0), stotal)
                if st:
                    prev_season_totals[name] = st
    except Exception:
        pass
    # Add aliases for any totals coming from the in-memory organized cache too.
    _add_prev_show_aliases()

    records = get_all_records()
    movies = {}
    shows = {}
    genre_counter = Counter()
    genre_breakdown = defaultdict(lambda: {"movies": [], "shows": []})

    for r in records:
        rtype = r.get("type")
        genres = _normalize_genres(r.get("genres", []))
        
        if rtype == "Movie":
            key = f"{r.get('name')}_{r.get('year')}"
            if key not in movies:
                movies[key] = {
                    "name": r.get("name"),
                    "year": r.get("year"),
                    "watch_count": 0,
                    "watches": [],
                    "genres": genres,
                    "poster": None,
                    "jellyfin_id": r.get("jellyfin_id"),
                }
            elif genres:
                merged = _normalize_genres((movies[key].get("genres") or []) + genres)
                movies[key]["genres"] = merged
            if not movies[key].get("jellyfin_id") and r.get("jellyfin_id"):
                movies[key]["jellyfin_id"] = r.get("jellyfin_id")
            movies[key]["watch_count"] += 1
            movies[key]["watches"].append({"timestamp": r.get("timestamp"), "user": r.get("user"), "jellyfin_id": r.get("jellyfin_id")})
            for g in genres:
                genre_counter[g] += 1

        elif rtype == "Episode":
            series = r.get("series_name") or "Unknown Series"
            if series not in shows:
                shows[series] = {
                    "series_name": series,
                    "seasons": {},
                    "genres": genres,
                    "poster": None,
                    "year": r.get("year"),
                    "jellyfin_series_id": r.get("jellyfin_series_id"),
                }
            elif not shows[series].get("year") and r.get("year"):
                shows[series]["year"] = r.get("year")
            if not shows[series].get("jellyfin_series_id") and r.get("jellyfin_series_id"):
                shows[series]["jellyfin_series_id"] = r.get("jellyfin_series_id")
            if genres:
                merged = _normalize_genres((shows[series].get("genres") or []) + genres)
                shows[series]["genres"] = merged
            
            season = _coerce_int(r.get("season"), 0) or 0
            
            if season == 999:
                continue
            
            if season not in shows[series]["seasons"]:
                shows[series]["seasons"][season] = {"season_number": season, "episodes": []}
            
            ep_list = shows[series]["seasons"][season]["episodes"]
            ep_no = _coerce_int(r.get("episode"), 0) or 0
            if ep_no <= 0:
                continue
            
            found = False
            for ep in ep_list:
                if ep["episode"] == ep_no:
                    ep["watch_count"] += 1
                    ep["watches"].append({"timestamp": r.get("timestamp"), "user": r.get("user"), "jellyfin_id": r.get("jellyfin_id")})
                    if r.get("jellyfin_id"):
                        ep["jellyfin_id"] = r.get("jellyfin_id")
                    found = True
                    break
            
            if not found:
                ep_list.append({
                    "name": r.get("name"),
                    "season": season,
                    "episode": ep_no,
                    "watch_count": 1,
                    "watches": [{"timestamp": r.get("timestamp"), "user": r.get("user"), "jellyfin_id": r.get("jellyfin_id")}],
                    "jellyfin_id": r.get("jellyfin_id"),
                })

    for m in movies.values():
        if not m["poster"]:
            # Don't block /api/history on TMDB network calls.
            m["poster"] = get_tmdb_poster(m["name"], m["year"], "movie", allow_network=False)
        for g in m["genres"]:
            genre_breakdown[g]["movies"].append(m)

    for series_name, s in shows.items():
        if not s["poster"]:
            # Don't block /api/history on TMDB network calls.
            s["poster"] = get_tmdb_poster(series_name, s.get("year"), "tv", allow_network=False)
        
        total_watched = 0
        seasons_list = []
        
        for snum, sdata in sorted(s["seasons"].items()):
            episodes = sorted(sdata["episodes"], key=lambda x: x["episode"])
            for ep in episodes:
                ep["watches"].sort(key=lambda x: x["timestamp"], reverse=True)
            
            watched_count = len(episodes)
            total_watched += watched_count
            
            season_key = f"{series_name}_{snum}"
            is_season_complete = season_key in season_complete
            
            season_obj = {
                "season_number": snum,
                "episodes": episodes,
                "episode_count": watched_count,
                "total_episodes": watched_count,
                "total_watches": sum(e["watch_count"] for e in episodes),
                "has_tmdb_data": False,
                "manually_completed": is_season_complete
            }
            
            seasons_list.append(season_obj)

        observed_by_season = {}
        for season in seasons_list:
            snum = _coerce_int(season.get("season_number"), 0) or 0
            if snum <= 0:
                continue
            mx = 0
            for ep in season.get("episodes", []):
                ep_no = _coerce_int(ep.get("episode"), 0) or 0
                if ep_no > mx:
                    mx = ep_no
            if mx > 0:
                observed_by_season[snum] = mx

        def _tmdb_matches_observed(info):
            if not info:
                return False
            seasons_meta = info.get("seasons", {})
            if not observed_by_season:
                return True
            for snum, observed_max in observed_by_season.items():
                meta = seasons_meta.get(snum)
                if not meta:
                    return False
                tmdb_total = _coerce_int(meta.get("total_episodes"), 0) or 0
                if tmdb_total < observed_max:
                    return False
            return True
        
        # Check if manually marked complete
        is_manually_complete = series_name in manual_complete
        # Use tolerant lookup for totals (case + optional year suffix variants).
        prev_total = 0
        prev_seasons = {}
        try:
            candidates = [
                str(series_name or "").strip(),
                str(series_name or "").strip().lower(),
                _norm_title(str(series_name or "").strip()),
            ]
            try:
                m = re.search(r"\s*\((\d{4})\)\s*$", str(series_name or "").strip())
                if m:
                    base = str(series_name or "").strip()[:m.start()].strip()
                    candidates.extend([base, base.lower(), _norm_title(base)])
            except Exception:
                pass
            for k in candidates:
                if not k:
                    continue
                v = _coerce_int(prev_show_totals.get(k), 0) or 0
                if v > 0:
                    prev_total = v
                    prev_seasons = prev_season_totals.get(k) or {}
                    break
        except Exception:
            prev_total = _coerce_int(prev_show_totals.get(series_name), 0) or 0
            prev_seasons = prev_season_totals.get(series_name) or {}
        
        if is_manually_complete:
            # Only use cached TMDB info here; never block /api/history on network.
            series_info = None
            if prev_total <= 0:
                series_info = get_tmdb_series_info(series_name, expected_year=s.get("year"), allow_network=False)
                if series_info and not _tmdb_matches_observed(series_info):
                    series_info = None

            total_possible = total_watched
            has_tmdb = False
            if series_info and series_info.get("total_episodes_in_series", 0) > 0:
                total_possible = series_info["total_episodes_in_series"]
                has_tmdb = True
                for season in seasons_list:
                    snum = season.get("season_number")
                    if snum in series_info.get("seasons", {}):
                        tmdb_total = series_info["seasons"][snum]["total_episodes"]
                        observed_max = observed_by_season.get(snum, 0)
                        prev_season_total = _coerce_int(prev_seasons.get(snum), 0) or 0
                        season["total_episodes"] = max(
                            _coerce_int(tmdb_total, 0) or 0,
                            season.get("episode_count", 0),
                            observed_max,
                            prev_season_total,
                        )
                        season["has_tmdb_data"] = True
            elif prev_total > 0:
                # Use persisted totals as a fast fallback.
                total_possible = max(prev_total, total_watched)
                has_tmdb = True
                for season in seasons_list:
                    snum = season.get("season_number")
                    observed_max = observed_by_season.get(snum, 0)
                    prev_season_total = _coerce_int(prev_seasons.get(snum), 0) or 0
                    season["total_episodes"] = max(
                        season.get("episode_count", 0),
                        observed_max,
                        prev_season_total,
                    )
                    season["has_tmdb_data"] = True if prev_season_total > 0 else False

            for season in seasons_list:
                season["completion_percentage"] = 100
                season["manually_completed"] = True

            s["total_episodes"] = total_watched
            s["total_episodes_possible"] = max(total_possible, total_watched, prev_total)
            s["total_watches"] = sum(x["total_watches"] for x in seasons_list)
            s["completion_percentage"] = 100
            s["has_tmdb_data"] = has_tmdb
            s["manually_completed"] = True
            s["seasons"] = seasons_list
            s["has_new_content"] = False  # NEW: Flag for new content after manual complete
        else:
            # Only use cached TMDB info here; never block /api/history on network.
            series_info = None
            if prev_total <= 0:
                series_info = get_tmdb_series_info(series_name, expected_year=s.get("year"), allow_network=False)
                if series_info and not _tmdb_matches_observed(series_info):
                    series_info = None
            
            if series_info and series_info.get("total_episodes_in_series", 0) > 0:
                total_possible = series_info["total_episodes_in_series"]
                
                for season in seasons_list:
                    snum = season["season_number"]
                    season_key = f"{series_name}_{snum}"
                    
                    if season_key in season_complete:
                        season["completion_percentage"] = 100
                        season["manually_completed"] = True
                    elif snum > 0 and snum in series_info.get("seasons", {}):
                        tmdb_total = series_info["seasons"][snum]["total_episodes"]
                        observed_max = observed_by_season.get(snum, 0)
                        prev_season_total = _coerce_int(prev_seasons.get(snum), 0) or 0
                        season["total_episodes"] = max(
                            _coerce_int(tmdb_total, 0) or 0,
                            season.get("episode_count", 0),
                            observed_max,
                            prev_season_total,
                        )
                        season["has_tmdb_data"] = True
                        season_watched = season["episode_count"]
                        season_total = season["total_episodes"]
                        pct = round((season_watched / season_total) * 100) if season_total > 0 else 0
                        # Don't auto-show 100% unless manually completed
                        if pct >= 100 and not season.get("manually_completed"):
                            pct = 99
                        season["completion_percentage"] = pct
                
                s["total_episodes"] = total_watched
                s["total_episodes_possible"] = max(total_possible, total_watched, prev_total)
                s["total_watches"] = sum(x["total_watches"] for x in seasons_list)
                
                pct_total = s["total_episodes_possible"]
                raw_pct = round((total_watched / pct_total) * 100) if pct_total > 0 else 0
                s["auto_all_added"] = (raw_pct >= 100 and not s.get("manually_completed"))

                # Don't auto-show 100% unless manually completed
                pct = raw_pct
                if pct >= 100 and not s.get("manually_completed"):
                    pct = 99
                    
                s["completion_percentage"] = pct
                s["has_tmdb_data"] = True
                s["manually_completed"] = False
                s["seasons"] = seasons_list
                s["has_new_content"] = False
            else:
                inferred_total = 0
                for season in seasons_list:
                    observed_max = 0
                    for ep in season.get("episodes", []):
                        ep_no = _coerce_int(ep.get("episode"), 0) or 0
                        observed_max = max(observed_max, ep_no)

                    prev_season_total = 0
                    try:
                        prev_season_total = _coerce_int(prev_seasons.get(season.get("season_number")), 0) or 0
                    except Exception:
                        prev_season_total = 0

                    season_total = max(season.get("episode_count", 0), observed_max, prev_season_total)
                    season["total_episodes"] = season_total
                    season["has_tmdb_data"] = True if (prev_total > 0 or prev_season_total > 0) else False

                    if season.get("manually_completed"):
                        season["completion_percentage"] = 100
                    else:
                        spct = round((season.get("episode_count", 0) / season_total) * 100) if season_total > 0 else 0
                        if spct >= 100:
                            spct = 99
                        season["completion_percentage"] = spct

                    inferred_total += season_total

                total_possible = inferred_total if inferred_total > 0 else total_watched
                total_possible = max(total_possible, prev_total)
                s["total_episodes"] = total_watched
                if prev_total > 0:
                    s["total_episodes_possible"] = max(total_possible, total_watched)
                    s["tmdb_pending"] = False
                else:
                    # Brand-new show: avoid incorrect 1/1 totals while TMDB totals are fetched
                    # asynchronously (webhook/prewarm will populate SHOW_TOTALS_FILE).
                    s["total_episodes_possible"] = 0
                    s["tmdb_pending"] = True
                s["total_watches"] = sum(x["total_watches"] for x in seasons_list)

                pct = round((total_watched / total_possible) * 100) if total_possible > 0 else 0
                if pct >= 100:
                    pct = 99

                # If TMDB totals are pending, don't pretend the show is nearly complete.
                s["completion_percentage"] = pct if prev_total > 0 else 0
                s["has_tmdb_data"] = True if prev_total > 0 else False
                s["manually_completed"] = False
                s["seasons"] = seasons_list
                s["has_new_content"] = False
        
        for g in s["genres"]:
            genre_breakdown[g]["shows"].append(s)

    movies_list = sorted(movies.values(), key=lambda x: x["watches"][0]["timestamp"] if x.get("watches") else "", reverse=True)
    shows_list = sorted(shows.values(), key=lambda x: x.get("total_watches", 0), reverse=True)

    # Persist latest non-zero totals for resilience across cache clears/restarts.
    try:
        totals_payload = {}
        for show in shows_list:
            name = show.get("series_name")
            if not name:
                continue
            # Avoid persisting inferred watched-only totals (causes 5/5 resets).
            if not (show.get("has_tmdb_data") or show.get("manually_completed")):
                continue
            total_possible = _coerce_int(show.get("total_episodes_possible"), 0) or 0
            season_map = {}
            for season in show.get("seasons", []):
                snum = _coerce_int(season.get("season_number"), 0) or 0
                stotal = _coerce_int(season.get("total_episodes") or season.get("episode_count"), 0) or 0
                if snum > 0 and stotal > 0:
                    season_map[str(snum)] = stotal
            if total_possible > 0 or season_map:
                totals_payload[name] = {"total": total_possible, "seasons": season_map}
        if totals_payload:
            existing_totals = load_json_file(SHOW_TOTALS_FILE)
            if existing_totals != totals_payload:
                save_json_file(SHOW_TOTALS_FILE, totals_payload)
    except Exception:
        pass

    # Attach rating and note for each movie and show from ratings file.  The ratings
    # are stored using keys "movie|<title>|<year>" or "show|<series_name>".  If no
    # rating exists then set None.
    try:
        ratings_cache = load_ratings()
        # Movies
        for m in movies_list:
            try:
                rkey = f"movie|{m.get('name')}|{m.get('year')}"
                rinfo = ratings_cache.get(rkey, None)
                if isinstance(rinfo, dict):
                    m["rating"] = rinfo.get("rating")
                    m["note"] = rinfo.get("note") or ""
                else:
                    m["rating"] = None
                    m["note"] = ""
            except Exception:
                m["rating"] = None
                m["note"] = ""
        # Shows
        for s in shows_list:
            try:
                rkey = f"show|{s.get('series_name')}"
                rinfo = ratings_cache.get(rkey, None)
                if isinstance(rinfo, dict):
                    s["rating"] = rinfo.get("rating")
                    s["note"] = rinfo.get("note") or ""
                else:
                    s["rating"] = None
                    s["note"] = ""
            except Exception:
                s["rating"] = None
                s["note"] = ""
    except Exception:
        # If ratings file cannot be loaded simply set rating to None and note to empty string
        for m in movies_list:
            m["rating"] = None
            m["note"] = ""
        for s in shows_list:
            s["rating"] = None
            s["note"] = ""

    week_ago = datetime.now() - timedelta(days=7)
    month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    this_week = 0
    this_month = 0
    week_content = []

    for r in records:
        try:
            dt = datetime.fromisoformat(r["timestamp"].replace("Z", "").split("+")[0])
            if dt >= week_ago:
                this_week += 1
                week_content.append(r)
            if dt >= month_start:
                this_month += 1
        except:
            pass

    week_movies = Counter()
    week_shows = Counter()
    for r in week_content:
        if r.get("type") == "Movie":
            week_movies[r.get("name")] += 1
        elif r.get("type") == "Episode":
            week_shows[r.get("series_name")] += 1
    
    trending = {
        "movies": [{"name": k, "count": v} for k, v in week_movies.most_common(5)],
        "shows": [{"name": k, "count": v} for k, v in week_shows.most_common(5)]
    }

    days_in_month = (datetime.now() - month_start).days + 1
    # Count unique items per genre (same as genre_breakdown for recommendations)
    genre_item_counts = {}
    for genre, items in genre_breakdown.items():
        genre_item_counts[genre] = len(items["movies"]) + len(items["shows"])

    genres = [{"genre": g, "count": c} for g, c in sorted(genre_item_counts.items(), 
    key=lambda x: x[1], reverse=True)[:10]]


    watch_streak = calculate_watch_streak(records)
    quick_stats = get_quick_stats(movies_list, shows_list, records)
    
    in_progress = [s for s in shows_list if 0 < s.get("completion_percentage", 0) < 100 and s.get("has_tmdb_data")]
    completed = [s for s in shows_list if s.get("completion_percentage", 0) == 100]

    result = {
        "movies": movies_list,
        "shows": shows_list,
        "stats": {
            "total_watches": len(records),
            "unique_movies": len(movies),
            "tv_shows": len(shows),
            "this_week": this_week,
            "total_hours": round(len(records) * 0.75),
            "avg_per_day": round(this_month / days_in_month, 1) if days_in_month > 0 else 0
        },
        "genres": genres,
        "genre_breakdown": dict(genre_breakdown),
        "trending": trending,
        "watch_streak": watch_streak,
        "quick_stats": quick_stats,
        "in_progress": in_progress,
        "completed": completed,
        "sig": sig
    }

    cache["data"] = result
    cache["time"] = datetime.now()
    cache["sig"] = sig
    
    return result


def jellyfin_import():
    print("\n" + "="*60)
    print("JELLYFIN IMPORT STARTING")
    print("="*60)
    
    if not JELLYFIN_URL:
        msg = "JELLYFIN_URL not set"
        print(f"ERROR: {msg}")
        return False, msg, 0
    
    if not JELLYFIN_API_KEY:
        msg = "JELLYFIN_API_KEY not set"
        print(f"ERROR: {msg}")
        return False, msg, 0
    
    try:
        headers = {"X-Emby-Token": JELLYFIN_API_KEY}
        
        print("\nFetching users...")
        r = requests.get(f"{JELLYFIN_URL}/Users", headers=headers, timeout=15)
        
        if r.status_code != 200:
            msg = f"Users fetch failed: HTTP {r.status_code}"
            print(f"ERROR: {msg}")
            return False, msg, 0
        
        users = r.json()
        if not users:
            msg = "No users found"
            print(f"ERROR: {msg}")
            return False, msg, 0
        
        user_id = users[0]["Id"]
        user_name = users[0]["Name"]
        print(f"User: {user_name} (ID: {user_id})")

        start = 0
        limit = 100
        imported = 0
        os.makedirs("/data", exist_ok=True)

        print("\nStarting item fetch...")
        
        with open(DATA_FILE, "w") as f:
            iteration = 0
            while True:
                iteration += 1
                print(f"\n[Batch {iteration}] Fetching items {start} to {start+limit}...")
                
                params = {
                    "Filters": "IsPlayed",
                    "Recursive": "true",
                    "Fields": "Genres,UserData",
                    "IncludeItemTypes": "Movie,Episode",
                    "StartIndex": start,
                    "Limit": limit
                }
                
                try:
                    r2 = requests.get(
                        f"{JELLYFIN_URL}/Users/{user_id}/Items",
                        headers=headers,
                        params=params,
                        timeout=30
                    )
                except Exception as e:
                    print(f"  ERROR: {e}")
                    print("  Retrying...")
                    continue
                
                if r2.status_code != 200:
                    msg = f"Items fetch failed at {start}: HTTP {r2.status_code}"
                    print(f"ERROR: {msg}")
                    return False, msg, imported
                
                page = r2.json()
                items = page.get("Items", [])
                total = page.get("TotalRecordCount", 0)
                
                print(f"  Got {len(items)} items (Total in library: {total})")

                for item in items:
                    last_played = item.get("UserData", {}).get("LastPlayedDate")
                    if not last_played:
                        continue
                     
                    record = {
                        "timestamp": last_played,
                        "type": item.get("Type"),
                        "name": item.get("Name"),
                        "year": item.get("ProductionYear"),
                        "series_name": item.get("SeriesName"),
                        "season": item.get("ParentIndexNumber"),
                        "episode": item.get("IndexNumber"),
                        "user": user_name,
                        "genres": item.get("Genres", []),
                        "source": "import",
                        "jellyfin_id": item.get("Id"),
                    }
                    f.write(json.dumps(record) + "\n")
                    imported += 1
                
                f.flush()
                print(f"  Imported so far: {imported}")

                start += len(items)
                if start >= total or not items:
                    break

        cache["data"] = None
        cache["time"] = None
        
        print("\n" + "="*60)
        print(f"IMPORT COMPLETE: {imported} items")
        print("="*60 + "\n")
        
        return True, f"Imported {imported} items successfully", imported
        
    except Exception as e:
        msg = f"Import error: {str(e)}"
        print(f"\nERROR: {msg}")
        print(traceback.format_exc())
        return False, msg, 0


def sonarr_import():
    print("\n" + "="*60)
    print("SONARR IMPORT STARTING")
    print("="*60)
    
    if not SONARR_URL or not SONARR_API_KEY:
        msg = "SONARR_URL or SONARR_API_KEY not set"
        print(f"ERROR: {msg}")
        return False, msg, 0
    
    try:
        headers = {"X-Api-Key": SONARR_API_KEY}
        
        print("\nFetching series from Sonarr...")
        r = requests.get(f"{SONARR_URL}/api/v3/series", headers=headers, timeout=15)
        
        if r.status_code != 200:
            msg = f"Sonarr fetch failed: HTTP {r.status_code}"
            print(f"ERROR: {msg}")
            return False, msg, 0
        
        series_list = r.json()
        print(f"Found {len(series_list)} series in Sonarr")
        
        imported = 0
        for series in series_list:
            title = series.get("title")
            year = series.get("year")
            
            if not title:
                continue
            
            series_id = series.get("id")
            r2 = requests.get(f"{SONARR_URL}/api/v3/episodefile", headers=headers, params={"seriesId": series_id}, timeout=15)
            
            if r2.status_code == 200:
                episode_files = r2.json()
                
                for ep_file in episode_files:
                    season_num = ep_file.get("seasonNumber")
                    episodes = ep_file.get("episodes", [])
                    
                    for ep in episodes:
                        ep_num = ep.get("episodeNumber")
                        ep_title = ep.get("title", f"Episode {ep_num}")
                        
                        record = {
                            "timestamp": datetime.now().isoformat(),
                            "type": "Episode",
                            "name": ep_title,
                            "year": year,
                            "series_name": title,
                            "season": season_num,
                            "episode": ep_num,
                            "user": "Sonarr Import",
                            "genres": [g for g in series.get("genres", [])],
                            "source": "sonarr_import"
                        }
                        append_record(record)
                        imported += 1
            
            print(f"  Imported: {title}")
        
        print("\n" + "="*60)
        print(f"SONARR IMPORT COMPLETE: {imported} episodes")
        print("="*60 + "\n")
        
        return True, f"Imported {imported} episodes from Sonarr", imported
        
    except Exception as e:
        msg = f"Sonarr import error: {str(e)}"
        print(f"\nERROR: {msg}")
        print(traceback.format_exc())
        return False, msg, 0


def radarr_import():
    print("\n" + "="*60)
    print("RADARR IMPORT STARTING")
    print("="*60)
    
    if not RADARR_URL or not RADARR_API_KEY:
        msg = "RADARR_URL or RADARR_API_KEY not set"
        print(f"ERROR: {msg}")
        return False, msg, 0
    
    try:
        headers = {"X-Api-Key": RADARR_API_KEY}
        
        print("\nFetching movies from Radarr...")
        r = requests.get(f"{RADARR_URL}/api/v3/movie", headers=headers, timeout=15)
        
        if r.status_code != 200:
            msg = f"Radarr fetch failed: HTTP {r.status_code}"
            print(f"ERROR: {msg}")
            return False, msg, 0
        
        movies_list = r.json()
        print(f"Found {len(movies_list)} movies in Radarr")
        
        imported = 0
        for movie in movies_list:
            title = movie.get("title")
            year = movie.get("year")
            has_file = movie.get("hasFile", False)
            
            if not title or not has_file:
                continue
            
            record = {
                "timestamp": datetime.now().isoformat(),
                "type": "Movie",
                "name": title,
                "year": year,
                "series_name": None,
                "season": None,
                "episode": None,
                "user": "Radarr Import",
                "genres": [g for g in movie.get("genres", [])],
                "source": "radarr_import"
            }
            append_record(record)
            imported += 1
            print(f"  Imported: {title} ({year})")
        
        print("\n" + "="*60)
        print(f"RADARR IMPORT COMPLETE: {imported} movies")
        print("="*60 + "\n")
        
        return True, f"Imported {imported} movies from Radarr", imported
        
    except Exception as e:
        msg = f"Radarr import error: {str(e)}"
        print(f"\nERROR: {msg}")
        print(traceback.format_exc())
        return False, msg, 0


@app.route("/")
def index():
    return """<!DOCTYPE html>
<html><head><meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1.0"/><title>Jellyfin Watch Tracker</title>
<style>html,body{background:#0a0e27;color:#fff;}</style>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
/* === Additional styles for advanced features === */
.rating-wrap{display:flex;flex-direction:column;gap:4px;margin-top:4px}
.rating{display:inline-flex;gap:2px;cursor:pointer}
.rating .star{font-size:1.1em;color:rgba(255,255,255,0.5);transition:color 0.2s;}
.rating .star.filled{color:#ffd93d;}
.rating-note{font-size:0.75em;opacity:0.75;line-height:1.2;word-break:break-word;}
.grid-item .rating-wrap{align-items:center;text-align:center}
.grid-item .rating{justify-content:center}
.advanced-filters{display:flex;flex-wrap:wrap;gap:10px;margin-top:10px;align-items:center;width:100%}
.advanced-filters select,.advanced-filters input{padding:8px 12px;border-radius:8px;border:2px solid rgba(255,255,255,0.1);background:rgba(20,25,45,0.6);color:#fff;font-size:13px;flex:1;min-width:120px}
.advanced-filters button{padding:8px 12px;border-radius:8px;border:2px solid rgba(255,255,255,0.1);background:rgba(139,156,255,0.15);color:#8b9cff;font-weight:600;cursor:pointer;transition:background 0.2s}
.advanced-filters button:hover{background:rgba(139,156,255,0.25)}
#resultsCounter{font-size:0.9em;font-weight:600;margin-left:auto;opacity:0.8;white-space:nowrap}
.auto-refresh{display:flex;align-items:center;gap:6px;margin-left:10px;}
.auto-refresh input[type="checkbox"]{transform:scale(1.2);}
.auto-refresh span{font-size:0.8em;opacity:0.7;}
/* Theme variations */
body.light{background:#f5f5f5;color:#1a1f3a;}
body.light .header,body.light .content,body.light .stats .stat-card,body.light .filters,body.light .movie-item,body.light .show-group,body.light .genre-item,body.light .season-group,body.light .episode-item{background:rgba(255,255,255,0.9);color:#1a1f3a;border-color:rgba(0,0,0,0.1)}
body.light .filter-btn{color:#8b9cff;background:rgba(139,156,255,0.15)}
body.amoled{background:#000;color:#fff;}
body.amoled .header,body.amoled .content,body.amoled .stats .stat-card,body.amoled .filters,body.amoled .movie-item,body.amoled .show-group,body.amoled .genre-item,body.amoled .season-group,body.amoled .episode-item{background:rgba(0,0,0,0.8);border-color:rgba(255,255,255,0.1);color:#fff}
body.solarized{background:#002b36;color:#93a1a1;}
body.solarized .header,body.solarized .content,body.solarized .stats .stat-card,body.solarized .filters,body.solarized .movie-item,body.solarized .show-group,body.solarized .genre-item,body.solarized .season-group,body.solarized .episode-item{background:rgba(0,43,54,0.8);color:#93a1a1;border-color:rgba(147,161,161,0.2)}
body.nord{background:#2e3440;color:#d8dee9;}
body.nord .header,body.nord .content,body.nord .stats .stat-card,body.nord .filters,body.nord .movie-item,body.nord .show-group,body.nord .genre-item,body.nord .season-group,body.nord .episode-item{background:rgba(46,52,64,0.8);border-color:rgba(216,222,233,0.1);color:#d8dee9}

/* Layout variants for compact and spacious modes */
:root.layout-compact .movie-item,:root.layout-compact .show-group,:root.layout-compact .genre-item,:root.layout-compact .season-group,:root.layout-compact .episode-item{
  padding:8px;
  margin-bottom:8px;
  gap:8px;
}
:root.layout-spacious .movie-item,:root.layout-spacious .show-group,:root.layout-spacious .genre-item,:root.layout-spacious .season-group,:root.layout-spacious .episode-item{
  padding:25px;
  margin-bottom:20px;
  gap:20px;
}
*{margin:0;padding:0;box-sizing:border-box}
html{font-size:16px;height:100%}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:linear-gradient(135deg,#0a0e27 0%,#1a1f3a 100%);color:#fff;padding:15px;min-height:100%;overflow-y:auto}
.container{max-width:1800px;margin:0 auto;width:100%}
.header{background:linear-gradient(135deg,rgba(139,156,255,0.15),rgba(255,107,107,0.15));backdrop-filter:blur(20px);border-radius:20px;padding:30px;margin-bottom:30px;border:1px solid rgba(255,255,255,0.1);box-shadow:0 8px 32px rgba(0,0,0,0.3);display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:20px}
.header-title{display:flex;align-items:center;gap:15px}
.logo{font-size:2.5em;filter:drop-shadow(0 4px 15px rgba(139,156,255,0.5))}
.header h1{font-size:2em;font-weight:700;background:linear-gradient(135deg,#8b9cff,#ff8585);-webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:-1px;line-height:1.4;padding-bottom:6px;overflow:visible}
.btn{background:linear-gradient(135deg,#8b9cff,#6b8cff);color:#fff;border:none;padding:10px 16px;border-radius:12px;cursor:pointer;font-weight:600;font-size:13px;margin:3px;transition:all 0.3s;box-shadow:0 4px 15px rgba(139,156,255,0.3)}
.btn:hover{transform:translate3d(0,-2px,0);}
.btn:disabled{opacity:0.5;cursor:not-allowed;transform:none}
.btn.secondary{background:transparent;border:2px solid #8b9cff;box-shadow:none}
.btn.secondary:hover{background:rgba(139,156,255,0.1)}
.btn.manual{background:linear-gradient(135deg,#6bcf7f,#5eb3f5);box-shadow:0 4px 15px rgba(107,207,127,0.3)}
.btn.manual:hover{box-shadow:0 6px 20px rgba(107,207,127,0.5)}
.btn.danger{background:linear-gradient(135deg,#ff6b6b,#ff8585);box-shadow:0 4px 15px rgba(255,107,107,0.3)}
.btn.danger:hover{box-shadow:0 6px 20px rgba(255,107,107,0.5)}
.btn.small{padding:6px 12px;font-size:11px;margin:2px}
.btn.warning{background:linear-gradient(135deg,#ffd93d,#ffb86c);color:#1a1f3a;box-shadow:0 4px 15px rgba(255,217,61,0.3)}
.btn.warning:hover{box-shadow:0 6px 20px rgba(255,217,61,0.5)}
.btn.purple{background:linear-gradient(135deg,#c78bff,#a855f7);box-shadow:0 4px 15px rgba(199,139,255,0.3)}
.btn.purple:hover{box-shadow:0 6px 20px rgba(199,139,255,0.5)}
.tabs{display:flex;gap:10px;margin-bottom:25px;background:rgba(20,25,45,0.6);backdrop-filter:blur(10px);padding:10px;border-radius:16px;border:1px solid rgba(255,255,255,0.08);flex-wrap:wrap}
.tab-btn{background:transparent;color:rgba(255,255,255,0.6);border:none;padding:10px 20px;border-radius:12px;cursor:pointer;font-weight:600;font-size:14px;transition:all 0.3s}
.tab-btn.active{background:linear-gradient(135deg,#8b9cff,#6b8cff);color:#fff;box-shadow:0 4px 15px rgba(139,156,255,0.4)}
.tab-btn:hover{color:#fff;background:rgba(139,156,255,0.2)}
.tab-content{display:none}
.tab-content.active{display:block}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:15px;margin-bottom:25px}
.stat-card{background:rgba(20,25,45,0.6);backdrop-filter:blur(10px);border-radius:16px;padding:20px;border:1px solid rgba(255,255,255,0.08);transition:all 0.3s}
.stat-card:hover{transform:translate3d(0,-5px,0);}
.stat-card h3{color:rgba(255,255,255,0.5);font-size:0.8em;margin-bottom:8px;font-weight:600;text-transform:uppercase;letter-spacing:1px}
.stat-card .value{color:#8b9cff;font-size:2em;font-weight:700;text-shadow:0 2px 10px rgba(139,156,255,0.3)}
.filters{background:rgba(20,25,45,0.6);backdrop-filter:blur(10px);border-radius:16px;padding:15px;margin-bottom:20px;border:1px solid rgba(255,255,255,0.08);display:flex;gap:10px;flex-wrap:wrap;align-items:center}
.filter-btn{background:rgba(139,156,255,0.15);color:#8b9cff;border:2px solid rgba(139,156,255,0.3);padding:8px 16px;border-radius:12px;cursor:pointer;font-weight:600;font-size:13px;transition:all 0.3s}
.filter-btn:hover{background:rgba(139,156,255,0.25);border-color:rgba(139,156,255,0.5)}
.filter-btn.active{background:linear-gradient(135deg,#8b9cff,#6b8cff);color:#fff;border-color:transparent;box-shadow:0 4px 15px rgba(139,156,255,0.4)}
.search-box{flex:1;min-width:200px}
.search-box input,select{width:100%;padding:10px 15px;border-radius:12px;border:2px solid rgba(255,255,255,0.08);background:rgba(20,25,45,0.6);color:#fff;font-size:13px;transition:all 0.3s;font-weight:500}
.search-box input:focus,select:focus{outline:none;border-color:#8b9cff;box-shadow:0 0 20px rgba(139,156,255,0.2)}
select{cursor:pointer}
.content{background:rgba(20,25,45,0.6);backdrop-filter:blur(10px);border-radius:16px;padding:15px;border:1px solid rgba(255,255,255,0.08)}
.movie-item,.show-group{background:rgba(20,25,45,0.8);backdrop-filter:blur(10px);border-radius:16px;padding:15px;margin-bottom:12px;display:flex;gap:15px;border:1px solid rgba(255,255,255,0.05);transition:all 0.3s;position:relative}
.movie-item:hover,.show-group:hover{transform:translate3d(5px,0,0);}
.movie-item{border-left:4px solid #ff8585}
.show-group{border-left:4px solid #5ef5e0}
.poster-container{position:relative}
.poster{width:90px;height:135px;border-radius:12px;object-fit:cover;box-shadow:0 4px 15px rgba(0,0,0,0.4)}
.poster-placeholder {
  width:90px;
  height:135px;
  border-radius:12px;
  background:linear-gradient(135deg, rgba(139,156,255,0.1), rgba(255,107,107,0.1));
  display:flex;
  align-items:center;
  justify-content:center;
  font-size:2.5em;
  min-height:135px;
}

.poster {
  min-height:135px;
  background:rgba(20,25,45,0.8);
}

.grid-item .poster {
  min-height:200px;
}

.grid-item .poster-placeholder {
  height:200px;
  min-height:200px;
}
.upload-poster-btn{position:absolute;bottom:5px;right:5px;background:rgba(139,156,255,0.9);color:#fff;border:none;padding:5px 8px;border-radius:8px;cursor:pointer;font-size:10px;font-weight:600;opacity:0;transition:opacity 0.3s}
.poster-container:hover .upload-poster-btn{opacity:1}
.item-content{flex:1;min-width:0}
.movie-title,.show-title{font-size:1.3em;font-weight:700;margin-bottom:8px;letter-spacing:-0.5px}
.movie-details{color:rgba(255,255,255,0.7);margin-bottom:8px;font-size:0.95em}
.badge{padding:4px 12px;border-radius:18px;font-size:0.8em;font-weight:700;margin-right:6px;display:inline-block;letter-spacing:0.3px}
.badge-success{background:rgba(107,207,127,0.25);color:#6bcf7f;border:2px solid rgba(107,207,127,0.4)}
.badge-warning{background:rgba(255,217,61,0.25);color:#ffd93d;border:2px solid rgba(255,217,61,0.4)}
.badge-info{background:rgba(94,179,245,0.25);color:#5eb3f5;border:2px solid rgba(94,179,245,0.4)}
.badge-alert{background:rgba(255,107,107,0.25);color:#ff6b6b;border:2px solid rgba(255,107,107,0.4)}
.show-header{cursor:pointer;padding:8px;border-radius:12px;display:flex;justify-content:space-between;align-items:center;transition:all 0.3s}
.show-header:hover{background:rgba(139,156,255,0.1)}
.progress-bar{width:100%;height:6px;background:rgba(255,255,255,0.08);border-radius:6px;overflow:hidden;margin-top:8px;box-shadow:inset 0 2px 5px rgba(0,0,0,0.2)}
.progress-fill{height:100%;background:linear-gradient(90deg,#8b9cff,#6bcf7f);border-radius:6px;box-shadow:0 2px 8px rgba(139,156,255,0.4)}
.seasons-list{margin-top:15px;padding-top:15px;border-top:1px solid rgba(255,255,255,0.08);display:none}
.season-group{background:rgba(255,255,255,0.05);backdrop-filter:blur(5px);border-radius:12px;padding:12px;margin-bottom:10px;border:1px solid rgba(255,255,255,0.05)}
.season-header{cursor:pointer;padding:6px;border-radius:10px;display:flex;justify-content:space-between;transition:all 0.3s}
.season-header:hover{background:rgba(139,156,255,0.15)}
.season-title{font-size:1.1em;font-weight:700;letter-spacing:-0.3px}
.episodes-list{margin-top:12px;padding-top:12px;border-top:1px solid rgba(255,255,255,0.08);display:none}
.episode-item{background:rgba(20,25,45,0.95);border-radius:10px;padding:10px 12px;margin-bottom:6px;font-size:0.95em;border:1px solid rgba(255,255,255,0.05);transition:all 0.3s;display:flex;justify-content:space-between;align-items:center}
.episode-item:hover{background:rgba(139,156,255,0.1);border-color:rgba(139,156,255,0.2)}
.episode-item .delete-btn{opacity:0;transition:opacity 0.3s}
.episode-item:hover .delete-btn{opacity:1}
.episode-item.selected{background:rgba(139,156,255,0.2);border-color:rgba(139,156,255,0.4)}
.loading{text-align:center;padding:40px;color:rgba(255,255,255,0.4);font-size:1em}
.genre-item{background:rgba(20,25,45,0.8);backdrop-filter:blur(10px);border-radius:16px;padding:15px;margin-bottom:12px;border-left:4px solid #8b9cff;border:1px solid rgba(255,255,255,0.05);transition:all 0.3s}
.genre-item:hover{transform:translate3d(5px,0,0);}
.genre-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}
.genre-name{font-size:1.2em;font-weight:700;letter-spacing:-0.5px}
.genre-count{background:rgba(139,156,255,0.2);color:#8b9cff;padding:5px 15px;border-radius:20px;border:2px solid rgba(139,156,255,0.3);font-weight:700;font-size:0.95em}
.genre-items{color:rgba(255,255,255,0.75);font-size:0.95em;line-height:1.6}
.genre-items strong{color:rgba(255,255,255,0.9);font-weight:600}
.chart-container{background:rgba(20,25,45,0.8);backdrop-filter:blur(10px);border-radius:16px;padding:25px;margin-bottom:25px;border:1px solid rgba(255,255,255,0.08);max-width:450px;margin-left:auto;margin-right:auto}

#genreChart {
  max-height: 300px !important;
  max-width: 300px !important;
  margin: 0 auto;
}
.modal{display:none;position:fixed;z-index:1000;left:0;top:0;width:100%;height:100%;background:rgba(0,0,0,0.8);backdrop-filter:blur(10px);overflow-y:auto}
.modal-content{background:rgba(20,25,45,0.95);backdrop-filter:blur(20px);border:1px solid rgba(255,255,255,0.1);margin:3% auto;padding:25px;border-radius:20px;width:90%;max-width:600px;box-shadow:0 10px 50px rgba(0,0,0,0.5)}
.close{color:rgba(255,255,255,0.6);float:right;font-size:26px;font-weight:bold;cursor:pointer;transition:color 0.3s}
.close:hover{color:#fff}
input[type="file"],input[type="text"],input[type="number"],input[type="date"]{width:100%;padding:12px;border-radius:12px;border:2px solid rgba(255,255,255,0.1);background:rgba(20,25,45,0.6);color:#fff;margin-top:8px;margin-bottom:12px;font-size:13px;font-weight:500}
input[type="file"]{border-style:dashed;border-color:rgba(139,156,255,0.5);cursor:pointer}
input[type="checkbox"]{width:auto;margin:0}
input::placeholder{color:rgba(255,255,255,0.4)}
.form-group{margin-bottom:15px}
.form-group label{display:block;color:rgba(255,255,255,0.8);font-weight:600;margin-bottom:6px;font-size:0.9em}
.zoom-control{display:flex;align-items:center;gap:10px;background:rgba(20,25,45,0.6);backdrop-filter:blur(10px);padding:10px 15px;border-radius:12px;border:1px solid rgba(255,255,255,0.08)}
.zoom-control label{color:rgba(255,255,255,0.7);font-size:0.85em;font-weight:600;white-space:nowrap}
.zoom-slider{
  width:140px;
  height:22px;
  background:transparent;
  outline:none;
  border:0;
  padding:0;
  -webkit-appearance:none;
  appearance:none;
}
.zoom-slider:focus{outline:none}
.zoom-slider::-webkit-slider-runnable-track{
  height:6px;
  border-radius:999px;
  background:rgba(255,255,255,0.14);
  border:1px solid rgba(255,255,255,0.10);
}
.zoom-slider::-webkit-slider-thumb{
  -webkit-appearance:none;
  appearance:none;
  width:16px;
  height:16px;
  border-radius:50%;
  background:linear-gradient(135deg,var(--accent,#8b9cff),var(--accent2,#6b8cff));
  cursor:pointer;
  box-shadow:0 2px 8px rgba(139,156,255,0.5);
  margin-top:-6px; /* center thumb on 6px track */
}
.zoom-slider::-moz-range-track{
  height:6px;
  border-radius:999px;
  background:rgba(255,255,255,0.14);
  border:1px solid rgba(255,255,255,0.10);
}
.zoom-slider::-moz-range-thumb{
  width:16px;
  height:16px;
  border-radius:50%;
  background:linear-gradient(135deg,var(--accent,#8b9cff),var(--accent2,#6b8cff));
  cursor:pointer;
  border:0;
  box-shadow:0 2px 8px rgba(139,156,255,0.5);
}
.zoom-slider::-moz-focus-outer{border:0}
.zoom-slider::-ms-track{background:transparent;border-color:transparent;color:transparent}
.zoom-value{color:#8b9cff;font-weight:700;font-size:0.9em;min-width:40px}
.view-toggle{display:flex;gap:6px}
.view-toggle button{background:rgba(139,156,255,0.15);border:2px solid rgba(139,156,255,0.3);color:#8b9cff;padding:7px 14px;border-radius:10px;cursor:pointer;font-weight:600;font-size:12px;transition:all 0.3s}
.view-toggle button.active{background:linear-gradient(135deg,#8b9cff,#6b8cff);color:#fff;border-color:transparent}
.grid-view{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:15px}
.grid-item{background:rgba(20,25,45,0.8);backdrop-filter:blur(10px);border-radius:16px;padding:12px;border:1px solid rgba(255,255,255,0.05);transition:all 0.3s;cursor:pointer;position:relative}
.grid-item .poster{width:100%;height:200px}
.grid-item .poster-placeholder{width:100%;height:200px}
.grid-item-title{font-size:1em;font-weight:600;margin-top:10px;text-align:center;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.grid-item-info{font-size:0.85em;text-align:center;margin-top:5px;color:rgba(255,255,255,0.6)}
.grid-item .grid-item-actions {
    position:absolute;
    top:8px;
    right:8px;
    opacity:0;  /* Hidden by default */
    transition:opacity 0.3s;
    z-index:10
}
.grid-item:hover .grid-item-actions {
    opacity:1;  /* Show on hover */
}
.grid-item:hover{transform:translate3d(0,-5px,0);}
/* ================= GRID MULTI-SELECT (Checkbox only when needed) ================= */
.grid-checkbox-wrap{
  position:absolute;
  top:10px;
  left:10px;
  z-index:15;
  display:none; /* hidden unless selection mode is ON */
}
body.grid-select-mode .grid-checkbox-wrap{
  display:block;
}
body.grid-select-mode .grid-item{
  cursor:default;
}
.grid-select-checkbox{
  width:22px !important;
  height:22px !important;
  min-width:22px !important;
  min-height:22px !important;
  margin:0 !important;
  cursor:pointer !important;
  appearance:none !important;
  -webkit-appearance:none !important;
  background:rgba(20,25,45,0.9) !important;
  border:2px solid rgba(255,255,255,0.35) !important;
  border-radius:6px !important;
  position:relative !important;
}
.grid-select-checkbox:checked{
  background:#8b9cff !important;
  border-color:#8b9cff !important;
}
.grid-select-checkbox:checked::after{
  content:"✓" !important;
  position:absolute !important;
  top:-2px !important;
  left:4px !important;
  color:#fff !important;
  font-size:18px !important;
  font-weight:800 !important;
}
.grid-item.grid-selected{
  outline:2px solid rgba(139,156,255,0.7);
  box-shadow:0 0 0 4px rgba(139,156,255,0.15);
}
.quick-stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:15px;margin-bottom:25px}
.quick-stat-card{background:rgba(20,25,45,0.8);backdrop-filter:blur(10px);border-radius:16px;padding:20px;border-left:4px solid #8b9cff;border:1px solid rgba(255,255,255,0.08);transition:all 0.3s}
.quick-stat-card:hover{transform:translateY(-3px);box-shadow:0 6px 20px rgba(0,0,0,0.3)}
.quick-stat-title{color:rgba(255,255,255,0.6);font-size:0.85em;margin-bottom:8px;text-transform:uppercase;letter-spacing:1px;font-weight:600}
.quick-stat-value{color:#8b9cff;font-size:1.6em;font-weight:700;margin-bottom:4px}
.quick-stat-label{color:rgba(255,255,255,0.8);font-size:1em}
.radio-group{display:flex;gap:15px;margin-top:8px;flex-wrap:wrap}
.radio-group label{display:flex;align-items:center;gap:6px;cursor:pointer}
.radio-group input[type="radio"]{width:auto;margin:0}
.checkbox-group{display:flex;align-items:center;gap:8px;margin-top:8px}
.checkbox-group input[type="checkbox"]{width:auto;margin:0}
.episode-fields{display:none;padding:12px;background:rgba(139,156,255,0.1);border-radius:12px;margin-top:12px}
.episode-fields.show{display:block}
.action-buttons{display:flex;gap:5px;flex-wrap:wrap;margin-top:8px}
.manage-actions{opacity:0;transition:opacity 0.3s}
.show-group:hover .manage-actions{opacity:1}
/* Movie actions were unintentionally hidden (no hover rule). Always show for movies. */
.movie-item .manage-actions{opacity:1}
.movie-item:hover .manage-actions{opacity:1}

/* Make Jellyfin link buttons more visible and touch-friendly */
.btn.linkout{
  background: linear-gradient(135deg, rgba(111,130,255,0.85), rgba(95,115,255,0.85)) !important;
  border: 1px solid rgba(255,255,255,0.14) !important;
  color: #fff !important;
  font-weight: 700;
}
.btn.linkout:hover{
  background: linear-gradient(135deg, rgba(111,130,255,0.92), rgba(95,115,255,0.92)) !important;
}
.btn.linkout.small{
  min-width: 46px;
}
.season-actions{opacity:0;transition:opacity 0.3s}
.season-group:hover .season-actions{opacity:1}
.import-section{background:rgba(20,25,45,0.8);backdrop-filter:blur(10px);border-radius:16px;padding:20px;margin-bottom:20px;border:1px solid rgba(255,255,255,0.08)}
.import-section h3{margin-bottom:15px;font-size:1.3em;color:#8b9cff}
.import-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:15px}
.import-card{background:rgba(255,255,255,0.05);border-radius:12px;padding:15px;border:1px solid rgba(255,255,255,0.1);transition:all 0.3s}
.import-card:hover{border-color:rgba(139,156,255,0.3);box-shadow:0 4px 15px rgba(0,0,0,0.2)}
.import-card h4{margin-bottom:10px;font-size:1.1em}
.import-card p{color:rgba(255,255,255,0.7);font-size:0.9em;margin-bottom:12px}
.bulk-bar{display:none;background:rgba(139,156,255,0.2);backdrop-filter:blur(10px);border-radius:12px;padding:12px 15px;margin-bottom:15px;border:2px solid rgba(139,156,255,0.4);align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px}
.bulk-bar.active{display:flex}
.bulk-info{color:#8b9cff;font-weight:700;font-size:0.95em}
.bulk-actions{display:flex;gap:8px;flex-wrap:wrap}
input.select-checkbox {
  width: 20px !important;
  height: 20px !important;
  min-width: 20px !important;
  min-height: 20px !important;
  cursor: pointer !important;
  margin: 0 12px 0 0 !important;
  padding: 0 !important;
  appearance: none !important;
  -webkit-appearance: none !important;
  -moz-appearance: none !important;
  background-color: #1a1a1a !important;
  border: 2px solid #555 !important;
  border-radius: 4px !important;
  position: relative !important;
  flex-shrink: 0 !important;
  display: inline-block !important;
  vertical-align: middle !important;
}

input.select-checkbox:hover {
  border-color: #8b9cff !important;
}

input.select-checkbox:checked {
  background-color: #8b9cff !important;
  border-color: #8b9cff !important;
}

input.select-checkbox:checked::after {
  content: "✓" !important;
  position: absolute !important;
  top: -2px !important;
  left: 2px !important;
  color: #fff !important;
  font-size: 18px !important;
  font-weight: bold !important;
  line-height: 1 !important;
}

.episode-item.selected {
  background: rgba(139,156,255,0.12) !important;
}

.content-loading {
  opacity: 0.5;
  pointer-events: none;
  transition: opacity 0.2s;
}

/* ================= THEME TOGGLE (Dark/Light) ================= */
body.light{
  background:linear-gradient(135deg,#f6f8ff 0%,#e9eefc 100%) !important;
  color:#101426 !important;
}
body.light .header{
  background:linear-gradient(135deg,rgba(60,80,255,0.12),rgba(255,120,120,0.10)) !important;
  border:1px solid rgba(0,0,0,0.08) !important;
  box-shadow:0 8px 32px rgba(0,0,0,0.10) !important;
}
body.light .tabs,
body.light .stat-card,
body.light .filters,
body.light .content,
body.light .movie-item,
body.light .show-group,
body.light .grid-item,
body.light .genre-item,
body.light .chart-container,
body.light .import-section,
body.light .import-card,
body.light .modal-content,
body.light .season-group,
body.light .episode-item{
  background:rgba(255,255,255,0.80) !important;
  border:1px solid rgba(0,0,0,0.08) !important;
  box-shadow:none !important;
}
body.light .tab-btn{ color:rgba(0,0,0,0.55) !important; }
body.light .tab-btn:hover{ color:#000 !important; background:rgba(60,80,255,0.10) !important; }

body.light .movie-details,
body.light .genre-items,
body.light .quick-stat-title,
body.light .grid-item-info,
body.light .form-group label,
body.light p,
body.light .loading{
  color:rgba(0,0,0,0.65) !important;
}
body.light .movie-title,
body.light .show-title,
body.light .season-title,
body.light .genre-name,
body.light h2, body.light h3, body.light h4{
  color:#101426 !important;
}
body.light .badge-info{
  background:rgba(94,179,245,0.20) !important;
  border-color:rgba(94,179,245,0.35) !important;
}
body.light input[type="text"],
body.light input[type="number"],
body.light input[type="date"],
body.light select{
  background:rgba(255,255,255,0.95) !important;
  color:#101426 !important;
  border:2px solid rgba(0,0,0,0.10) !important;
}
body.light input::placeholder{ color:rgba(0,0,0,0.40) !important; }
body.light .close{ color:rgba(0,0,0,0.55) !important; }
body.light .close:hover{ color:#000 !important; }

/* chart legend readability in light mode */
body.light canvas{ filter:none !important; }
/* ================= HEADER TOOLBAR LOOK (like screenshot) ================= */

.header{
  padding:22px 26px !important;
  border-radius:22px !important;
  gap:18px !important;
}

.header-title{
  gap:14px !important;
}

.logo{
  font-size:2.1em !important;
}

.header h1{
  font-size:2.1em !important;
  line-height:1.05 !important;
}

/* Right side toolbar container */
.header-toolbar{
  display:flex;
  align-items:center;
  gap:14px;
  flex-wrap:wrap;
  justify-content:flex-end;
}

/* Make toolbar buttons like compact tiles */
.header-toolbar .btn{
  padding:10px 12px !important;
  border-radius:14px !important;
  min-width:74px;
  height:58px;
  display:flex;
  flex-direction:column;
  align-items:center;
  justify-content:center;
  line-height:1.05;
  font-size:12px !important;
  margin:0 !important;
}

/* Keep “Add Watch” slightly more prominent like screenshot */
.header-toolbar .btn.manual{
  min-width:82px;
  height:62px;
}

/* Zoom control styled like a pill */
.zoom-control{
  height:58px;
  padding:10px 14px !important;
  border-radius:14px !important;
  gap:12px !important;
}

/* Keep zoom label + percent aligned nicely */
.zoom-control label{
  margin:0 !important;
  font-size:12px !important;
}
.zoom-value{
  font-size:12px !important;
}

/* Make the header less tall on small screens */
@media (max-width: 900px){
  .header{
    padding:18px 18px !important;
  }
  .header-toolbar{
    justify-content:flex-start;
  }
  .header-toolbar .btn{
    min-width:70px;
    height:54px;
  }
  .zoom-control{
    height:54px;
  }
}
/* ================= SOFTER DARK THEME (Less eye strain) ================= */

/* Overall page background like your screenshot (muted slate) */
body{
  background: radial-gradient(1200px 700px at 20% 0%, #141a28 0%, #0b0f18 60%, #070a11 100%) !important;
  color: rgba(255,255,255,0.92) !important;
}

/* Header slab - calmer, less colorful */
.header{
  background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.03)) !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  box-shadow: 0 10px 40px rgba(0,0,0,0.45) !important;
}

/* Tabs + panels: same family as screenshot */
.tabs,
.stat-card,
.filters,
.content,
.movie-item,
.show-group,
.grid-item,
.genre-item,
.chart-container,
.import-section,
.import-card,
.modal-content,
.season-group,
.episode-item{
  background: linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.03)) !important;
  border: 1px solid rgba(255,255,255,0.075) !important;
  box-shadow: none !important;
}

/* Text contrast softened */
.movie-details,
.genre-items,
.quick-stat-title,
.grid-item-info,
.form-group label,
.loading{
  color: rgba(255,255,255,0.68) !important;
}

/* Accent color: soft blue-violet (less neon) */
:root{
  --accent: #6f82ff;
  --accent2: #5f73ff;
  --accentSoft: rgba(111,130,255,0.18);
  --borderSoft: rgba(111,130,255,0.28);
}

/* Stat value color (was very bright) */
.stat-card .value{ color: var(--accent) !important; text-shadow: none !important; }

/* Buttons: darker “tile” look like screenshot */
.btn{
  background: linear-gradient(180deg, rgba(255,255,255,0.07), rgba(255,255,255,0.04)) !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  box-shadow: none !important;
}
.btn:hover{
  transform: translateY(-1px) !important;
  background: linear-gradient(180deg, rgba(255,255,255,0.09), rgba(255,255,255,0.05)) !important;
}

/* Keep special buttons but reduce “neon” intensity */
.btn.manual{
  background: linear-gradient(135deg, rgba(111,130,255,0.85), rgba(95,115,255,0.85)) !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  box-shadow: none !important;
}
.btn.danger{
  background: linear-gradient(135deg, rgba(255,107,107,0.75), rgba(255,133,133,0.75)) !important;
  box-shadow: none !important;
}
.btn.warning{
  background: linear-gradient(135deg, rgba(255,217,61,0.80), rgba(255,184,108,0.80)) !important;
  box-shadow: none !important;
}
.btn.purple{
  background: linear-gradient(135deg, rgba(199,139,255,0.75), rgba(168,85,247,0.75)) !important;
  box-shadow: none !important;
}
.btn.secondary{
  background: transparent !important;
  border: 1px solid rgba(255,255,255,0.18) !important;
}

/* Tabs: calmer active state */
.tab-btn.active{
  background: linear-gradient(135deg, rgba(111,130,255,0.95), rgba(95,115,255,0.95)) !important;
  box-shadow: none !important;
}
.tab-btn{ color: rgba(255,255,255,0.62) !important; }
.tab-btn:hover{ background: rgba(111,130,255,0.16) !important; color: rgba(255,255,255,0.92) !important; }

/* Filters: match muted style */
.filter-btn{
  background: rgba(111,130,255,0.10) !important;
  color: rgba(210,218,255,0.95) !important;
  border: 1px solid rgba(255,255,255,0.14) !important;
}
.filter-btn.active{
  background: linear-gradient(135deg, rgba(111,130,255,0.95), rgba(95,115,255,0.95)) !important;
  border-color: transparent !important;
  box-shadow: none !important;
}

/* Inputs: darker, softer borders */
.search-box input, select,
input[type="text"], input[type="number"], input[type="date"]{
  background: rgba(10,14,22,0.55) !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
}
.search-box input:focus, select:focus,
input[type="text"]:focus, input[type="number"]:focus, input[type="date"]:focus{
  border-color: rgba(111,130,255,0.55) !important;
  box-shadow: 0 0 0 3px rgba(111,130,255,0.10) !important;
}

/* Progress bar less “glowy” */
.progress-bar{
  background: rgba(255,255,255,0.07) !important;
}
.progress-fill{
  background: linear-gradient(90deg, rgba(111,130,255,0.95), rgba(107,207,127,0.75)) !important;
  box-shadow: none !important;
}

/* Badges calmer */
.badge-info{ background: rgba(95,179,245,0.18) !important; border-color: rgba(95,179,245,0.28) !important; }
.badge-success{ background: rgba(107,207,127,0.18) !important; border-color: rgba(107,207,127,0.28) !important; }
.badge-warning{ background: rgba(255,217,61,0.16) !important; border-color: rgba(255,217,61,0.26) !important; }
.badge-alert{ background: rgba(255,107,107,0.16) !important; border-color: rgba(255,107,107,0.26) !important; }

/* H1 gradient less intense */
.header h1{
  background: linear-gradient(135deg, rgba(160,175,255,0.95), rgba(140,160,255,0.75)) !important;
  -webkit-background-clip: text !important;
  -webkit-text-fill-color: transparent !important;
}

html, body{
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-rendering: geometricPrecision;
}

/* ===== Desktop icon sharpness fix ===== */
.material-icons,
.mdl-button .material-icons,
i.md-icon,
.icon,
.emby-icon,
button i,
.headerButton i,
.btn i {
  -webkit-font-smoothing: antialiased;
  text-rendering: geometricPrecision;
  transform: translateZ(0);
  backface-visibility: hidden;
}

/* ===== Desktop text clarity ===== */
html, body {
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-rendering: geometricPrecision;
}

/* ===== Performance & smoothness ===== */
.grid-view,
.movie-item,
.show-group,
.grid-item,
.stat-card {
  will-change: transform;
}

.content {
  contain: layout paint style;
}

/* ===== 4K Crisp + Smooth Desktop Pack ===== */

/* crisp text */
html, body {
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-rendering: geometricPrecision;
}

/* crisp icons (font icons + svg) */
i, .material-icons, svg, .icon, .emby-icon {
  shape-rendering: geometricPrecision;
  text-rendering: geometricPrecision;
  transform: translateZ(0);
  backface-visibility: hidden;
}

/* make posters look sharper (browser scaling) */
img {
  image-rendering: auto; /* keep default (best for photos) */
  transform: translateZ(0);
}

/* smooth scrolling */
* {
  scroll-behavior: smooth;
}

.grid-view, .list-view, .content, .page, .main {
  contain: layout paint style;
}

/* reduce jank on hover animations */
.grid-item, .movie-item, .show-group, .stat-card, .btn, button {
  will-change: transform;
}

/* if you have any scale() hover effects, keep them tiny */
.grid-item:hover, .movie-item:hover {
  transform: translate3d(0,0,0) scale(1.01);
}

.btn.success.small{
  background:#2ecc71;
  color:#fff;
  font-weight:600;
}
.btn.success.small:hover{
  background:#27ae60;
}

.btn.success{
  background:linear-gradient(135deg,#3ad67a,#20b15b);
  color:#fff;
  box-shadow:0 4px 15px rgba(58,214,122,0.25);
}
.btn.success:hover{
  box-shadow:0 6px 20px rgba(58,214,122,0.35);
}

/* ========================================
   AUTO PERFORMANCE OPTIMIZATION
   ======================================== */

/* Detect reduced motion preference (accessibility + performance) */
@media (prefers-reduced-motion: reduce) {
  * {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}

/* Detect lower-end devices (small viewport + touch) */
@media (max-width: 768px) and (hover: none) {
  /* Remove expensive backdrop filters on mobile */
  .header,
  .tabs,
  .filters,
  .stat-card,
  .content,
  .movie-item,
  .show-group,
  .modal-content,
  .season-group,
  .chart-container,
  .import-section {
    backdrop-filter: none !important;
    background: rgba(20, 25, 45, 0.95) !important;
  }
  
  /* Simplify shadows */
  .header,
  .stat-card,
  .movie-item,
  .show-group,
  .btn {
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2) !important;
  }
  
  /* Disable hover effects on touch devices */
  .stat-card:hover,
  .movie-item:hover,
  .show-group:hover,
  .grid-item:hover {
    transform: none !important;
  }
  
  /* Faster transitions */
  * {
    transition-duration: 0.15s !important;
  }

  /* No hover on touch: keep actions visible */
  .manage-actions{ opacity: 1 !important; }
  .episode-item .delete-btn{ opacity: 1 !important; }
  .grid-item-actions{ opacity: 1 !important; }
  .grid-item-actions .btn{ min-width: 44px; min-height: 44px; }
}

/* Detect very low-end devices (old Android, budget phones) */
@media (max-width: 480px) {
  /* Remove all transforms */
  .stat-card:hover,
  .movie-item:hover,
  .show-group:hover,
  .btn:hover,
  .grid-item:hover {
    transform: none !important;
  }
  
  /* Remove text shadows */
  .stat-card .value,
  .header h1 {
    text-shadow: none !important;
  }
  
  /* Disable all transitions */
  * {
    transition: none !important;
  }
  
  /* Remove blur from all elements */
  * {
    backdrop-filter: none !important;
    filter: none !important;
  }
}
/* Add this right after your existing grid-view styles */
.grid-view {
  display:grid;
  grid-template-columns:repeat(auto-fill,minmax(150px,1fr));
  gap:15px;
  contain: layout style paint;
  will-change: scroll-position;
}

.grid-item {
  contain: layout style paint;
  content-visibility: auto;
}

.movie-item, .show-group {
  contain: layout style;
  content-visibility: auto;
}

/* ===== PERFORMANCE MODE ===== */
body.performance-mode .header,
body.performance-mode .tabs,
body.performance-mode .stat-card,
body.performance-mode .filters,
body.performance-mode .content,
body.performance-mode .movie-item,
body.performance-mode .show-group,
body.performance-mode .grid-item,
body.performance-mode .genre-item,
body.performance-mode .chart-container,
body.performance-mode .import-section,
body.performance-mode .import-card,
body.performance-mode .modal-content,
body.performance-mode .season-group,
body.performance-mode .episode-item {
  backdrop-filter: none !important;
  background: rgba(20,25,45,0.95) !important;
}

body.performance-mode .header {
  box-shadow: none !important;
}

body.performance-mode * {
  transition-duration: 0.1s !important;
  animation-duration: 0s !important;
}

body.performance-mode .grid-view,
body.performance-mode .movie-item,
body.performance-mode .show-group,
body.performance-mode .grid-item {
  will-change: auto !important;
}

/* Performance mode button active state */
#perfModeBtn.active {
  background: linear-gradient(135deg, rgba(107,207,127,0.85), rgba(94,179,245,0.85)) !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
}

.genre-item {
  background: rgba(139, 156, 255, 0.1);
  border-radius: 6px;
  padding: 2px 10px; /* Ultra-thin padding */
  margin-bottom: 2px; /* Minimal gap */
  border-left: 3px solid #8b9cff;
  transition: all 0.3s ease;
}

.genre-item:hover {
  background: rgba(139, 156, 255, 0.15);
}

.genre-header {
  cursor: pointer;
  user-select: none;
  line-height: 1.2; /* Tight line height */
}

.genre-name {
  font-weight: 700;
  font-size: 0.85em;
}

.genre-count {
  color: rgba(255, 255, 255, 0.7);
  font-size: 0.8em;
  font-weight: 600;
}

.genre-items {
  font-size: 0.85em;
  color: rgba(255, 255, 255, 0.7);
  line-height: 1.6;
}
/* Responsive adjustments for small screens.  On narrow devices like mobile,
   the two-column layout in the Genres tab can overflow or cause the pie
   chart to be cut off.  Use a media query to stack the genre list and
   chart vertically and shrink the chart so it remains fully visible.
   The !important overrides are necessary because inline styles set in
   JavaScript have higher specificity. */
@media (max-width: 700px) {
  #genres-content > div {
    grid-template-columns: 1fr !important;
  }
  #genreChart {
    /* Ensure the chart scales to the full width of the container on small screens */
    width: 100% !important;
    max-width: 100% !important;
    max-height: 250px !important;
    height: auto !important;
  }
  /* Reduce padding in the genre list for compactness on mobile */
  .genre-item {
    padding: 12px !important;
  }
  /* Remove sticky positioning for the pie chart container on small screens to
     prevent jittery scrolling.  The sticky behavior works well on large
     displays but causes a trembling effect on mobile devices as the
     viewport scrolls. */
  .genre-chart-container {
    position: static !important;
    top: auto !important;
  }
}

html.layout-compact .movie-item,
html.layout-compact .show-group{
  padding:10px !important;
  margin-bottom:8px !important;
  gap:10px !important;
}

html.layout-compact .poster,
html.layout-compact .poster-placeholder{
  width:75px !important;
  height:112px !important;
}

html.layout-spacious .movie-item,
html.layout-spacious .show-group{
  padding:22px !important;
  margin-bottom:16px !important;
  gap:18px !important;
}

html.layout-spacious .poster,
html.layout-spacious .poster-placeholder{
  width:110px !important;
  height:165px !important;
}

body.light { background:#f5f5f5 !important; color:#1a1f3a !important; }
body.amoled { background:#000 !important; color:#fff !important; }
body.solarized { background:#002b36 !important; color:#93a1a1 !important; }
body.nord { background:#2e3440 !important; color:#d8dee9 !important; }

/* ===== GRID default sizing (comfortable) ===== */
.grid-view{
  display:grid;
  grid-template-columns:repeat(auto-fill,minmax(180px,1fr));
  gap:16px;
}

.grid-item .poster-container{
  width:100%;
  aspect-ratio:2/3;
  border-radius:14px;
  overflow:hidden;
  background:rgba(255,255,255,0.06);
}

.grid-item .poster-container img.poster{
  width:100%;
  height:100%;
  object-fit:cover;
  display:block;
}

/* Make titles under posters more readable */
.grid-item-title{
  margin-top:10px;
  font-weight:800;
  font-size:14px;
  line-height:1.25;
  color:rgba(255,255,255,0.95);
  text-shadow:0 1px 2px rgba(0,0,0,0.55);
  overflow:hidden;
  display:-webkit-box;
  -webkit-line-clamp:2;
  -webkit-box-orient:vertical;
}

.grid-item-info{
  color:rgba(255,255,255,0.70);
  font-size:12px;
  margin-top:6px;
}

/* ===== COMPACT ===== */
html.layout-compact .grid-view{
  grid-template-columns:repeat(auto-fill,minmax(150px,1fr));
  gap:12px;
}

html.layout-compact .grid-item-title{
  font-size:13px;
}

/* ===== SPACIOUS (bigger posters + bigger titles) ===== */
html.layout-spacious .grid-view{
  grid-template-columns:repeat(auto-fill,minmax(220px,1fr));
  gap:20px;
}

html.layout-spacious .grid-item-title{
  font-size:16px;
}

html.layout-spacious .grid-item-info{
  font-size:13px;
}

/* ===== LIST poster sizing ===== */
.movie-item .poster-container,
.show-group .poster-container{
  width:110px;
  min-width:110px;
}

.movie-item .poster,
.show-group .poster{
  width:110px;
  height:165px;
  object-fit:cover;
}

html.layout-compact .movie-item .poster-container,
html.layout-compact .show-group .poster-container{
  width:90px;
  min-width:90px;
}

html.layout-compact .movie-item .poster,
html.layout-compact .show-group .poster{
  width:90px;
  height:135px;
}

html.layout-spacious .movie-item .poster-container,
html.layout-spacious .show-group .poster-container{
  width:150px;
  min-width:150px;
}

html.layout-spacious .movie-item .poster,
html.layout-spacious .show-group .poster{
  width:150px;
  height:225px;
}

/* ================================
   GRID LAYOUT: Posters MUST fill
   (Fix compact/spacious half posters)
   ================================ */

/* Make each grid card a vertical flex layout */
.grid-item{
  display: flex;
  flex-direction: column;
}

/* Poster area should take consistent space */
.grid-item .poster-container{
  width: 100%;
  aspect-ratio: 2 / 3;
  min-height: 240px;           /* desktop safety */
  border-radius: 18px;
  overflow: hidden;
  position: relative;
  flex: 0 0 auto;              /* don't collapse */
}

/* FORCE the poster image to fill the container */
.grid-item .poster-container img.poster{
  width: 100% !important;
  height: 100% !important;
  display: block !important;
  object-fit: cover !important;
  object-position: center !important;
}

/* If you use placeholders, make them fill too */
.grid-item .poster-container .poster-placeholder{
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
}

/* Grid column sizing per layout */
.grid-view{
  display: grid;
  gap: 16px;
  grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); /* Comfortable */
}

html.layout-compact .grid-view{
  grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
}

html.layout-spacious .grid-view{
  grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
}

/* Mobile adjustments */
@media (max-width: 480px){
  .grid-item .poster-container{ min-height: 200px; }
  html.layout-spacious .grid-view{ grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); }
}

/* ================================
   Make names under posters clearer
   ================================ */
.grid-item-title{
  font-weight: 800;
  font-size: 1.05rem;
  letter-spacing: 0.2px;
  color: rgba(255,255,255,0.92);
  text-shadow: 0 2px 10px rgba(0,0,0,0.55);
  margin-top: 10px;
}

.grid-item-info{
  color: rgba(255,255,255,0.70);
  font-weight: 600;
}

.layout-row{
  display:flex;
  align-items:center;
  gap:10px;
  margin:10px 0 0 0;      /* spacing below grid controls */
  padding:10px 12px;
  border-radius:12px;
  background: rgba(20,25,45,0.35);
  border: 1px solid rgba(255,255,255,0.12);
  width: fit-content;
}

.layout-label{
  font-weight:700;
  font-size:12px;
  opacity:0.85;
}

.layout-select{
  padding:8px 12px;
  border-radius:10px;
  border:1px solid rgba(255,255,255,0.18);
  background: rgba(20,25,45,0.6);
  color: inherit;
  font-size: 13px;
}

/* Mobile: full width */
@media (max-width: 520px){
  .layout-row{ width:100%; }
  .layout-select{ flex:1; }
}

@media (max-width: 768px) {
  /* GRID: make the 3 modes clearly different on phones */
  html.layout-compact .grid-view {
    grid-template-columns: repeat(auto-fill, minmax(125px, 1fr)) !important;
    gap: 10px !important;
  }
  html:not(.layout-compact):not(.layout-spacious) .grid-view {
    /* Comfortable */
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)) !important;
    gap: 14px !important;
  }
  html.layout-spacious .grid-view {
    grid-template-columns: repeat(auto-fill, minmax(175px, 1fr)) !important;
    gap: 18px !important;
  }

  /* LIST: make padding/poster sizes differ so it’s obvious */
  html.layout-compact .movie-item,
  html.layout-compact .show-group {
    padding: 10px !important;
    gap: 10px !important;
  }
  html.layout-compact .movie-item .poster,
  html.layout-compact .show-group .poster,
  html.layout-compact .movie-item .poster-placeholder,
  html.layout-compact .show-group .poster-placeholder {
    width: 80px !important;
    height: 120px !important;
  }

  html.layout-spacious .movie-item,
  html.layout-spacious .show-group {
    padding: 18px !important;
    gap: 16px !important;
  }
  html.layout-spacious .movie-item .poster,
  html.layout-spacious .show-group .poster,
  html.layout-spacious .movie-item .poster-placeholder,
  html.layout-spacious .show-group .poster-placeholder {
    width: 120px !important;
    height: 180px !important;
  }
}

/* ================= RESPONSIVE POLISH PACK =================
   Goals:
   - Fluid spacing + typography across phone/tablet/desktop
   - Touch-friendly controls and better small-screen layout
   - Overrides only (keeps existing behavior/JS intact)
   ========================================================== */

:root{
  --pagePad: clamp(10px, 2.2vw, 18px);
  --panelPad: clamp(12px, 2.0vw, 18px);
  --radius: 16px;
  --radiusLg: 20px;
}

body{
  /* Respect mobile safe areas (iOS notch), env() resolves to 0 elsewhere */
  padding: calc(var(--pagePad) + env(safe-area-inset-top))
           calc(var(--pagePad) + env(safe-area-inset-right))
           calc(var(--pagePad) + env(safe-area-inset-bottom))
           calc(var(--pagePad) + env(safe-area-inset-left)) !important;
}

.container{
  max-width: 1600px;
  padding-inline: clamp(0px, 1vw, 10px);
}

.header{
  padding: clamp(16px, 2.4vw, 26px) !important;
  border-radius: var(--radiusLg) !important;
}

.header h1{
  font-size: clamp(1.35rem, 2.4vw, 2.1rem) !important;
}

.logo{
  font-size: clamp(1.8rem, 3.2vw, 2.2rem) !important;
}

/* Tabs: allow horizontal scroll on small screens instead of awkward wrapping */
.tabs{
  overflow-x: auto;
  -webkit-overflow-scrolling: touch;
  scroll-snap-type: x mandatory;
  scrollbar-width: none;
}
.tabs::-webkit-scrollbar{ display:none; }
.tab-btn{
  scroll-snap-align: start;
  white-space: nowrap;
}

/* Touch targets: make tap interactions reliable */
button, .btn, select, input{
  touch-action: manipulation;
}
.btn, button, .tab-btn, select, input{
  min-height: 42px;
}

:focus-visible{
  outline: 2px solid var(--borderSoft);
  outline-offset: 2px;
}

/* Stats grid: keep readable on phones */
@media (max-width: 900px){
  .stats{ grid-template-columns: repeat(3, minmax(0, 1fr)) !important; }
}
@media (max-width: 640px){
  .tabs{ flex-wrap: nowrap !important; }
  .stats{ grid-template-columns: repeat(2, minmax(0, 1fr)) !important; }
}
@media (max-width: 420px){
  .stats{ grid-template-columns: 1fr !important; }
}

/* Header toolbar: turn into a clean grid on phones */
@media (max-width: 900px){
  .header-toolbar{ width:100%; justify-content:flex-start; }
}
@media (max-width: 640px){
  .header-toolbar{
    display:grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 10px;
    justify-content: stretch;
    width: 100%;
  }
  .zoom-control{
    grid-column: 1 / -1;
    height: auto;
  }
  .auto-refresh{
    grid-column: 1 / -1;
    justify-content: space-between;
    width: 100%;
  }
  .header-toolbar .btn{
    width: 100%;
    min-width: 0;
    height: 52px;
  }
}

/* Filters: make inputs wrap cleanly on smaller screens */
.filters, .advanced-filters{
  padding: var(--panelPad) !important;
}
.search-box{
  min-width: min(320px, 100%) !important;
}

/* Pending TMDB totals: make it discoverable + usable */
.tmdb-pending-total{
  cursor:pointer;
  text-decoration: underline dotted rgba(255,255,255,0.35);
  text-underline-offset: 3px;
  -webkit-touch-callout: none;
}

.tmdb-pending-total:hover{
  text-decoration-color: rgba(255,255,255,0.55);
}

/* List view: stack poster/content on phones to avoid cramped rows */
@media (max-width: 640px){
  .movie-item, .show-group{
    flex-direction: column;
    align-items: stretch;
  }
  .poster-container{
    display:flex;
    justify-content:center;
  }
  .movie-item .poster,
  .show-group .poster,
  .movie-item .poster-placeholder,
  .show-group .poster-placeholder{
    width: min(220px, 100%) !important;
    height: auto !important;
    aspect-ratio: 2 / 3;
  }
}

/* Grid view: predictable columns across device sizes */
@media (max-width: 640px){
  .grid-view{
    grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
    gap: 12px !important;
  }
  html.layout-compact .grid-view,
  html.layout-spacious .grid-view{
    grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
  }
  .grid-item .poster,
  .grid-item .poster-placeholder{
    height: 180px !important;
  }
}
@media (max-width: 420px){
  .grid-view{ grid-template-columns: 1fr !important; }
  .grid-item .poster,
  .grid-item .poster-placeholder{
    height: 220px !important;
  }
}

/* Chart panels: full width on phones */
@media (max-width: 640px){
  .chart-container{
    max-width: 100% !important;
    padding: 16px !important;
  }
  #genreChart{
    max-width: 240px !important;
    max-height: 240px !important;
  }
}

/* === FIX 1: force Layout dropdown row below the view-toggle buttons === */
#layoutRow {
  flex-basis: 100% !important;
  width: 100% !important;
  margin-top: 6px !important;
  margin-left: auto !important;
  justify-content: flex-end !important;
  padding: 0 !important;
  background: transparent !important;
  border: none !important;
}
#layoutRow .layout-select{
  width: auto;
  min-width: 160px;
}

/* === FIX 3: mobile layout mode must visibly change (compact/comfortable/spacious) === */
@media (max-width: 768px) {
  /* GRID: make the 3 modes clearly different on phones */
  html.layout-compact .grid-view {
    grid-template-columns: repeat(auto-fill, minmax(125px, 1fr)) !important;
    gap: 10px !important;
  }

  /* Comfortable (default) */
  html:not(.layout-compact):not(.layout-spacious) .grid-view {
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)) !important;
    gap: 14px !important;
  }

  html.layout-spacious .grid-view {
    grid-template-columns: repeat(auto-fill, minmax(175px, 1fr)) !important;
    gap: 18px !important;
  }

  /* LIST: make padding/poster sizes differ so it’s obvious */
  html.layout-compact .movie-item,
  html.layout-compact .show-group {
    padding: 10px !important;
    gap: 10px !important;
  }

  html.layout-compact .movie-item .poster,
  html.layout-compact .show-group .poster,
  html.layout-compact .movie-item .poster-placeholder,
  html.layout-compact .show-group .poster-placeholder {
    width: 80px !important;
    height: 120px !important;
  }

  html.layout-spacious .movie-item,
  html.layout-spacious .show-group {
    padding: 18px !important;
    gap: 16px !important;
  }

  html.layout-spacious .movie-item .poster,
  html.layout-spacious .show-group .poster,
  html.layout-spacious .movie-item .poster-placeholder,
  html.layout-spacious .show-group .poster-placeholder {
    width: 120px !important;
    height: 180px !important;
  }
}
</style></head>
<body><div class="container">
<div class="header">
  <div class="header-title">
    <div class="logo">🎬</div>
    <h1>Jellyfin Watch Tracker</h1>
  </div>
    <div class="header-toolbar">
    <div class="zoom-control">
      <label>Zoom</label>
      <input type="range" class="zoom-slider" id="zoomSlider" min="70" max="150" value="100" step="5">
      <span class="zoom-value" id="zoomValue">100%</span>
    </div>
    <button class="btn manual" onclick="openManualEntry()">➕ Add Watch</button>
    <button class="btn" onclick="location.reload()">🔄 Refresh</button>
    <button class="btn secondary" id="themeToggleBtn" onclick="toggleTheme()">🌙 Theme</button>
    <button class="btn secondary" id="perfModeBtn" onclick="togglePerformanceMode()">⚡ Performance</button>
    <!-- Single Export button with inline options for JSON and CSV.  When the
         Export button is clicked, two small buttons appear side by side
         allowing the user to choose the format. -->
    <div id="exportWrapper" style="display:flex;align-items:center;gap:8px;">
      <button class="btn" id="exportBtn" onclick="toggleExportOptions()">📤 Export</button>
      <div id="exportOptions" style="display:none;gap:6px;">
        <button class="btn small" onclick="exportData('json');toggleExportOptions();">📥 JSON</button>
        <button class="btn small" onclick="exportData('csv');toggleExportOptions();">📊 CSV</button>
      </div>
    </div>

      <!-- Undo button -->
      <button class="btn secondary" id="undoBtn" onclick="undoAction()">↩️ Undo</button>

      <!-- Auto-refresh controls -->
      <div class="auto-refresh">
        <label style="display:flex;align-items:center;gap:4px;">
          <input type="checkbox" id="autoRefreshToggle" onchange="toggleAutoRefresh()"> Auto-Refresh
        </label>
        <select id="autoRefreshInterval" onchange="updateAutoRefreshInterval()" style="padding:4px 6px;border-radius:6px;border:1px solid rgba(255,255,255,0.2);background:rgba(20,25,45,0.6);color:inherit;font-size:12px;">
          <option value="1">1m</option>
          <option value="5" selected>5m</option>
          <option value="10">10m</option>
          <option value="15">15m</option>
          <option value="30">30m</option>
        </select>
        <span id="refreshCountdown"></span>
      </div>

      <!-- Theme selector removed: theme is toggled via the Theme button -->

  </div>
</div>

<div class="tabs">
<button class="tab-btn active" onclick="switchTab(event,'history')">📚 Library</button>
<button class="tab-btn" onclick="switchTab(event,'genres')">🎭 Genres</button>
<button class="tab-btn" onclick="switchTab(event,'analytics')">📊 Insights</button>
<button class="tab-btn" onclick="switchTab(event,'progress')">📈 Progress</button>
<button class="tab-btn" onclick="switchTab(event,'settings')">⚙ Settings</button>
</div>

<div id="bulk-bar" class="bulk-bar">
  <div class="bulk-info"><span id="bulk-count">0</span> selected</div>
  <div class="bulk-actions">
    <button class="btn small" onclick="selectAll()">✓ Select All</button>
    <button class="btn small secondary" onclick="clearSelection()">✕ Clear</button>
    <button class="btn small danger" onclick="bulkDelete()">🗑️ Delete Selected</button>
  </div>
</div>

<div id="history-tab" class="tab-content active">
<div class="stats">
<div class="stat-card"><h3>Total Watches</h3><div class="value" id="total">0</div></div>
<div class="stat-card"><h3>Movies</h3><div class="value" id="movies">0</div></div>
<div class="stat-card"><h3>TV Shows</h3><div class="value" id="shows">0</div></div>
<div class="stat-card"><h3>This Week</h3><div class="value" id="week">0</div></div>
<div class="stat-card"><h3>Watch Time</h3><div class="value" id="hours">0h</div></div>
<div class="stat-card"><h3>Daily Avg</h3><div class="value" id="avg">0</div></div>
</div>

<div class="filters">
  <button class="filter-btn active" onclick="setFilter('all', event)">All</button>
  <button class="filter-btn" onclick="setFilter('movies', event)">Movies</button>
  <button class="filter-btn" onclick="setFilter('shows', event)">TV Shows</button>
  <button class="filter-btn" onclick="setFilter('incomplete', event)">Incomplete</button>

  <select id="sort" onchange="renderHistory()">
    <option value="recent">Most Recent</option>
    <option value="oldest">Oldest First</option>
    <option value="mostwatched">Most Watched</option>
    <option value="alphabetical">A-Z</option>
    <option value="incompletefirst">Incomplete First</option>
  </select>

  <div class="search-box">
    <input type="text" id="search" placeholder="Search..." oninput="renderHistory()">
  </div>

  <div class="view-toggle">
    <button class="active" onclick="setView('list', event)">List</button>
    <button onclick="setView('grid', event)">Grid</button>

    <button id="gridSelectBtn" onclick="toggleGridSelectMode(event)" style="display:none;">Select</button>
    <button id="gridSelectAllBtn" onclick="gridSelectAll(event)" style="display:none;">All</button>
    <button id="gridClearBtn" onclick="gridClearSelection(event)" style="display:none;">Clear</button>
  </div>

  <!-- Layout control row below Grid Select -->
  <div id="layoutRow" class="layout-row">
    <label for="layoutSelect" class="layout-label">Layout</label>
    <select id="layoutSelect" onchange="changeLayout(this.value)" class="layout-select">
      <option value="comfortable">Comfortable</option>
      <option value="compact">Compact</option>
      <option value="spacious">Spacious</option>
    </select>
  </div>
</div>

<!-- Advanced search and filter controls -->
<div class="advanced-filters">
  <select id="genreFilter" onchange="applyFilters()">
    <option value="">All Genres</option>
  </select>

  <input type="number" id="yearFrom" placeholder="From Year" onchange="applyFilters()">
  <input type="number" id="yearTo" placeholder="To Year" onchange="applyFilters()">

  <select id="statusFilter" onchange="applyFilters()">
    <option value="">All Statuses</option>
    <option value="completed">Completed</option>
    <option value="inprogress">In Progress</option>
  </select>

  <select id="typeFilter" onchange="applyFilters()">
    <option value="all">All Types</option>
    <option value="movie">Movies Only</option>
    <option value="show">Shows Only</option>
  </select>

  <button onclick="clearFilters()">Clear</button>

  <span id="resultsCounter">Found 0 movies, 0 shows</span>
</div>


<div id="grid-bulk-bar" class="bulk-bar">
  <div class="bulk-info"><span id="grid-bulk-count">0</span> selected</div>
  <div id="grid-bulk-status" style="font-size:12px;opacity:0.75;margin-top:4px;display:none"></div>
  <div class="bulk-actions">
    <button class="btn small" onclick="gridSelectAll()">✓ Select All</button>
    <button class="btn small secondary" onclick="gridClearSelection()">✕ Clear</button>

    <button class="btn small success" onclick="gridBulkMarkComplete()">✓ Mark Completed</button>

    <button class="btn small danger" onclick="gridBulkDelete()">🗑️ Delete Selected</button>
  </div>
</div>

<div class="content" id="content"><div class="loading">Loading...</div></div>
</div>

<div id="genres-tab" class="tab-content">
<h2 style="margin-bottom:25px;font-size:1.8em;font-weight:700">🎭 Genre Analysis</h2>
<div id="genres-content"><div class="loading">Loading genres...</div></div>
</div>

<div id="analytics-tab" class="tab-content">
<h2 style="margin-bottom:25px;font-size:1.8em;font-weight:700">📊 Watch Insights</h2>

<div class="quick-stats-grid" id="quick-stats"></div>

<div style="background:rgba(20,25,45,0.8);backdrop-filter:blur(10px);border-radius:16px;padding:20px;margin-bottom:25px;border:1px solid rgba(255,255,255,0.08)">
<h3 style="font-size:1.4em;margin-bottom:15px;color:#8b9cff">🔥 Trending This Week</h3>
<div id="trending-content"></div>
</div>

<!-- Watch Time Analytics section -->
<div style="background:rgba(20,25,45,0.8);backdrop-filter:blur(10px);border-radius:16px;padding:20px;margin-bottom:25px;border:1px solid rgba(255,255,255,0.08)">
  <h3 style="font-size:1.4em;margin-bottom:15px;color:#8b9cff">⏱️ Watch Time Analytics</h3>
  <div style="display:flex;flex-wrap:wrap;gap:20px">
    <div style="flex:1;min-width:240px">
      <p>Total: <span id="wt-total">0h</span></p>
      <p>Movies: <span id="wt-movies">0h</span></p>
      <p>Shows: <span id="wt-shows">0h</span></p>
      <p>Daily Avg: <span id="wt-daily">0h</span></p>
      <p>Binge Sessions: <span id="wt-binge">0</span></p>
    </div>
    <div style="flex:2;min-width:300px">
      <canvas id="watchTimeChart" height="200"></canvas>
    </div>
  </div>
</div>
</div>

<div id="progress-tab" class="tab-content">
<h2 style="margin-bottom:25px;font-size:1.8em;font-weight:700">📈 Progress Tracker</h2>

<div style="background:rgba(20,25,45,0.8);backdrop-filter:blur(10px);border-radius:16px;padding:20px;margin-bottom:25px;border:1px solid rgba(255,255,255,0.08)">
  <div style="display:flex;justify-content:space-between;align-items:center;cursor:pointer" onclick="toggleSection('movies-progress-list','movies-progress-arrow')">
    <h3 style="font-size:1.4em;margin:0;color:#8b9cff">🎬 Movies Progress</h3>
    <span id="movies-progress-arrow" style="transition:transform 0.3s;display:inline-block">&#9654;</span>
  </div>
  <div id="movies-progress-list" style="display:none;margin-top:15px">
    <div id="movies-progress-controls" style="display:flex;flex-wrap:wrap;gap:10px;margin-bottom:10px">
      <input type="text" id="moviesSearchInput" placeholder="Search movies..." oninput="setMovieProgressSearch(this.value)" style="flex:1;min-width:150px;background:rgba(255,255,255,0.1);border:none;color:inherit;padding:6px 8px;border-radius:8px">
      <select id="moviesSortSelect" onchange="setMovieProgressSort(this.value)" style="background:rgba(255,255,255,0.1);border:none;color:inherit;padding:6px 8px;border-radius:8px">
        <option value="lastWatched">Recent</option>
        <option value="alphabetical">A-Z</option>
        <option value="watchCount">Watch Count</option>
      </select>
    </div>
    <div id="movies-list-container"></div>
  </div>
</div>

<div style="background:rgba(20,25,45,0.8);backdrop-filter:blur(10px);border-radius:16px;padding:20px;border:1px solid rgba(255,255,255,0.08)">
  <div style="display:flex;justify-content:space-between;align-items:center;cursor:pointer" onclick="toggleSection('shows-progress-list','shows-progress-arrow')">
    <h3 style="font-size:1.4em;margin:0;color:#6bcf7f">📺 Shows Progress</h3>
    <span id="shows-progress-arrow" style="transition:transform 0.3s;display:inline-block">&#9654;</span>
  </div>
  <div id="shows-progress-list" style="display:none;margin-top:15px">
    <div id="shows-progress-controls" style="display:flex;flex-wrap:wrap;gap:10px;margin-bottom:10px">
      <input type="text" id="showsSearchInput" placeholder="Search shows..." oninput="setShowProgressSearch(this.value)" style="flex:1;min-width:150px;background:rgba(255,255,255,0.1);border:none;color:inherit;padding:6px 8px;border-radius:8px">
      <select id="showsSortSelect" onchange="setShowProgressSort(this.value)" style="background:rgba(255,255,255,0.1);border:none;color:inherit;padding:6px 8px;border-radius:8px">
        <option value="lastWatched">Recent</option>
        <option value="alphabetical">A-Z</option>
        <option value="completionDesc">Completion % (desc)</option>
        <option value="episodesLeftAsc">Episodes Left (asc)</option>
      </select>
      <select id="showsFilterSelect" onchange="setShowProgressFilter(this.value)" style="background:rgba(255,255,255,0.1);border:none;color:inherit;padding:6px 8px;border-radius:8px">
        <option value="all">All</option>
        <option value="inprogress">In Progress</option>
        <option value="complete">Completed</option>
      </select>
    </div>
    <div id="shows-list-container"></div>
  </div>
</div>
</div>

<div id="settings-tab" class="tab-content">
<h2 style="margin-bottom:25px;font-size:1.8em;font-weight:700">⚙ Settings</h2>
<h3 style="margin:0 0 15px">📥 Import</h3>

<div class="import-section">
<h3>🐟 Jellyfin</h3>
<div class="import-grid">
<div class="import-card">
<h4>Import Watch History</h4>
<p>Import all watched movies and episodes from your Jellyfin server.</p>
<button class="btn" id="jellyfinImportBtn" onclick="importJellyfin()">Import from Jellyfin</button>
</div>
</div>
</div>

<div class="import-section">
<h3>📺 Sonarr & Radarr</h3>
<div class="import-grid">
<div class="import-card">
<h4>Sonarr TV Shows</h4>
<p>Import all downloaded TV shows from Sonarr. Great for marking old shows as watched!</p>
<button class="btn manual" id="sonarrImportBtn" onclick="importSonarr()">Import from Sonarr</button>
</div>
<div class="import-card">
<h4>Radarr Movies</h4>
<p>Import all downloaded movies from Radarr.</p>
<button class="btn manual" id="radarrImportBtn" onclick="importRadarr()">Import from Radarr</button>
</div>
</div>
<p style="color:rgba(255,255,255,0.6);font-size:0.9em;margin-top:15px">💡 Set SONARR_URL, SONARR_API_KEY, RADARR_URL, RADARR_API_KEY environment variables</p>
</div>

<div class="import-section">
<h3>🧹 Maintenance</h3>
<div class="import-grid">
<div class="import-card">
<h4>Clear TMDB Cache</h4>
<p>Clear cached TMDB data to force refresh. Useful if show lookups are failing.</p>
<button class="btn danger" onclick="clearTMDBCache()">Clear TMDB Cache</button>
</div>
<div class="import-card">
<h4>Clear Watch History</h4>
<p>Remove all watch records and reset charts.</p>
<button class="btn danger" onclick="clearWatchHistory()">Clear Watch History</button>
</div>
<div class="import-card">
<h4>Clear Insights</h4>
<p>Reset genre insights and recommendations.</p>
<button class="btn danger" onclick="clearInsights()">Clear Insights</button>
</div>
<div class="import-card">
<h4>Clear Progress</h4>
<p>Reset manual completions and progress tracking.</p>
<button class="btn danger" onclick="clearProgress()">Clear Progress</button>
</div>
</div>
</div>
</div>

</div>

<div id="posterModal" class="modal">
<div class="modal-content">
<span class="close" onclick="closePosterModal()">&times;</span>
<h2 style="margin-bottom:15px;font-size:1.6em">Upload Custom Poster</h2>
<p id="posterItemName" style="color:rgba(255,255,255,0.7);margin-bottom:12px"></p>
<input type="file" id="posterFile" accept="image/*">
<button class="btn" style="width:100%;margin-top:15px" onclick="uploadPoster()">Upload Poster</button>
</div>
</div>

<div id="manualEntryModal" class="modal">
<div class="modal-content">
<span class="close" onclick="closeManualEntry()">&times;</span>
<h2 style="margin-bottom:20px;font-size:1.6em">➕ Add Manual Watch Entry</h2>

<div class="form-group">
<label>Content Type</label>
<div class="radio-group">
<label><input type="radio" name="entryType" value="Movie" checked onchange="toggleEpisodeFields()"> 🎬 Movie</label>
<label><input type="radio" name="entryType" value="Episode" onchange="toggleEpisodeFields()"> 📺 Show</label>
</div>
</div>

<div id="movieFields" class="movie-fields">
<div class="form-group">
<label>Movie Title</label>
<input type="text" id="entryMovieTitle" placeholder="e.g., The Dark Knight">
</div>
</div>

<div id="episodeFields" class="episode-fields">
<div class="form-group">
<label>Series Name</label>
<input type="text" id="entrySeriesName" placeholder="e.g., Breaking Bad">
</div>

<div class="form-group">
<label>Season Number</label>
<input type="number" id="entrySeason" placeholder="e.g., 1" min="0">
</div>

<div class="form-group">
<label>Episode Number / Range (optional)</label>
<input type="text" id="entryEpisodeSpec" placeholder="e.g., 1  or  1,2,5-8">
</div>

<div class="form-group">
<label>Episode Name (optional)</label>
<input type="text" id="entryEpisodeName" placeholder="Optional (TMDB will be used when available)">
</div>
</div>

<div class="form-group">
<label>Year (optional)</label>
<input type="number" id="entryYear" placeholder="e.g., 2008" min="1900" max="2026">
</div>

<div class="form-group">
<label>Genres (optional, comma-separated)</label>
<input type="text" id="entryGenres" placeholder="e.g., Drama, Thriller">
</div>

<div class="form-group">
<label>Watch Date</label>
<input type="date" id="entryDate">
</div>

<button class="btn" id="manualEntrySubmitBtn" style="width:100%;margin-top:15px" onclick="submitManualEntry()">✓ Add Watch Record</button>
</div>
</div>

<div id="addEpisodeModal" class="modal">
<div class="modal-content">
<span class="close" onclick="closeAddEpisode()">&times;</span>
<h2 style="margin-bottom:20px;font-size:1.6em" id="addEpisodeTitle">➕ Add Episode</h2>

<div class="form-group">
<label>Series Name</label>
<input type="text" id="addEpSeriesName" readonly style="opacity:0.7">
</div>

<div class="form-group">
<label>Season Number</label>
<input type="number" id="addEpSeason" placeholder="Season number" min="1">
</div>

<div class="form-group">
<label>Episode Number</label>
<input type="number" id="addEpEpisode" placeholder="e.g., 1" min="1" step="1">
</div>

<div class="form-group checkbox-group">
<input type="checkbox" id="skipTMDB">
<label for="skipTMDB">Skip TMDB (use "Episode X" as name)</label>
</div>

<div class="form-group">
<label>Watch Date</label>
<input type="date" id="addEpDate">
</div>

<button class="btn" style="width:100%;margin-top:15px" onclick="submitAddEpisode()">✓ Add Episode</button>
</div>
</div>

<div id="addSeasonModal" class="modal">
<div class="modal-content">
<span class="close" onclick="closeAddSeason()">&times;</span>
<h2 style="margin-bottom:20px;font-size:1.6em" id="addSeasonTitle">➕ Add Entire Season</h2>

<div class="form-group">
<label>Series Name</label>
<input type="text" id="addSeasonSeriesName" readonly style="opacity:0.7">
</div>

<div class="form-group">
<label>Season Number</label>
<input type="number" id="addSeasonNumber" placeholder="Season number" min="1">
</div>

<div class="form-group">
<label>Number of Episodes (if TMDB fails)</label>
<input type="number" id="addSeasonEpCount" placeholder="e.g., 10" min="1">
</div>

<div class="form-group checkbox-group">
<input type="checkbox" id="skipTMDBSeason">
<label for="skipTMDBSeason">Skip TMDB (use manual episode count)</label>
</div>

<div class="form-group">
<label>Watch Date (for all episodes)</label>
<input type="date" id="addSeasonDate">
</div>

<button class="btn" style="width:100%;margin-top:15px" onclick="submitAddSeason()">✓ Add All Episodes</button>
</div>
</div>

<div id="settingsModal" class="modal">
<div class="modal-content">
<span class="close" onclick="closeSettings()">&times;</span>
<h2 style="margin-bottom:20px;font-size:1.6em">⚙️ Settings</h2>

<div class="import-section" style="margin:0">
<h3>Environment Variables</h3>
<p style="color:rgba(255,255,255,0.7);margin-bottom:15px">Configure these in your Docker/system environment:</p>

<div style="background:rgba(255,255,255,0.05);padding:15px;border-radius:12px;font-family:monospace;font-size:0.85em">
<div style="margin-bottom:10px"><strong>JELLYFIN_URL</strong>: Jellyfin server URL</div>
<div style="margin-bottom:10px"><strong>JELLYFIN_API_KEY</strong>: Jellyfin API key</div>
<div style="margin-bottom:10px"><strong>TMDB_API_KEY</strong>: TMDB API key (get from themoviedb.org)</div>
<div style="margin-bottom:10px"><strong>SONARR_URL</strong>: Sonarr server URL (optional)</div>
<div style="margin-bottom:10px"><strong>SONARR_API_KEY</strong>: Sonarr API key (optional)</div>
<div style="margin-bottom:10px"><strong>RADARR_URL</strong>: Radarr server URL (optional)</div>
<div><strong>RADARR_API_KEY</strong>: Radarr API key (optional)</div>
</div>
</div>

<button class="btn" style="width:100%;margin-top:20px" onclick="closeSettings()">Close</button>
</div>
</div>

<script>
let data={};let filter='all';let genreChart=null;let currentPosterItem=null;let viewMode_movies = "list";let viewMode_shows = "list";let viewMode_all = "list";let viewMode_incomplete = "list";let currentManageShow=null;let selectedEpisodes=new Set();let genresRenderedSig=null;
let historySig = null;
const HISTORY_FETCH_TIMEOUT_MS = 120000;
const RECS_FETCH_TIMEOUT_MS = 25000;
const INSIGHTS_FETCH_TIMEOUT_MS = 20000;
let sigPollTimer = null;
let sigPollBusy = false;
let sigPollDelay = 1500;
let sigPollFailures = 0;
let quickReloadBusy = false;
let eventsSource = null;
let realtimeConnected = false;
let lastRealtimeSeq = 0;
let realtimeFailures = 0;
let loadRetryTimer = null;
let loadRetryDelayMs = 1200;
let reloadFailureCount = 0;
let genreRenderToken = 0;
let recommendationsRequestToken = 0;
// Compatibility fallback for older cached UI fragments that reference `insights`.
let insights = { success: false, combos: [], moods: {} };
// Chart instance for watch time analytics
let watchTimeChart = null;
let uiTheme = localStorage.getItem('theme') || localStorage.getItem('uiTheme') || 'dark';
let openPanels = new Set();
let scrollPosition = 0;
let gridSelectMode = false;
let selectedGridItems = new Set(); // key format: movie|<name>|<year>  or  show|<series>
let performanceMode = false;  // ADD THIS LINE
let currentPage = 1;  // ADD THIS LINE
let enableLoadMoreInQuality = true; // set false if you ever want to disable it
const itemsPerPage = 50;
// Offsets for fetching new recommendation batches.  Separate counters are
// maintained for movies and shows so that the user can request more
// recommendations for each category independently.  The API uses a
// single numeric offset parameter to fetch different recommendation
// batches, so when only one category is refreshed we still update
// that category without affecting the other.
let recommendationOffset = 0; // legacy offset (unused but kept for compatibility)
let recommendationOffsetMovies = 0;
let recommendationOffsetShows = 0;

// Hold the currently displayed lists of movie and show recommendations.  These
// arrays are updated independently when the user requests more movies or shows.
let currentMoviesRecs = [];
let currentShowsRecs = [];
let lastRecsTopGenre = null;
let lastRecsError = null;
let lastRecsLoadingSection = null; // 'all' | 'movies' | 'shows' | null

function getCachedRecs(){
  try{
    const raw = sessionStorage.getItem('recsCache');
    if(!raw) return null;
    const obj = JSON.parse(raw);
    if(!obj || !obj.data) return null;
    // Expire after 12 hours to avoid very stale UI
    if(obj.ts && (Date.now() - obj.ts) > (12*60*60*1000)) return null;
    return obj.data;
  }catch(e){
    return null;
  }
}

function storeCachedRecs(payload){
  try{
    if(!payload || !payload.success) return;
    const wrapper = { ts: Date.now(), data: payload };
    const raw = JSON.stringify(wrapper);
    if(raw.length < 1_000_000){
      sessionStorage.setItem('recsCache', raw);
    }
  }catch(e){}
}

function scrollToRecommendations(){
  const el = document.getElementById('recommendations-container');
  if(!el) return;
  el.scrollIntoView({ behavior: 'smooth', block: 'start' });
  // account for the fixed header
  setTimeout(() => { window.scrollBy(0, -90); }, 0);
}

// Progress list search, sort and filter settings. These are persisted to
// localStorage so that the user's preferences are remembered across page
// reloads. Default values are used if nothing is saved.  The search
// strings are stored in lowercase for easier case-insensitive comparison.
let progressMoviesSearch = (localStorage.getItem('progressMoviesSearch') || '').toLowerCase();
let progressMoviesSort = localStorage.getItem('progressMoviesSort') || 'lastWatched';
let progressShowsSearch = (localStorage.getItem('progressShowsSearch') || '').toLowerCase();
let progressShowsSort = localStorage.getItem('progressShowsSort') || 'lastWatched';
let progressShowsFilter = localStorage.getItem('progressShowsFilter') || 'all';

// Pagination settings for the progress lists.  To keep the progress
// section manageable, movies and shows are paginated client‑side.
// The page size constants define how many items are shown initially
// and how many additional items are loaded each time "Load More" is
// clicked.  Separate limits are maintained for movies, in‑progress
// shows and completed shows so that each category can be loaded
// independently.
const MOVIES_PROGRESS_PAGE_SIZE = 20;
const SHOWS_PROGRESS_PAGE_SIZE = 10;
let moviesProgressLimit = MOVIES_PROGRESS_PAGE_SIZE;
let showsInProgressLimit = SHOWS_PROGRESS_PAGE_SIZE;
let showsCompletedLimit = SHOWS_PROGRESS_PAGE_SIZE;

// === Genre/Mood filter support ===
// Global variables to store active mood or genre-combo filters for recommendations.
// When either of these is non-null, the recommendations section will filter
// results accordingly.  Only one filter can be active at a time; selecting
// a mood clears the combo filter and vice versa.
let currentMoodFilter = null;
let currentGenreComboFilter = null;

// Define a mapping from mood names to arrays of genre names.  This map
// associates each mood with one or more related genres.  When a mood is
// selected, recommendations will be filtered to include items whose
// genres intersect with the corresponding array.
const moodGenreMap = {
  "intense": ["Thriller", "Crime", "Mystery"],
  "fun": ["Comedy", "Animation", "Family"],
  "emotional": ["Drama", "Romance"],
  "exciting": ["Action", "Adventure", "Sci-Fi"],
  "scary": ["Horror", "Thriller"]
};

// Track the currently active top-level tab (history, genres, analytics, progress, import).
// This allows us to detect when the user leaves the progress tab so we can
// reset any search filter applied while navigating from the progress view.
let currentTab = 'history';

function genreKeyFromName(name){
  return encodeURIComponent(String(name == null ? '' : name));
}

function genreNameFromKey(key){
  try {
    return decodeURIComponent(String(key == null ? '' : key));
  } catch(_) {
    return String(key == null ? '' : key);
  }
}

// Collapsed state flags for the shows progress categories.  These
// variables control whether the In Progress and Completed lists are
// expanded (false) or collapsed (true).  They are reset in
// resetProgressPagination() based on the current progress filter.
let showsInProgressCollapsed = false;
let showsCompletedCollapsed = false;

// Reset the pagination limits when filters, search or sort settings
// change.  This ensures the lists start from the beginning when the
// user adjusts their criteria.
function resetProgressPagination(){
  moviesProgressLimit = MOVIES_PROGRESS_PAGE_SIZE;
  showsInProgressLimit = SHOWS_PROGRESS_PAGE_SIZE;
  showsCompletedLimit = SHOWS_PROGRESS_PAGE_SIZE;

  // Reset category collapsed states based on the current show filter.  When
  // viewing only in‑progress items, collapse the completed section.  When
  // viewing only completed items, collapse the in‑progress section.  For
  // the "all" filter, expand both sections by default.  This keeps the
  // interface tidy when switching between filters.
  if(progressShowsFilter === 'inprogress'){
    showsInProgressCollapsed = false;
    showsCompletedCollapsed = true;
  } else if(progressShowsFilter === 'complete'){
    showsInProgressCollapsed = true;
    showsCompletedCollapsed = false;
  } else {
    // For the 'all' filter, expand the in-progress section and collapse
    // the completed section by default to keep the list concise.
    showsInProgressCollapsed = false;
    showsCompletedCollapsed = true;
  }
}

function toggleGridSelectMode(e){
  if(e){ e.preventDefault(); e.stopPropagation(); }
  gridSelectMode = !gridSelectMode;
  document.body.classList.toggle('grid-select-mode', gridSelectMode);

  if(!gridSelectMode){
    selectedGridItems.clear();
    updateGridBulkBar();
    renderHistory();
  }else{
    renderHistory();
  }

  const btn = document.getElementById('gridSelectBtn');
  if(btn){
    btn.classList.toggle('active', gridSelectMode);
    btn.textContent = gridSelectMode ? '✅ Selecting' : '☑️ Select';
  }
}

function showGridSelectButton(){
  const btn = document.getElementById('gridSelectBtn');
  const allBtn = document.getElementById('gridSelectAllBtn');
  const clrBtn = document.getElementById('gridClearBtn');

  // Determine which viewMode to use based on current filter
  let viewMode;
  if (filter === 'movies') {
    viewMode = viewMode_movies;
  } else if (filter === 'shows') {
    viewMode = viewMode_shows;
  } else if (filter === 'all') {
    viewMode = viewMode_all;
  } else if (filter === 'incomplete') {
    viewMode = viewMode_incomplete;
  }

  const inGrid = (viewMode === 'grid');

  if(btn) btn.style.display = inGrid ? 'inline-block' : 'none';

  const showExtra = inGrid && gridSelectMode;
  if(allBtn) allBtn.style.display = showExtra ? 'inline-block' : 'none';
  if(clrBtn) clrBtn.style.display = showExtra ? 'inline-block' : 'none';

  document.getElementById('grid-bulk-bar').style.display = 'none';
}

function gridKeyMovie(name, year){
  return `movie|${name}|${year||''}`;
}
function gridKeyShow(series){
  return `show|${series}`;
}

function toggleGridItemSelection(key){
  if(selectedGridItems.has(key)) selectedGridItems.delete(key);
  else selectedGridItems.add(key);
  updateGridBulkBar();
  // Update visuals without re-render
  const el = document.querySelector(`[data-grid-key="${CSS.escape(key)}"]`);
  if(el){
    el.classList.toggle('grid-selected', selectedGridItems.has(key));
    const cb = el.querySelector('.grid-select-checkbox');
    if(cb) cb.checked = selectedGridItems.has(key);
  }
}

function updateGridBulkBar(){
  const total = selectedGridItems.size;

  const bar = document.getElementById('grid-bulk-bar');
  const count = document.getElementById('grid-bulk-count');

  if(!bar || !count) return;

  if(total > 0){
    bar.style.display = 'flex';
    count.textContent = total;
  }else{
    bar.style.display = 'none';
  }
}

function setGridBulkStatus(msg){
  const el = document.getElementById('grid-bulk-status');
  if(!el) return;
  if(!msg){
    el.textContent = '';
    el.style.display = 'none';
  }else{
    el.textContent = msg;
    el.style.display = 'block';
  }
}

function clearGridBulkStatusSoon(ms){
  setTimeout(()=>setGridBulkStatus(''), ms || 1200);
}

function gridSelectAll(e){
  if(e){ e.preventDefault(); e.stopPropagation(); }
  selectedGridItems.clear();

  const search=(document.getElementById('search').value||'').toLowerCase();
  let movies=(data.movies||[]).filter(m=>!search||m.name.toLowerCase().includes(search));
  let shows=(data.shows||[]).filter(s=>!search||s.series_name.toLowerCase().includes(search));

  if(filter==='movies'){ shows=[]; }
  else if(filter==='shows'){ movies=[]; }
  else if(filter==='incomplete'){ movies=[]; shows=shows.filter(s=>s.completion_percentage<100&&s.has_tmdb_data); }

  for(const m of movies){
    selectedGridItems.add(gridKeyMovie(m.name, m.year));
  }
  for(const s of shows){
    selectedGridItems.add(gridKeyShow(s.series_name));
  }

  updateGridBulkBar();
  renderHistory();
}

function gridClearSelection(e){
  if(e){ e.preventDefault(); e.stopPropagation(); }
  selectedGridItems.clear();
  updateGridBulkBar();
  setGridBulkStatus('');
  renderHistory();
}

async function gridBulkDelete(){
  if(selectedGridItems.size === 0) return;
  if(!confirm(`Delete ${selectedGridItems.size} selected items?`)) return;

  saveState();

  const items = Array.from(selectedGridItems);
  try{
    for(const key of items){
      const parts = key.split('|');
      const type = parts[0];

      if(type === 'movie'){
        const name = parts[1];
        const year = parts[2] ? parseInt(parts[2]) : null;
        await fetch('/api/delete_movie',{
          method:'POST',
          headers:{'Content-Type':'application/json'},
          body:JSON.stringify({name:name, year:year})
        });
      }else if(type === 'show'){
        const series = parts.slice(1).join('|'); // safe join
        await fetch('/api/delete_show',{
          method:'POST',
          headers:{'Content-Type':'application/json'},
          body:JSON.stringify({series_name:series})
        });
      }
    }

    selectedGridItems.clear();
    quickReload();
  }catch(e){
    alert('Error: '+e.message);
  }
}

function saveState(targetKey) {
  scrollPosition = window.pageYOffset || document.documentElement.scrollTop;
  saveOpenPanels();
  localStorage.setItem('filter', filter);
  
  // Use sessionStorage for scroll - only persists for this session
  sessionStorage.setItem('scrollPosition', scrollPosition);
  sessionStorage.setItem('shouldRestore', 'true');  // Flag that we should restore
  
  // Save which show/season was in view for better restoration
  const shows = document.querySelectorAll('.show-group');
  for(const show of shows) {
    const rect = show.getBoundingClientRect();
    if(rect.top >= 0 && rect.top < window.innerHeight / 2) {
      const showId = show.querySelector('.seasons-list')?.id;
      if(showId) {
        sessionStorage.setItem('lastViewedShow', showId);
      }
      break;
    }
  }
  if(targetKey){
    sessionStorage.setItem('restoreTarget', targetKey);
  }
}

function restoreState() {
  // Only restore if this was triggered by a delete operation
  const shouldRestore = sessionStorage.getItem('shouldRestore');
  if(!shouldRestore) return;
  
  // Allow a one-time panel restore, then clear the flag immediately
  sessionStorage.setItem('restorePanelsOnce', 'true');
  sessionStorage.removeItem('shouldRestore');
  
  // Get saved scroll position
  const savedScroll = parseInt(sessionStorage.getItem('scrollPosition')) || scrollPosition;
  const lastViewedShow = sessionStorage.getItem('lastViewedShow');
  const restoreTarget = sessionStorage.getItem('restoreTarget');

  function findRestoreTarget(key){
    if(!key) return null;
    try{
      if(viewMode === 'grid'){
        return document.querySelector(`[data-grid-key="${CSS.escape(key)}"]`);
      }
    }catch(e){}

    if(key.startsWith('show|')){
      const name = key.slice('show|'.length);
      const enc = encodeURIComponent(name || '').split("'").join('%27');
      return document.querySelector(`.show-group[data-show-name="${enc}"]`);
    }
    if(key.startsWith('movie|')){
      const parts = key.split('|');
      const yearStr = parts.pop();
      const name = parts.slice(1).join('|');
      const enc = encodeURIComponent(name || '').split("'").join('%27');
      const items = Array.from(document.querySelectorAll(`.movie-item[data-movie-name="${enc}"]`));
      if(items.length === 0) return null;
      if(!yearStr) return items[0];
      return items.find(el => (el.querySelector('.movie-title')?.textContent || '').includes(`(${yearStr})`)) || items[0];
    }
    return null;
  }
  
  // Restore panels FIRST, then scroll (important order!)
  setTimeout(() => {
    // Restore all open panels
    restoreOpenPanels();
    
    // Small delay to let panels open, then scroll
    setTimeout(() => {
      let restored = false;
      if(restoreTarget){
        const targetEl = findRestoreTarget(restoreTarget);
        if(targetEl){
          targetEl.scrollIntoView({ behavior: 'instant', block: 'center' });
          window.scrollBy(0, -100);
          restored = true;
        }
      }
      if(!restored && lastViewedShow && document.getElementById(lastViewedShow)) {
        // Scroll to the specific show they were viewing
        const element = document.getElementById(lastViewedShow).closest('.show-group');
        if(element) {
          element.scrollIntoView({ behavior: 'instant', block: 'start' });
          window.scrollBy(0, -100);
        }
      } else if(!restored) {
        // Fallback to exact pixel position
        window.scrollTo({
          top: savedScroll,
          left: 0,
          behavior: 'instant'
        });
      }
      
      // Clean up after restoration
      sessionStorage.removeItem('scrollPosition');
      sessionStorage.removeItem('lastViewedShow');
      sessionStorage.removeItem('restoreTarget');
      sessionStorage.removeItem('openPanels');
    }, 100);
  }, 150);
}

// Themes available for cycling via the theme toggle button.  'dark' is the
// default base theme; other values apply a class to the <body> element.
const availableThemes = ['dark','light','amoled','solarized','nord'];

/**
 * Apply the given theme. Removes any previous theme classes and adds the
 * appropriate class if the theme is not 'dark'. Also updates the
 * persistent storage and the theme toggle button label.
 *
 * @param {string} theme The desired theme name.
 */
function applyTheme(theme){
  uiTheme = availableThemes.includes(theme) ? theme : 'dark';

  // Clear theme classes (dark is the base)
  document.body.classList.remove('light','amoled','solarized','nord');
  if(uiTheme !== 'dark') document.body.classList.add(uiTheme);

  // Save in BOTH keys (important)
  localStorage.setItem('theme', uiTheme);
  localStorage.setItem('uiTheme', uiTheme);

  // Update button label
  const btn = document.getElementById('themeToggleBtn');
  if(btn){
    let emoji = '🌙';
    if(uiTheme === 'light') emoji = '☀️';
    else if(uiTheme === 'amoled') emoji = '🖤';
    else if(uiTheme === 'solarized') emoji = '🌅';
    else if(uiTheme === 'nord') emoji = '❄️';
    btn.textContent = `${emoji} Theme`;
  }
}

function toggleTheme(){
  // If uiTheme was never set properly, recover from storage
  if(!uiTheme) uiTheme = localStorage.getItem('theme') || localStorage.getItem('uiTheme') || 'dark';

  const idx = availableThemes.indexOf(uiTheme);
  const next = availableThemes[(idx + 1) % availableThemes.length];
  applyTheme(next);
}

function togglePerformanceMode() {
  performanceMode = !performanceMode;
  localStorage.setItem('performanceMode', performanceMode);
  
  const btn = document.getElementById('perfModeBtn');
  if(btn) {
    btn.textContent = performanceMode ? '🚀 Quality' : '⚡ Performance';
    btn.classList.toggle('active', performanceMode);
  }
  
  // Apply or remove performance mode
  document.body.classList.toggle('performance-mode', performanceMode);
  
  // Reset pagination when toggling
  currentPage = 1;
  
  // Re-render
  renderHistory();
  
  // Show notification
  const mode = performanceMode ? 'Performance Mode ON - Faster scrolling!' : 'Quality Mode ON - Beautiful effects!';
  console.log(mode);
}

function saveOpenPanels() {
  openPanels.clear();
  document.querySelectorAll('.seasons-list, .episodes-list').forEach(el => {
    if(el.style.display === 'block') {
      openPanels.add(el.id);
    }
  });
  // Note: We intentionally do not include the progress lists (movies-progress-list and
  // shows-progress-list) here. These collapsible sections are toggled independently
  // and do not need to persist their state via openPanels.
  
  // Also save to sessionStorage for reliability
  sessionStorage.setItem('openPanels', JSON.stringify(Array.from(openPanels)));
}

function restoreOpenPanels() {
  // Only restore panels when explicitly requested (e.g. after delete/undo)
  const restoreOnce = sessionStorage.getItem('restorePanelsOnce');
  if(!restoreOnce) return;
  sessionStorage.removeItem('restorePanelsOnce');
  // Load from sessionStorage
  const savedPanels = sessionStorage.getItem('openPanels');
  if(savedPanels) {
    try {
      const panelIds = JSON.parse(savedPanels);
      panelIds.forEach(id => {
        openPanels.add(id);
        const el = document.getElementById(id);
        if(el) {
          // restore display for standard panels
          el.style.display = 'block';
        }
      });
      // Note: We no longer restore progress lists here. Only seasons/episode
      // panels are persisted. The progress lists are toggled independently and
      // start collapsed on each page load.
    } catch(e) {
      console.error('Failed to restore panels:', e);
    }
  }
}

function applyStatsFromData(){
  if(!data || !data.stats) return;
  document.getElementById('total').textContent = data.stats.total_watches || 0;
  document.getElementById('movies').textContent = data.stats.unique_movies || 0;
  const showCount = (data.stats.tv_shows != null) ? data.stats.tv_shows : (data.stats.unique_shows || 0);
  document.getElementById('shows').textContent = showCount;
  document.getElementById('week').textContent = data.stats.this_week || 0;
  document.getElementById('hours').textContent = formatDurationHours(data.stats.total_hours || 0);
  document.getElementById('avg').textContent = data.stats.avg_per_day || 0;
}

function renderAllFromData(){
  if(!data) return;
  applyStatsFromData();
  renderHistory();
  renderAnalytics();
  // Reset progress pagination and collapsed states on initial load to
  // ensure the correct sections expand/collapse according to the
  // saved filter (inprogress/complete/all) before rendering.
  resetProgressPagination();
  renderProgress();
  try { populateGenreFilter(); } catch(e) { console.error(e); }
  showGridSelectButton();
  updateGridBulkBar();
  // Restore state after a longer delay to ensure DOM is ready
  setTimeout(() => {
    try { restoreState(); } catch(e) {}
  }, 200);
}

function getCachedHistory(){
  try {
    const raw = sessionStorage.getItem('historyCache');
    if(!raw) return null;
    const obj = JSON.parse(raw);
    if(!obj || !obj.data) return null;
    return obj;
  } catch(e) {
    return null;
  }
}

function storeCachedHistory(payload){
  try {
    if(!payload) return;
    const wrapper = { ts: Date.now(), sig: payload.sig || null, data: payload };
    const raw = JSON.stringify(wrapper);
    // Avoid exceeding storage limits
    if(raw.length < 4_000_000){
      sessionStorage.setItem('historyCache', raw);
    }
  } catch(e) {}
}

async function fetchJsonWithTimeout(url, timeoutMs){
  if(!timeoutMs || timeoutMs <= 0){
    const r = await fetch(url, { cache: 'no-store' });
    if(!r.ok){
      throw new Error(`HTTP ${r.status}`);
    }
    return await r.json();
  }
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const r = await fetch(url, { cache: 'no-store', signal: controller.signal });
    if(!r.ok){
      throw new Error(`HTTP ${r.status}`);
    }
    return await r.json();
  } finally {
    clearTimeout(timer);
  }
}

function clearLoadRetry(){
  if(loadRetryTimer){
    clearTimeout(loadRetryTimer);
    loadRetryTimer = null;
  }
  loadRetryDelayMs = 1200;
}

function scheduleLoadRetry(){
  if(loadRetryTimer) return;
  const delay = loadRetryDelayMs;
  loadRetryDelayMs = Math.min(loadRetryDelayMs * 2, 15000);
  loadRetryTimer = setTimeout(async () => {
    loadRetryTimer = null;
    try {
      await quickReload(false);
    } catch(_) {}
  }, delay);
}

async function load(){
  // Use cached history for instant render while fresh data loads
  const cached = getCachedHistory();
  if(cached && cached.data){
    data = cached.data;
    historySig = data.sig || historySig;
  }
  // Load safe server config (e.g., Jellyfin public URL for deep links)
  try{
    const cfg = await fetchJsonWithTimeout('/api/config', 2500);
    window.__appConfig = cfg || {};
  }catch(e){
    window.__appConfig = window.__appConfig || {};
  }
  const savedTab = localStorage.getItem('currentTab');
  
  // Theme & layout restore: prefer saved theme in localStorage
  const savedTheme = localStorage.getItem('theme') || localStorage.getItem('uiTheme') || 'dark';
  applyTheme(savedTheme);
  const themeSelectEl = document.getElementById('themeSelect');
  if (themeSelectEl) themeSelectEl.value = savedTheme;
  // Layout restore
  const savedLayout = localStorage.getItem('layout') || 'comfortable';
  applyLayout(savedLayout);
  const layoutSelectEl = document.getElementById('layoutSelect');
  if (layoutSelectEl) layoutSelectEl.value = savedLayout;
  console.log('Loaded data:', data);
  
  // Performance mode restore - ADD THIS
  const savedPerfMode = localStorage.getItem('performanceMode') === 'true';
  if(savedPerfMode) {
    performanceMode = true;
    document.body.classList.add('performance-mode');
    const btn = document.getElementById('perfModeBtn');
    if(btn) {
      btn.textContent = '🚀 Quality';
      btn.classList.add('active');
    }
  }
  
  // Restore filter state FIRST
  const savedFilter = localStorage.getItem('filter');
  if(savedFilter) {
    filter = savedFilter;
    // Update filter button states
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    const filterButtons = document.querySelectorAll('.filter-btn');
    if(filter === 'all') filterButtons[0]?.classList.add('active');
    else if(filter === 'movies') filterButtons[1]?.classList.add('active');
    else if(filter === 'shows') filterButtons[2]?.classList.add('active');
    else if(filter === 'incomplete') filterButtons[3]?.classList.add('active');
  }
  
  // Restore separate view modes BEFORE rendering
  const savedViewMode_movies = localStorage.getItem('viewMode_movies');
  const savedViewMode_shows = localStorage.getItem('viewMode_shows');
  const savedViewMode_all = localStorage.getItem('viewMode_all');
  const savedViewMode_incomplete = localStorage.getItem('viewMode_incomplete');
  if (savedViewMode_movies) {
    viewMode_movies = savedViewMode_movies;
  }
  if (savedViewMode_shows) {
    viewMode_shows = savedViewMode_shows;
  }
  if (savedViewMode_all) {
  viewMode_all = savedViewMode_all;
  }
  if (savedViewMode_incomplete) {
    viewMode_incomplete = savedViewMode_incomplete;
  }

  // Update view toggle buttons based on current filter
  let currentViewMode;
  if (filter === 'movies') {
    currentViewMode = viewMode_movies;
  } else if (filter === 'shows') {
    currentViewMode = viewMode_shows;
  } else if (filter === 'all') {
    currentViewMode = viewMode_all;
  } else if (filter === 'incomplete') {
    currentViewMode = viewMode_incomplete;
  }

  document.querySelectorAll('.view-toggle button').forEach(b => b.classList.remove('active'));
  if(currentViewMode === 'grid') {
    document.querySelectorAll('.view-toggle button')[1].classList.add('active');
  } else {
    document.querySelectorAll('.view-toggle button')[0].classList.add('active');
  }
  
  // Restore auto-refresh controls
      const autoToggle = document.getElementById('autoRefreshToggle');
      if (autoToggle) {
        autoToggle.checked = autoRefreshEnabled;
      }
      const autoIntervalSel = document.getElementById('autoRefreshInterval');
      if (autoIntervalSel) {
        autoIntervalSel.value = String(autoRefreshInterval);
      }
      // Start auto-refresh if enabled
  if (autoRefreshEnabled) { startAutoRefresh(); } else { updateRefreshIndicator(0); }
  // Register service worker for PWA
  if ('serviceWorker' in navigator) {
        if ('caches' in window) {
          caches.keys().then(function(keys){
            keys.forEach(function(k){
              if (k && k.startsWith('watch-tracker-') && k !== 'watch-tracker-v21') {
                caches.delete(k).catch(function(){});
              }
            });
          }).catch(function(){});
        }
    navigator.serviceWorker.register('/sw.js', { updateViaCache: 'none' })
      .then(function(reg){ try { reg.update(); } catch(e) {} })
      .catch(function(e){ console.warn('SW registration failed', e); });
  }

  if(cached && cached.data){
    renderAllFromData();
    if(savedTab) setActiveTab(savedTab);
  }

  // Fetch fresh data and re-render only if changed
  try {
    const fresh = await fetchJsonWithTimeout('/api/history', HISTORY_FETCH_TIMEOUT_MS);
    const prevSig = historySig;
    const nextSig = fresh && fresh.sig ? fresh.sig : null;
    data = fresh || data;
    if(nextSig) historySig = nextSig;
    const shouldRender = (!cached || !cached.sig || !nextSig || cached.sig !== nextSig || !prevSig);
    if(shouldRender){
      data = fresh;
      renderAllFromData();
      if(savedTab) setActiveTab(savedTab);
    }
    storeCachedHistory(fresh);
    clearLoadRetry();
    reloadFailureCount = 0;
  } catch(e) {
    reloadFailureCount += 1;
    if(e && e.name === 'AbortError'){
      // transient timeout/abort, keep cached UI
    } else if(reloadFailureCount % 5 === 1){
      console.warn('Load error:', (e && e.message) ? e.message : e);
    }
    scheduleLoadRetry();
  }

  startSigPolling();
}

/**
 * Reload history data from the server and update UI. During regular reloads (e.g. manual
 * refresh or first load) we show the content-loading spinner. When performing an
 * auto-refresh in the background we avoid showing the spinner to prevent layout
 * contraction on desktop.
 *
 * @param {boolean} showLoading If true, show the loading spinner; otherwise reload silently.
 */
async function quickReload(showLoading = false) {
  if (quickReloadBusy) return;
  quickReloadBusy = true;
  const content = document.getElementById('content');

  if (showLoading && content) {
    content.classList.add('content-loading');
  }

  try {
    if (showLoading) {
      const cached = getCachedHistory();
      if (cached && cached.data) {
        data = cached.data;
        applyStatsFromData();
        setActiveTab(currentTab);
        try { restoreState(); } catch(e) { console.error('Restore state failed:', e); }
      }
    }

    const prevSig = historySig;
    const fresh = await fetchJsonWithTimeout('/api/history', HISTORY_FETCH_TIMEOUT_MS);
    const nextSig = fresh && fresh.sig ? fresh.sig : null;
    const sigChanged = !prevSig || !nextSig || prevSig !== nextSig;

    data = fresh || data;
    if(nextSig) historySig = nextSig;
    storeCachedHistory(data);
    clearLoadRetry();
    reloadFailureCount = 0;

    if(sigChanged){
      applyStatsFromData();
      setActiveTab(currentTab);
      try { restoreState(); } catch(e) { console.error('Restore state failed:', e); }
    }
  } catch (e) {
    reloadFailureCount += 1;
    if (e && e.name === 'AbortError') {
      // Ignore fetch aborts caused by browser/network transitions.
    } else {
      if(reloadFailureCount % 5 === 1){
        console.warn('Auto-refresh error', (e && e.message) ? e.message : e);
      }
    }
    scheduleLoadRetry();
  } finally {
    quickReloadBusy = false;
    if (showLoading && content) {
      content.classList.remove('content-loading');
    }
  }
}

function toggleEpisodeSelection(series, season, episode, event) {
  event.stopPropagation();
  event.preventDefault();
  
  const key = `${series}_${season}_${episode}`;
  const checkbox = event.target;
  
  // Determine new state
  const shouldBeChecked = !selectedEpisodes.has(key);
  
  // Update the Set
  if(shouldBeChecked) {
    selectedEpisodes.add(key);
  } else {
    selectedEpisodes.delete(key);
  }
  
  // Force the checkbox state
  setTimeout(() => {
    checkbox.checked = shouldBeChecked;
    const episodeItem = checkbox.closest('.episode-item');
    if(episodeItem) {
      if(shouldBeChecked) {
        episodeItem.classList.add('selected');
      } else {
        episodeItem.classList.remove('selected');
      }
    }
  }, 0);
  
  updateBulkBar();
  
  return false;
}

function updateBulkBar() {
  const count = selectedEpisodes.size;
  const bar = document.getElementById('bulk-bar');
  const countSpan = document.getElementById('bulk-count');
  
  if(count > 0) {
    bar.style.display = 'flex';
    countSpan.textContent = count;
  } else {
    bar.style.display = 'none';
  }
}

function selectAll() {
  const shows = data.shows || [];
  selectedEpisodes.clear();
  for(const show of shows) {
    for(const season of show.seasons || []) {
      for(const ep of season.episodes || []) {
        const key = `${show.series_name}_${season.season_number}_${ep.episode}`;
        selectedEpisodes.add(key);
      }
    }
  }
  saveOpenPanels();
  updateBulkBar();
  renderHistory();
  restoreOpenPanels();
}

function clearSelection() {
  selectedEpisodes.clear();
  saveOpenPanels();
  updateBulkBar();
  renderHistory();
  restoreOpenPanels();
}

async function bulkDelete(){
  if(selectedEpisodes.size===0)return;
  if(!confirm(`Delete ${selectedEpisodes.size} selected episodes?`))return;
  saveState();
  
  const episodes=Array.from(selectedEpisodes).map(key=>{
    const parts=key.split('_');
    return{
      series_name:parts.slice(0,-2).join('_'),
      season:parseInt(parts[parts.length-2]),
      episode:parseInt(parts[parts.length-1])
    };
  });
  
  try{
    const r=await fetch('/api/bulk_delete_episodes',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({episodes})
    });
    const result=await r.json();
    if(result.success){
      selectedEpisodes.clear();
      quickReload();
    }else{
      alert('Failed: '+result.error);
    }
  }catch(e){
    alert('Error: '+e.message);
  }
}

async function gridBulkMarkComplete(){
  const items = Array.from(selectedGridItems);
  const showKeys = items.filter(k => k.startsWith('show|'));
  if(showKeys.length === 0) return; // silent

  const firstKey = showKeys[0];
  saveState(firstKey || undefined);

  try{
    setGridBulkStatus(`Marking 0/${showKeys.length}…`);

    let done = 0;
    for(const key of showKeys){
      const series = key.slice('show|'.length);

      setGridBulkStatus(`Marking ${done+1}/${showKeys.length}…`);

      const r = await fetch('/api/mark_complete', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ series_name: series })
      });

      const j = await r.json();
      if(!j.success){
        throw new Error(j.error || `Failed for: ${series}`);
      }

      done++;
    }

    setGridBulkStatus(`Done ✓ Marked ${done}/${showKeys.length}`);
    clearGridBulkStatusSoon(1200);

    selectedGridItems.clear();
    updateGridBulkBar();
    quickReload();

  }catch(e){
    console.error('gridBulkMarkComplete error:', e);
    setGridBulkStatus('Failed (see console)');
    clearGridBulkStatusSoon(1800);
  }
}

function gridItemClick(seriesName, isShow) {
  if (!isShow) return;

  // Switch to the Library (history) tab so the user can see the show.
  const historyBtn = document.querySelector(".tab-btn[onclick*='history']");
  if (historyBtn) {
    switchTab({ target: historyBtn }, 'history');
  }

  filter = 'shows';
  localStorage.setItem('filter', 'shows');
  viewMode_shows = 'list';
  localStorage.setItem('viewMode_shows', 'list');

  // Reset view toggle buttons to reflect list mode
  document.querySelectorAll('.view-toggle button').forEach(b => b.classList.remove('active'));
  const firstToggle = document.querySelector('.view-toggle button:first-child');
  if (firstToggle) firstToggle.classList.add('active');

  // Clear the search input so the show appears in the list
  const searchInput = document.getElementById('search');
  if (searchInput && searchInput.value) {
    searchInput.value = '';
  }

  // Render history with the updated filter and view
  renderHistory();

  // Expand + scroll to the show after render
  setTimeout(() => {
    const id = 'show-' + seriesName.replace(/[^a-zA-Z0-9]/g, '');
    const seasonEl = document.getElementById(id);
    if (!seasonEl) return;

    // Expand first (height changes)
    if (seasonEl.style.display === 'none' || seasonEl.style.display === '') {
      toggle(id);
    }

    // After expansion, re-measure + scroll with header offset
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        // Find show-group container (walk up)
        let groupEl = seasonEl;
        for (let i = 0; i < 6; i++) {
          if (groupEl && groupEl.classList && groupEl.classList.contains('show-group')) break;
          groupEl = groupEl ? groupEl.parentElement : null;
        }

        const target = groupEl || seasonEl;

        const header = document.querySelector('.header');
        const headerOffset = header ? header.getBoundingClientRect().height + 12 : 120;

        const rect = target.getBoundingClientRect();
        const y = window.scrollY + rect.top - headerOffset;

        window.scrollTo({ top: y, behavior: 'smooth' });
      });
    });
  }, 50);
}

function gridMovieClick(movieName, movieYear) {
  navigateToMovie(movieName, movieYear);
}

function switchTab(e,t){
  // If leaving the progress tab, clear any search filter applied when
  // navigating from progress to the library.  Then re-render the history
  // view so that the cleared search takes effect.
  if(currentTab === 'progress' && t !== 'progress'){
    const searchInput = document.getElementById('search');
    if(searchInput && searchInput.value){
      searchInput.value = '';
    }
    renderHistory();
  }
  localStorage.setItem('currentTab', t);
  setActiveTab(t);
}

function setActiveTab(t){
  const tab = t || 'history';
  currentTab = tab;
  document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c=>c.classList.remove('active'));
  const btn = document.querySelector(`.tab-btn[onclick*="'${tab}'"]`) || document.querySelector(`.tab-btn[onclick*="${tab}"]`);
  if(btn) btn.classList.add('active');
  const content = document.getElementById(tab+'-tab');
  if(content) content.classList.add('active');
  if(tab==='genres')renderGenresLazy();
  if(tab==='analytics')renderAnalytics();
  if(tab==='progress')renderProgress();
  if(tab==='history')renderHistory();
}

function getLatestShowTimestamp(s){
  let latest='';
  for(const season of s.seasons||[]){
    for(const ep of season.episodes||[]){
      if(ep.watches&&ep.watches[0]&&ep.watches[0].timestamp>latest){
        latest=ep.watches[0].timestamp;
      }
    }
  }
  return latest;
}

function formatTimeAmPm(ts){
  if(!ts) return '';
  const d = new Date(ts);
  if(isNaN(d)) return '';
  let h = d.getHours();
  const m = d.getMinutes().toString().padStart(2,'0');
  const ampm = h >= 12 ? 'PM' : 'AM';
  h = h % 12;
  if(h === 0) h = 12;
  return `${h}:${m} ${ampm}`;
}

function formatDateTimeAmPm(ts){
  if(!ts) return '';
  const d = new Date(ts);
  if(isNaN(d)) return '';
  const date = d.toLocaleDateString(undefined,{year:'numeric',month:'short',day:'numeric'});
  return `${date} ${formatTimeAmPm(ts)}`;
}

function formatDurationHours(hours){
  const h = Number(hours);
  if(!isFinite(h) || h <= 0) return '0h';
  if(h < 24){
    const rounded = Math.round(h * 10) / 10;
    return `${rounded}h`;
  }
  const totalDays = Math.floor(h / 24);
  const years = Math.floor(totalDays / 365);
  const months = Math.floor((totalDays % 365) / 30);
  const days = totalDays % 30;
  const parts = [];
  if(years) parts.push(`${years}y`);
  if(months) parts.push(`${months}mo`);
  if(days || parts.length === 0) parts.push(`${days}d`);
  return parts.join(' ');
}

function getLastEpisodeInfo(show){
  let best = null;
  for(const season of show.seasons||[]){
    for(const ep of season.episodes||[]){
      for(const w of (ep.watches||[])){
        const t = w && w.timestamp ? Date.parse(w.timestamp) : NaN;
        if(isNaN(t)) continue;
        const seasonNum = Number(season.season_number) || 0;
        const epNum = Number(ep.episode) || 0;
        const betterByTime = !best || t > best.t;
        const sameTimeBetterEpisode = best && t === best.t && (
          seasonNum > best.season ||
          (seasonNum === best.season && epNum > best.episode)
        );
        if(betterByTime || sameTimeBetterEpisode){
          const epName = ep.name || ep.title || ep.episode_name || ep.episode_title || '';
          best = { t, season: seasonNum, episode: epNum, name: epName, timestamp: w.timestamp, jellyfin_id: w.jellyfin_id || ep.jellyfin_id || null };
        }
      }
    }
  }
  return best;
}

function getJellyfinItemUrl(itemId){
  const cfg = window.__appConfig || {};
  const baseRaw = (cfg.jellyfin_public_url || '').trim();
  const base = baseRaw.replace(new RegExp('/+$'), '');
  if(!base || !itemId) return '';
  // Support both "https://host/jellyfin" and "https://host/jellyfin/web"
  const isWebBase = new RegExp('/web$', 'i').test(base);
  const prefix = isWebBase ? (base + '/index.html') : (base + '/web/index.html');
  return `${prefix}#!/details?id=${encodeURIComponent(String(itemId))}`;
}

function openInJellyfin(itemId){
  const url = getJellyfinItemUrl(itemId);
  if(!url){
    alert('Jellyfin link not available. Set JELLYFIN_PUBLIC_URL (or JELLYFIN_URL) and ensure this item has a Jellyfin id.');
    return;
  }
  // On mobile/touch devices, use same-window navigation so the OS can hand off
  // to an installed app if it has registered to open Jellyfin URLs.
  try{
    const touchLike = (window.matchMedia && window.matchMedia('(hover: none)').matches) || (navigator.maxTouchPoints && navigator.maxTouchPoints > 0);
    if(touchLike){
      window.location.assign(url);
      return;
    }
  }catch(e){}
  window.open(url, '_blank', 'noopener');
}

function getTotalPossibleEpisodes(show){
  if(show && show.tmdb_pending) return 0;
  const direct = show && show.total_episodes_possible ? show.total_episodes_possible : 0;
  if(direct && direct > 0) return direct;
  let sum = 0;
  let has = false;
  (show && show.seasons ? show.seasons : []).forEach(season => {
    const total = season.total_episodes || season.episode_count || 0;
    if(total > 0){ sum += total; has = true; }
  });
  if(has) return sum;
  return show && show.total_episodes ? show.total_episodes : 0;
}
function setView(mode, e) {
  // Save separate view modes for each filter
  if (filter === 'movies') {
    viewMode_movies = mode;
    localStorage.setItem('viewMode_movies', mode);
  } else if (filter === 'shows') {
    viewMode_shows = mode;
    localStorage.setItem('viewMode_shows', mode);
  } else if (filter === 'all') {
    viewMode_all = mode;
    localStorage.setItem('viewMode_all', mode);
  } else if (filter === 'incomplete') {
    viewMode_incomplete = mode;
    localStorage.setItem('viewMode_incomplete', mode);
  }

  document.querySelectorAll('.view-toggle button').forEach(b => b.classList.remove('active'));
  if (e && e.target) e.target.classList.add('active');

  // IMPORTANT: reset paging when switching list/grid
  currentPage = 1;

  renderHistory();
  showGridSelectButton();
  updateGridBulkBar();
}

function openPosterModal(item,type){
  currentPosterItem={...item,type};
  document.getElementById('posterItemName').textContent=type==='movie'?item.name+' ('+(item.year||'')+')':(item.series_name||item.name);
  document.getElementById('posterModal').style.display='block';
}

function closePosterModal(){
  document.getElementById('posterModal').style.display='none';
  currentPosterItem=null;
  document.getElementById('posterFile').value='';
}

function openManualEntry(){
  document.getElementById('manualEntryModal').style.display='block';
  document.getElementById('entryDate').valueAsDate=new Date();
  try{ toggleEpisodeFields(); }catch(e){}
}

function closeManualEntry(){
  document.getElementById('manualEntryModal').style.display='none';
  document.getElementById('entryMovieTitle').value='';
  document.getElementById('entryYear').value='';
  document.getElementById('entrySeriesName').value='';
  document.getElementById('entrySeason').value='';
  document.getElementById('entryEpisodeSpec').value='';
  document.getElementById('entryEpisodeName').value='';
  document.getElementById('entryGenres').value='';
  document.querySelectorAll('input[name="entryType"]')[0].checked=true;
  toggleEpisodeFields();
}

function toggleEpisodeFields(){
  const type=document.querySelector('input[name="entryType"]:checked').value;
  const fields=document.getElementById('episodeFields');
  const movie=document.getElementById('movieFields');
  if(type==='Episode'){
    fields.classList.add('show');
    if(movie) movie.style.display='none';
  }else{
    fields.classList.remove('show');
    if(movie) movie.style.display='block';
  }
}

function openAddEpisode(seriesName){
  currentManageShow=seriesName;
  document.getElementById('addEpSeriesName').value=seriesName;
  document.getElementById('addEpisodeTitle').textContent=`➕ Add Episode to ${seriesName}`;
  document.getElementById('addEpDate').valueAsDate=new Date();
  document.getElementById('skipTMDB').checked=false;
  document.getElementById('addEpisodeModal').style.display='block';
}

function closeAddEpisode(){
  document.getElementById('addEpisodeModal').style.display='none';
  document.getElementById('addEpSeason').value='';
  document.getElementById('addEpEpisode').value='';
  currentManageShow=null;
}

function openAddSeason(seriesName){
  currentManageShow=seriesName;
  document.getElementById('addSeasonSeriesName').value=seriesName;
  document.getElementById('addSeasonTitle').textContent=`➕ Add Season to ${seriesName}`;
  document.getElementById('addSeasonDate').valueAsDate=new Date();
  document.getElementById('skipTMDBSeason').checked=false;
  document.getElementById('addSeasonModal').style.display='block';
}

function closeAddSeason(){
  document.getElementById('addSeasonModal').style.display='none';
  document.getElementById('addSeasonNumber').value='';
  document.getElementById('addSeasonEpCount').value='';
  currentManageShow=null;
}

function openSettings(){
  document.getElementById('settingsModal').style.display='block';
}

function closeSettings(){
  document.getElementById('settingsModal').style.display='none';
}

async function submitManualEntry(){
  const type=document.querySelector('input[name="entryType"]:checked').value;
  const year=document.getElementById('entryYear').value;
  const date=document.getElementById('entryDate').value;
  const genresStr=document.getElementById('entryGenres').value.trim();
  
  const yearInt = year ? parseInt(year) : null;
  const genresList = genresStr ? genresStr.split(',').map(g=>g.trim()).filter(Boolean) : [];

  const btn = document.getElementById('manualEntrySubmitBtn');
  const btnOldText = btn ? btn.textContent : '';
  const setBusy = (on, text) => {
    if(!btn) return;
    if(on){
      btn.disabled = true;
      btn.textContent = text || 'Adding...';
      btn.style.opacity = '0.75';
      btn.style.cursor = 'default';
    }else{
      btn.disabled = false;
      btn.textContent = btnOldText || '✓ Add Watch Record';
      btn.style.opacity = '';
      btn.style.cursor = '';
    }
  };
  const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));
  const beginBusy = async (text) => {
    setBusy(true, text);
    // Give the browser a tick to paint the disabled/loading state.
    await sleep(0);
  };

  if(type==='Movie'){
    const title=document.getElementById('entryMovieTitle').value.trim();
    if(!title){
    alert('Please enter a movie title');
    return;
  }
    if(!date){
      alert('Please select a watch date');
      return;
    }

    const record={
      timestamp:new Date(date+'T12:00:00').toISOString(),
      type:'Movie',
      name:title,
      year:yearInt,
      user:'Manual Entry',
      genres:genresList,
      source:'manual'
    };

    try{
      const t0 = (performance && performance.now) ? performance.now() : Date.now();
      await beginBusy('Adding movie...');
      const r=await fetch('/api/manual_entry',{
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify(record)
      });
      const result=await r.json();
      if(result.success){
        const t1 = (performance && performance.now) ? performance.now() : Date.now();
        const elapsed = t1 - t0;
        if(elapsed < 350) await sleep(350 - elapsed);
        saveState(gridKeyMovie(title, yearInt));
        closeManualEntry();
        location.reload();
      }else{
        setBusy(false);
        alert('Failed: '+result.error);
      }
    }catch(e){
      setBusy(false);
      alert('Error: '+e.message);
    }
    return;
  }
  
  if(!date){
    alert('Please select a watch date');
    return;
  }

  // Show (Episode records under the hood)
  const seriesName=document.getElementById('entrySeriesName').value.trim();
  const seasonRaw=(document.getElementById('entrySeason').value||'').trim();
  const episodeSpec=(document.getElementById('entryEpisodeSpec').value||'').trim();
  const episodeName=(document.getElementById('entryEpisodeName').value||'').trim();

    if(!seriesName){
      alert('Please enter series name');
      return;
    }

    // CASE A: Show only (no season, no episode) -> add ALL seasons+episodes
    // CASE B: Show + season only -> add ALL episodes in that season
    // CASE C: Show + season + episodeSpec (1,2,5-8) -> add those episodes
    const isBulkAllSeries = (!seasonRaw && !episodeSpec);
    const isBulkSeason = (seasonRaw && !episodeSpec);
    const isBulkEpisodeSpec = (seasonRaw && episodeSpec);

    // If episodeSpec provided but no season, block (no sense)
    if(!seasonRaw && episodeSpec){
      alert('Season number is required when specifying episode numbers');
      return;
    }

    if(isBulkAllSeries || isBulkSeason || isBulkEpisodeSpec){
      try{
        const t0 = (performance && performance.now) ? performance.now() : Date.now();
        await beginBusy('Adding show...');
        const r = await fetch('/api/manual_tv_bulk', {
          method:'POST',
          headers:{'Content-Type':'application/json'},
          body: JSON.stringify({
            series_name: seriesName,
            season: seasonRaw ? parseInt(seasonRaw) : null,
            episode_spec: episodeSpec,
            episode_name: episodeName,
            year: yearInt,
            genres: genresList,
            date: date,
            skip_tmdb: false
          })
        });
        const j = await r.json();
        if(j.success){
          const t1 = (performance && performance.now) ? performance.now() : Date.now();
          const elapsed = t1 - t0;
          if(elapsed < 350) await sleep(350 - elapsed);
          alert(`✓ Added ${j.added} episodes`);
          saveState(gridKeyShow(j.series_name || seriesName));
          closeManualEntry();
          location.reload();
        }else{
          setBusy(false);
          alert('Failed: ' + (j.error || 'Unknown error'));
        }
      }catch(e){
        setBusy(false);
        alert('Error: ' + e.message);
      }
      return;
    }

    // Fallback (should not reach here)
    alert('Invalid episode input');
    return;
}

async function submitAddEpisode(){
  const seriesName=document.getElementById('addEpSeriesName').value.trim();
  const season=document.getElementById('addEpSeason').value;
  const episode=document.getElementById('addEpEpisode').value;
  const date=document.getElementById('addEpDate').value;
  const skipTMDB=document.getElementById('skipTMDB').checked;
  
  if(!season||!episode||!date){
    alert('Please fill all fields');
    return;
  }
  
  saveState(gridKeyShow(seriesName));
  try{
    const r=await fetch('/api/add_episode',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({
        series_name:seriesName,
        season:parseInt(season),
        episode:parseInt(episode),
        date:date,
        skip_tmdb:skipTMDB
      })
    });
    const result=await r.json();
    if(result.success){
      alert(`✓ Episode added: ${result.episode_name}`);
      closeAddEpisode();
      location.reload();
    }else{
      alert('Failed: '+result.error);
    }
  }catch(e){
    alert('Error: '+e.message);
  }
}

async function submitAddSeason(){
  const seriesName=document.getElementById('addSeasonSeriesName').value.trim();
  const seasonNum=document.getElementById('addSeasonNumber').value;
  const date=document.getElementById('addSeasonDate').value;
  const epCount=document.getElementById('addSeasonEpCount').value;
  const skipTMDB=document.getElementById('skipTMDBSeason').checked;
  
  if(!seasonNum||!date){
    alert('Please fill all fields');
    return;
  }
  
  if(skipTMDB&&!epCount){
    alert('Please enter number of episodes');
    return;
  }
  
  if(!skipTMDB&&!confirm(`Add ALL episodes from Season ${seasonNum}?`))return;
  
  saveState(gridKeyShow(seriesName));
  try{
    const r=await fetch('/api/add_season',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({
        series_name:seriesName,
        season:parseInt(seasonNum),
        date:date,
        skip_tmdb:skipTMDB,
        episode_count:epCount?parseInt(epCount):null
      })
    });
    const result=await r.json();
    if(result.success){
      closeAddSeason();
      location.reload();
    }else{
      alert('Failed: '+result.error);
    }
  }catch(e){
    alert('Error: '+e.message);
  }
}

async function markComplete(seriesName){
  if(!confirm(`Mark "${seriesName}" as 100% complete?\\n\\nThis will override the progress bar.`))return;
  
  saveState(gridKeyShow(seriesName));
  try{
    const r=await fetch('/api/mark_complete',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({series_name:seriesName})
    });
    const result=await r.json();
    if(result.success){
      location.reload();
    }else{
      alert('Failed: '+result.error);
    }
  }catch(e){
    alert('Error: '+e.message);
  }
}

async function markSeasonComplete(seriesName,seasonNum){
  if(!confirm(`Mark "${seriesName}" Season ${seasonNum} as 100% complete?`))return;
  
  saveState(gridKeyShow(seriesName));
  try{
    const r=await fetch('/api/mark_season_complete',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({series_name:seriesName,season:seasonNum})
    });
    const result=await r.json();
    if(result.success){
      location.reload();
    }else{
      alert('Failed: '+result.error);
    }
  }catch(e){
    alert('Error: '+e.message);
  }
}

async function clearTMDBCache(){
  if(!confirm('Clear all cached TMDB data? This will force fresh lookups.'))return;
  
  try{
    const r=await fetch('/api/clear_tmdb_cache',{method:'POST'});
    const result=await r.json();
    if(result.success){
      try { sessionStorage.removeItem('historyCache'); } catch(e) {}
      try { sessionStorage.removeItem('recsCache'); } catch(e) {}
      try {
        if ('caches' in window) {
          const keys = await caches.keys();
          await Promise.all(keys.map(k => (k && k.startsWith('watch-tracker-')) ? caches.delete(k) : Promise.resolve()));
        }
      } catch (e) {}
      alert('✓ TMDB cache cleared! Reloading...');
      location.reload();
    }else{
      alert('Failed: '+result.error);
    }
  }catch(e){
    alert('Error: '+e.message);
  }
}

// Clear all watch history records and reload the UI.  This will remove
// everything from your watch log, including movies and episodes, and
// reset charts and progress.  Use with caution!
async function clearWatchHistory(){
  if(!confirm('Clear all watch history? This will remove all records and progress.')) return;
  try{
    const r = await fetch('/api/clear_history',{method:'POST'});
    const result = await r.json();
    if(result.success){
      alert('✓ Watch history cleared! Reloading...');
      // Force a full reload so all data (stats, charts, progress) refreshes
      location.reload();
    }else{
      alert('Failed: '+result.error);
    }
  }catch(e){
    alert('Error: '+e.message);
  }
}

// Clear manual completion and progress data.  This resets all manual
// completions and season completions and then reloads the UI so the
// progress trackers rebuild from scratch.
async function clearProgress(){
  if(!confirm('Clear all manual progress? This will reset manual completions and progress tracking.')) return;
  try{
    const r = await fetch('/api/clear_progress',{method:'POST'});
    const result = await r.json();
    if(result.success){
      alert('✓ Progress cleared! Reloading...');
      location.reload();
    }else{
      alert('Failed: '+result.error);
    }
  }catch(e){
    alert('Error: '+e.message);
  }
}

// Clear cached insights.  Currently insights are computed from watch
// history on the fly, but this will reset any cached data and reload
// the UI to reflect a fresh state.
async function clearInsights(){
  if(!confirm('Clear insights? This will reset genre insights and recommendations.')) return;
  try{
    const r = await fetch('/api/clear_insights',{method:'POST'});
    const result = await r.json();
    if(result.success){
      alert('✓ Insights cleared! Reloading...');
      location.reload();
    }else{
      alert('Failed: '+result.error);
    }
  }catch(e){
    alert('Error: '+e.message);
  }
}

async function importJellyfin(){
  if(!confirm('Import from Jellyfin? This will OVERWRITE existing data!'))return;
  const btn = document.getElementById('jellyfinImportBtn');
  let prevText;
  if(btn){
    prevText = btn.textContent;
    btn.disabled = true;
    btn.textContent = 'Importing...';
  }
  try{
    const r=await fetch('/api/jellyfin_import',{method:'POST'});
    const j=await r.json();
    alert(j.success?'SUCCESS: '+j.message:'FAILED: '+j.error);
    if(j.success)location.reload();
  }catch(e){
    alert('Import failed: '+e.message);
  }finally{
    if(btn){
      btn.disabled = false;
      btn.textContent = prevText || 'Import from Jellyfin';
    }
  }
}

async function importSonarr(){
  if(!confirm('Import from Sonarr? This will add all downloaded TV shows.'))return;
  const btn=document.getElementById('sonarrImportBtn');
  btn.disabled=true;
  btn.textContent='Importing...';
  
  try{
    const r=await fetch('/api/sonarr_import',{method:'POST'});
    const j=await r.json();
    alert(j.success?'SUCCESS: '+j.message:'FAILED: '+j.error);
    if(j.success)location.reload();
  }catch(e){
    alert('Import failed: '+e.message);
  }finally{
    btn.disabled=false;
    btn.textContent='Import from Sonarr';
  }
}

async function importRadarr(){
  if(!confirm('Import from Radarr? This will add all downloaded movies.'))return;
  const btn=document.getElementById('radarrImportBtn');
  btn.disabled=true;
  btn.textContent='Importing...';
  
  try{
    const r=await fetch('/api/radarr_import',{method:'POST'});
    const j=await r.json();
    alert(j.success?'SUCCESS: '+j.message:'FAILED: '+j.error);
    if(j.success)location.reload();
  }catch(e){
    alert('Import failed: '+e.message);
  }finally{
    btn.disabled=false;
    btn.textContent='Import from Radarr';
  }
}

async function deleteMovie(movieName,movieYear){
  if(!confirm(`Delete ALL watch records for "${movieName}" (${movieYear})?`))return;
  saveState(gridKeyMovie(movieName, movieYear));
  
  try{
    const r=await fetch('/api/delete_movie',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({name:movieName,year:movieYear})
    });
    const result=await r.json();
    if(result.success){
      quickReload();
    }else{
      alert('Failed: '+result.error);
    }
  }catch(e){
    alert('Error: '+e.message);
  }
}

async function deleteShow(seriesName){
  if(!confirm(`Delete ALL watch records for "${seriesName}"?`))return;
  saveState(gridKeyShow(seriesName));
  
  try{
    const r=await fetch('/api/delete_show',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({series_name:seriesName})
    });
    const result=await r.json();
    if(result.success){
      quickReload();
    }else{
      alert('Failed: '+result.error);
    }
  }catch(e){
    alert('Error: '+e.message);
  }
}

async function deleteSeason(seriesName,seasonNum){
  if(!confirm(`Delete ALL episodes from "${seriesName}" Season ${seasonNum}?`))return;
  saveState(gridKeyShow(seriesName));
  
  try{
    const r=await fetch('/api/delete_season',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({series_name:seriesName,season:seasonNum})
    });
    const result=await r.json();
    if(result.success){
      quickReload();
    }else{
      alert('Failed: '+result.error);
    }
  }catch(e){
    alert('Error: '+e.message);
  }
}

async function deleteEpisode(seriesName,seasonNum,episodeNum){
  if(!confirm(`Delete "${seriesName}" S${seasonNum}E${episodeNum}?`))return;
  saveState(gridKeyShow(seriesName));
  
  try{
    const r=await fetch('/api/delete_episode',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({series_name:seriesName,season:seasonNum,episode:episodeNum})
    });
    const result=await r.json();
    if(result.success){
      quickReload();
    }else{
      alert('Failed: '+result.error);
    }
  }catch(e){
    alert('Error: '+e.message);
  }
}

async function uploadPoster(){
  const file=document.getElementById('posterFile').files[0];
  if(!file){
    alert('Please select a file');
    return;
  }
  
  const formData=new FormData();
  formData.append('poster',file);
  formData.append('name',currentPosterItem.name||currentPosterItem.series_name);
  formData.append('year',currentPosterItem.year||'');
  formData.append('type',currentPosterItem.type);
  
  try{
    const r=await fetch('/api/upload_poster',{method:'POST',body:formData});
    const result=await r.json();
    if(result.success){
      closePosterModal();
      location.reload();
    }else{
      alert('Upload failed: '+result.error);
    }
  }catch(e){
    alert('Upload error: '+e.message);
  }
}

async function setManualShowTotal(seriesName){
  try{
    if(!seriesName) return;
    const raw = prompt(`Set total episodes for "${seriesName}" (positive number):`, '');
    if(raw === null) return;
    const total = parseInt(String(raw).trim(), 10);
    if(!Number.isFinite(total) || total <= 0){
      alert('Please enter a positive number.');
      return;
    }
    const r = await fetch('/api/set_show_total', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ series_name: seriesName, total: total })
    });
    const j = await r.json();
    if(j && j.success){
      quickReload();
    } else {
      alert('Failed: ' + ((j && j.error) || 'Unknown error'));
    }
  }catch(e){
    alert('Error: ' + (e && e.message ? e.message : e));
  }
}

function setupPendingTotalsInteractions(){
  const root = document.getElementById('content');
  if(!root || root.__pendingTotalsSetup) return;
  root.__pendingTotalsSetup = true;

  // Desktop: double click the pending totals text
  root.addEventListener('dblclick', function(e){
    const el = e.target && e.target.closest ? e.target.closest('.tmdb-pending-total') : null;
    if(!el) return;
    e.preventDefault();
    e.stopPropagation();
    const enc = el.getAttribute('data-series-name') || '';
    const seriesName = enc ? decodeURIComponent(enc) : '';
    setManualShowTotal(seriesName);
  });

  // Touch: press and hold on the pending totals text
  let lpTimer = null;
  let startX = 0;
  let startY = 0;
  let lpFired = false;
  let lpEl = null;

  function clearLp(){
    if(lpTimer){
      clearTimeout(lpTimer);
      lpTimer = null;
    }
    lpEl = null;
  }

  root.addEventListener('pointerdown', function(e){
    const el = e.target && e.target.closest ? e.target.closest('.tmdb-pending-total') : null;
    if(!el) return;
    if(e.pointerType === 'mouse') return; // use dblclick for mouse
    lpFired = false;
    lpEl = el;
    startX = e.clientX || 0;
    startY = e.clientY || 0;
    clearLp();
    lpEl = el;
    lpTimer = setTimeout(() => {
      lpFired = true;
      const enc = el.getAttribute('data-series-name') || '';
      const seriesName = enc ? decodeURIComponent(enc) : '';
      setManualShowTotal(seriesName);
    }, 650);
  }, { passive: true });

  root.addEventListener('pointermove', function(e){
    if(!lpTimer || !lpEl) return;
    const dx = Math.abs((e.clientX || 0) - startX);
    const dy = Math.abs((e.clientY || 0) - startY);
    if(dx > 12 || dy > 12){
      clearLp();
    }
  }, { passive: true });

  root.addEventListener('pointerup', clearLp, { passive: true });
  root.addEventListener('pointercancel', clearLp, { passive: true });

  // Suppress the click that often follows a long-press so it doesn't toggle the show accordion.
  root.addEventListener('click', function(e){
    if(!lpFired) return;
    const el = e.target && e.target.closest ? e.target.closest('.tmdb-pending-total') : null;
    if(!el) return;
    e.preventDefault();
    e.stopPropagation();
    lpFired = false;
  }, true);

  // Prevent mobile context menu from stealing the long-press gesture.
  root.addEventListener('contextmenu', function(e){
    const el = e.target && e.target.closest ? e.target.closest('.tmdb-pending-total') : null;
    if(!el) return;
    e.preventDefault();
  });
}

function renderHistory(){
  const search=(document.getElementById('search').value||'').toLowerCase();
  const sort=document.getElementById('sort').value;
  
  let viewMode;
  if (filter === 'movies') {
    viewMode = viewMode_movies;
  } else if (filter === 'shows') {
    viewMode = viewMode_shows;
  } else if (filter === 'all') {
    viewMode = viewMode_all;
  } else if (filter === 'incomplete') {
    viewMode = viewMode_incomplete;
  }

  document.querySelectorAll('.view-toggle button').forEach(b => b.classList.remove('active'))
  if (viewMode === 'grid') {
      document.querySelectorAll('.view-toggle button')[1].classList.add('active')
  } else {
      document.querySelectorAll('.view-toggle button')[0].classList.add('active')
  }
  
  let movies=(data.movies||[]).filter(m=>!search||m.name.toLowerCase().includes(search));
  let shows=(data.shows||[]).filter(s=>!search||s.series_name.toLowerCase().includes(search));
  try {
    const adv = typeof advancedFilters !== 'undefined' && advancedFilters ? advancedFilters : (JSON.parse(localStorage.getItem('advancedFilters') || '{}') || {});
    if (adv.genre) {
      movies = movies.filter(m => (m.genres || []).includes(adv.genre));
      shows = shows.filter(s => (s.genres || []).includes(adv.genre));
    }
    const yf = parseInt(adv.yearFrom, 10);
    if (!isNaN(yf)) {
      movies = movies.filter(m => m.year && m.year >= yf);
      shows = shows.filter(s => s.year && s.year >= yf);
    }
    const yt = parseInt(adv.yearTo, 10);
    if (!isNaN(yt)) {
      movies = movies.filter(m => m.year && m.year <= yt);
      shows = shows.filter(s => s.year && s.year <= yt);
    }
    // FIX 2: Change 'in_progress' to 'inprogress'
    if (adv.status === 'completed') {
      shows = shows.filter(s => s.completion_percentage >= 100);
    } else if (adv.status === 'inprogress') {
      shows = shows.filter(s => s.completion_percentage > 0 && s.completion_percentage < 100);
    }
    if (adv.type === 'movie') {
      shows = [];
    } else if (adv.type === 'show') {
      movies = [];
    }
  } catch(err) { console.error(err); }
  
  if(filter==='movies'){
    shows=[];
  }else if(filter==='shows'){
    movies=[];
  }else if(filter==='incomplete'){
    movies=[];
    shows=shows.filter(s=>s.completion_percentage<100&&s.has_tmdb_data);
  }
  
  // FIX 3: Change 'most_watched' to 'mostwatched' and 'incomplete_first' to 'incompletefirst'
  if(sort==='recent'){
    movies.sort((a,b)=>(b.watches[0]?.timestamp||'').localeCompare(a.watches[0]?.timestamp||''));
    shows.sort((a,b)=>getLatestShowTimestamp(b).localeCompare(getLatestShowTimestamp(a)));
  }else if(sort==='oldest'){
    movies.sort((a,b)=>(a.watches[0]?.timestamp||'').localeCompare(b.watches[0]?.timestamp||''));
    shows.sort((a,b)=>getLatestShowTimestamp(a).localeCompare(getLatestShowTimestamp(b)));
  }else if(sort==='mostwatched'){
    movies.sort((a,b)=>b.watch_count-a.watch_count);
    shows.sort((a,b)=>b.total_watches-a.total_watches);
  }else if(sort==='alphabetical'){
    movies.sort((a,b)=>a.name.localeCompare(b.name));
    shows.sort((a,b)=>a.series_name.localeCompare(b.series_name));
  }else if(sort==='incompletefirst'){
    shows.sort((a,b)=>a.completion_percentage-b.completion_percentage);
  }
  
  const shouldShowLoadMore = (performanceMode || enableLoadMoreInQuality);
  const totalMoviesAll = movies.length;
  const totalShowsAll  = shows.length;
  const totalItemsAll  = totalMoviesAll + totalShowsAll;
  let endIdx = totalItemsAll;
  if (shouldShowLoadMore) endIdx = currentPage * itemsPerPage;
  const slicedMovies = movies.slice(0, endIdx);
  const remainingForShows = Math.max(0, endIdx - totalMoviesAll);
  const slicedShows = shows.slice(0, remainingForShows);
  movies = slicedMovies;
  shows = slicedShows;
  const remainingCount = Math.max(0, totalItemsAll - endIdx);
  const showLoadMoreButton = shouldShowLoadMore && (remainingCount > 0);

  if(viewMode==='grid'){
   let html='<div class="grid-view">';
   
   movies.forEach(m=>{
    const poster=m.poster?`<img src="${m.poster}" class="poster" loading="lazy" decoding="async">`:`<div class="poster-placeholder">🎬</div>`;
    const movieNameEnc = encodeURIComponent(m.name || '').split("'").join('%27');
    const yearArg = (m.year === null || m.year === undefined) ? 'null' : m.year;
    const lastWatch = (m.watches && m.watches[0] && m.watches[0].timestamp) ? m.watches[0].timestamp : '';
    const lastDateTime = lastWatch ? formatDateTimeAmPm(lastWatch) : '';
    const gkey = gridKeyMovie(m.name, m.year);
    const gkeyAttr = gkey.replace(/"/g, '&quot;');
    const gkeyJs = encodeURIComponent(gkey).split("'").join('%27');
    const gsel = selectedGridItems.has(gkey) ? ' grid-selected' : '';
    const gchk = selectedGridItems.has(gkey) ? 'checked' : '';
     const itemJson = JSON.stringify(m).replace(/"/g, '&quot;');
     const gridOnclick = gridSelectMode ? `event.stopPropagation();toggleGridItemSelection(decodeURIComponent('${gkeyJs}'))` : `gridMovieClick(decodeURIComponent('${movieNameEnc}'),${yearArg})`;
     const checkboxOnclick = `event.stopPropagation();toggleGridItemSelection(decodeURIComponent('${gkeyJs}'))`;
     const jfBtnGridMovie = m.jellyfin_id
       ? `<button class="btn linkout small" title="Open in Jellyfin" onclick="event.stopPropagation();openInJellyfin(decodeURIComponent('${encodeURIComponent(String(m.jellyfin_id))}'))">🔗</button>`
       : ``;
     
     html += `<div class="grid-item${gsel}" data-grid-key="${gkeyAttr}" onclick="${gridOnclick}"><div class="grid-checkbox-wrap"><input type="checkbox" class="grid-select-checkbox" ${gchk} onclick="${checkboxOnclick}"></div><div class="grid-item-actions"><button class="btn manual small" title="Upload poster" onclick="event.stopPropagation();openPosterModal(${itemJson},'movie')">📷</button>${jfBtnGridMovie}<button class="btn danger small" onclick="event.stopPropagation();deleteMovie(decodeURIComponent('${movieNameEnc}'),${yearArg})">✕</button></div><div class="poster-container">${poster}</div><div class="grid-item-title">${m.name}</div><div class="grid-item-info">${m.year||''} • ${m.watch_count}x</div><div class="grid-item-info">${lastDateTime ? `Last: ${lastDateTime}` : ``}</div></div>`;
   });
   
   shows.forEach(s=>{
    const poster=s.poster?`<img src="${s.poster}" class="poster" loading="lazy" decoding="async">`:`<div class="poster-placeholder">📺</div>`;
    const comp=s.completion_percentage||0;
    const seasons=s.seasons||[];
    const seasonInfo=seasons.map(se=>`S${se.season_number}`).join(', ');
    const lastInfo = getLastEpisodeInfo(s);
    const lastGridDate = formatDateTimeAmPm(lastInfo && lastInfo.timestamp);
    const lastGrid = lastInfo ? `Last: S${lastInfo.season}E${lastInfo.episode}${lastGridDate ? ` | ${lastGridDate}` : ''}` : ''; 
    const seriesNameEnc = encodeURIComponent(s.series_name).split("'").join('%27');
    const key = `show|${s.series_name}`;
    const keyAttr = key.replace(/"/g, '&quot;');
    const keyJs = encodeURIComponent(key).split("'").join('%27');
    const isSelected = selectedGridItems.has(key);
    const allAdded = s.auto_all_added;
    const gsel = isSelected ? ' grid-selected' : '';
    const showOnclick = gridSelectMode ? `event.stopPropagation();toggleGridItemSelection(decodeURIComponent('${keyJs}'))` : `gridItemClick(decodeURIComponent('${seriesNameEnc}'),true)`;
     const showCheckbox = `event.stopPropagation();toggleGridItemSelection(decodeURIComponent('${keyJs}'))`;
     const jfGridShowId = s.jellyfin_series_id || (lastInfo && lastInfo.jellyfin_id) || null;
     const jfBtnGridShow = jfGridShowId
       ? `<button class="btn linkout small" title="Open in Jellyfin" onclick="event.stopPropagation();openInJellyfin(decodeURIComponent('${encodeURIComponent(String(jfGridShowId))}'))">🔗</button>`
       : ``;
     
     html+=`<div class="grid-item${gsel}" data-grid-key="${keyAttr}" onclick="${showOnclick}">${gridSelectMode ? `<div class="grid-checkbox-wrap"><input type="checkbox" class="grid-select-checkbox" ${isSelected ? 'checked' : ''} onclick="${showCheckbox}" ></div>` : ``}<div class="grid-item-actions"><button class="btn success small" title="Mark completed" onclick="event.stopPropagation();markComplete(decodeURIComponent('${seriesNameEnc}'))">✓</button>${jfBtnGridShow}<button class="btn danger small" title="Delete show" onclick="event.stopPropagation();deleteShow(decodeURIComponent('${seriesNameEnc}'))">🗑️</button></div><div class="poster-container">${poster}</div><div class="grid-item-title">${s.series_name}</div><div class="grid-item-info">${seasonInfo}</div>${lastGrid ? `<div class="grid-item-info">${lastGrid}</div>` : ``}<div class="grid-item-info">${allAdded ? `<span class="badge badge-info">All episodes added</span>` : ``} <span class="badge ${comp>=100 ? 'badge-success' : 'badge-warning'}">${comp}%</span></div></div>`;
   });
  
  html+='</div>';
  
  if(showLoadMoreButton){
    html += `<div style="text-align:center;padding:30px">
      <button class="btn manual" onclick="loadMore()" style="font-size:16px;padding:15px 30px">
        📥 Load More (${remainingCount} remaining)
      </button>
    </div>`;
  }
  
  document.getElementById('content').innerHTML=html||'<div class="loading">No items found</div>';
  updateGridBulkBar();
  try { injectRatings(); } catch(e) { console.error(e); }
  try { updateResultsCounter(totalMoviesAll, totalShowsAll); } catch(e) {}
  return;
}

  
  let html='';
  movies.forEach(m=>{
    const poster = m.poster
      ? `<img src="${m.poster}" class="poster" loading="lazy" decoding="async">`
      : `<div class="poster-placeholder">🎬</div>`;
    const itemJson = JSON.stringify(m).replace(/"/g, '&quot;');
    const movieNameEnc = encodeURIComponent(m.name || '').split("'").join('%27');
    const yearArg = (m.year === null || m.year === undefined) ? 'null' : m.year;
    const watchCount = m.watch_count || 0;
    const lastWatch = (m.watches && m.watches[0] && m.watches[0].timestamp) ? m.watches[0].timestamp : '';
    const lastDateTime = lastWatch ? formatDateTimeAmPm(lastWatch) : '';
    const details = `${watchCount} watch${watchCount===1?'':'es'}${lastDateTime ? ` • Last: ${lastDateTime}` : ''}`;
    const genres = (m.genres || []).join(', ');
    
    const jfBtnMovie = m.jellyfin_id
      ? `<button class="btn linkout small" onclick="event.stopPropagation();openInJellyfin(decodeURIComponent('${encodeURIComponent(String(m.jellyfin_id))}'))">🔗 Jellyfin</button>`
      : ``;
    html += `<div class="movie-item" data-movie-name="${movieNameEnc}">
      <div class="poster-container">
        ${poster}
        <button class="upload-poster-btn" onclick="event.stopPropagation();openPosterModal(${itemJson},'movie')">📤</button>
      </div>
      <div class="item-content">
        <div class="movie-title">${m.name}${m.year ? ` (${m.year})` : ''}</div>
        <div class="movie-details">${details}</div>
        ${genres ? `<div class="movie-details">${genres}</div>` : ``}
        <div class="manage-actions">
          ${jfBtnMovie}
          <button class="btn danger small" onclick="event.stopPropagation();deleteMovie(decodeURIComponent('${movieNameEnc}'),${yearArg})">🗑️ Delete Movie</button>
        </div>
      </div>
    </div>`;
  });
  
  shows.forEach(s=>{
    const poster = s.poster
      ? `<img src="${s.poster}" class="poster" loading="lazy" decoding="async">`
      : `<div class="poster-placeholder">📺</div>`;
    const itemJson = JSON.stringify(s).replace(/"/g, '&quot;');
    const seriesNameEnc = encodeURIComponent(s.series_name || '').split("'").join('%27');
    const safeId = (s.series_name || '').replace(/[^a-zA-Z0-9]/g, '');
    const showId = `show-${safeId}`;
    const comp = s.completion_percentage || 0;
    const compBadge = comp >= 100 ? 'badge-success' : 'badge-warning';
    const totalPossible = getTotalPossibleEpisodes(s);
    const totalWatched = s.total_episodes || 0;
    const detailHtml = totalPossible
      ? `<span class="episode-totals">${totalWatched}/${totalPossible} episodes watched</span>`
      : (
          s.tmdb_pending
            ? `<span class="episode-totals tmdb-pending-total" data-series-name="${seriesNameEnc}" title="Double-click (desktop) or press and hold (touch) to set total episodes">${totalWatched}/? episodes watched (fetching TMDB totals...)</span>`
            : `<span class="episode-totals">${totalWatched} episodes watched</span>`
        );
    const lastInfo = getLastEpisodeInfo(s);
    const lastDate = formatDateTimeAmPm(lastInfo && lastInfo.timestamp);
    const lastLine = lastInfo ? `Last watched: S${lastInfo.season}E${lastInfo.episode}${lastInfo.name ? ` - ${lastInfo.name}` : ``}${lastDate ? ` | ${lastDate}` : ''}` : ``;
    const allAddedBadge = s.auto_all_added ? `<span class="badge badge-info">All episodes added</span>` : ``;
    const newContentBadge = s.has_new_content ? `<span class="badge badge-alert">New Content</span>` : ``;
    const jfShowId = s.jellyfin_series_id || (lastInfo && lastInfo.jellyfin_id) || null;
    const jfBtnShow = jfShowId
      ? `<button class="btn linkout small" onclick="event.stopPropagation();openInJellyfin(decodeURIComponent('${encodeURIComponent(String(jfShowId))}'))">🔗 Jellyfin</button>`
      : ``;
    
    let seasonsHtml = '';
    (s.seasons || []).forEach(season => {
      const sid = `season-${safeId}-${season.season_number}`;
      const spct = season.completion_percentage || 0;
      const spctBadge = spct >= 100 ? 'badge-success' : 'badge-warning';
      const seasonTitle = season.name || `Season ${season.season_number}`;
      const seasonTotal = season.total_episodes || season.episode_count || 0;
      const seasonDetail = seasonTotal ? `${season.episode_count}/${seasonTotal} eps` : `${season.episode_count} eps`;
      
      let episodesHtml = '';
      (season.episodes || []).forEach(ep => {
        const key = `${s.series_name}_${season.season_number}_${ep.episode}`;
        const checkboxId = `chk_${safeId}_${season.season_number}_${ep.episode}`;
        const isSelected = selectedEpisodes.has(key);
        const selectedClass = isSelected ? ' selected' : '';
        const checked = isSelected ? 'checked' : '';
        const rewatched = ep.watch_count > 1 ? ` <span class="badge badge-info">Rewatched ${ep.watch_count}x</span>` : '';
        const jfEpBtn = ep.jellyfin_id
          ? `<button class="btn linkout small" title="Open in Jellyfin" onclick="event.stopPropagation();openInJellyfin(decodeURIComponent('${encodeURIComponent(String(ep.jellyfin_id))}'))">🔗</button>`
          : ``;
        episodesHtml += `<div class="episode-item${selectedClass}" onclick="event.stopPropagation();">
          <span>
            <input type="checkbox" class="select-checkbox" id="${checkboxId}" ${checked} onclick="event.stopPropagation();toggleEpisodeSelection(decodeURIComponent('${seriesNameEnc}'),${season.season_number},${ep.episode},event);return false;">
            <label for="${checkboxId}" style="cursor:pointer;user-select:none;">E${ep.episode}: ${ep.name}${rewatched}</label>
          </span>
          <span style="display:flex;gap:6px;align-items:center;">
            ${jfEpBtn}
            <button class="btn danger small delete-btn" onclick="event.stopPropagation();deleteEpisode(decodeURIComponent('${seriesNameEnc}'),${season.season_number},${ep.episode})">🗑️</button>
          </span>
        </div>`;
      });
      if(!episodesHtml){
        episodesHtml = `<div class="loading">No episodes</div>`;
      }
      
      seasonsHtml += `<div class="season-group">
        <div class="season-header" onclick="toggle('${sid}')">
          <div class="season-title">${seasonTitle}</div>
          <div style="display:flex;gap:6px;align-items:center;">
            <span class="badge ${spctBadge}">${spct}%</span>
            <span class="badge badge-info">${seasonDetail}</span>
            <button class="btn warning small" onclick="event.stopPropagation();markSeasonComplete(decodeURIComponent('${seriesNameEnc}'),${season.season_number})">✓ Mark 100%</button>
            <button class="btn danger small" onclick="event.stopPropagation();deleteSeason(decodeURIComponent('${seriesNameEnc}'),${season.season_number})">🗑️ Delete Season</button>
          </div>
        </div>
        <div class="episodes-list" id="${sid}" style="display:none">${episodesHtml}</div>
      </div>`;
    });
    
    html += `<div class="show-group" data-show-name="${seriesNameEnc}">
      <div class="poster-container">
        ${poster}
        <button class="upload-poster-btn" onclick="event.stopPropagation();openPosterModal(${itemJson},'tv')">📤</button>
      </div>
      <div class="item-content">
        <div class="show-header" onclick="toggle('${showId}')">
          <div>
            <div class="show-title">${s.series_name}</div>
            <div class="movie-details">${detailHtml}</div>
            ${lastLine ? `<div class="movie-details">${lastLine}</div>` : ``}
          </div>
          <div style="display:flex;gap:6px;align-items:center;">
            ${newContentBadge}
            ${allAddedBadge}
            <span class="badge ${compBadge}">${comp}%</span>
          </div>
        </div>
        <div class="progress-bar"><div class="progress-fill" style="width:${Math.min(comp,100)}%"></div></div>
        <div class="manage-actions" style="margin-top:8px;display:flex;gap:6px;flex-wrap:wrap;">
          ${jfBtnShow}
          <button class="btn manual small" onclick="event.stopPropagation();openAddEpisode(decodeURIComponent('${seriesNameEnc}'))">➕ Episode</button>
          <button class="btn manual small" onclick="event.stopPropagation();openAddSeason(decodeURIComponent('${seriesNameEnc}'))">➕ Season</button>
          <button class="btn warning small" onclick="event.stopPropagation();markComplete(decodeURIComponent('${seriesNameEnc}'))">✓ Mark 100%</button>
          <button class="btn danger small" onclick="event.stopPropagation();deleteShow(decodeURIComponent('${seriesNameEnc}'))">🗑️ Delete</button>
        </div>
        <div class="seasons-list" id="${showId}">${seasonsHtml}</div>
      </div>
    </div>`;
  });

  if(showLoadMoreButton){
    html += `<div style="text-align:center;padding:30px">
      <button class="btn manual" onclick="loadMore()" style="font-size:16px;padding:15px 30px">
        📥 Load More (${remainingCount} remaining)
      </button>
    </div>`;
  }
  
  document.getElementById('content').innerHTML=html||'<div class="loading">No items found</div>';
  updateBulkBar();
  try { injectRatings(); } catch(e) { console.error(e); }
  try { updateResultsCounter(totalMoviesAll, totalShowsAll); } catch(e) {}
  try { restoreOpenPanels(); } catch(e) {}
}

function loadMore() {
  const prevScroll = window.scrollY;
  currentPage++;
  renderHistory();

  // keep user roughly in the same place (prevents overshoot feeling)
  requestAnimationFrame(() => {
    window.scrollTo({ top: prevScroll, behavior: 'instant' });
  });
}

async function renderGenres(){
  const renderToken = ++genreRenderToken;
  const genres=data.genres||[];
  const breakdown=data.genre_breakdown||{};
  
  if(genreChart)genreChart.destroy();
  
  const fetchInsights = () => fetchJsonWithTimeout('/api/genre_insights', INSIGHTS_FETCH_TIMEOUT_MS);
  const insightsPromise = fetchInsights()
    .then(j => {
      if(renderToken !== genreRenderToken || currentTab !== 'genres') return insights;
      insights = (j && typeof j === 'object') ? j : insights;
      return insights;
    })
    .catch(async e => {
      // One fast retry helps when the tab is opened during a transient reconnect.
      try {
        await new Promise(res => setTimeout(res, 350));
        const j = await fetchInsights();
        if(renderToken !== genreRenderToken || currentTab !== 'genres') return insights;
        insights = (j && typeof j === 'object') ? j : insights;
        return insights;
      } catch (e2) {
        if(renderToken === genreRenderToken && currentTab === 'genres'){
          console.error('Failed to load insights:', (e2 && e2.message) || (e && e.message) || e2 || e);
        }
        return insights;
      }
    });

  function buildGenreInsightsParts(insights){
    let combosHtml = '';
    let moodsHtml = '';
    if(insights && insights.success) {
      if(insights.combos && insights.combos.length > 0) {
        combosHtml += `<div style="background:linear-gradient(135deg,rgba(94,245,224,.13),rgba(94,245,224,.06));padding:14px;border-radius:12px;border:1px solid rgba(94,245,224,.35)">`;
        combosHtml += `<h3 style="margin:0 0 10px 0;color:#5ef5e0;font-size:0.98em;font-weight:700">Genre Combos You Love</h3>`;
        insights.combos.forEach(c => {
          const comboGenres = Array.isArray(c.genres) ? c.genres : [];
          const comboLabel = comboGenres.join(' + ');
          const comboParam = encodeURIComponent(comboGenres.join('|'));
          combosHtml += `<div style="margin:5px 0;font-size:0.9em">
            <button class="btn secondary small" onclick="filterByComboEncoded('${comboParam}')" style="padding:6px 12px;font-size:0.85em">
              ${escapeHtml(comboLabel)} (${c.count} items)
            </button>
          </div>`;
        });
        combosHtml += `</div>`;
      }

      if(insights.moods && Object.keys(insights.moods).length > 0) {
        moodsHtml += `<div style="background:linear-gradient(135deg,rgba(255,107,157,.13),rgba(255,107,157,.06));padding:14px;border-radius:12px;border:1px solid rgba(255,107,157,.35)">`;
        moodsHtml += `<h3 style="margin:0 0 10px 0;color:#ff6b9d;font-size:0.98em;font-weight:700">Mood Explorer</h3>`;
        const moodIcons = {
          intense: '⚡',
          fun: '😄',
          emotional: '💔',
          exciting: '🚀',
          scary: '👻'
        };
        for(const [mood, info] of Object.entries(insights.moods)) {
          const icon = moodIcons[mood] || '🎬';
          moodsHtml += `<div style="margin:8px 0">
            <button class="btn secondary small" onclick="filterByMood('${mood}')" style="padding:6px 12px;font-size:0.85em">
              ${icon} Feeling ${mood}? (${info.count} items)
            </button>
          </div>`;
        }
        moodsHtml += `</div>`;
      }
    }
    return { combosHtml, moodsHtml };
  }
  
  // Genres dashboard layout:
  // Row 1: recommendations (left) + pie chart (right)
  // Row 2: genre combos + mood explorer (side by side)
  // Row 3: full genre list
  const colsTop = window.innerWidth < 1100 ? '1fr' : 'minmax(0,1fr) 420px';
  const colsMid = window.innerWidth < 900 ? '1fr' : '1fr 1fr';
  let layoutHtml = `
    <div style="display:grid;gap:18px;align-items:start">
      <div id="genres-top" style="display:grid;grid-template-columns:${colsTop};gap:18px;align-items:start">
        <section style="min-width:0">
          <div id="recommendations-container" style="background:linear-gradient(180deg,rgba(25,33,60,.86),rgba(17,24,46,.86));padding:14px;border-radius:14px;border:1px solid rgba(139,156,255,.28)">
            <div class="loading">Loading recommendations...</div>
          </div>
        </section>
        <aside style="position:sticky;top:14px;align-self:start;display:grid;gap:12px;height:fit-content">
          <div style="background:linear-gradient(180deg,rgba(16,22,40,.92),rgba(11,16,31,.92));padding:14px;border-radius:14px;border:1px solid rgba(139,156,255,.25)">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
              <strong style="font-size:0.95rem">Genre Distribution</strong>
              <span style="font-size:0.8rem;opacity:0.75">Click chart slices to jump</span>
            </div>
            <canvas id="genreChart" width="420" height="420"></canvas>
          </div>
        </aside>
      </div>

      <section style="display:grid;grid-template-columns:${colsMid};gap:12px">
        <div id="genre-combos-container"></div>
        <div id="genre-moods-container"></div>
      </section>

      <section style="min-width:0">
        <div id="genre-list-container" style="display:grid;gap:10px"></div>
      </section>
    </div>
  `;
  
  document.getElementById('genres-content').innerHTML = layoutHtml;

  // Render the list/chart immediately; combos/moods and recommendations load in the background.
  try {
    const combosEl = document.getElementById('genre-combos-container');
    const moodsEl = document.getElementById('genre-moods-container');
    if(combosEl) combosEl.innerHTML = `<div class="loading" style="padding:10px 0">Loading genre combos...</div>`;
    if(moodsEl) moodsEl.innerHTML = `<div class="loading" style="padding:10px 0">Loading moods...</div>`;

    if(combosEl || moodsEl){
      const insightsTimeout = setTimeout(() => {
        if(renderToken !== genreRenderToken || currentTab !== 'genres') return;
        if(combosEl && combosEl.innerHTML && combosEl.innerHTML.includes('Loading genre combos')){
          combosEl.innerHTML = `<div class="loading" style="padding:10px 0">Genre combos unavailable</div>`;
        }
        if(moodsEl && moodsEl.innerHTML && moodsEl.innerHTML.includes('Loading moods')){
          moodsEl.innerHTML = `<div class="loading" style="padding:10px 0">Mood explorer unavailable</div>`;
        }
      }, 6000);
      insightsPromise.then(insights => {
        if(renderToken !== genreRenderToken || currentTab !== 'genres') return;
        const parts = buildGenreInsightsParts(insights);
        if(combosEl) combosEl.innerHTML = parts.combosHtml || `<div class="loading" style="padding:10px 0">No genre combos available</div>`;
        if(moodsEl) moodsEl.innerHTML = parts.moodsHtml || `<div class="loading" style="padding:10px 0">No mood data available</div>`;
      }).finally(() => clearTimeout(insightsTimeout));
    }
  } catch(e) {}
  
  // Small delay to ensure canvas is in DOM
  setTimeout(() => {
    if(renderToken !== genreRenderToken || currentTab !== 'genres') return;
    const canvas = document.getElementById('genreChart');
    if(!canvas) {
      console.error('Canvas not found!');
      return;
    }
    
    const ctx = canvas.getContext('2d');
    // Before creating a new chart, destroy any existing chart instance associated
    // with this canvas.  Chart.js maintains a registry keyed by the canvas ID,
    // and attempting to instantiate a second chart on the same element will
    // trigger a "Chart id already in use" error.  Use Chart.getChart if
    // available to retrieve and destroy the existing chart.  Also destroy
    // our own genreChart reference if set.
    try {
      if(typeof Chart !== 'undefined' && typeof Chart.getChart === 'function') {
        const existing = Chart.getChart('genreChart');
        if(existing) existing.destroy();
      }
    } catch(e) {
      // ignore errors from Chart.getChart
    }
    if(genreChart) {
      try { genreChart.destroy(); } catch(e) {}
      genreChart = null;
    }
    // Define a palette of colors for the genre chart. If there are more genres than colors,
    // cycle through the palette so each segment still has a visible color. Chart.js expects
    // the number of backgroundColor entries to match or exceed the data length; otherwise
    // segments may not render correctly. Construct a list matching genres.length.
    const baseColors=['#8b9cff','#ff8585','#6bcf7f','#ffd93d','#5ef5e0','#ff6b9d','#c78bff','#ffb86c','#8bffd9','#ff8b8b'];
    const bgColors = genres.map((_, idx) => baseColors[idx % baseColors.length]);

    genreChart=new Chart(ctx,{
      type:'doughnut',
      data:{
        labels:genres.map(g=>g.genre),
        datasets:[{
          data:genres.map(g=>g.count),
          backgroundColor:bgColors,
          borderColor:'rgba(20,25,45,0.8)',
          borderWidth:3
        }]
      },
      options:{
        responsive:true,
        maintainAspectRatio:true,
        onClick: (evt, activeEls) => {
          if(activeEls.length > 0) {
            const index = activeEls[0].index;
            const genre = genres[index].genre;
            scrollToGenre(genre);
          }
        },
        plugins:{
          legend:{
            position:'bottom',
            labels:{
              color:'#fff',
              font:{size:12,weight:'600'},
              padding:10
            }
          },
          tooltip:{
            backgroundColor:'rgba(20,25,45,0.95)',
            titleFont:{size:14,weight:'700'},
            bodyFont:{size:12},
            padding:8,
            cornerRadius:8
          }
        }
      }
    });
  }, 100);
  
  let html='';

  // Genre list
  genres.forEach(g=>{
    const b=breakdown[g.genre]||{movies:[],shows:[]};
    const genreName = String(g.genre || '');
    const genreKey = genreKeyFromName(genreName);
    const movieNames=b.movies.slice(0,3).map(m=>escapeHtml(m.name || m.title || 'Unknown')).join(', ');
    const showNames=b.shows.slice(0,3).map(s=>escapeHtml(s.series_name || s.name || 'Unknown')).join(', ');
    let items='';
    if(movieNames)items+=`<strong>Movies:</strong> ${movieNames}${b.movies.length>3?' & '+(b.movies.length-3)+' more':''}<br>`;
    if(showNames)items+=`<strong>Shows:</strong> ${showNames}${b.shows.length>3?' & '+(b.shows.length-3)+' more':''}`;
    
    html+=`<div class="genre-item" id="genre-item-${genreKey}" style="background:rgba(20,25,45,0.72);border:1px solid rgba(139,156,255,0.2);border-radius:12px;padding:12px">
      <div class="genre-header" style="cursor:pointer;display:flex;justify-content:space-between;align-items:center;gap:10px" onclick="toggleGenreDetailsByKey('${genreKey}')">
        <div style="display:flex;align-items:center;gap:10px;min-width:0">
          <span id="genre-arrow-${genreKey}" style="transition:transform 0.3s;display:inline-block;opacity:0.85">&#9654;</span>
          <div class="genre-name" style="font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${escapeHtml(genreName)}</div>
        </div>
        <div class="genre-count" style="font-size:0.82rem;padding:4px 10px;border-radius:999px;background:rgba(139,156,255,0.18);border:1px solid rgba(139,156,255,0.35)">${g.count} items</div>
      </div>
      <div id="genre-details-${genreKey}" class="genre-items" style="display:none;margin-top:10px;padding-top:10px;border-top:1px solid rgba(255,255,255,0.08)">${items||'No items'}</div>
    </div>`;
  });
  
  document.getElementById('genre-list-container').innerHTML=html||'<div class="loading">No genre data available</div>';

  // Reset recommendation offsets and lists when the Genres tab is opened.
  // Each category (movies and shows) maintains its own offset so the
  // user can request additional recommendations independently.  Clear
  // any previously stored results before loading a fresh batch.
  recommendationOffsetMovies = 0;
  recommendationOffsetShows = 0;
  currentMoviesRecs = [];
  currentShowsRecs = [];
  const recEl = document.getElementById('recommendations-container');
  if(recEl){
    recEl.innerHTML = `<div class="loading">Loading recommendations...</div>`;
  }
  recommendationsRequestToken += 1;
  // Trigger recommendations immediately after the genres UI is mounted.
  setTimeout(() => {
    if(currentTab === 'genres' && renderToken === genreRenderToken){
      loadRecommendations();
    }
  }, 0);
}

function renderGenresLazy(){
  const sig = data && data.sig ? data.sig : null;
  const hasLayout =
    !!document.getElementById('genre-list-container') &&
    !!document.getElementById('recommendations-container');
  if(hasLayout && genresRenderedSig && sig && genresRenderedSig === sig){
    return;
  }
  genresRenderedSig = sig || 'no-sig';
  renderGenres();
}

function scrollToGenre(genre) {
  const genreKey = genreKeyFromName(genre);
  const element = document.getElementById(`genre-item-${genreKey}`);
  if(element) {
    element.scrollIntoView({behavior: 'smooth', block: 'center'});
    toggleGenreDetailsByKey(genreKey);
    // Highlight briefly
    element.style.background = 'rgba(139,156,255,0.3)';
    setTimeout(() => {
      element.style.background = '';
    }, 1500);
  }
}

function filterByMood(mood) {
  // Set the mood filter and clear any genre-combo filter.  When a
  // mood is selected, recommendations will be filtered to items whose
  // genre lists intersect with the genres associated with the chosen
  // mood (see moodGenreMap).  Reset the recommendation offset so
  // fresh results are loaded, then load recommendations.  We still
  // expand the associated genres and scroll to the first one for
  // context.
  currentMoodFilter = mood;
  currentGenreComboFilter = null;
  // Reset both recommendation offsets when a new mood is selected.  This
  // ensures that the first batch of filtered results is loaded for both
  // movies and shows.
  recommendationOffsetMovies = 0;
  recommendationOffsetShows = 0;
  // Jump to recommendations and reload to apply the filter.
  scrollToRecommendations();
  loadRecommendations();
}

function filterByComboEncoded(encodedComboString) {
  filterByCombo(genreNameFromKey(encodedComboString));
}

// Apply a genre-combo filter to recommendations.  The comboString
// parameter should be a string with individual genres separated by a
// vertical bar (e.g. "Action|Adventure").  The filter will be applied
// such that only movies and shows containing all of the specified
// genres will appear in the recommendations section.  Selecting a
// combo clears any active mood filter.  The recommendationOffset is
// reset so a fresh batch of results is loaded.
function filterByCombo(comboString) {
  // Parse the combo string into an array of genres
  // Split the combo string on the "|" character and trim whitespace from each
  const genres = (comboString || '').split('|').map(g => g.trim()).filter(g => g);
  if(genres.length === 0) {
    return;
  }
  currentGenreComboFilter = genres;
  currentMoodFilter = null;
  // Reset both recommendation offsets when a new genre combo is selected.
  // This ensures fresh batches of filtered results for both categories.
  recommendationOffsetMovies = 0;
  recommendationOffsetShows = 0;
  // Jump to recommendations and reload to apply the filter.
  scrollToRecommendations();
  loadRecommendations();
}

function toggleGenreDetails(genre) {
  return toggleGenreDetailsByKey(genreKeyFromName(genre));
}

function toggleGenreDetailsByKey(genreKey) {
  const details = document.getElementById(`genre-details-${genreKey}`);
  const arrow = document.getElementById(`genre-arrow-${genreKey}`);
  if(!details) return;
  const genre = genreNameFromKey(genreKey);
  // Fetch full breakdown from the global data so we can render all movies and shows
  const breakdown = (data && data.genre_breakdown) || {};
  const b = breakdown[genre] || { movies: [], shows: [] };
  
  if(details.style.display === 'none') {
    // Build a complete list of movies and shows for this genre.  We only do
    // this when expanding the section to avoid unnecessary DOM updates.
    let contentHtml = '';
    // Movies section
    contentHtml += '<div style="margin-bottom:12px">';
    contentHtml += '<div style="font-weight:600;color:#8b9cff;margin-bottom:6px">🎬 Movies</div>';
    if(b.movies && b.movies.length > 0) {
      // Container with scroll if there are many items.  Limit height to
      // prevent the list from overtaking the page; overflow-y allows
      // scrolling within the container.
      contentHtml += '<div style="max-height:200px; overflow-y:auto; padding-left:10px">';
      b.movies.forEach(m => {
        const title = escapeHtml(m.name || m.title || 'Unknown');
        const year = m.year ? ` (${m.year})` : '';
        // Bolden the movie name so that it stands out better in the list
        contentHtml += `<div style="font-weight:600;margin-bottom:6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${title}${year}</div>`;
      });
      contentHtml += '</div>';
    } else {
      contentHtml += '<div style="color:rgba(255,255,255,0.5);padding-left:10px">No movies</div>';
    }
    contentHtml += '</div>';
    // Shows section
    contentHtml += '<div>';
    contentHtml += '<div style="font-weight:600;color:#5ef5e0;margin-bottom:6px">📺 Shows</div>';
    if(b.shows && b.shows.length > 0) {
      contentHtml += '<div style="max-height:200px; overflow-y:auto; padding-left:10px">';
      b.shows.forEach(s => {
        const title = escapeHtml(s.series_name || s.name || 'Unknown');
        const year = s.year ? ` (${s.year})` : '';
        // Bolden the show name so that it stands out better in the list
        contentHtml += `<div style="font-weight:600;margin-bottom:6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${title}${year}</div>`;
      });
      contentHtml += '</div>';
    } else {
      contentHtml += '<div style="color:rgba(255,255,255,0.5);padding-left:10px">No shows</div>';
    }
    contentHtml += '</div>';
    details.innerHTML = contentHtml;
    details.style.display = 'block';
    if(arrow) arrow.style.transform = 'rotate(90deg)';
  } else {
    // Collapse the details
    details.style.display = 'none';
    if(arrow) arrow.style.transform = 'rotate(0deg)';
  }
}

function renderRecommendationsPanel(){
  const recEl = document.getElementById('recommendations-container');
  if(!recEl) return;

  let heading;
  if(currentMoodFilter){
    heading = `🎭 Recommendations for your ${currentMoodFilter} mood`;
  } else if(currentGenreComboFilter){
    heading = `🎬 Recommendations: ${currentGenreComboFilter.join(' + ')}`;
  } else if(lastRecsTopGenre){
    heading = `🎯 Recommended for you (Based on ${lastRecsTopGenre})`;
  } else {
    heading = `🎯 Recommended for you`;
  }

  const loadingAll = lastRecsLoadingSection === 'all';
  const loadingMovies = loadingAll || lastRecsLoadingSection === 'movies';
  const loadingShows  = loadingAll || lastRecsLoadingSection === 'shows';

  let msgHtml = '';
  if(lastRecsError){
    msgHtml = `<div style="margin:10px 0 14px 0;padding:10px 12px;border-radius:10px;background:rgba(255,107,157,.10);border:1px solid rgba(255,107,157,.28);color:rgba(255,255,255,0.85);font-size:0.9em">
      ${escapeHtml(lastRecsError)}
    </div>`;
  }

  const moviesBtnDisabled = loadingMovies ? 'disabled' : '';
  const showsBtnDisabled  = loadingShows ? 'disabled' : '';

  let recHtml = ``;
  recHtml += `<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px">
    <h3 style="margin:0;color:#8b9cff">${heading}</h3>
  </div>`;
  recHtml += msgHtml;
  recHtml += '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:20px">';

  // Movies column
  recHtml += '<div style="display:flex;flex-direction:column">';
  recHtml += `<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
    <h4 style="margin:0;color:rgba(255,255,255,0.8)">🎬 Movies</h4>
    <button class="btn secondary small" onclick="refreshMovieRecommendations()" ${moviesBtnDisabled} style="padding:6px 12px;${loadingMovies ? 'opacity:.65;cursor:default;' : ''}">🔄 More Movies${loadingMovies ? ' ⏳' : ''}</button>
  </div>`;
  if(loadingMovies && currentMoviesRecs.length === 0){
    recHtml += '<div class="loading">Loading movies...</div>';
  } else if(currentMoviesRecs.length === 0) {
    if(currentMoodFilter || currentGenreComboFilter) {
      recHtml += '<p style="color:rgba(255,255,255,0.5)">No movies match this selection</p>';
    } else if(recommendationOffsetMovies === 0) {
      recHtml += '<p style="color:rgba(255,255,255,0.5)">No unwatched movies found in Radarr</p>';
    } else {
      recHtml += '<p style="color:rgba(255,255,255,0.5)">No more movies to show</p>';
    }
  } else {
    currentMoviesRecs.forEach(m => {
      recHtml += `<div style="background:rgba(20,25,45,0.8);padding:15px;border-radius:12px;margin-bottom:10px;border-left:3px solid #ff8585">
        <div style="font-weight:700;margin-bottom:5px">${escapeHtml(m.title || '')} ${m.year ? '('+escapeHtml(String(m.year))+')' : ''}</div>
        <div style="font-size:0.85em;color:rgba(255,255,255,0.6);margin-bottom:8px">${escapeHtml(m.overview || '')}</div>
        <div>${(m.genres||[]).slice(0,3).map(g => `<span class="badge badge-info">${escapeHtml(g)}</span>`).join(' ')}</div>
      </div>`;
    });
  }
  recHtml += '</div>';

  // Shows column
  recHtml += '<div style="display:flex;flex-direction:column">';
  recHtml += `<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
    <h4 style="margin:0;color:rgba(255,255,255,0.8)">📺 Shows</h4>
    <button class="btn secondary small" onclick="refreshShowRecommendations()" ${showsBtnDisabled} style="padding:6px 12px;${loadingShows ? 'opacity:.65;cursor:default;' : ''}">🔄 More Shows${loadingShows ? ' ⏳' : ''}</button>
  </div>`;
  if(loadingShows && currentShowsRecs.length === 0){
    recHtml += '<div class="loading">Loading shows...</div>';
  } else if(currentShowsRecs.length === 0) {
    if(currentMoodFilter || currentGenreComboFilter) {
      recHtml += '<p style="color:rgba(255,255,255,0.5)">No shows match this selection</p>';
    } else if(recommendationOffsetShows === 0) {
      recHtml += '<p style="color:rgba(255,255,255,0.5)">No unwatched shows found in Sonarr</p>';
    } else {
      recHtml += '<p style="color:rgba(255,255,255,0.5)">No more shows to show</p>';
    }
  } else {
    currentShowsRecs.forEach(s => {
      recHtml += `<div style="background:rgba(20,25,45,0.8);padding:15px;border-radius:12px;margin-bottom:10px;border-left:3px solid #5ef5e0">
        <div style="font-weight:700;margin-bottom:5px">${escapeHtml(s.title || '')} ${s.year ? '('+escapeHtml(String(s.year))+')' : ''}</div>
        <div style="font-size:0.85em;color:rgba(255,255,255,0.6);margin-bottom:8px">${escapeHtml(s.overview || '')}</div>
        <div>${(s.genres||[]).slice(0,3).map(g => `<span class="badge badge-info">${escapeHtml(g)}</span>`).join(' ')}</div>
      </div>`;
    });
  }
  recHtml += '</div>';
  recHtml += '</div>';

  recEl.innerHTML = recHtml;
}

async function loadRecommendations(section) {
  // On first paint, show cached recs instantly if available.
  if((!currentMoviesRecs || currentMoviesRecs.length === 0) && (!currentShowsRecs || currentShowsRecs.length === 0)){
    const cached = getCachedRecs();
    if(cached && cached.success){
      lastRecsTopGenre = cached.top_genre || lastRecsTopGenre;
      if(Array.isArray(cached.movies)) currentMoviesRecs = cached.movies;
      if(Array.isArray(cached.shows)) currentShowsRecs = cached.shows;
      try { renderRecommendationsPanel(); } catch(e) {}
    }
  }

  try {
    const reqToken = ++recommendationsRequestToken;
    let offset;
    if(section === 'movies') {
      offset = recommendationOffsetMovies;
    } else if(section === 'shows') {
      offset = recommendationOffsetShows;
    } else {
      offset = Math.max(recommendationOffsetMovies, recommendationOffsetShows);
    }

    lastRecsError = null;
    lastRecsLoadingSection = section || 'all';
    renderRecommendationsPanel();

    const recs = await fetchJsonWithTimeout(`/api/genre_recommendations?offset=${offset}`, RECS_FETCH_TIMEOUT_MS);
    if(reqToken !== recommendationsRequestToken || currentTab !== 'genres') return;

    if(!recs || !recs.success) {
      lastRecsError = 'Recommendations temporarily unavailable (server busy)';
      lastRecsLoadingSection = null;
      renderRecommendationsPanel();
      return;
    }

    if(!recs.top_genre) {
      lastRecsTopGenre = null;
      lastRecsError = (currentMoodFilter || currentGenreComboFilter)
        ? 'No recommendations match this selection'
        : 'No recommendations available yet';
      lastRecsLoadingSection = null;
      renderRecommendationsPanel();
      return;
    }

    lastRecsTopGenre = recs.top_genre;
    storeCachedRecs(recs);

    let movies = Array.isArray(recs.movies) ? [...recs.movies] : [];
    let shows = Array.isArray(recs.shows) ? [...recs.shows] : [];

    if(currentMoodFilter) {
      const mg = moodGenreMap[currentMoodFilter] || [];
      const mgLower = mg.map(x => (x || '').toLowerCase());
      movies = movies.filter(m => ((m.genres||[]).map(x => (x||'').toLowerCase())).some(g => mgLower.includes(g)));
      shows = shows.filter(s => ((s.genres||[]).map(x => (x||'').toLowerCase())).some(g => mgLower.includes(g)));
    } else if(currentGenreComboFilter) {
      movies = movies.filter(m => {
        const gs = (m.genres || []).map(x => (x || '').toLowerCase());
        return currentGenreComboFilter.every(g => gs.includes((g || '').toLowerCase()));
      });
      shows = shows.filter(s => {
        const gs = (s.genres || []).map(x => (x || '').toLowerCase());
        return currentGenreComboFilter.every(g => gs.includes((g || '').toLowerCase()));
      });
    }

    if(!section || section === 'movies') currentMoviesRecs = movies;
    if(!section || section === 'shows') currentShowsRecs = shows;

    lastRecsError = null;
    lastRecsLoadingSection = null;
    renderRecommendationsPanel();
  } catch(e) {
    if(currentTab !== 'genres') return;
    console.error('Failed to load recommendations:', (e && e.message) ? e.message : e);
    lastRecsError = 'Recommendations temporarily unavailable (network slow)';
    lastRecsLoadingSection = null;
    try { renderRecommendationsPanel(); } catch(_) {}
  }
}

function refreshRecommendations() {
  // Increment both offsets and reload recommendations for both categories.
  recommendationOffsetMovies++;
  recommendationOffsetShows++;
  loadRecommendations();
}

// Fetch the next batch of movie recommendations.  When the user clicks
// the "More Movies" button this function is invoked.  It increments
// the movie offset and calls loadRecommendations() specifying the
// 'movies' section so that only the movies list is refreshed.
function refreshMovieRecommendations() {
  if(lastRecsLoadingSection) return;
  recommendationOffsetMovies++;
  loadRecommendations('movies');
}

// Fetch the next batch of show recommendations.  When the user clicks
// the "More Shows" button this function is invoked.  It increments
// the show offset and calls loadRecommendations() specifying the
// 'shows' section so that only the shows list is refreshed.
function refreshShowRecommendations() {
  if(lastRecsLoadingSection) return;
  recommendationOffsetShows++;
  loadRecommendations('shows');
}

function renderAnalytics(){
  const streak=data.watch_streak||{current:0,longest:0};
  const quick=data.quick_stats||{};
  const trending=data.trending||{movies:[],shows:[]};
  
  let statsHtml='';
  statsHtml+=`<div class="quick-stat-card"><div class="quick-stat-title">🔥 Current Streak</div><div class="quick-stat-value">${streak.current}</div><div class="quick-stat-label">days</div></div>`;
  statsHtml+=`<div class="quick-stat-card"><div class="quick-stat-title">🏆 Longest Streak</div><div class="quick-stat-value">${streak.longest}</div><div class="quick-stat-label">days</div></div>`;
  statsHtml+=`<div class="quick-stat-card"><div class="quick-stat-title">🔁 Most Rewatched</div><div class="quick-stat-value">${quick.most_rewatched?.count||0}x</div><div class="quick-stat-label">${quick.most_rewatched?.name||'N/A'}</div></div>`;
  statsHtml+=`<div class="quick-stat-card"><div class="quick-stat-title">🍿 Most Binged</div><div class="quick-stat-value">${quick.most_binged?.count||0}</div><div class="quick-stat-label">${quick.most_binged?.name||'N/A'}</div></div>`;
  // Display fastest completion in days. If no show qualifies (days === 0 or undefined) then show 'N/A'.
  const fcDays = quick.fastest_completion?.days;
  const fcName = quick.fastest_completion?.name || 'N/A';
  const fcDisplay = fcDays && fcDays > 0 ? fcDays : 'N/A';
  statsHtml+=`<div class="quick-stat-card"><div class="quick-stat-title">⚡ Fastest Completion</div><div class="quick-stat-value">${fcDisplay}</div><div class="quick-stat-label">${fcName}</div></div>`;
  statsHtml+=`<div class="quick-stat-card"><div class="quick-stat-title">📺 Avg Episodes/Day</div><div class="quick-stat-value">${quick.avg_episodes_per_day||0}</div><div class="quick-stat-label">episodes</div></div>`;
  
  document.getElementById('quick-stats').innerHTML=statsHtml;
  
  let trendingHtml='<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:15px">';
  trendingHtml+='<div><h4 style="margin-bottom:12px;color:rgba(255,255,255,0.7)">🎬 Movies</h4>';
  if(trending.movies.length===0){
    trendingHtml+='<p style="color:rgba(255,255,255,0.5)">No movies watched this week</p>';
  }else{
    trending.movies.forEach(m=>{
      trendingHtml+=`<div style="background:rgba(20,25,45,0.8);padding:12px 15px;border-radius:12px;margin-bottom:10px;border-left:3px solid #ff8585;display:flex;justify-content:space-between;align-items:center"><span>${m.name}</span><span class="badge badge-info">${m.count} watches</span></div>`;
    });
  }
  trendingHtml+='</div><div><h4 style="margin-bottom:12px;color:rgba(255,255,255,0.7)">📺 Shows</h4>';
  if(trending.shows.length===0){
    trendingHtml+='<p style="color:rgba(255,255,255,0.5)">No shows watched this week</p>';
  }else{
    trending.shows.forEach(s=>{
      trendingHtml+=`<div style="background:rgba(20,25,45,0.8);padding:12px 15px;border-radius:12px;margin-bottom:10px;border-left:3px solid #5ef5e0;display:flex;justify-content:space-between;align-items:center"><span>${s.name}</span><span class="badge badge-info">${s.count} episodes</span></div>`;
    });
  }
  trendingHtml+='</div></div>';
  document.getElementById('trending-content').innerHTML=trendingHtml;

  // Render watch time analytics chart and summary
  try { renderWatchTime(); } catch(e) { console.error(e); }
}

function renderProgress(){
  // Build movies and shows progress lists sorted by last watched time
  const movies=data.movies||[];
  const shows=data.shows||[];
  
  // Compute movie progress with last watched timestamp
  const movieProgress=movies.map(m=>{
    let lastWatched='';
    if(m.watches && m.watches.length>0){
      m.watches.forEach(w=>{
        const ts=w.timestamp||'';
        if(!lastWatched || ts>lastWatched) lastWatched=ts;
      });
    }
    return {...m,lastWatched};
  }).sort((a,b)=>{
    if(!a.lastWatched && !b.lastWatched) return 0;
    if(!a.lastWatched) return 1;
    if(!b.lastWatched) return -1;
    return b.lastWatched.localeCompare(a.lastWatched);
  });
  
  // Compute show progress with last watched timestamp and episodes left
  const showProgress=shows.map(s=>{
    let lastWatched='';
    if(s.seasons){
      Object.values(s.seasons).forEach(season=>{
        if(season.episodes){
          season.episodes.forEach(ep=>{
            if(ep.watches){
              ep.watches.forEach(w=>{
                const ts=w.timestamp||'';
                if(!lastWatched || ts>lastWatched) lastWatched=ts;
              });
            }
          });
        }
      });
    }
    const totalPossible = getTotalPossibleEpisodes(s);
    const totalWatched=s.total_episodes||0;
    const episodesLeft=Math.max(0,totalPossible-totalWatched);
    return {...s,lastWatched,episodesLeft};
  }).sort((a,b)=>{
    if(!a.lastWatched && !b.lastWatched) return 0;
    if(!a.lastWatched) return 1;
    if(!b.lastWatched) return -1;
    return b.lastWatched.localeCompare(a.lastWatched);
  });
  
  // Set search and sort inputs to reflect current preferences
  const mSearchInput=document.getElementById('moviesSearchInput');
  if(mSearchInput){mSearchInput.value=progressMoviesSearch;}
  const mSortSelect=document.getElementById('moviesSortSelect');
  if(mSortSelect){mSortSelect.value=progressMoviesSort;}
  const sSearchInput=document.getElementById('showsSearchInput');
  if(sSearchInput){sSearchInput.value=progressShowsSearch;}
  const sSortSelect=document.getElementById('showsSortSelect');
  if(sSortSelect){sSortSelect.value=progressShowsSort;}
  const sFilterSelect=document.getElementById('showsFilterSelect');
  if(sFilterSelect){sFilterSelect.value=progressShowsFilter;}

  // Filter and sort movies based on search and sort preferences
  let moviesFiltered = movieProgress.filter(m => (m.name || '').toLowerCase().includes(progressMoviesSearch || ''));
  if(progressMoviesSort === 'alphabetical'){
    moviesFiltered.sort((a,b)=> (a.name||'').localeCompare(b.name||''));
  } else if(progressMoviesSort === 'watchCount'){
    moviesFiltered.sort((a,b)=> (b.watch_count||0) - (a.watch_count||0));
  } else {
    // default recent sort by lastWatched
    moviesFiltered.sort((a,b)=>{
      if(!a.lastWatched && !b.lastWatched) return 0;
      if(!a.lastWatched) return 1;
      if(!b.lastWatched) return -1;
      return b.lastWatched.localeCompare(a.lastWatched);
    });
  }
  // Build movies HTML using filtered and sorted list with pagination and clickable cards
  let moviesHtml='';
  if(moviesFiltered.length===0){
    moviesHtml = '<p style="color:rgba(255,255,255,0.5)">No movies to display</p>';
  } else {
    const visibleMovies = moviesFiltered.slice(0, moviesProgressLimit);
    moviesHtml = '<div style="display:grid;gap:12px">';
    visibleMovies.forEach(m => {
      const comp = 100;
      const barColor = '#8b9cff';
      const escapedName = (m.name || '').replace(/'/g,"\\'");
      moviesHtml += `<div onclick="navigateToMovie('${escapedName}')" style="cursor:pointer;background:rgba(20,25,45,0.9);padding:15px;border-radius:12px;border-left:4px solid ${barColor}">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
          <strong style="font-size:1.1em">${m.name}</strong>
          <span class="badge badge-warning">${comp}%</span>
        </div>
        <div style="color:rgba(255,255,255,0.7);margin-bottom:8px">Watched ${m.watch_count} time${m.watch_count===1?'':'s'}</div>
        <div class="progress-bar"><div class="progress-fill" style="width:${comp}%;background:${barColor}"></div></div>
      </div>`;
    });
    moviesHtml += '</div>';
    if(moviesFiltered.length > moviesProgressLimit){
      moviesHtml += '<div style="text-align:center;margin-top:10px"><button class="btn small" onclick="loadMoreMovies()">Load More</button></div>';
    }
  }
  const moviesContainer = document.getElementById('movies-list-container');
  if(moviesContainer){ moviesContainer.innerHTML = moviesHtml; }

  // Filter shows based on search
  let showsFiltered = showProgress.filter(s => (s.series_name || '').toLowerCase().includes(progressShowsSearch || ''));
  // Separate into completed and in-progress
  let completedShows = showsFiltered.filter(s => (s.episodesLeft===0) || ((s.completion_percentage||0)>=100));
  let inProgressShows = showsFiltered.filter(s => !((s.episodesLeft===0) || ((s.completion_percentage||0)>=100)));
  // Apply filter option (all/inprogress/complete)
  if(progressShowsFilter === 'inprogress'){
    completedShows = [];
  } else if(progressShowsFilter === 'complete'){
    inProgressShows = [];
  }
  // Sorting function for shows
  function sortShows(list){
    return list.sort((a,b) => {
      if(progressShowsSort === 'alphabetical'){
        return (a.series_name||'').localeCompare(b.series_name||'');
      } else if(progressShowsSort === 'completionDesc'){
        const ca = a.completion_percentage||0;
        const cb = b.completion_percentage||0;
        if(cb !== ca) return cb - ca;
        // tie-breaker: lastWatched
        const la = a.lastWatched || '';
        const lb = b.lastWatched || '';
        return lb.localeCompare(la);
      } else if(progressShowsSort === 'episodesLeftAsc'){
        const ela = a.episodesLeft||0;
        const elb = b.episodesLeft||0;
        if(ela !== elb) return ela - elb;
        const la = a.lastWatched || '';
        const lb = b.lastWatched || '';
        return lb.localeCompare(la);
      } else {
        // default recent by lastWatched
        const la = a.lastWatched || '';
        const lb = b.lastWatched || '';
        if(!la && !lb) return 0;
        if(!la) return 1;
        if(!lb) return -1;
        return lb.localeCompare(la);
      }
    });
  }
  inProgressShows = sortShows(inProgressShows);
  completedShows = sortShows(completedShows);
  // Build shows HTML with collapsible categories and pagination
  let showsHtml = '';
  // Determine visible items based on pagination limits
  const visibleInProgress = inProgressShows.slice(0, showsInProgressLimit);
  const visibleCompleted = completedShows.slice(0, showsCompletedLimit);
  if(visibleInProgress.length === 0 && visibleCompleted.length === 0){
    showsHtml = '<p style="color:rgba(255,255,255,0.5)">No shows to display</p>';
  } else {
    // In Progress category
    if(inProgressShows.length > 0){
      const arrowIcon = showsInProgressCollapsed ? '▶' : '▼';
      showsHtml += `<div style="display:flex;justify-content:space-between;align-items:center;cursor:pointer;margin-bottom:5px" onclick="toggleShowCategory('inprogress')"><h4 style="margin:0;color:#6bcf7f;font-size:1.2em;font-weight:bold">In Progress</h4><span>${arrowIcon}</span></div>`;
      if(!showsInProgressCollapsed){
        showsHtml += `<div style="display:grid;gap:10px">`;
        visibleInProgress.forEach(s => {
          const comp = s.completion_percentage || 0;
          const totalPossible = getTotalPossibleEpisodes(s);
          const totalWatched = s.total_episodes || 0;
          const episodesLeft = s.episodesLeft != null ? s.episodesLeft : Math.max(0, totalPossible - totalWatched);
          let detailText = '';
          if(totalPossible > 0){ detailText = `${totalWatched}/${totalPossible} episodes watched - ${episodesLeft} left`; } else { detailText = `${totalWatched} episodes watched`; }
          let barColor;
          if(comp >= 100) barColor = '#2ecc71';
          else if(comp >= 75) barColor = '#27ae60';
          else if(comp >= 50) barColor = '#f1c40f';
          else if(comp >= 25) barColor = '#e67e22';
          else barColor = '#e74c3c';
          const escapedName = (s.series_name || '').replace(/'/g, "\\'");
          const trophy = (episodesLeft === 0 || comp >= 100) ? '🏆' : '';
              showsHtml += `<div onclick="navigateToShow('${escapedName}')" style="cursor:pointer;background:rgba(20,25,45,0.9);padding:6px 8px;border-radius:12px;border-left:4px solid ${barColor}">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:3px">
                  <strong style="font-size:1.1em;font-weight:bold">${s.series_name}</strong>
                  <span class="badge badge-warning">${comp}%</span>
                </div>
                <div style="color:rgba(255,255,255,0.7);margin-bottom:3px">${detailText}</div>
                <div class="progress-bar"><div class="progress-fill" style="width:${comp}%;background:${barColor}"></div></div>
                <div style="margin-top:3px;text-align:right">
                  ${trophy ? `<span style="margin-right:5px">${trophy}</span>` : ''}
                  <button class="btn success small" onclick="event.stopPropagation();markComplete('${escapedName}')" title="Mark completed">✓</button>
                </div>
              </div>`;
        });
        showsHtml += '</div>';
        if(inProgressShows.length > showsInProgressLimit){
          showsHtml += '<div style="text-align:center;margin-top:10px"><button class="btn small" onclick="loadMoreInProgressShows()">Load More</button></div>';
        }
      }
    }
    // Completed category
    if(completedShows.length > 0){
      const arrowIcon2 = showsCompletedCollapsed ? '▶' : '▼';
      showsHtml += `<div style="display:flex;justify-content:space-between;align-items:center;cursor:pointer;margin-top:15px;margin-bottom:5px" onclick="toggleShowCategory('completed')"><h4 style="margin:0;color:#8b9cff;font-size:1.2em;font-weight:bold">Completed</h4><span>${arrowIcon2}</span></div>`;
      if(!showsCompletedCollapsed){
        showsHtml += `<div style="display:grid;gap:10px">`;
        visibleCompleted.forEach(s => {
          const comp = s.completion_percentage || 100;
          const totalPossible = getTotalPossibleEpisodes(s);
          const totalWatched = s.total_episodes || 0;
          const episodesLeft = s.episodesLeft != null ? s.episodesLeft : Math.max(0, totalPossible - totalWatched);
          let detailText = '';
          if(totalPossible > 0){ detailText = `${totalWatched}/${totalPossible} episodes watched - ${episodesLeft} left`; } else { detailText = `${totalWatched} episodes watched`; }
          let barColor;
          if(comp >= 100) barColor = '#2ecc71';
          else if(comp >= 75) barColor = '#27ae60';
          else if(comp >= 50) barColor = '#f1c40f';
          else if(comp >= 25) barColor = '#e67e22';
          else barColor = '#e74c3c';
          const escapedName = (s.series_name || '').replace(/'/g, "\\'");
          const trophy = (episodesLeft === 0 || comp >= 100) ? '🏆' : '';
              showsHtml += `<div onclick="navigateToShow('${escapedName}')" style="cursor:pointer;background:rgba(20,25,45,0.9);padding:6px 8px;border-radius:12px;border-left:4px solid ${barColor}">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:3px">
                  <strong style="font-size:1.1em;font-weight:bold">${s.series_name}</strong>
                  <span class="badge badge-warning">${comp}%</span>
                </div>
                <div style="color:rgba(255,255,255,0.7);margin-bottom:3px">${detailText}</div>
                <div class="progress-bar"><div class="progress-fill" style="width:${comp}%;background:${barColor}"></div></div>
                <div style="margin-top:3px;text-align:right">
                  ${trophy ? `<span style="margin-right:5px">${trophy}</span>` : ''}
                  <button class="btn success small" onclick="event.stopPropagation();markComplete('${escapedName}')" title="Mark completed">✓</button>
                </div>
              </div>`;
        });
        showsHtml += '</div>';
        if(completedShows.length > showsCompletedLimit){
          showsHtml += '<div style="text-align:center;margin-top:10px"><button class="btn small" onclick="loadMoreCompletedShows()">Load More</button></div>';
        }
      }
    }
  }
  const showsContainer=document.getElementById('shows-list-container');
  if(showsContainer){ showsContainer.innerHTML = showsHtml; }
}

function toggle(id) {
  const el = document.getElementById(id);
  if(el.style.display === 'block') {
    el.style.display = 'none';
    openPanels.delete(id);
  } else {
    el.style.display = 'block';
    openPanels.add(id);
  }
}

// Toggle collapsible sections for movies and shows progress.  This function
// shows or hides the list with the given `listId` and rotates the arrow
// associated with `arrowId`.  It also updates `openPanels` and saves
// the state in sessionStorage so that open/closed sections persist
// across page reloads.  A section is considered open if its element's
// display style is set to "block".  When the section is opened, the arrow
// rotates 90 degrees; when closed, it resets to its original orientation.
function toggleSection(listId, arrowId){
  // Simple toggle for collapsible progress sections.  It does not interact
  // with openPanels or sessionStorage; progress panels do not persist
  // their open/closed state across reloads.  The arrow rotates to
  // indicate the section's state.
  const el = document.getElementById(listId);
  const arrow = document.getElementById(arrowId);
  if(!el) return;
  const isOpen = el.style.display === 'block';
  if(isOpen){
    el.style.display = 'none';
    if(arrow) arrow.style.transform = 'rotate(0deg)';
  } else {
    el.style.display = 'block';
    if(arrow) arrow.style.transform = 'rotate(90deg)';
  }
}

    // Toggle the collapsed state of a show category ("inprogress" or "completed").
    // When a header is clicked, the corresponding boolean flag is flipped
    // and the progress lists are re-rendered.  This ensures the arrow
    // orientation and visible items update immediately.
    function toggleShowCategory(cat){
      if(cat === 'inprogress'){
        // Toggle the in-progress category.  If it becomes expanded, collapse
        // the completed category for a cleaner single-view experience.  If
        // it becomes collapsed, leave the other category unchanged.
        const wasCollapsed = showsInProgressCollapsed;
        showsInProgressCollapsed = !showsInProgressCollapsed;
        if(!showsInProgressCollapsed){
          // We just expanded the in-progress section; collapse completed
          showsCompletedCollapsed = true;
        }
      } else if(cat === 'completed'){
        // Toggle the completed category.  If it becomes expanded, collapse
        // the in-progress category.
        const wasCollapsed = showsCompletedCollapsed;
        showsCompletedCollapsed = !showsCompletedCollapsed;
        if(!showsCompletedCollapsed){
          showsInProgressCollapsed = true;
        }
      }
      renderProgress();
    }

function setFilter(f,e){
  filter = f;
  localStorage.setItem('filter', f);
  // Clear any search term when switching the filter to ensure a clean view.
  const searchInput = document.getElementById('search');
  if(searchInput && searchInput.value){
    searchInput.value = '';
  }
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  e.target.classList.add('active');
  currentPage = 1;
  renderHistory();
}

function exportData(f){window.location.href='/api/export/'+f}

    // Toggle the inline export options for JSON and CSV.  When the
    // Export button is clicked, the small JSON/CSV buttons are shown
    // or hidden.  Selecting either option will hide the buttons again.
    function toggleExportOptions(){
      const opts = document.getElementById('exportOptions');
      if(!opts) return;
      if(opts.style.display === 'flex'){
        opts.style.display = 'none';
      } else {
        opts.style.display = 'flex';
      }
    }

// Progress list control handlers.  These functions update the global
// search/sort/filter variables, persist them to localStorage, and
// re-render the progress lists.  They are bound to input/select
// change events in the HTML.
function setMovieProgressSearch(val){
  progressMoviesSearch = (val || '').toLowerCase();
  localStorage.setItem('progressMoviesSearch', progressMoviesSearch);
  resetProgressPagination();
  renderProgress();
}
function setMovieProgressSort(val){
  progressMoviesSort = val || 'lastWatched';
  localStorage.setItem('progressMoviesSort', progressMoviesSort);
  resetProgressPagination();
  renderProgress();
}
function setShowProgressSearch(val){
  progressShowsSearch = (val || '').toLowerCase();
  localStorage.setItem('progressShowsSearch', progressShowsSearch);
  resetProgressPagination();
  renderProgress();
}
function setShowProgressSort(val){
  progressShowsSort = val || 'lastWatched';
  localStorage.setItem('progressShowsSort', progressShowsSort);
  resetProgressPagination();
  renderProgress();
}
function setShowProgressFilter(val){
  progressShowsFilter = val || 'all';
  localStorage.setItem('progressShowsFilter', progressShowsFilter);
  resetProgressPagination();
  renderProgress();
}

// Client-side export for movies and shows as JSON.  These functions
// serialize the relevant portion of the dataset into a JSON blob and
// trigger a download.  This provides partial export functionality
// without relying on server-side endpoints.
function exportMoviesJson(){
  if(!data || !data.movies){ alert('No movie data to export'); return; }
  const blob = new Blob([JSON.stringify(data.movies, null, 2)], {type:'application/json'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'movies_watch_data.json';
  document.body.appendChild(a);
  a.click();
  setTimeout(() => {
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, 0);
}
function exportShowsJson(){
  if(!data || !data.shows){ alert('No show data to export'); return; }
  const blob = new Blob([JSON.stringify(data.shows, null, 2)], {type:'application/json'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'shows_watch_data.json';
  document.body.appendChild(a);
  a.click();
  setTimeout(() => {
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, 0);
}

// Load more items in the progress lists.  These functions increase
// the respective limits and then re-render the progress view.  The
// limits are increased by the page size constants defined above.
function loadMoreMovies(){
  moviesProgressLimit += MOVIES_PROGRESS_PAGE_SIZE;
  renderProgress();
}
function loadMoreInProgressShows(){
  showsInProgressLimit += SHOWS_PROGRESS_PAGE_SIZE;
  renderProgress();
}
function loadMoreCompletedShows(){
  showsCompletedLimit += SHOWS_PROGRESS_PAGE_SIZE;
  renderProgress();
}

// Navigate to a specific movie within the library.  When the user
// clicks a movie in the progress view, we switch to the Library tab,
// set the filter to movies and the view mode to list, populate the
// search box with the movie name and then re-render the history view.
// This allows the user to quickly jump to the detailed view for a
// particular film without manually searching.
function navigateToMovie(movieName, movieYear){
  // Ensure the history tab is active
  const historyBtn = document.querySelector(".tab-btn[onclick*='history']");
  if(historyBtn){
    // Use switchTab directly to avoid triggering other click handlers
    switchTab({target: historyBtn}, 'history');
  }
  // Set the filter to movies and save it
  filter = 'movies';
  localStorage.setItem('filter', 'movies');
  // Ensure movies are shown in list view
  viewMode_movies = 'list';
  localStorage.setItem('viewMode_movies', 'list');
  // Update the search input to filter the movie list
  const searchInput = document.getElementById('search');
  if(searchInput){
    searchInput.value = movieName || '';
  }
  // Re-render the history view to apply filter and search
  renderHistory();
  // After render, scroll to the specific movie item if possible
  setTimeout(() => {
    let target = null;
    document.querySelectorAll('.movie-item').forEach(item => {
      const enc = item.getAttribute('data-movie-name');
      const name = enc ? decodeURIComponent(enc) : (item.querySelector('.movie-title')?.textContent || '').split(' (')[0];
      if (name === movieName) {
        if (movieYear === undefined || movieYear === null) {
          target = item;
        } else {
          const titleText = item.querySelector('.movie-title')?.textContent || '';
          if (titleText.includes(`(${movieYear})`)) target = item;
        }
      }
    });
    if (target) {
      target.scrollIntoView({ behavior: 'smooth', block: 'center' });
    } else {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  }, 120);
}

    // Navigate to a specific show within the library.  This function is
    // similar to navigateToMovie but tailored for TV shows.  When the
    // user clicks a show in the progress view, we switch to the Library
    // tab, filter for shows, set list view, populate the search box
    // with the show name to narrow the list, then render and scroll to
    // the selected show.  We also expand the seasons list if necessary.
    function navigateToShow(showName){
      // Ensure the history tab is active
      const historyBtn = document.querySelector(".tab-btn[onclick*='history']");
      if(historyBtn){
        switchTab({ target: historyBtn }, 'history');
      }
      // Set filter to shows and persist
      filter = 'shows';
      localStorage.setItem('filter', 'shows');
      // List view for shows
      viewMode_shows = 'list';
      localStorage.setItem('viewMode_shows', 'list');
      // Update search to the show name for quick filtering
      const searchInput = document.getElementById('search');
      if(searchInput){
        searchInput.value = showName || '';
      }
      // Render history with the updated filter and search
      renderHistory();
      // After rendering, expand the show and scroll to its header
      setTimeout(() => {
        const id = 'show-' + (showName || '').replace(/[^a-zA-Z0-9]/g, '');
        const seasonEl = document.getElementById(id);
        if(seasonEl){
          // Expand the show if collapsed
          if(seasonEl.style.display === 'none' || seasonEl.style.display === ''){
            toggle(id);
          }
          // Ascend to show-group container (3 levels up)
          let groupEl = seasonEl;
          for(let i=0; i<3; i++){
            if(groupEl && groupEl.parentElement){
              groupEl = groupEl.parentElement;
            }
          }
          if(groupEl){
            groupEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
          } else {
            seasonEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
          }
        }
      }, 200);
    }

window.onclick=function(e){
  const posterModal=document.getElementById('posterModal');
  const manualModal=document.getElementById('manualEntryModal');
  const addEpModal=document.getElementById('addEpisodeModal');
  const addSeasonModal=document.getElementById('addSeasonModal');
  const settingsModal=document.getElementById('settingsModal');
  if(e.target==posterModal)closePosterModal();
  if(e.target==manualModal)closeManualEntry();
  if(e.target==addEpModal)closeAddEpisode();
  if(e.target==addSeasonModal)closeAddSeason();
  if(e.target==settingsModal)closeSettings();
}

const zoomSlider=document.getElementById('zoomSlider');
const zoomValue=document.getElementById('zoomValue');
const htmlRoot=document.documentElement;

function applyZoomPercent(pct){
  const base = 16;
  const px = Math.max(10, Math.round(base * (pct / 100))); // WHOLE pixels only
  htmlRoot.style.fontSize = px + 'px';
  zoomValue.textContent = pct + '%';
  localStorage.setItem('uiZoom', String(pct));
}

const savedZoom = parseInt(localStorage.getItem('uiZoom') || '100', 10);
zoomSlider.value = savedZoom;
applyZoomPercent(savedZoom);

zoomSlider.addEventListener('input', function(){
  applyZoomPercent(parseInt(this.value, 10));
});

/* === Advanced Feature Scripts === */
// Render rating stars HTML
function escapeHtml(str){
  return (str || '')
    .replace(/&/g,'&amp;')
    .replace(/</g,'&lt;')
    .replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;')
    .replace(/'/g,'&#39;');
}

function renderRatingHTML(current, itemKey, note) {
  let stars = '';
  for (let i = 1; i <= 5; i++) {
    const filled = current && i <= current ? 'filled' : '';
    stars += `<span class="star ${filled}" onclick="rateItem('${itemKey}', ${i}, event)">★</span>`;
  }
  const safeNote = escapeHtml(note || '');
  const noteHtml = safeNote ? `<div class="rating-note">${safeNote}</div>` : '';
  return `<div class="rating-wrap"><div class="rating">${stars}</div>${noteHtml}</div>`;
}

// Rate an item (movie or show)
function rateItem(key, rating, event) {
  if (event && event.stopPropagation) event.stopPropagation();
  let existingNote = '';
  if (key.startsWith('show|')) {
    const show = (data.shows || []).find(s => ('show|' + s.series_name) === key);
    existingNote = show && show.note ? show.note : '';
  } else {
    (data.movies || []).forEach(m => {
      if (gridKeyMovie(m.name, m.year) === key) {
        if (m.note) existingNote = m.note;
      }
    });
  }
  const note = prompt('Optional note:', existingNote) || '';
  fetch('/api/rate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ id: key, rating: rating, note: note })
  }).then(r => r.json()).then(j => {
    if (j && j.success) {
      if (key.startsWith('show|')) {
        const show = data.shows.find(s => ('show|' + s.series_name) === key);
        if (show) { show.rating = rating; show.note = note; }
      } else {
        data.movies.forEach(m => {
          if (gridKeyMovie(m.name, m.year) === key) {
            m.rating = rating;
            m.note = note;
          }
        });
      }
      renderHistory();
    } else {
      alert('Rating failed: ' + ((j && j.error) || 'Unknown error'));
    }
  }).catch(e => alert('Rating error: ' + e));
}

// Inject rating stars into grid and list views
function injectRatings() {
  // Grid view
  document.querySelectorAll('.grid-item').forEach(item => {
    const key = item.getAttribute('data-grid-key');
    if (!key) return;
    let currentRating = null;
    let currentNote = '';
    if (key.startsWith('show|')) {
      const show = (data.shows || []).find(s => ('show|' + s.series_name) === key);
      currentRating = show && show.rating;
      currentNote = show && show.note ? show.note : '';
    } else {
      const movie = (data.movies || []).find(m => gridKeyMovie(m.name, m.year) === key);
      currentRating = movie && movie.rating;
      currentNote = movie && movie.note ? movie.note : '';
    }
    const existing = item.querySelector('.rating-wrap') || item.querySelector('.rating');
    if (existing) existing.remove();
    const infoEl = item.querySelector('.grid-item-info');
    const ratingHTML = renderRatingHTML(currentRating, key, currentNote);
    const temp = document.createElement('div');
    temp.innerHTML = ratingHTML;
    const el = temp.firstElementChild;
    if (infoEl && infoEl.parentNode) {
      infoEl.parentNode.insertBefore(el, infoEl.nextSibling);
    } else {
      item.appendChild(el);
    }
  });
  // Movie list
  document.querySelectorAll('.movie-item').forEach(item => {
    const titleEl = item.querySelector('.movie-title');
    if (!titleEl) return;
    const text = titleEl.textContent;
    const match = text.match(/^(.*?)(?:\\s*\\((\\d{4})\\))?$/);
    const name = match ? match[1].trim() : text.trim();
    const year = match && match[2] ? parseInt(match[2], 10) : null;
    let key = null;
    let rating = null;
    let note = '';
    (data.movies || []).forEach(m => {
      if (m.name === name && (!year || m.year == year)) {
        key = gridKeyMovie(m.name, m.year);
        rating = m.rating;
        note = m.note || '';
      }
    });
    if (!key) return;
    const existing = item.querySelector('.rating-wrap') || item.querySelector('.rating');
    if (existing) existing.remove();
    const ratingHTML = renderRatingHTML(rating, key, note);
    const temp = document.createElement('div');
    temp.innerHTML = ratingHTML;
    titleEl.parentNode.insertBefore(temp.firstElementChild, titleEl.nextSibling);
  });
  // Show list
  document.querySelectorAll('.show-group').forEach(group => {
    const header = group.querySelector('.show-title');
    if (!header) return;
    const seriesName = header.textContent;
    const key = 'show|' + seriesName;
    const show = (data.shows || []).find(s => s.series_name === seriesName);
    const rating = show && show.rating;
    const note = show && show.note ? show.note : '';
    const existing = group.querySelector('.rating-wrap') || group.querySelector('.rating');
    if (existing) existing.remove();
    const ratingHTML = renderRatingHTML(rating, key, note);
    const temp = document.createElement('div');
    temp.innerHTML = ratingHTML;
    header.parentNode.insertBefore(temp.firstElementChild, header.nextSibling);
  });
}

// Advanced filters state
let advancedFilters = { genre: '', yearFrom: '', yearTo: '', status: '', type: 'all' };
function applyFilters() {
  advancedFilters.genre = document.getElementById('genreFilter').value;
  advancedFilters.yearFrom = document.getElementById('yearFrom').value;
  advancedFilters.yearTo = document.getElementById('yearTo').value;
  advancedFilters.status = document.getElementById('statusFilter').value;
  advancedFilters.type = document.getElementById('typeFilter').value;
  localStorage.setItem('advancedFilters', JSON.stringify(advancedFilters));
  currentPage = 1;
  renderHistory();
}

function clearFilters() {
  document.getElementById('genreFilter').value = '';
  document.getElementById('yearFrom').value = '';
  document.getElementById('yearTo').value = '';
  document.getElementById('statusFilter').value = '';
  document.getElementById('typeFilter').value = 'all';
  applyFilters();
}

function populateGenreFilter() {
  const select = document.getElementById('genreFilter');
  if (!select) return;
  while (select.options.length > 1) select.remove(1);
  (data.genres || []).forEach(g => {
    const opt = document.createElement('option');
    opt.value = g.genre;
    opt.textContent = g.genre;
    select.appendChild(opt);
  });
  try {
    const saved = JSON.parse(localStorage.getItem('advancedFilters') || '{}');
    if (saved) {
      advancedFilters = saved;
      if (saved.genre) select.value = saved.genre;
      const yfEl = document.getElementById('yearFrom');
      if (yfEl) yfEl.value = saved.yearFrom || '';
      const ytEl = document.getElementById('yearTo');
      if (ytEl) ytEl.value = saved.yearTo || '';
      const sfEl = document.getElementById('statusFilter');
      if (sfEl) sfEl.value = saved.status || '';
      const tfEl = document.getElementById('typeFilter');
      if (tfEl) tfEl.value = saved.type || 'all';
    }
  } catch (e) {}
}

function updateResultsCounter(movieCount, showCount) {
  const counter = document.getElementById('resultsCounter');
  if (counter) {
    counter.textContent = `Found ${movieCount} movies, ${showCount} shows`;
  }
}

// Undo functionality
async function undoAction() {
  if (!confirm('Undo last action?')) return;
  try {
    const r = await fetch('/api/undo', { method: 'POST' });
    let j = null;
    try {
      j = await r.json();
    } catch (e) {
      throw new Error('Undo failed: invalid response');
    }
    if (!r.ok || !j || !j.success) {
      throw new Error((j && j.error) || 'Undo failed');
    }
    alert('Action undone successfully');
    try {
      await load();
    } catch (e) {
      console.error('Reload after undo failed:', e);
      location.reload();
    }
  } catch (e) {
    alert('Undo error: ' + e.message);
  }
}

// Auto-refresh controls
let autoRefreshTimer = null;
function updateRefreshIndicator(timeLeft) {
  const el = document.getElementById('refreshCountdown');
  if (!el) return;
  if (timeLeft <= 0) {
    el.textContent = '';
  } else {
    const m = Math.floor(timeLeft / 60000);
    const s = Math.floor((timeLeft % 60000) / 1000);
    const pad = n => n.toString().padStart(2, '0');
    el.textContent = `Refreshing in ${pad(m)}:${pad(s)}`;
  }
}

function startAutoRefresh() {
  if (!autoRefreshEnabled) return;

  // Clear any existing timer
  if (autoRefreshTimer) {
    clearTimeout(autoRefreshTimer);
    autoRefreshTimer = null;
  }

  const intervalMs = autoRefreshInterval * 60000;
  const startTime = Date.now();

  function tick() {
    try {
      const elapsed = Date.now() - startTime;
      const remaining = intervalMs - elapsed;

      updateRefreshIndicator(remaining);

      if (remaining <= 0) {
        const modalOpen = document.querySelector('.modal[style*="display: block"]');
        const searchActive = document.activeElement && document.activeElement.id === 'search';

        if (!modalOpen && !searchActive) {
          // Background reload without spinner (prevents desktop "contraction")
          quickReload(false);
        }

        // Restart cycle cleanly
        startAutoRefresh();
        return;
      }

      autoRefreshTimer = setTimeout(tick, 1000);
    } catch (e) {
      console.error('Auto-refresh tick error:', e);
      // Keep ticking even if something errors once
      autoRefreshTimer = setTimeout(tick, 1000);
    }
  }

  tick();
}

function toggleAutoRefresh() {
  const toggle = document.getElementById('autoRefreshToggle');
  autoRefreshEnabled = toggle && toggle.checked;
  localStorage.setItem('autoRefreshEnabled', autoRefreshEnabled ? '1' : '0');
  if (autoRefreshEnabled) {
    startAutoRefresh();
  } else {
    if (autoRefreshTimer) clearTimeout(autoRefreshTimer);
    autoRefreshTimer = null;
    updateRefreshIndicator(0);
  }
}

function updateAutoRefreshInterval() {
  const val = parseInt(document.getElementById('autoRefreshInterval').value, 10);
  autoRefreshInterval = val;
  localStorage.setItem('autoRefreshInterval', String(val));
  if (autoRefreshEnabled) startAutoRefresh();
}

let autoRefreshEnabled = localStorage.getItem('autoRefreshEnabled') === '1';
let autoRefreshInterval = parseInt(localStorage.getItem('autoRefreshInterval') || '5', 10);

function startRealtimeUpdates(){
  if(!('EventSource' in window)) return false;
  if(eventsSource) return true;
  try {
    const es = new EventSource('/api/events');
    eventsSource = es;
    es.onopen = () => {
      realtimeConnected = true;
      realtimeFailures = 0;
      sigPollFailures = 0;
      if(sigPollTimer){
        clearTimeout(sigPollTimer);
        sigPollTimer = null;
      }
    };
    es.addEventListener('history_update', async () => {
      try {
        await quickReload(false);
      } catch (_) {}
    });
    es.onerror = () => {
      realtimeConnected = false;
      realtimeFailures += 1;
      try { es.close(); } catch(_) {}
      if(eventsSource === es) eventsSource = null;
      if(realtimeFailures < 3){
        setTimeout(() => {
          if(!eventsSource) startRealtimeUpdates();
        }, 1200);
      } else if(!sigPollTimer){
        scheduleSigPoll(1500);
      }
    };
    return true;
  } catch(e) {
    return false;
  }
}

async function fetchHistorySig(){
  return await fetchJsonWithTimeout('/api/realtime_seq', 3500);
}

function scheduleSigPoll(delayMs){
  if(sigPollTimer) clearTimeout(sigPollTimer);
  sigPollTimer = setTimeout(runSigPoll, delayMs);
}

async function runSigPoll(){
  if(realtimeConnected) return;
  if(document.hidden) return scheduleSigPoll(sigPollDelay);
  if(sigPollBusy) return scheduleSigPoll(sigPollDelay);
  sigPollBusy = true;
  try {
    const j = await fetchHistorySig();
    if(j && typeof j.seq === 'number'){
      if(typeof lastRealtimeSeq === 'number' && j.seq > lastRealtimeSeq){
        await quickReload(false);
      }
      lastRealtimeSeq = j.seq;
      if(j.sig) historySig = j.sig;
    } else if(j && j.sig){
      if(historySig && j.sig !== historySig){
        await quickReload(false);
      }
      historySig = j.sig;
    }
    sigPollFailures = 0;
    sigPollDelay = 1500;
  } catch(e) {
    sigPollFailures += 1;
    if(sigPollFailures >= 6){
      sigPollDelay = 15000;
    } else if(sigPollFailures >= 3){
      sigPollDelay = 8000;
    } else {
      sigPollDelay = Math.min(sigPollDelay + 1000, 8000);
    }
  } finally {
    sigPollBusy = false;
    scheduleSigPoll(sigPollDelay);
  }
}

function startSigPolling(){
  if(startRealtimeUpdates()) return;
  if(sigPollTimer) return;
  sigPollDelay = 1500;
  sigPollFailures = 0;
  scheduleSigPoll(500);
}

// Theme and layout helpers
// Removed duplicate applyTheme and changeTheme definitions. The theme
// handling is implemented earlier in the script using availableThemes,
// applyTheme() and toggleTheme(). To programmatically set a theme call
// applyTheme(theme).
function applyLayout(layout) {
  const root = document.documentElement;

  const mode = (layout === 'compact' || layout === 'spacious') ? layout : 'comfortable';

  root.classList.remove('layout-compact', 'layout-spacious');
  if (mode === 'compact') root.classList.add('layout-compact');
  if (mode === 'spacious') root.classList.add('layout-spacious');

  localStorage.setItem('layout', mode);

  // Force visual update (some CSS/layout depends on render)
  try { renderHistory(); } catch(e) {}
}

function changeLayout(layout) {
  localStorage.setItem('layout', layout || 'comfortable');
  applyLayout(layout);
}

// Watch time analytics
async function renderWatchTime() {
  try {
    const res = await fetch('/api/watch_time');
    const j = await res.json();
    if (!j || !j.success) return;
    document.getElementById('wt-total').textContent = formatDurationHours(j.total);
    document.getElementById('wt-movies').textContent = formatDurationHours(j.movies);
    document.getElementById('wt-shows').textContent = formatDurationHours(j.shows);
    document.getElementById('wt-daily').textContent = formatDurationHours(j.avg_per_day);
    document.getElementById('wt-binge').textContent = (j.binges || []).length;
    const labels = j.monthly.map(x => x.month);
    const hours = j.monthly.map(x => x.hours);
    if (watchTimeChart) { try { watchTimeChart.destroy(); } catch (e) {} }
    const ctx = document.getElementById('watchTimeChart').getContext('2d');
    watchTimeChart = new Chart(ctx, {
      type: 'bar',
      data: { labels: labels, datasets: [{ label: 'Hours', data: hours, backgroundColor: '#8b9cff' }] },
      options: {
        responsive: true,
        scales: {
          x: { ticks: { color: '#fff' } },
          y: { ticks: { color: '#fff' } }
        },
        plugins: { legend: { display: false } }
      }
    });
  } catch (e) {
    console.error('Watch time load error', e);
  }
}

/* === End of advanced feature scripts === */

load();
setupPendingTotalsInteractions();
</script></body></html>"""

@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        payload = request.get_json(force=True, silent=True) or {}
        notification_type = (payload.get("NotificationType") or payload.get("Event") or "").lower()
        
        if notification_type and (
            "playbackstop" not in notification_type and
            "stop" not in notification_type and
            "ended" not in notification_type
        ):
            return jsonify({"success": True, "ignored": True})

        item_type = payload.get("ItemType") or payload.get("Type") or (payload.get("Item", {}) or {}).get("Type") or "Unknown"
        name = payload.get("Name") or (payload.get("Item", {}) or {}).get("Name") or "Unknown"
        year = payload.get("Year") or (payload.get("Item", {}) or {}).get("ProductionYear")
        series_name = payload.get("SeriesName") or (payload.get("Item", {}) or {}).get("SeriesName")
        season = payload.get("SeasonNumber") or (payload.get("Item", {}) or {}).get("ParentIndexNumber")
        episode = payload.get("EpisodeNumber") or (payload.get("Item", {}) or {}).get("IndexNumber")
        user = payload.get("NotificationUsername") or (payload.get("User", {}) or {}).get("Name") or "Unknown"
        genres = payload.get("Genres") or (payload.get("Item", {}) or {}).get("Genres") or []
        genres_norm = _normalize_genres(genres)

        dedupe_key = (
            str(notification_type or "").strip().lower(),
            str(item_type or "").strip().lower(),
            str(name or "").strip().lower(),
            str(series_name or "").strip().lower(),
            _coerce_int(season, None),
            _coerce_int(episode, None),
            str(user or "").strip().lower(),
        )
        if _is_duplicate_webhook_event(dedupe_key):
            return jsonify({"success": True, "ignored": True, "duplicate": True})

        # Jellyfin ids for deep-linking back to Jellyfin Web UI ("Open in Jellyfin").
        try:
            item_obj = (payload.get("Item") or {}) if isinstance(payload.get("Item"), dict) else {}
        except Exception:
            item_obj = {}
        jellyfin_id = (
            payload.get("ItemId") or payload.get("Id") or
            item_obj.get("Id") or item_obj.get("ItemId")
        )
        jellyfin_series_id = (
            payload.get("SeriesId") or item_obj.get("SeriesId") or
            payload.get("SeriesID") or item_obj.get("SeriesID")
        )

        record = {
            "timestamp": datetime.now().isoformat(),
            "type": item_type,
            "name": name,
            "year": year,
            "series_name": series_name,
            "season": season,
            "episode": episode,
            "user": user,
            "genres": genres_norm,
            "source": "webhook",
            "jellyfin_id": jellyfin_id,
            "jellyfin_series_id": jellyfin_series_id,
        }

        append_record(record)
        # Keep the previous organized payload so we can reuse TMDB totals while
        # recomputing; the signature change will still force a refresh.
        cache["time"] = None
        cache["sig"] = None
        # Rebuild missing posters/episode totals in the background (does not block webhook).
        try:
            if str(item_type or "").strip().lower() == "movie":
                _queue_tmdb_poster_fetch(name, year, "movie")
            elif str(item_type or "").strip().lower() == "episode":
                _queue_tmdb_poster_fetch(series_name or name, year, "tv")
                # Jellyfin webhooks frequently provide episode-year values here; use title-only matching.
                _queue_tmdb_series_fetch(series_name or name, expected_year=None)
        except Exception:
            pass
        prewarm_organized_cache_async()
        return jsonify({"success": True})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/history")
def api_history():
    return jsonify(organize_data())

@app.route("/api/config")
def api_config():
    # Do not return secrets (API keys). Only return safe configuration.
    base = (JELLYFIN_PUBLIC_URL or JELLYFIN_URL or "").rstrip("/")
    return jsonify({
        "jellyfin_public_url": base,
        "jellyfin_configured": True if base else False,
    })

@app.route("/api/history_sig")
def api_history_sig():
    try:
        global history_sig_fast
        sig = history_sig_fast or cache.get("sig")
        if not sig:
            sig = _data_signature()
            history_sig_fast = sig
        return jsonify({"sig": sig})
    except Exception as e:
        return jsonify({"sig": None, "error": str(e)}), 500

@app.route("/api/realtime_seq")
def api_realtime_seq():
    try:
        return jsonify({
            "seq": realtime_seq,
            "sig": history_sig_fast or cache.get("sig")
        })
    except Exception as e:
        return jsonify({"seq": None, "sig": None, "error": str(e)}), 500

@app.route("/api/events")
def api_events():
    try:
        last_id_raw = request.headers.get("Last-Event-ID") or request.args.get("last")
        last_id = _coerce_int(last_id_raw, None)
        if last_id is None:
            last_id = realtime_seq

        def _stream(start_seq):
            current = start_seq
            # Open the stream immediately
            yield "event: ready\ndata: {}\n\n"
            while True:
                changed = False
                with realtime_cv:
                    if realtime_seq <= current:
                        realtime_cv.wait(timeout=20.0)
                    if realtime_seq > current:
                        current = realtime_seq
                        changed = True
                if changed:
                    payload = json.dumps({"seq": current})
                    yield f"id: {current}\nevent: history_update\ndata: {payload}\n\n"
                else:
                    yield "event: ping\ndata: {}\n\n"

        resp = Response(_stream(last_id), mimetype="text/event-stream")
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        resp.headers["X-Accel-Buffering"] = "no"
        return resp
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/genre_recommendations', methods=['GET'])
def get_genre_recommendations():
    try:
        global genre_recs_cache
        # Get offset parameter (for pagination)
        offset = int(request.args.get('offset', 0))
        
        # Get organized data (this calls organize_data() which processes your records)
        org_data = organize_data()
        
        # Use the SAME genre breakdown that the pie chart uses
        genre_breakdown = org_data.get('genre_breakdown', {})
        
        if not genre_breakdown:
            return jsonify({'success': True, 'top_genre': None, 'movies': [], 'shows': [], 'has_more': False})
        
        # Count total items per genre (same as pie chart)
        genre_counts = {}
        for genre, items in genre_breakdown.items():
            total_count = len(items.get('movies', [])) + len(items.get('shows', []))
            genre_counts[genre] = total_count
        
        # Get top genre (will match pie chart now!)
        top_genre = max(genre_counts, key=genre_counts.get)
        top_genre_norm = str(top_genre or "").strip().lower()
        if not top_genre_norm:
            return jsonify({'success': True, 'top_genre': None, 'movies': [], 'shows': [], 'has_more': False})

        now_ts = time.time()
        cache_fresh = (
            genre_recs_cache.get("top_genre") == top_genre and
            (now_ts - float(genre_recs_cache.get("ts") or 0)) < 1800
        )

        if cache_fresh:
            all_recommended_movies = list(genre_recs_cache.get("movies") or [])
            all_recommended_shows = list(genre_recs_cache.get("shows") or [])
        else:
            radarr_url = os.getenv("RADARR_URL")
            radarr_key = os.getenv("RADARR_API_KEY")
            sonarr_url = os.getenv("SONARR_URL")
            sonarr_key = os.getenv("SONARR_API_KEY")

            watched_movie_titles = {str(m.get('name', '')).lower() for m in org_data.get('movies', [])}
            watched_show_titles = {str(s.get('series_name', '')).lower() for s in org_data.get('shows', [])}

            def _fetch_radarr():
                out = []
                if not (radarr_url and radarr_key):
                    return out
                try:
                    radarr_resp = requests.get(
                        f'{radarr_url}/api/v3/movie',
                        headers={'X-Api-Key': radarr_key},
                        timeout=8
                    )
                    if not radarr_resp.ok:
                        return out
                    for movie in radarr_resp.json() or []:
                        movie_genres = _normalize_genres(movie.get('genres', []) or [])
                        if not movie_genres:
                            continue
                        movie_genres_norm = {str(g or "").strip().lower() for g in movie_genres}
                        if top_genre_norm not in movie_genres_norm:
                            continue
                        title = str(movie.get('title', '') or '')
                        if title.lower() in watched_movie_titles:
                            continue
                        out.append({
                            'title': title,
                            'year': movie.get('year'),
                            'overview': (movie.get('overview', '') or 'No description')[:150] + '...',
                            'genres': movie_genres,
                            'tmdbId': movie.get('tmdbId')
                        })
                except Exception as e:
                    print(f"Radarr error: {e}")
                return out

            def _fetch_sonarr():
                out = []
                if not (sonarr_url and sonarr_key):
                    return out
                try:
                    sonarr_resp = requests.get(
                        f'{sonarr_url}/api/v3/series',
                        headers={'X-Api-Key': sonarr_key},
                        timeout=8
                    )
                    if not sonarr_resp.ok:
                        return out
                    for show in sonarr_resp.json() or []:
                        show_genres = _normalize_genres(show.get('genres', []) or [])
                        if not show_genres:
                            continue
                        show_genres_norm = {str(g or "").strip().lower() for g in show_genres}
                        if top_genre_norm not in show_genres_norm:
                            continue
                        title = str(show.get('title', '') or '')
                        if title.lower() in watched_show_titles:
                            continue
                        out.append({
                            'title': title,
                            'year': show.get('year'),
                            'overview': (show.get('overview', '') or 'No description')[:150] + '...',
                            'genres': show_genres,
                            'tvdbId': show.get('tvdbId')
                        })
                except Exception as e:
                    print(f"Sonarr error: {e}")
                return out

            with ThreadPoolExecutor(max_workers=2) as ex:
                fm = ex.submit(_fetch_radarr)
                fs = ex.submit(_fetch_sonarr)
                all_recommended_movies = fm.result()
                all_recommended_shows = fs.result()

            all_recommended_movies.sort(key=lambda x: (x.get('title') or '').lower())
            all_recommended_shows.sort(key=lambda x: (x.get('title') or '').lower())

            genre_recs_cache = {
                "sig": org_data.get("sig"),
                "top_genre": top_genre,
                "ts": now_ts,
                "movies": all_recommended_movies,
                "shows": all_recommended_shows,
            }
        
        # Paginate results (5 movies + 5 shows per page)
        start_idx = offset * 5
        recommended_movies = all_recommended_movies[start_idx:start_idx + 5]
        recommended_shows = all_recommended_shows[start_idx:start_idx + 5]
        
        # Check if there are more results
        has_more_movies = len(all_recommended_movies) > (start_idx + 5)
        has_more_shows = len(all_recommended_shows) > (start_idx + 5)
        has_more = has_more_movies or has_more_shows
        
        print(f"Showing movies {start_idx+1}-{start_idx+len(recommended_movies)} of {len(all_recommended_movies)}")
        print(f"Showing shows {start_idx+1}-{start_idx+len(recommended_shows)} of {len(all_recommended_shows)}")
        
        return jsonify({
            'success': True,
            'top_genre': top_genre,
            'movies': recommended_movies,
            'shows': recommended_shows,
            'has_more': has_more,
            'total_movies': len(all_recommended_movies),
            'total_shows': len(all_recommended_shows)
        })
    
    except Exception as e:
        print(f"Genre recommendations error: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/genre_insights', methods=['GET'])
def get_genre_insights():
    try:
        global genre_insights_cache
        sig = None
        try:
            sig = _data_signature()
            if genre_insights_cache.get("sig") == sig and genre_insights_cache.get("data"):
                return jsonify(genre_insights_cache["data"])
        except Exception:
            sig = None

        org_data = organize_data()
        # Keep /api/genre_insights fast: UI doesn't render trends, and scanning
        # large history files here causes slow loads/timeouts.
        trends = []
        
        # Genre combinations you like
        genre_combos = Counter()
        
        for movie in org_data.get('movies', []):
            genres = movie.get('genres', [])
            watch_count = movie.get('watch_count', 1)
            if len(genres) >= 2:
                # Create all pairs
                for i in range(len(genres)):
                    for j in range(i+1, len(genres)):
                        combo = tuple(sorted([genres[i], genres[j]]))
                        genre_combos[combo] += watch_count
        
        for show in org_data.get('shows', []):
            genres = show.get('genres', [])
            total_eps = show.get('total_episodes', 0)
            if len(genres) >= 2 and total_eps > 0:
                for i in range(len(genres)):
                    for j in range(i+1, len(genres)):
                        combo = tuple(sorted([genres[i], genres[j]]))
                        genre_combos[combo] += 1  # Count show once, not by episodes
        
        top_combos = [
            {"genres": list(combo), "count": count}
            for combo, count in genre_combos.most_common(5)
        ]
        
        # Mood-based recommendations
        mood_map = {
            "intense": ["Thriller", "Crime", "Mystery"],
            "fun": ["Comedy", "Animation", "Family"],
            "emotional": ["Drama", "Romance"],
            "exciting": ["Action", "Adventure", "Sci-Fi"],
            "scary": ["Horror", "Thriller"]
        }
        
        genre_breakdown = org_data.get('genre_breakdown', {})
        moods = {}
        
        for mood, mood_genres in mood_map.items():
            count = 0
            for g in mood_genres:
                if g in genre_breakdown:
                    count += len(genre_breakdown[g].get('movies', [])) + len(genre_breakdown[g].get('shows', []))
            if count > 0:
                moods[mood] = {"count": count, "genres": mood_genres}
        
        insights_payload = {
            "success": True,
            "trends": trends,
            "combos": top_combos,
            "moods": moods
        }

        try:
            genre_insights_cache["sig"] = sig
            genre_insights_cache["data"] = insights_payload
        except Exception:
            pass

        return jsonify(insights_payload)
    
    except Exception as e:
        print(f"Genre insights error: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route("/api/manual_entry", methods=["POST"])
def api_manual_entry():
    try:
        record = request.get_json()
        
        if not record or not record.get("name") or not record.get("timestamp"):
            return jsonify({"success": False, "error": "Missing required fields"}), 400
        
        if record.get("type") not in ["Movie", "Episode"]:
            return jsonify({"success": False, "error": "Type must be Movie or Episode"}), 400

        # Canonicalize Movie titles via TMDB so casing/format matches TMDB (and posters resolve).
        if record.get("type") == "Movie":
            try:
                raw_title = str(record.get("name") or "").strip()
                exp_year = _coerce_int(record.get("year"), None)
                info = get_tmdb_movie_info(raw_title, expected_year=exp_year, allow_network=True)
                if isinstance(info, dict) and info.get("tmdb_name"):
                    record["name"] = info.get("tmdb_name") or raw_title
                    # Only fill year when user didn't provide one.
                    if record.get("year") in (None, "", 0) and info.get("tmdb_year"):
                        record["year"] = _coerce_int(info.get("tmdb_year"), None)
            except Exception:
                pass
        
        if record.get("type") == "Episode":
            if not record.get("series_name") or record.get("season") is None or record.get("episode") is None:
                return jsonify({"success": False, "error": "Episodes require series_name, season, and episode"}), 400

            series_name = (record.get("series_name") or "").strip()
            season = _coerce_int(record.get("season"), None)
            ep_num = _coerce_int(record.get("episode"), None)
            if not series_name or season is None or ep_num is None:
                return jsonify({"success": False, "error": "Episodes require series_name, season, and episode"}), 400
            record["season"] = season
            record["episode"] = ep_num

            # Canonicalize series name via TMDB when possible.
            canonical_series = series_name
            series_info = None
            try:
                series_info = get_tmdb_series_info(series_name, expected_year=record.get("year"))
                if isinstance(series_info, dict) and series_info.get("tmdb_name"):
                    canonical_series = series_info.get("tmdb_name") or series_name
            except Exception:
                series_info = None
            record["series_name"] = canonical_series
            try:
                if isinstance(series_info, dict):
                    _update_show_totals_from_tmdb(canonical_series, record.get("year"), series_info)
            except Exception:
                pass

            # Prefer TMDB episode title over user-provided casing/title when possible.
            try:
                tmdb_name = get_tmdb_episode_name(canonical_series, season, ep_num)
                record["name"] = tmdb_name if tmdb_name else (record.get("name") or f"Episode {ep_num}")
            except Exception:
                record["name"] = record.get("name") or f"Episode {ep_num}"
        
        append_record(record)
        print(f"✓ Manual entry added: {record.get('type')} - {record.get('name')}")

        # Queue TMDB work (posters/totals) for manual adds. /api/history stays cache-only,
        # so without this you can get placeholders until the next background prewarm.
        try:
            rtype = str(record.get("type") or "").strip().lower()
            if rtype == "movie":
                _queue_tmdb_poster_fetch(record.get("name"), record.get("year"), "movie")
            elif rtype == "episode":
                _queue_tmdb_poster_fetch(record.get("series_name") or record.get("name"), record.get("year"), "tv")
                _queue_tmdb_series_fetch(record.get("series_name") or record.get("name"), expected_year=None)
        except Exception:
            pass
        
        return jsonify({"success": True, "message": "Watch record added successfully"})
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/add_episode", methods=["POST"])
def api_add_episode():
    try:
        data = request.get_json()
        series_name = (data.get("series_name") or "").strip()
        season = data.get("season")
        episode = data.get("episode")
        date = data.get("date")
        skip_tmdb = data.get("skip_tmdb", False)
        
        if not all([series_name, season is not None, episode is not None, date]):
            return jsonify({"success": False, "error": "Missing required fields"}), 400

        canonical_series = series_name
        series_info = None
        if TMDB_API_KEY and not skip_tmdb:
            try:
                series_info = get_tmdb_series_info(series_name)
                if isinstance(series_info, dict) and series_info.get("tmdb_name"):
                    canonical_series = series_info.get("tmdb_name") or series_name
            except Exception:
                series_info = None
        
        if not skip_tmdb:
            episode_name = get_tmdb_episode_name(canonical_series, season, episode)
            if not episode_name:
                episode_name = f"Episode {episode}"
        else:
            episode_name = f"Episode {episode}"

        try:
            if isinstance(series_info, dict):
                _update_show_totals_from_tmdb(canonical_series, None, series_info)
        except Exception:
            pass
        
        record = {
            "timestamp": datetime.fromisoformat(date + "T12:00:00").isoformat(),
            "type": "Episode",
            "name": episode_name,
            "year": None,
            "series_name": canonical_series,
            "season": season,
            "episode": episode,
            "user": "Manual Entry",
            "genres": [],
            "source": "manual_episode"
        }
        
        append_record(record)
        print(f"✓ Added episode: {canonical_series} S{season}E{episode} - {episode_name}")

        # Queue poster + totals fetch for this show.
        try:
            _queue_tmdb_poster_fetch(canonical_series, None, "tv")
            _queue_tmdb_series_fetch(canonical_series, expected_year=None)
        except Exception:
            pass
        
        return jsonify({"success": True, "series_name": canonical_series, "episode_name": episode_name})
    
    except Exception as e:
        print(f"✗ Add episode error: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/add_season", methods=["POST"])
def api_add_season():
    try:
        data = request.get_json()
        series_name = (data.get("series_name") or "").strip()
        season = data.get("season")
        date = data.get("date")
        skip_tmdb = data.get("skip_tmdb", False)
        episode_count = data.get("episode_count")
        
        if not all([series_name, season is not None, date]):
            return jsonify({"success": False, "error": "Missing required fields"}), 400

        canonical_series = series_name
        series_info = None
        if TMDB_API_KEY and not skip_tmdb:
            try:
                series_info = get_tmdb_series_info(series_name)
                if isinstance(series_info, dict) and series_info.get("tmdb_name"):
                    canonical_series = series_info.get("tmdb_name") or series_name
            except Exception:
                series_info = None
        
        if not skip_tmdb:
            episodes = get_tmdb_season_episodes(canonical_series, season)
            
            if not episodes or len(episodes) == 0:
                if episode_count:
                    episodes = [{"episode_number": i, "name": f"Episode {i}"} for i in range(1, episode_count + 1)]
                else:
                    return jsonify({"success": False, "error": "Could not fetch episodes from TMDB. Try manual episode count or skip TMDB option."}), 400
        else:
            if not episode_count:
                return jsonify({"success": False, "error": "Episode count required when skipping TMDB"}), 400
            episodes = [{"episode_number": i, "name": f"Episode {i}"} for i in range(1, episode_count + 1)]

        try:
            if isinstance(series_info, dict):
                _update_show_totals_from_tmdb(canonical_series, None, series_info)
        except Exception:
            pass
        
        count = 0
        for ep in episodes:
            record = {
                "timestamp": datetime.fromisoformat(date + "T12:00:00").isoformat(),
                "type": "Episode",
                "name": ep["name"],
                "year": None,
                "series_name": canonical_series,
                "season": season,
                "episode": ep["episode_number"],
                "user": "Manual Entry",
                "genres": [],
                "source": "manual_season"
            }
            append_record(record)
            count += 1
        
        print(f"✓ Added season: {canonical_series} S{season} - {count} episodes")

        # Queue poster + totals fetch for this show.
        try:
            _queue_tmdb_poster_fetch(canonical_series, None, "tv")
            _queue_tmdb_series_fetch(canonical_series, expected_year=None)
        except Exception:
            pass
        
        return jsonify({"success": True, "series_name": canonical_series, "count": count})
    
    except Exception as e:
        print(f"✗ Add season error: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

def parse_episode_spec(spec):
    """
    Supports:
      "1" -> [1]
      "1,2,5-8" -> [1,2,5,6,7,8]
      " 1 , 3 - 4 " -> [1,3,4]
    """
    if not spec:
        return []
    s = spec.replace(" ", "")
    out = []
    parts = [p for p in s.split(",") if p]
    for p in parts:
        if "-" in p:
            a, b = p.split("-", 1)
            if a.isdigit() and b.isdigit():
                a = int(a); b = int(b)
                if a <= b:
                    out.extend(range(a, b + 1))
                else:
                    out.extend(range(b, a + 1))
        else:
            if p.isdigit():
                out.append(int(p))
    # unique + sorted
    return sorted(set([x for x in out if x > 0]))

@app.route("/api/manual_tv_bulk", methods=["POST"])
def api_manual_tv_bulk():
    """
    Bulk add episodes using TMDB:
    - series_name only -> ALL seasons + ALL episodes
    - series_name + season -> ALL episodes in that season
    - series_name + season + episode_spec (e.g. 1,2,5-8) -> those episodes
    """
    try:
        payload = request.get_json(force=True, silent=True) or {}
        series_name = (payload.get("series_name") or "").strip()
        season = payload.get("season")
        episode_spec = (payload.get("episode_spec") or "").strip()
        episode_name_override = (payload.get("episode_name") or "").strip()
        year = _coerce_int(payload.get("year"), None)
        raw_genres = payload.get("genres", [])
        date = (payload.get("date") or "").strip()
        skip_tmdb = bool(payload.get("skip_tmdb", False))

        # Normalize genres input (accept list or comma-separated string).
        genres = []
        try:
            if isinstance(raw_genres, str):
                genres = [g.strip() for g in raw_genres.split(",") if g.strip()]
            elif isinstance(raw_genres, list):
                genres = [str(g).strip() for g in raw_genres if str(g).strip()]
        except Exception:
            genres = []
        genres = _normalize_genres(genres)

        if not series_name:
            return jsonify({"success": False, "error": "Missing series name"}), 400
        if not date:
            return jsonify({"success": False, "error": "Missing watch date"}), 400

        # Normalize
        season = int(season) if season not in (None, "", "null") else None

        # Determine mode
        episodes_list = parse_episode_spec(episode_spec)

        # If user provided episode_spec, season must exist
        if episodes_list and (season is None):
            return jsonify({"success": False, "error": "Season number is required when specifying episode numbers"}), 400

        # Build list of (season, episode, name)
        to_add = []
        canonical_series = series_name
        series_info = None

        if season is None:
            # ALL seasons + ALL episodes
            if skip_tmdb:
                return jsonify({"success": False, "error": "Cannot fetch all seasons when Skip TMDB is enabled"}), 400

            series_info = get_tmdb_series_info(series_name, expected_year=year)
            if not series_info or not series_info.get("seasons"):
                return jsonify({"success": False, "error": "TMDB could not find this series"}), 400
            if series_info.get("tmdb_name"):
                canonical_series = series_info.get("tmdb_name") or series_name
            try:
                _update_show_totals_from_tmdb(canonical_series, year, series_info)
            except Exception:
                pass

            # Iterate seasons in TMDB info (season_number > 0 already in your cache builder)
            for snum in sorted(series_info["seasons"].keys()):
                eps = get_tmdb_season_episodes(canonical_series, snum)
                if not eps:
                    continue
                for ep in eps:
                    epno = ep.get("episode_number")
                    nm = ep.get("name") or f"Episode {epno}"
                    to_add.append((snum, epno, nm))

        else:
            # Season is provided
            if TMDB_API_KEY and not skip_tmdb:
                try:
                    series_info = get_tmdb_series_info(series_name, expected_year=year)
                    if isinstance(series_info, dict) and series_info.get("tmdb_name"):
                        canonical_series = series_info.get("tmdb_name") or series_name
                    if isinstance(series_info, dict):
                        _update_show_totals_from_tmdb(canonical_series, year, series_info)
                except Exception:
                    series_info = None

            if episodes_list:
                # Specific episodes in season
                if skip_tmdb:
                    for epno in episodes_list:
                        to_add.append((season, epno, f"Episode {epno}"))
                else:
                    eps = get_tmdb_season_episodes(canonical_series, season) or []
                    ep_map = {}
                    try:
                        for ep in eps:
                            epno = _coerce_int(ep.get("episode_number"), 0) or 0
                            if epno > 0 and ep.get("name"):
                                ep_map[epno] = ep.get("name")
                    except Exception:
                        ep_map = {}
                    for epno in episodes_list:
                        nm = ep_map.get(epno) or f"Episode {epno}"
                        if (not ep_map.get(epno)) and episode_name_override and len(episodes_list) == 1:
                            nm = episode_name_override
                        to_add.append((season, epno, nm))
            else:
                # ALL episodes in that season
                if skip_tmdb:
                    return jsonify({"success": False, "error": "Skip TMDB needs an episode list (e.g. 1,2,5-8)"}), 400

                eps = get_tmdb_season_episodes(canonical_series, season)
                if not eps:
                    return jsonify({"success": False, "error": "TMDB could not fetch episodes for that season"}), 400
                for ep in eps:
                    epno = ep.get("episode_number")
                    nm = ep.get("name") or f"Episode {epno}"
                    to_add.append((season, epno, nm))

        if not to_add:
            return jsonify({"success": False, "error": "No episodes found to add"}), 400

        # Write records
        ts = datetime.fromisoformat(date + "T12:00:00").isoformat()
        added = 0
        for snum, epno, nm in to_add:
            record = {
                "timestamp": ts,
                "type": "Episode",
                "name": nm,
                "year": year,
                "series_name": canonical_series,
                "season": int(snum),
                "episode": int(epno),
                "user": "Manual Entry",
                "genres": genres,
                "source": "manual_bulk_tv"
            }
            append_record(record)
            added += 1

        # Queue poster + totals fetch for this show.
        try:
            _queue_tmdb_poster_fetch(canonical_series, year, "tv")
            _queue_tmdb_series_fetch(canonical_series, expected_year=year)
        except Exception:
            pass

        return jsonify({"success": True, "series_name": canonical_series, "added": added})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/mark_complete", methods=["POST"])
def api_mark_complete():
    try:
        data = request.get_json()
        series_name = data.get("series_name")
        
        if not series_name:
            return jsonify({"success": False, "error": "Missing series name"}), 400
        
        manual_complete[series_name] = True
        save_cache("complete")

        # Also mark all seasons complete for this show
        season_keys = []
        try:
            data_all = organize_data()
            for show in data_all.get("shows", []):
                if show.get("series_name") == series_name:
                    for season in show.get("seasons", []):
                        snum = season.get("season_number")
                        if snum is None:
                            continue
                        season_key = f"{series_name}_{snum}"
                        season_complete[season_key] = True
                        season_keys.append(season_key)
                    break
        except Exception:
            season_keys = []
        if season_keys:
            save_cache("season_complete")

        # Record action for undo
        record_action("mark_complete", {"series_name": series_name, "season_keys": season_keys})
        cache["data"] = None
        cache["time"] = None
        print(f"✓ Marked as complete: {series_name}")
        return jsonify({"success": True})
    
    except Exception as e:
        print(f"✗ Mark complete error: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/mark_season_complete", methods=["POST"])
def api_mark_season_complete():
    """Mark a season as 100% complete"""
    try:
        data = request.get_json()
        series_name = data.get("series_name")
        season = data.get("season")
        
        if not series_name or season is None:
            return jsonify({"success": False, "error": "Missing series name or season"}), 400
        
        season_key = f"{series_name}_{season}"
        season_complete[season_key] = True
        save_cache("season_complete")
        # Record action for undo
        record_action("mark_season_complete", {"series_name": series_name, "season": season, "season_key": season_key})
        cache["data"] = None
        cache["time"] = None
        print(f"✓ Marked season as complete: {series_name} S{season}")
        return jsonify({"success": True})
    
    except Exception as e:
        print(f"✗ Mark season complete error: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/clear_tmdb_cache", methods=["POST"])
def api_clear_tmdb_cache():
    try:
        global series_cache, poster_cache, tmdb_retry, genre_recs_cache
        series_cache = {}
        poster_cache = {}
        tmdb_retry = set()
        genre_recs_cache = {"sig": None, "top_genre": None, "ts": 0, "movies": [], "shows": []}
        save_cache("series")
        save_cache("poster")

        # Also reset persisted show totals so they can be rebuilt from fresh TMDB series info.
        try:
            save_json_file(SHOW_TOTALS_FILE, {})
        except Exception:
            pass
        
        cache["data"] = None
        cache["time"] = None
        
        print("✓ TMDB cache cleared")
        # Kick off background rebuild so posters/totals repopulate without blocking /api/history.
        prewarm_tmdb_cache_async(force=True)
        
        return jsonify({"success": True})
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# API: Manually set total episode count for a show
#
# This is used when TMDB matching fails for a title and the UI shows
# "watched/? (fetching TMDB totals...)". It persists the provided total in
# SHOW_TOTALS_FILE (with tolerant aliases) so the UI can show watched/total.
@app.route("/api/set_show_total", methods=["POST"])
def api_set_show_total():
    try:
        data = request.get_json(force=True, silent=True) or {}
        series_name = (data.get("series_name") or "").strip()
        total = _coerce_int(data.get("total"), 0) or 0
        if not series_name or total <= 0:
            return jsonify({"success": False, "error": "Missing series_name or invalid total"}), 400

        # Reuse the same persistence logic as TMDB results.
        _update_show_totals_from_tmdb(series_name, None, {"total_episodes_in_series": total, "seasons": {}})

        # Ensure the organized cache is invalidated and clients refresh immediately.
        _touch_tmdb_cache_and_notify()
        return jsonify({"success": True})
    except Exception as e:
        print(f"✗ set_show_total error: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# API: Clear watch history
#
# This endpoint deletes all watch records from the history file and resets
# internal caches.  It does not touch manual completions or season progress.
# When invoked from the UI, a full page reload will occur so charts and
# statistics refresh to their empty state.
@app.route("/api/clear_history", methods=["POST"])
def api_clear_history():
    try:
        # Overwrite the watch history file with an empty list
        save_all_records([])
        # Reset cached data so future calls reload from disk
        cache["data"] = None
        cache["time"] = None
        print("✓ Watch history cleared")
        return jsonify({"success": True})
    except Exception as e:
        print(f"Clear history error: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# API: Clear manual progress
#
# This endpoint resets all manual completion markers and season completion
# flags.  It clears the in-memory dictionaries and writes empty JSON files
# back to disk.  After clearing, the UI should reload to rebuild progress
# from watch history alone.
@app.route("/api/clear_progress", methods=["POST"])
def api_clear_progress():
    try:
        global manual_complete, season_complete
        manual_complete = {}
        season_complete = {}
        # Persist the cleared structures to disk
        save_cache("complete")
        save_cache("season_complete")
        print("✓ Progress cleared")
        return jsonify({"success": True})
    except Exception as e:
        print(f"Clear progress error: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# API: Clear insights
#
# Currently genre insights and recommendations are computed dynamically from
# watch history.  There is no separate persistent cache for insights, but
# clearing this endpoint can be used to reset any cached data structures in
# memory and force a refresh on the next request.
@app.route("/api/clear_insights", methods=["POST"])
def api_clear_insights():
    try:
        # Reset data cache so that subsequent requests recompute insights
        cache["data"] = None
        cache["time"] = None
        try:
            global genre_insights_cache, genre_recs_cache
            genre_insights_cache = {"sig": None, "data": None}
            genre_recs_cache = {"sig": None, "top_genre": None, "ts": 0, "movies": [], "shows": []}
        except Exception:
            pass
        print("✓ Insights cleared")
        return jsonify({"success": True})
    except Exception as e:
        print(f"Clear insights error: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/bulk_delete_episodes", methods=["POST"])
def api_bulk_delete_episodes():
    """Bulk delete episodes"""
    try:
        data = request.get_json()
        episodes_to_delete = data.get("episodes", [])
        
        if not episodes_to_delete:
            return jsonify({"success": False, "error": "No episodes provided"}), 400
        
        records = get_all_records()
        to_delete = []
        filtered = []
        for r in records:
            should_delete = False
            if r.get("type") == "Episode":
                for ep_data in episodes_to_delete:
                    if (r.get("series_name") == ep_data["series_name"] and 
                        r.get("season") == ep_data["season"] and 
                        r.get("episode") == ep_data["episode"]):
                        should_delete = True
                        to_delete.append(r)
                        break
            if not should_delete:
                filtered.append(r)
        deleted = len(to_delete)
        if deleted > 0:
            record_action("bulk_delete", {"records": to_delete})
        save_all_records(filtered)
        cache["data"] = None
        cache["time"] = None
        print(f"✓ Bulk deleted {deleted} episodes")
        return jsonify({"success": True, "deleted": deleted})
    
    except Exception as e:
        print(f"✗ Bulk delete error: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/delete_movie", methods=["POST"])
def api_delete_movie():
    try:
        data = request.get_json()
        movie_name = data.get("name")
        movie_year = data.get("year")
        # Gather all records to be deleted so we can restore them on undo
        records = get_all_records()
        to_delete = []
        filtered = []
        for r in records:
            if r.get("type") == "Movie" and r.get("name") == movie_name and r.get("year") == movie_year:
                to_delete.append(r)
            else:
                filtered.append(r)
        deleted = len(to_delete)
        # Record the action for undo
        if deleted > 0:
            record_action("delete_movie", {"name": movie_name, "year": movie_year, "records": to_delete})
        save_all_records(filtered)
        cache["data"] = None
        cache["time"] = None
        print(f"✓ Deleted movie: {movie_name} ({movie_year}) - {deleted} records")
        return jsonify({"success": True, "deleted": deleted})
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/delete_show", methods=["POST"])
def api_delete_show():
    try:
        data = request.get_json()
        series_name = data.get("series_name")
        # If the series was manually marked complete remove the flag, record it for undo
        if series_name in manual_complete:
            del manual_complete[series_name]
            save_cache("complete")
        # Gather all episode records to be deleted
        records = get_all_records()
        to_delete = []
        filtered = []
        for r in records:
            if r.get("type") == "Episode" and r.get("series_name") == series_name:
                to_delete.append(r)
            else:
                filtered.append(r)
        deleted = len(to_delete)
        if deleted > 0:
            record_action("delete_show", {"series_name": series_name, "records": to_delete})
        save_all_records(filtered)
        cache["data"] = None
        cache["time"] = None
        print(f"✓ Deleted show: {series_name} - {deleted} records")
        return jsonify({"success": True, "deleted": deleted})
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/delete_season", methods=["POST"])
def api_delete_season():
    try:
        data = request.get_json()
        series_name = data.get("series_name")
        season = data.get("season")
        
        records = get_all_records()
        to_delete = []
        filtered = []
        for r in records:
            if r.get("type") == "Episode" and r.get("series_name") == series_name and r.get("season") == season:
                to_delete.append(r)
            else:
                filtered.append(r)
        deleted = len(to_delete)
        if deleted > 0:
            record_action("delete_season", {"series_name": series_name, "season": season, "records": to_delete})
        save_all_records(filtered)
        cache["data"] = None
        cache["time"] = None
        print(f"✓ Deleted season: {series_name} S{season} - {deleted} records")
        return jsonify({"success": True, "deleted": deleted})
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/delete_episode", methods=["POST"])
def api_delete_episode():
    try:
        data = request.get_json()
        series_name = data.get("series_name")
        season = data.get("season")
        episode = data.get("episode")
        
        records = get_all_records()
        to_delete = []
        filtered = []
        for r in records:
            if (r.get("type") == "Episode" and 
                r.get("series_name") == series_name and 
                r.get("season") == season and 
                r.get("episode") == episode):
                to_delete.append(r)
            else:
                filtered.append(r)
        deleted = len(to_delete)
        if deleted > 0:
            record_action("delete_episode", {"series_name": series_name, "season": season, "episode": episode, "records": to_delete})
        save_all_records(filtered)
        cache["data"] = None
        cache["time"] = None
        print(f"✓ Deleted episode: {series_name} S{season}E{episode} - {deleted} records")
        return jsonify({"success": True, "deleted": deleted})
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# NEW API ENDPOINTS FOR ADVANCED FEATURES

@app.route("/api/ratings", methods=["GET"])
def api_get_ratings():
    """
    Return the stored ratings for all movies and shows.  The returned JSON
    is keyed by item identifier (e.g. "movie|Inception|2010" or "show|Breaking Bad").
    """
    try:
        return jsonify(load_ratings())
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/rate", methods=["POST"])
def api_rate():
    """
    Submit or update a rating for a movie or show.  Expects JSON with
    fields 'id' (the unique item identifier), 'rating' (1-5), and
    optional 'note'.  On success the data is saved to RATINGS_FILE and
    the cache is cleared so the rating is included on next load.
    """
    try:
        data = request.get_json() or {}
        item_id = data.get("id") or data.get("key")
        rating = data.get("rating")
        note = data.get("note", "")
        if not item_id or rating is None:
            return jsonify({"success": False, "error": "Missing id or rating"}), 400
        # Ensure rating is within 1-5
        try:
            rating_int = int(rating)
        except Exception:
            return jsonify({"success": False, "error": "Rating must be a number"}), 400
        if rating_int < 1 or rating_int > 5:
            return jsonify({"success": False, "error": "Rating must be between 1 and 5"}), 400
        ratings_cache = load_ratings()
        ratings_cache[item_id] = {"rating": rating_int, "note": note}
        save_ratings(ratings_cache)
        # clear cache so UI picks up new rating
        cache["data"] = None
        cache["time"] = None
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/watch_time", methods=["GET"])
def api_watch_time():
    """
    Compute overall watch time analytics.  The total watch time and breakdown
    by movies and TV shows are calculated in hours.  A monthly trend
    indicates total hours watched per month.  Average watch time per day
    is computed, as well as a count of binge sessions (defined as days
    where at least 3 hours of content were watched).
    """
    try:
        records = get_all_records()
        total_minutes = 0
        movie_minutes = 0
        show_minutes = 0
        monthly_minutes = defaultdict(int)
        daily_minutes = defaultdict(int)
        for rec in records:
            ts = rec.get("timestamp")
            if not ts:
                continue
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "").split("+")[0])
            except Exception:
                continue
            month_key = dt.strftime("%Y-%m")
            day_key = dt.date()
            if rec.get("type") == "Movie":
                # Use runtime from record if available; otherwise assume 110 minutes
                runtime = rec.get("runtime")
                minutes = runtime if isinstance(runtime, (int, float)) else 110
                movie_minutes += minutes
            elif rec.get("type") == "Episode":
                # Use runtime from record if available; otherwise assume 45 minutes
                runtime = rec.get("runtime")
                minutes = runtime if isinstance(runtime, (int, float)) else 45
                show_minutes += minutes
            else:
                continue
            total_minutes += minutes
            monthly_minutes[month_key] += minutes
            daily_minutes[day_key] += minutes
        # Convert aggregated minutes into hours with rounding
        total_hours = round(total_minutes / 60.0, 2)
        movies_hours = round(movie_minutes / 60.0, 2)
        shows_hours = round(show_minutes / 60.0, 2)
        # monthly list of dicts for chart consumption
        monthly_list = []
        for m, mins in sorted(monthly_minutes.items()):
            monthly_list.append({"month": m, "hours": round(mins / 60.0, 2)})
        avg_hours_per_day = round((total_minutes / 60.0) / len(daily_minutes), 2) if daily_minutes else 0
        # Determine binge sessions: list of dates where >= 3 hours (180 minutes) watched
        binge_dates = [day.isoformat() for day, mins in daily_minutes.items() if mins >= 180]
        return jsonify({
            "success": True,
            "total": total_hours,
            "movies": movies_hours,
            "shows": shows_hours,
            "monthly": monthly_list,
            "avg_per_day": avg_hours_per_day,
            # use 'binges' key to align with frontend expectation (plural)
            "binges": binge_dates
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/undo", methods=["POST"])
def api_undo():
    """
    Undo the last user action (e.g., delete or mark complete).  The action
    history stores the last 20 actions and their payloads.  Depending on
    the action type the undo operation will restore deleted records or
    reverse the completed status.  If no actions are available to undo an
    error is returned.
    """
    try:
        history = load_action_history()
        if not history:
            return jsonify({"success": False, "error": "No actions to undo"}), 400
        last = history.pop()
        save_action_history(history)
        action_type = last.get("type")
        payload = last.get("payload", {})
        if action_type in ("delete_movie", "delete_show", "delete_season", "delete_episode", "bulk_delete"):
            # Restore deleted records
            recs = payload.get("records", [])
            existing = get_all_records()
            for rec in recs:
                existing.append(rec)
            save_all_records(existing)
            # Clearing cache ensures UI refreshes
            cache["data"] = None
            cache["time"] = None
        elif action_type == "mark_complete":
            # Remove manual completion flag for a series
            series_name = payload.get("series_name")
            if series_name and series_name in manual_complete:
                del manual_complete[series_name]
                save_cache("complete")
            season_keys = payload.get("season_keys") or []
            if season_keys:
                for season_key in season_keys:
                    if season_key in season_complete:
                        del season_complete[season_key]
                save_cache("season_complete")
            cache["data"] = None
            cache["time"] = None
        elif action_type == "mark_season_complete":
            # Remove manual completion flag for a season
            season_key = payload.get("season_key")
            if season_key and season_key in season_complete:
                del season_complete[season_key]
                save_cache("season_complete")
            cache["data"] = None
            cache["time"] = None
        return jsonify({"success": True, "undone": action_type})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/manifest.json')
def manifest():
    """
    Serve a minimal web app manifest for PWA installation.  The icons
    referenced here use placeholder images served from placeholder.com.
    """
    try:
        manifest_data = {
            "name": "Jellyfin Watch Tracker",
            "short_name": "Watch Tracker",
            "start_url": "/",
            "display": "standalone",
            "background_color": "#0a0e27",
            "theme_color": "#8b9cff",
            "icons": [
                {"src": "https://via.placeholder.com/192", "sizes": "192x192", "type": "image/png"},
                {"src": "https://via.placeholder.com/512", "sizes": "512x512", "type": "image/png"}
            ]
        }
        return jsonify(manifest_data)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/sw.js')
def service_worker():
    """
    Serve a simple service worker script enabling offline caching of the
    application shell.  The worker caches the homepage and manifest on
    install and serves cached responses on subsequent requests when
    available.  Update CACHE_NAME to invalidate the cache when code
    changes.
    """
    sw_script = """
const CACHE_NAME = 'watch-tracker-v21';
const urlsToCache = [
  '/manifest.json'
];
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(urlsToCache);
    })
  );
  self.skipWaiting();
});
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((names) => Promise.all(
      names.map((n) => (n !== CACHE_NAME ? caches.delete(n) : Promise.resolve()))
    ))
  );
  self.clients.claim();
});
self.addEventListener('fetch', (event) => {
  if (event.request.mode === 'navigate') {
    // Always prefer the network for HTML so UI/JS updates aren't stuck behind cache.
    event.respondWith(fetch(event.request, { cache: 'no-store' }));
    return;
  }
  const url = new URL(event.request.url);
  if (urlsToCache.includes(url.pathname)) {
    event.respondWith(
      caches.match(event.request).then((response) => response || fetch(event.request))
    );
    return;
  }
  // Do not cache other requests (including /api/*) to avoid stale data.
  event.respondWith(fetch(event.request));
});
"""
    return Response(sw_script, mimetype='application/javascript')


@app.route("/api/jellyfin_import", methods=["POST"])
def api_jellyfin_import():
    ok, msg, count = jellyfin_import()
    return jsonify({"success": ok, "message": msg if ok else None, "error": None if ok else msg, "imported": count})

@app.route("/api/jellyfin_poll_now", methods=["GET", "POST"])
def api_jellyfin_poll_now():
    """
    Debug/utility endpoint: run the Jellyfin poll sync once and report how many
    records were appended. Useful when webhooks are missed (autoplay/credits skip).
    """
    try:
        if not JELLYFIN_POLL_ENABLED:
            return jsonify({
                "success": False,
                "error": "Jellyfin polling is disabled (set JELLYFIN_POLL_ENABLED=true to enable).",
                "config": {
                    "enabled": bool(JELLYFIN_POLL_ENABLED),
                    "interval_s": int(JELLYFIN_POLL_INTERVAL_S or 0),
                    "limit": int(JELLYFIN_POLL_LIMIT or 0),
                    "has_url": bool(JELLYFIN_URL),
                    "has_api_key": bool(JELLYFIN_API_KEY),
                },
            }), 400
        if not JELLYFIN_URL:
            return jsonify({
                "success": False,
                "error": "JELLYFIN_URL is not set.",
                "config": {
                    "enabled": bool(JELLYFIN_POLL_ENABLED),
                    "interval_s": int(JELLYFIN_POLL_INTERVAL_S or 0),
                    "limit": int(JELLYFIN_POLL_LIMIT or 0),
                    "has_url": bool(JELLYFIN_URL),
                    "has_api_key": bool(JELLYFIN_API_KEY),
                },
            }), 400
        if not JELLYFIN_API_KEY:
            return jsonify({
                "success": False,
                "error": "JELLYFIN_API_KEY is not set.",
                "config": {
                    "enabled": bool(JELLYFIN_POLL_ENABLED),
                    "interval_s": int(JELLYFIN_POLL_INTERVAL_S or 0),
                    "limit": int(JELLYFIN_POLL_LIMIT or 0),
                    "has_url": bool(JELLYFIN_URL),
                    "has_api_key": bool(JELLYFIN_API_KEY),
                },
            }), 400

        added = jellyfin_poll_sync_once()
        state = load_json_file(JELLYFIN_POLL_STATE_FILE)
        # Include a small "peek" so users can confirm Jellyfin is returning played items.
        peek = []
        try:
            headers = {"X-Emby-Token": JELLYFIN_API_KEY}
            uid = (state or {}).get("user_id")
            if uid:
                params = {
                    "Filters": "IsPlayed",
                    "Recursive": "true",
                    "Fields": "Genres,UserData",
                    "IncludeItemTypes": "Movie,Episode",
                    "StartIndex": 0,
                    "Limit": 10,
                    "SortBy": "DatePlayed",
                    "SortOrder": "Descending",
                }
                r2 = requests.get(f"{JELLYFIN_URL}/Users/{uid}/Items", headers=headers, params=params, timeout=20)
                if r2.status_code == 200:
                    page = r2.json() or {}
                    for it in (page.get("Items", []) or [])[:10]:
                        lp = (it.get("UserData") or {}).get("LastPlayedDate")
                        peek.append({
                            "id": it.get("Id"),
                            "type": it.get("Type"),
                            "name": it.get("Name"),
                            "series_name": it.get("SeriesName"),
                            "season": it.get("ParentIndexNumber"),
                            "episode": it.get("IndexNumber"),
                            "last_played": lp,
                            "last_played_epoch": _parse_iso_to_epoch_seconds(lp),
                        })
        except Exception:
            peek = []

        return jsonify({
            "success": True,
            "added": int(added or 0),
            "state": state,
            "peek": peek,
            "config": {
                "enabled": bool(JELLYFIN_POLL_ENABLED),
                "interval_s": int(JELLYFIN_POLL_INTERVAL_S or 0),
                "limit": int(JELLYFIN_POLL_LIMIT or 0),
                "lookback_s": int(JELLYFIN_POLL_LOOKBACK_S or 0),
                "has_url": bool(JELLYFIN_URL),
                "has_api_key": bool(JELLYFIN_API_KEY),
            },
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/sonarr_import", methods=["POST"])
def api_sonarr_import():
    ok, msg, count = sonarr_import()
    return jsonify({"success": ok, "message": msg if ok else None, "error": None if ok else msg, "imported": count})


@app.route("/api/radarr_import", methods=["POST"])
def api_radarr_import():
    ok, msg, count = radarr_import()
    return jsonify({"success": ok, "message": msg if ok else None, "error": None if ok else msg, "imported": count})


@app.route("/api/upload_poster", methods=["POST"])
def api_upload_poster():
    try:
        if 'poster' not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400
        
        file = request.files['poster']
        name = request.form.get('name', '')
        year = request.form.get('year', '')
        media_type = request.form.get('type', 'movie')
        
        if file.filename == '':
            return jsonify({"success": False, "error": "Empty filename"}), 400
        
        os.makedirs(POSTER_DIR, exist_ok=True)
        
        filename = secure_filename(f"{media_type}_{name}_{year}_{file.filename}")
        filepath = os.path.join(POSTER_DIR, filename)
        file.save(filepath)
        
        poster_url = f"/poster/{filename}"
        custom_key = f"{media_type}_{name}_{year}"
        custom_posters[custom_key] = poster_url
        save_cache("custom")
        
        cache["data"] = None
        cache["time"] = None
        
        return jsonify({"success": True, "url": poster_url})
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/poster/<filename>")
def serve_poster(filename):
    return send_file(os.path.join(POSTER_DIR, filename))


@app.route("/api/export/<fmt>")
def api_export(fmt):
    if fmt == "json":
        # Build a JSON array of records with optional rating and note fields
        try:
            records = get_all_records()
            ratings_cache = load_ratings()
            enriched = []
            for r in records:
                entry = r.copy()
                # Determine rating key based on type
                if r.get("type") == "Movie":
                    key = f"movie|{r.get('name')}|{r.get('year')}"
                elif r.get("type") == "Episode":
                    key = f"show|{r.get('series_name')}"
                else:
                    key = None
                if key and key in ratings_cache:
                    entry["rating"] = ratings_cache[key].get("rating")
                    entry["note"] = ratings_cache[key].get("note")
                enriched.append(entry)
            content = json.dumps(enriched, ensure_ascii=False)
            return send_file(io.BytesIO(content.encode()), mimetype="application/json", as_attachment=True, download_name=f"watch_history_{datetime.now().strftime('%Y%m%d')}.json")
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    elif fmt == "csv":
        try:
            records = get_all_records()
            ratings_cache = load_ratings()
            # Define CSV columns including rating and note
            fieldnames = ["timestamp", "type", "name", "year", "series_name", "season", "episode", "user", "genres", "source", "rating", "note"]
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                row = {k: r.get(k) for k in ["timestamp", "type", "name", "year", "series_name", "season", "episode", "user", "source"]}
                # Convert genres list to string
                row["genres"] = ", ".join(r.get("genres", []))
                # Determine rating key
                if r.get("type") == "Movie":
                    key = f"movie|{r.get('name')}|{r.get('year')}"
                elif r.get("type") == "Episode":
                    key = f"show|{r.get('series_name')}"
                else:
                    key = None
                if key and key in ratings_cache:
                    row["rating"] = ratings_cache[key].get("rating")
                    row["note"] = ratings_cache[key].get("note")
                else:
                    row["rating"] = ''
                    row["note"] = ''
                writer.writerow(row)
            return send_file(
                io.BytesIO(output.getvalue().encode()),
                mimetype="text/csv",
                as_attachment=True,
                download_name=f"watch_history_{datetime.now().strftime('%Y%m%d')}.csv"
            )
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": "invalid format"}), 400


if __name__ == "__main__":
    os.makedirs("/data", exist_ok=True)
    os.makedirs(POSTER_DIR, exist_ok=True)
    try:
        if not os.path.exists(TMDB_TOUCH_FILE):
            with open(TMDB_TOUCH_FILE, "w") as f:
                f.write(str(time.time_ns()))
    except Exception:
        pass
    if not os.path.exists(DATA_FILE):
        open(DATA_FILE, "a").close()
    load_caches()
    refresh_history_sig_fast()
    # Rebuild posters/series totals in the background on startup.
    prewarm_tmdb_cache_async()
    # Poll Jellyfin for newly played items as a safety net when webhooks are missed
    # (e.g. autoplay "Next episode" flows).
    start_jellyfin_poll_thread()
    
    print("=" * 60)
    print("JELLYFIN WATCH TRACKER - VERSION 35")
    print("ADVANCED SEARCH, RATINGS & PWA EDITION")
    print("=" * 60)
    print(f"UI:      http://0.0.0.0:5000")
    print(f"Webhook: http://0.0.0.0:5000/webhook")
    print(f"Records: {len(get_all_records())}")
    print("=" * 60)
    print("\n✨ NEW FEATURES:")
    print("  • 🔍 Advanced Search & Filters (genre, year range, status, type)")
    print("  • ⭐ Rating System with notes and export support")
    print("  • 📱 Better Mobile Experience with responsive design")
    print("  • ⏱️ Watch Time Analytics with monthly trends and binge detection")
    print("  • ↩️ Undo Button to revert recent actions")
    print("  • 🔄 Auto-Refresh with configurable intervals and smart pausing")
    print("  • 🎨 Themes & Customization including dark/light/amoled/solarized/nord")
    print("  • 📲 Progressive Web App support with offline caching")
    print("=" * 60 + "\n")
    
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)