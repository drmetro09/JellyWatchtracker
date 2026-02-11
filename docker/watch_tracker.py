from flask import Flask, request, jsonify, send_file
from datetime import datetime, timedelta
import json
import os
from collections import Counter, defaultdict
import io
import requests
import traceback
from werkzeug.utils import secure_filename
import csv
import time
import re
import unicodedata
import requests

app = Flask(__name__)

DATA_FILE = "/data/watch_history.json"
POSTER_DIR = "/data/custom_posters"
JELLYFIN_URL = os.getenv("JELLYFIN_URL", "").rstrip("/")
JELLYFIN_API_KEY = os.getenv("JELLYFIN_API_KEY", "")
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
SONARR_URL = os.getenv("SONARR_URL", "").rstrip("/")
SONARR_API_KEY = os.getenv("SONARR_API_KEY", "")
RADARR_URL = os.getenv("RADARR_URL", "").rstrip("/")
RADARR_API_KEY = os.getenv("RADARR_API_KEY", "")

cache = {"data": None, "time": None}
poster_cache = {}
series_cache = {}
custom_posters = {}
manual_complete = {}
season_complete = {}


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


def get_tmdb_poster(title, year=None, media_type="movie"):
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

    
def get_tmdb_series_info(series_name, force_refresh=False):
    if not TMDB_API_KEY:
        return None
    
    if not force_refresh and series_name in series_cache:
        cached = series_cache.get(series_name)
        if cached:
            return cached
        elif cached is None:
            return None
    
    try:
        search_queries = [
            series_name,
            series_name.replace(":", ""),
            series_name.replace("-", " "),
            series_name.replace("'", ""),
            series_name.split(":")[0].strip() if ":" in series_name else series_name,
        ]
        
        best_result = None
        best_score = 0
        
        for query in search_queries:
            if not query.strip():
                continue
                
            r = requests.get(
                "https://api.themoviedb.org/3/search/tv",
                params={"api_key": TMDB_API_KEY, "query": query, "language": "en-US"},
                timeout=10
            )
            
            if r.status_code != 200:
                continue
            
            data = r.json()
            results = data.get("results", [])
            
            if results and len(results) > 0:
                for result in results[:10]:
                    result_name = result.get("name", "").lower()
                    query_lower = query.lower()
                    
                    if result_name == query_lower:
                        best_result = result
                        best_score = 100
                        break
                    
                    if query_lower in result_name or result_name in query_lower:
                        score = 50
                        if score > best_score:
                            best_result = result
                            best_score = score
                
                if best_score == 100:
                    break
        
        if not best_result:
            series_cache[series_name] = None
            save_cache("series")
            return None
        
        tv_id = best_result.get("id")
        
        r2 = requests.get(
            f"https://api.themoviedb.org/3/tv/{tv_id}",
            params={"api_key": TMDB_API_KEY, "language": "en-US"},
            timeout=10
        )
        
        if r2.status_code != 200:
            series_cache[series_name] = None
            save_cache("series")
            return None
        
        detail = r2.json()
        seasons_info = {}
        total_all_episodes = 0
        
        seasons = detail.get("seasons", [])
        for season in seasons:
            snum = season.get("season_number")
            ep_count = season.get("episode_count", 0)
            
            if snum is not None and snum > 0:
                seasons_info[snum] = {
                    "total_episodes": ep_count,
                    "name": season.get("name", f"Season {snum}")
                }
                total_all_episodes += ep_count
        
        info = {
            "seasons": seasons_info,
            "total_episodes_in_series": total_all_episodes,
            "tmdb_name": best_result.get("name", ""),
            "tmdb_id": tv_id
        }
        
        series_cache[series_name] = info
        save_cache("series")
        return info
        
    except Exception as e:
        series_cache[series_name] = None
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
    cache["data"] = None
    cache["time"] = None


def append_record(record):
    os.makedirs("/data", exist_ok=True)
    with open(DATA_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())
    cache["data"] = None
    cache["time"] = None


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
    if cache["data"] and cache["time"]:
        if (datetime.now() - cache["time"]).total_seconds() < 3:
            return cache["data"]

    records = get_all_records()
    movies = {}
    shows = {}
    genre_counter = Counter()
    genre_breakdown = defaultdict(lambda: {"movies": [], "shows": []})

    for r in records:
        rtype = r.get("type")
        
        if rtype == "Movie":
            key = f"{r.get('name')}_{r.get('year')}"
            if key not in movies:
                movies[key] = {
                    "name": r.get("name"),
                    "year": r.get("year"),
                    "watch_count": 0,
                    "watches": [],
                    "genres": r.get("genres", []),
                    "poster": None
                }
            movies[key]["watch_count"] += 1
            movies[key]["watches"].append({"timestamp": r.get("timestamp"), "user": r.get("user")})
            for g in r.get("genres", []):
                genre_counter[g] += 1

        elif rtype == "Episode":
            series = r.get("series_name") or "Unknown Series"
            if series not in shows:
                shows[series] = {
                    "series_name": series,
                    "seasons": {},
                    "genres": r.get("genres", []),
                    "poster": None,
                    "year": r.get("year")
                }
            
            season = r.get("season") or 0
            
            if season == 999:
                continue
            
            if season not in shows[series]["seasons"]:
                shows[series]["seasons"][season] = {"season_number": season, "episodes": []}
            
            ep_list = shows[series]["seasons"][season]["episodes"]
            ep_no = r.get("episode") or 0
            
            found = False
            for ep in ep_list:
                if ep["episode"] == ep_no:
                    ep["watch_count"] += 1
                    ep["watches"].append({"timestamp": r.get("timestamp"), "user": r.get("user")})
                    found = True
                    break
            
            if not found:
                ep_list.append({
                    "name": r.get("name"),
                    "season": season,
                    "episode": ep_no,
                    "watch_count": 1,
                    "watches": [{"timestamp": r.get("timestamp"), "user": r.get("user")}]
                })

    for m in movies.values():
        if not m["poster"]:
            m["poster"] = get_tmdb_poster(m["name"], m["year"], "movie")
        for g in m["genres"]:
            genre_breakdown[g]["movies"].append(m)

    for series_name, s in shows.items():
        if not s["poster"]:
            s["poster"] = get_tmdb_poster(series_name, s.get("year"), "tv")
        
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
        
        # Check if manually marked complete
        is_manually_complete = series_name in manual_complete
        
        if is_manually_complete:
            s["total_episodes"] = total_watched
            s["total_episodes_possible"] = total_watched
            s["total_watches"] = sum(x["total_watches"] for x in seasons_list)
            s["completion_percentage"] = 100
            s["has_tmdb_data"] = False
            s["manually_completed"] = True
            s["seasons"] = seasons_list
            s["has_new_content"] = False  # NEW: Flag for new content after manual complete
        else:
            series_info = get_tmdb_series_info(series_name)
            
            if series_info and series_info.get("total_episodes_in_series", 0) > 0:
                total_possible = series_info["total_episodes_in_series"]
                
                for season in seasons_list:
                    snum = season["season_number"]
                    season_key = f"{series_name}_{snum}"
                    
                    if season_key in season_complete:
                        season["completion_percentage"] = 100
                        season["manually_completed"] = True
                    elif snum > 0 and snum in series_info.get("seasons", {}):
                        season["total_episodes"] = series_info["seasons"][snum]["total_episodes"]
                        season["has_tmdb_data"] = True
                        season_watched = season["episode_count"]
                        season_total = season["total_episodes"]
                        pct = round((season_watched / season_total) * 100) if season_total > 0 else 0
                        # Don't auto-show 100% unless manually completed
                        if pct >= 100 and not season.get("manually_completed"):
                            pct = 99
                        season["completion_percentage"] = pct
                
                s["total_episodes"] = total_watched
                s["total_episodes_possible"] = total_possible
                s["total_watches"] = sum(x["total_watches"] for x in seasons_list)
                
                raw_pct = round((total_watched / total_possible) * 100) if total_possible > 0 else 0
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
                total_possible = total_watched
                s["total_episodes"] = total_watched
                s["total_episodes_possible"] = total_possible
                s["total_watches"] = sum(x["total_watches"] for x in seasons_list)
                s["completion_percentage"] = 100
                s["has_tmdb_data"] = False
                s["manually_completed"] = False
                s["seasons"] = seasons_list
                s["has_new_content"] = False
        
        for g in s["genres"]:
            genre_breakdown[g]["shows"].append(s)

    movies_list = sorted(movies.values(), key=lambda x: x["watches"][0]["timestamp"] if x.get("watches") else "", reverse=True)
    shows_list = sorted(shows.values(), key=lambda x: x.get("total_watches", 0), reverse=True)

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
        "completed": completed
    }

    cache["data"] = result
    cache["time"] = datetime.now()
    
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
<html><head><meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1.0,maximum-scale=1.0,user-scalable=no"/><title>Jellyfin Watch Tracker</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
html{font-size:16px;height:100%}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:linear-gradient(135deg,#0a0e27 0%,#1a1f3a 100%);color:#fff;padding:15px;min-height:100%;overflow-y:auto}
.container{max-width:1800px;margin:0 auto;width:100%}
.header{background:linear-gradient(135deg,rgba(139,156,255,0.15),rgba(255,107,107,0.15));backdrop-filter:blur(20px);border-radius:20px;padding:30px;margin-bottom:30px;border:1px solid rgba(255,255,255,0.1);box-shadow:0 8px 32px rgba(0,0,0,0.3);display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:20px}
.header-title{display:flex;align-items:center;gap:15px}
.logo{font-size:2.5em;filter:drop-shadow(0 4px 15px rgba(139,156,255,0.5))}
.header h1{font-size:2em;font-weight:700;background:linear-gradient(135deg,#8b9cff,#ff8585);-webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:-1px}
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
.zoom-slider{width:130px;height:5px;border-radius:5px;background:rgba(255,255,255,0.1);outline:none;-webkit-appearance:none}
.zoom-slider::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:16px;height:16px;border-radius:50%;background:linear-gradient(135deg,#8b9cff,#6b8cff);cursor:pointer;box-shadow:0 2px 8px rgba(139,156,255,0.5)}
.zoom-slider::-moz-range-thumb{width:16px;height:16px;border-radius:50%;background:linear-gradient(135deg,#8b9cff,#6b8cff);cursor:pointer;border:none;box-shadow:0 2px 8px rgba(139,156,255,0.5)}
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
<button class="filter-btn active" onclick="setFilter('all',event)">All</button>
<button class="filter-btn" onclick="setFilter('movies',event)">Movies</button>
<button class="filter-btn" onclick="setFilter('shows',event)">TV Shows</button>
<button class="filter-btn" onclick="setFilter('incomplete',event)">📋 Incomplete</button>
<select id="sort" onchange="renderHistory()">
<option value="recent">Most Recent</option>
<option value="oldest">Oldest First</option>
<option value="most_watched">Most Watched</option>
<option value="alphabetical">A-Z</option>
<option value="incomplete_first">Incomplete First</option>
</select>
<div class="search-box">
<input type="text" id="search" placeholder="🔍 Search..." oninput="renderHistory()">
</div>
<div class="view-toggle">
<button class="active" onclick="setView('list',event)">📋 List</button>
<button onclick="setView('grid',event)">⊞ Grid</button>
<button id="gridSelectBtn" onclick="toggleGridSelectMode(event)" style="display:none">☑️ Select</button>
<button id="gridSelectAllBtn" onclick="gridSelectAll(event)" style="display:none">✓ All</button>
<button id="gridClearBtn" onclick="gridClearSelection(event)" style="display:none">✕ Clear</button>
</div>
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
</div>

<div id="progress-tab" class="tab-content">
<h2 style="margin-bottom:25px;font-size:1.8em;font-weight:700">📈 Progress Tracker</h2>

<div style="background:rgba(20,25,45,0.8);backdrop-filter:blur(10px);border-radius:16px;padding:20px;margin-bottom:25px;border:1px solid rgba(255,255,255,0.08)">
  <div style="display:flex;justify-content:space-between;align-items:center;cursor:pointer" onclick="toggleSection('movies-progress-list','movies-progress-arrow')">
    <h3 style="font-size:1.4em;margin:0;color:#8b9cff">🎬 Movies Progress</h3>
    <span id="movies-progress-arrow" style="transition:transform 0.3s;display:inline-block">▶</span>
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
    <span id="shows-progress-arrow" style="transition:transform 0.3s;display:inline-block">▶</span>
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
<label><input type="radio" name="entryType" value="Episode" onchange="toggleEpisodeFields()"> 📺 TV Episode</label>
</div>
</div>

<div class="form-group">
<label>Title / Episode Name</label>
<input type="text" id="entryName" placeholder="e.g., The Dark Knight">
</div>

<div class="form-group">
<label>Year (optional)</label>
<input type="number" id="entryYear" placeholder="e.g., 2024" min="1900" max="2026">
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
<label>Episode Number</label>
<input type="number" id="entryEpisode" placeholder="e.g., 1" min="1">
</div>
</div>

<div class="form-group">
<label>Watch Date</label>
<input type="date" id="entryDate">
</div>

<div class="form-group">
<label>Genres (optional, comma-separated)</label>
<input type="text" id="entryGenres" placeholder="e.g., Drama, Thriller">
</div>

<button class="btn" style="width:100%;margin-top:15px" onclick="submitManualEntry()">✓ Add Watch Record</button>
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
<input type="text" id="entryEpisode" placeholder="e.g., 1  or  1,2,5-8">
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
let data={};let filter='all';let genreChart=null;let currentPosterItem=null;let viewMode_movies = "list";let viewMode_shows = "list";let viewMode_all = "list";let viewMode_incomplete = "list";let currentManageShow=null;let selectedEpisodes=new Set();
let uiTheme = 'dark';
let openPanels = new Set();
let scrollPosition = 0;
let gridSelectMode = false;
let selectedGridItems = new Set(); // key format: movie|<name>|<year>  or  show|<series>
let performanceMode = false;  // ADD THIS LINE
let currentPage = 1;  // ADD THIS LINE
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

function saveState() {
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
}

function restoreState() {
  // Only restore if this was triggered by a delete operation
  const shouldRestore = sessionStorage.getItem('shouldRestore');
  if(!shouldRestore) return;
  
  // Clear the flag immediately
  sessionStorage.removeItem('shouldRestore');
  
  // Get saved scroll position
  const savedScroll = parseInt(sessionStorage.getItem('scrollPosition')) || scrollPosition;
  const lastViewedShow = sessionStorage.getItem('lastViewedShow');
  
  // Restore panels FIRST, then scroll (important order!)
  setTimeout(() => {
    // Restore all open panels
    restoreOpenPanels();
    
    // Small delay to let panels open, then scroll
    setTimeout(() => {
      if(lastViewedShow && document.getElementById(lastViewedShow)) {
        // Scroll to the specific show they were viewing
        const element = document.getElementById(lastViewedShow).closest('.show-group');
        if(element) {
          element.scrollIntoView({ behavior: 'instant', block: 'start' });
          window.scrollBy(0, -100);
        }
      } else {
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
      sessionStorage.removeItem('openPanels');
    }, 100);
  }, 150);
}

function applyTheme(theme){
  uiTheme = theme === 'light' ? 'light' : 'dark';
  document.body.classList.toggle('light', uiTheme === 'light');
  localStorage.setItem('uiTheme', uiTheme);

  const btn = document.getElementById('themeToggleBtn');
  if(btn){
    btn.textContent = (uiTheme === 'light') ? '☀️ Theme' : '🌙 Theme';
  }
}

function toggleTheme(){
  applyTheme(uiTheme === 'light' ? 'dark' : 'light');
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

async function load(){
  const r=await fetch('/api/history');
  data=await r.json();
  
  // Theme restore
  const savedTheme = localStorage.getItem('uiTheme') || 'dark';
  applyTheme(savedTheme);
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
  
  showGridSelectButton();
  
  document.getElementById('total').textContent=data.stats.total_watches;
  document.getElementById('movies').textContent=data.stats.unique_movies;
  document.getElementById('shows').textContent=data.stats.tv_shows;
  document.getElementById('week').textContent=data.stats.this_week;
  document.getElementById('hours').textContent=data.stats.total_hours+'h';
  document.getElementById('avg').textContent=data.stats.avg_per_day;
  
      renderHistory();
      renderGenres();
      renderAnalytics();
      // Reset progress pagination and collapsed states on initial load to
      // ensure the correct sections expand/collapse according to the
      // saved filter (inprogress/complete/all) before rendering.
      resetProgressPagination();
      renderProgress();
  
  // Restore state after a longer delay to ensure DOM is ready
  setTimeout(() => {
    restoreState();
  }, 200);
}

async function quickReload(){
  const content = document.getElementById('content');
  content.classList.add('content-loading');
  
  try {
    const r = await fetch('/api/history');
    data = await r.json();
    
    document.getElementById('total').textContent = data.stats.total_watches;
    document.getElementById('movies').textContent = data.stats.unique_movies;
    document.getElementById('shows').textContent = data.stats.tv_shows;
    document.getElementById('week').textContent = data.stats.this_week;
    document.getElementById('hours').textContent = data.stats.total_hours + 'h';
    document.getElementById('avg').textContent = data.stats.avg_per_day;
    
    renderHistory();
    restoreState();
  } catch(e) {
    console.error('Reload failed:', e);
  } finally {
    content.classList.remove('content-loading');
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

  saveState();

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

function gridItemClick(seriesName,isShow){
      if(isShow){
        // When a show card is clicked in the progress view, ensure the library
        // shows tab is active and the filter is set to shows.  Switch to list
        // view and persist these settings.  Clear the search box so all
        // shows are visible, then re-render the history view and scroll
        // to the target show element.
        // Switch to the Library (history) tab so the user can see the show.
        const historyBtn = document.querySelector(".tab-btn[onclick*='history']");
        if(historyBtn){
          switchTab({ target: historyBtn }, 'history');
        }

        filter = 'shows';
        localStorage.setItem('filter', 'shows');
        viewMode_shows = 'list';
        localStorage.setItem('viewMode_shows', 'list');

        // Reset view toggle buttons to reflect list mode
        document.querySelectorAll('.view-toggle button').forEach(b=>b.classList.remove('active'));
        document.querySelector('.view-toggle button:first-child').classList.add('active');

        // Clear the search input so the show appears in the list
        const searchInput = document.getElementById('search');
        if(searchInput){
          // Only clear if it currently has text; this prevents unnecessary
          // re-render when no search term is present
          if(searchInput.value){
            searchInput.value = '';
          }
        }

        // Render history with the updated filter and view
        renderHistory();
        // After rendering, expand the show if collapsed and scroll to its
        // header.  The seasons-list element has id "show-{slug}", but the
        // show header lives two levels up.  We toggle the seasons list
        // open if necessary so that scrollIntoView positions correctly.
        setTimeout(() => {
          const id = 'show-' + seriesName.replace(/[^a-zA-Z0-9]/g, '');
          const seasonEl = document.getElementById(id);
          if(seasonEl){
            // Expand the show if it's currently collapsed
            if(seasonEl.style.display === 'none' || seasonEl.style.display === ''){
              toggle(id);
            }
            // Ascend to the show-group container.  The hierarchy is:
            // seasons-list -> action-buttons -> item-content -> show-group
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
        }, 100);
      }
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
  currentTab = t;
  document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c=>c.classList.remove('active'));
  e.target.classList.add('active');
  document.getElementById(t+'-tab').classList.add('active');
  if(t==='genres')renderGenres();
  if(t==='analytics')renderAnalytics();
  if(t==='progress')renderProgress();
  if(t==='history')renderHistory();
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

function setView(mode,e) {
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
  
  document.querySelectorAll('.view-toggle button').forEach(b=>b.classList.remove('active'));
  e.target.classList.add('active');
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
}

function closeManualEntry(){
  document.getElementById('manualEntryModal').style.display='none';
  document.getElementById('entryName').value='';
  document.getElementById('entryYear').value='';
  document.getElementById('entrySeriesName').value='';
  document.getElementById('entrySeason').value='';
  document.getElementById('entryEpisode').value='';
  document.getElementById('entryGenres').value='';
  document.querySelectorAll('input[name="entryType"]')[0].checked=true;
  toggleEpisodeFields();
}

function toggleEpisodeFields(){
  const type=document.querySelector('input[name="entryType"]:checked').value;
  const fields=document.getElementById('episodeFields');
  if(type==='Episode'){
    fields.classList.add('show');
  }else{
    fields.classList.remove('show');
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
  const name=document.getElementById('entryName').value.trim();
  const year=document.getElementById('entryYear').value;
  const date=document.getElementById('entryDate').value;
  const genresStr=document.getElementById('entryGenres').value.trim();
  
    // Movie needs a title, Episode does not (we can use TMDB or fallback)
  if(type==='Movie' && !name){
    alert('Please enter a movie title');
    return;
  }
  
  if(!date){
    alert('Please select a watch date');
    return;
  }
  
  const record={
    timestamp:new Date(date+'T12:00:00').toISOString(),
    type:type,
    name: (type==='Episode' && !name) ? 'Episode' : name,
    year:year?parseInt(year):null,
    user:'Manual Entry',
    genres:genresStr?genresStr.split(',').map(g=>g.trim()):[],
    source:'manual'
  };
  
  if(type==='Episode'){
    const seriesName=document.getElementById('entrySeriesName').value.trim();
    const seasonRaw=(document.getElementById('entrySeason').value||'').trim();
    const episodeSpec=(document.getElementById('entryEpisode').value||'').trim();

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
        const r = await fetch('/api/manual_tv_bulk', {
          method:'POST',
          headers:{'Content-Type':'application/json'},
          body: JSON.stringify({
            series_name: seriesName,
            season: seasonRaw ? parseInt(seasonRaw) : null,
            episode_spec: episodeSpec,
            date: date,
            skip_tmdb: false
          })
        });
        const j = await r.json();
        if(j.success){
          alert(`✓ Added ${j.added} episodes`);
          closeManualEntry();
          location.reload();
        }else{
          alert('Failed: ' + (j.error || 'Unknown error'));
        }
      }catch(e){
        alert('Error: ' + e.message);
      }
      return;
    }

    // Fallback (should not reach here)
    alert('Invalid episode input');
    return;
  }
  
  try{
    const r=await fetch('/api/manual_entry',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify(record)
    });
    const result=await r.json();
    if(result.success){
      closeManualEntry();
      location.reload();
    }else{
      alert('Failed: '+result.error);
    }
  }catch(e){
    alert('Error: '+e.message);
  }
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
      alert('✓ TMDB cache cleared! Refresh to reload.');
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
  saveState();
  
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
  saveState();
  
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
  saveState();
  
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
  saveState();
  
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

function renderHistory(){
  const search=(document.getElementById('search').value||'').toLowerCase();
  const sort=document.getElementById('sort').value;
  
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

  // Update button states to match current filter's view mode
  document.querySelectorAll('.view-toggle button').forEach(b => b.classList.remove('active'))
  if (viewMode === 'grid') {
      document.querySelectorAll('.view-toggle button')[1].classList.add('active')
  } else {
      document.querySelectorAll('.view-toggle button')[0].classList.add('active')
  }
  
  let movies=(data.movies||[]).filter(m=>!search||m.name.toLowerCase().includes(search));
  let shows=(data.shows||[]).filter(s=>!search||s.series_name.toLowerCase().includes(search));
  
  if(filter==='movies'){
    shows=[];
  }else if(filter==='shows'){
    movies=[];
  }else if(filter==='incomplete'){
    movies=[];
    shows=shows.filter(s=>s.completion_percentage<100&&s.has_tmdb_data);
  }
  
  if(sort==='recent'){
    movies.sort((a,b)=>(b.watches[0]?.timestamp||'').localeCompare(a.watches[0]?.timestamp||''));
    shows.sort((a,b)=>getLatestShowTimestamp(b).localeCompare(getLatestShowTimestamp(a)));
  }else if(sort==='oldest'){
    movies.sort((a,b)=>(a.watches[0]?.timestamp||'').localeCompare(b.watches[0]?.timestamp||''));
    shows.sort((a,b)=>getLatestShowTimestamp(a).localeCompare(getLatestShowTimestamp(b)));
  }else if(sort==='most_watched'){
    movies.sort((a,b)=>b.watch_count-a.watch_count);
    shows.sort((a,b)=>b.total_watches-a.total_watches);
  }else if(sort==='alphabetical'){
    movies.sort((a,b)=>a.name.localeCompare(b.name));
    shows.sort((a,b)=>a.series_name.localeCompare(b.series_name));
  }else if(sort==='incomplete_first'){
    shows.sort((a,b)=>a.completion_percentage-b.completion_percentage);
  }
  
  // ========== ADD THIS PAGINATION SECTION ==========
  // PAGINATION: Only if performance mode is ON
  let totalItems = 0;
  let endIdx = 999999; // No limit by default

  if(performanceMode) {
    const totalMovies = movies.length;
    const totalShows = shows.length;
    totalItems = totalMovies + totalShows;
  
    const startIdx = 0;
    endIdx = currentPage * itemsPerPage;
  
    movies = movies.slice(startIdx, endIdx);
    shows = shows.slice(startIdx, Math.max(0, endIdx - movies.length));
  }

if(viewMode==='grid'){
  let html='<div class="grid-view">';
  
  movies.forEach(m=>{
    const poster=m.poster?`<img src="${m.poster}" class="poster" loading="lazy" decoding="async">`:`<div class="poster-placeholder">🎬</div>`;
    const movieName=m.name.replace(/'/g,"\\'").replace(/"/g,'&quot;');
    const gkey = gridKeyMovie(m.name, m.year);
    const gsel = selectedGridItems.has(gkey) ? ' grid-selected' : '';
    const gchk = selectedGridItems.has(gkey) ? 'checked' : '';
    const itemJson = JSON.stringify(m).replace(/"/g, '&quot;');
    
    html += `<div class="grid-item${gsel}" data-grid-key="${gkey}" onclick="${gridSelectMode ? `event.stopPropagation();toggleGridItemSelection('${gkey.replace(/'/g, '\\\\\\'')}')`  : ''}"><div class="grid-checkbox-wrap"><input type="checkbox" class="grid-select-checkbox" ${gchk} onclick="event.stopPropagation();toggleGridItemSelection('${gkey.replace(/'/g, '\\\\\\'')}')"></div><div class="grid-item-actions"><button class="btn manual small" title="Upload poster" onclick="event.stopPropagation();openPosterModal(${itemJson},'movie')">📷</button><button class="btn danger small" onclick="event.stopPropagation();deleteMovie('${movieName}',${m.year})">✕</button></div><div class="poster-container">${poster}</div><div class="grid-item-title">${m.name}</div><div class="grid-item-info">${m.year||''} • ${m.watch_count}x</div></div>`;
  });
  
  shows.forEach(s=>{
    const poster=s.poster?`<img src="${s.poster}" class="poster" loading="lazy" decoding="async">`:`<div class="poster-placeholder">📺</div>`;
    const comp=s.completion_percentage||0;
    const seasons=s.seasons||[];
    const seasonInfo=seasons.map(se=>`S${se.season_number}`).join(', ');
    const seriesName=s.series_name.replace(/'/g,"\\'").replace(/"/g,'&quot;');
    const key = `show|${s.series_name}`;
    const isSelected = selectedGridItems.has(key);
    const allAdded = s.auto_all_added;
    const gsel = isSelected ? ' grid-selected' : '';
    
    html+=`<div class="grid-item${gsel}" data-grid-key="${key}" onclick="${gridSelectMode ? `event.stopPropagation();toggleGridItemSelection('${key.replace(/'/g,"\\'")}')`  : `gridItemClick('${seriesName}',true)`}">${gridSelectMode ? `<div class="grid-checkbox-wrap"><input type="checkbox" class="grid-select-checkbox" ${isSelected ? 'checked' : ''} onclick="event.stopPropagation();toggleGridItemSelection('${key.replace(/'/g,"\\'")}')" ></div>` : ``}<div class="grid-item-actions"><button class="btn success small" title="Mark completed" onclick="event.stopPropagation();markComplete('${seriesName}')">✓</button><button class="btn danger small" title="Delete show" onclick="event.stopPropagation();deleteShow('${seriesName}')">🗑️</button></div><div class="poster-container">${poster}</div><div class="grid-item-title">${s.series_name}</div><div class="grid-item-info">${seasonInfo}</div><div class="grid-item-info">${allAdded ? `<span class="badge badge-info">All episodes added</span>` : ``} <span class="badge ${comp>=100 ? 'badge-success' : 'badge-warning'}">${comp}%</span></div></div>`;
  });
  
  html+='</div>';
  
  // Add Load More button only in performance mode
  if(performanceMode && totalItems > endIdx) {
    html += `<div style="text-align:center;padding:30px"><button class="btn manual" onclick="loadMore()" style="font-size:16px;padding:15px 30px">📥 Load More (${totalItems - endIdx} remaining)</button></div>`;
  }
  
  document.getElementById('content').innerHTML=html||'<div class="loading">No items found</div>';
  updateGridBulkBar();
  return;
}

  
  // LIST VIEW
  let html='';
  movies.forEach(m=>{
    const poster=m.poster?`<img src="${m.poster}" class="poster" loading="lazy" decoding="async">`:`<div class="poster-placeholder">🎬</div>`;
    const badge=m.watch_count>1?`<span class="badge badge-success">Rewatched ${m.watch_count}x</span>`:'';
    const itemJson=JSON.stringify(m).replace(/"/g,'&quot;');
    const movieName=m.name.replace(/'/g,"\\'");
    html+=`<div class="movie-item">
      <div class="poster-container">${poster}
        <button class="upload-poster-btn" onclick='openPosterModal(${itemJson},"movie")'>📤</button>
      </div>
      <div class="item-content">
        <div class="movie-title">${m.name} ${m.year?'('+m.year+')':''}</div>
        <div class="movie-details">${badge}</div>
        <div class="action-buttons manage-actions">
          <button class="btn danger small" onclick="deleteMovie('${movieName}',${m.year})">🗑️ Delete Movie</button>
        </div>
      </div>
    </div>`;
  });
  
  shows.forEach(s=>{
    const poster=s.poster?`<img src="${s.poster}" class="poster" loading="lazy" decoding="async">`:`<div class="poster-placeholder">📺</div>`;
    const id='show-'+s.series_name.replace(/[^a-zA-Z0-9]/g,'');
    const comp=s.completion_percentage||0;
    const seriesName=s.series_name.replace(/'/g,"\\'");
    
    let compBadge='';
    if(s.manually_completed){
      compBadge=`<span class="badge badge-success">✓ Complete</span>`;
    }else if(s.has_tmdb_data){
      if(comp===100){
        compBadge=`<span class="badge badge-success">✓ Complete</span>`;
    }else{
      const allAdded = s.auto_all_added;
      compBadge = allAdded
        ? `<span class="badge badge-info">All episodes added</span> <span class="badge badge-warning">${comp}%</span>`
        : `<span class="badge badge-warning">${comp}%</span>`;
      }
    }else{
      compBadge=`<span class="badge badge-info">Tracking</span>`;
    }
    
    const itemJson=JSON.stringify(s).replace(/"/g,'&quot;');
    html+=`<div class="show-group">
      <div class="poster-container">${poster}
        <button class="upload-poster-btn" onclick='openPosterModal(${itemJson},"tv")'>📤</button>
      </div>
      <div class="item-content">
        <div class="show-header" onclick="toggle('${id}')">
          <div style="flex:1">
            <div class="show-title">${s.series_name}</div>
            <div class="movie-details">${compBadge} <span class="badge badge-info">${s.total_episodes}/${s.total_episodes_possible} Episodes</span></div>
            <div class="progress-bar"><div class="progress-fill" style="width:${comp}%"></div></div>
          </div>
          <div style="font-size:1.3em">▼</div>
        </div>
        <div class="action-buttons manage-actions">
          <button class="btn manual small" onclick="event.stopPropagation();openAddEpisode('${seriesName}')">➕ Episode</button>
          <button class="btn manual small" onclick="event.stopPropagation();openAddSeason('${seriesName}')">➕ Season</button>
          <button class="btn warning small" onclick="event.stopPropagation();markComplete('${seriesName}')">✓ Mark 100%</button>
          <button class="btn danger small" onclick="event.stopPropagation();deleteShow('${seriesName}')">🗑️ Delete</button>
        </div>
        <div class="seasons-list" id="${id}">`;
    
    (s.seasons||[]).forEach(season=>{
      const sid=id+'-s'+season.season_number;
      const seasonEp=season.episode_count||0;
      const seasonTotal=season.total_episodes||seasonEp;
      const seasonPercent=season.completion_percentage||0;
      
      let seasonInfo='';
      if(season.manually_completed){
        seasonInfo=`<span class="badge badge-success">✓ Complete</span>`;
      }else if(season.has_tmdb_data&&seasonTotal>0){
        if(seasonPercent===100){
          seasonInfo=`<span class="badge badge-info">${seasonEp}/${seasonTotal}</span> <span class="badge badge-success">✓ ${seasonPercent}%</span>`;
        }else{
          seasonInfo=`<span class="badge badge-info">${seasonEp}/${seasonTotal}</span> <span class="badge badge-warning">${seasonPercent}%</span>`;
        }
      }else{
        seasonInfo=`<span class="badge badge-info">${seasonEp} watched</span>`;
      }
      
      html+=`<div class="season-group">
        <div class="season-header" onclick="toggle('${sid}')">
          <div><span class="season-title">Season ${season.season_number}</span> ${seasonInfo}</div>
          <div style="font-size:1.1em">▼</div>
        </div>
        <div class="action-buttons season-actions">
          <button class="btn warning small" onclick="event.stopPropagation();markSeasonComplete('${seriesName}',${season.season_number})">✓ Mark 100%</button>
          <button class="btn danger small" onclick="event.stopPropagation();deleteSeason('${seriesName}',${season.season_number})">🗑️ Delete Season</button>
        </div>
        <div class="episodes-list" id="${sid}">`;
      
      (season.episodes||[]).forEach(ep=>{
        const rewatched=ep.watch_count>1?` <span class="badge badge-success">×${ep.watch_count}</span>`:'';
        const key=`${s.series_name}_${season.season_number}_${ep.episode}`;
        const isSelected=selectedEpisodes.has(key);
        const selectedClass=isSelected?' selected':'';
        const checkboxId=`chk-${key.replace(/[^a-zA-Z0-9]/g,'')}`;
        html+=`<div class="episode-item${selectedClass}" onclick="event.stopPropagation();">
          <span><input type="checkbox" class="select-checkbox" id="${checkboxId}" ${isSelected?'checked':''} onclick="event.stopPropagation();toggleEpisodeSelection('${seriesName}',${season.season_number},${ep.episode},event);return false;"> <label for="${checkboxId}" style="cursor:pointer;user-select:none;">E${ep.episode}: ${ep.name}${rewatched}</label></span>
          <button class="btn danger small delete-btn" onclick="event.stopPropagation();deleteEpisode('${seriesName}',${season.season_number},${ep.episode})">🗑️</button>
        </div>`;
      });
      html+=`</div></div>`;
    });
    
    html+=`</div></div></div>`;
  });
  
  document.getElementById('content').innerHTML=html||'<div class="loading">No items found</div>';
}


function loadMore() {
  currentPage++;
  renderHistory();
  window.scrollTo({
    top: document.body.scrollHeight - 1000,
    behavior: 'smooth'
  });
}

async function renderGenres(){
  const genres=data.genres||[];
  const breakdown=data.genre_breakdown||{};
  
  if(genreChart)genreChart.destroy();
  
  // Fetch insights
  let insights = null;
  try {
    const insightsResp = await fetch('/api/genre_insights');
    insights = await insightsResp.json();
  } catch(e) {
    console.error('Failed to load insights:', e);
  }
  
  // Create side-by-side layout - pie chart bigger, genre list smaller
  let layoutHtml = `
      <div style="display:grid;grid-template-columns:0.8fr 550px;gap:30px;margin-bottom:30px">
      <div id="genre-list-container"></div>
      <!-- Wrap the chart in a container with a class so we can disable sticky positioning on mobile -->
      <div class="genre-chart-container" style="position:sticky;top:20px;height:fit-content">
        <div style="background:rgba(139,156,255,0.05);padding:20px;border-radius:16px;border:2px solid rgba(139,156,255,0.2)">
          <canvas id="genreChart" width="500" height="500"></canvas>
        </div>
      </div>
    </div>
  `;
  
  document.getElementById('genres-content').innerHTML = layoutHtml;
  
  // Small delay to ensure canvas is in DOM
  setTimeout(() => {
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
  
  // Add insights sections at the top
  if(insights && insights.success) {
    // The "Genre Trends" section was deemed unnecessary.  We intentionally
    // skip rendering any trending information here so the genres page
    // remains focused on the recommendations and your personal genre data.
    
    // Genre combos
    if(insights.combos && insights.combos.length > 0) {
      html += `<div style="background:rgba(94,245,224,0.1);padding:15px;border-radius:12px;margin-bottom:15px;border-left:4px solid #5ef5e0">`;
      html += `<h3 style="margin:0 0 10px 0;color:#5ef5e0;font-size:1em">🎬 Genre Combos You Love</h3>`;
      // Each combo is turned into a button.  Clicking the button will
      // activate a filter so that the recommendations below show only
      // movies and shows matching all genres in the combo.
      insights.combos.forEach(c => {
        const comboLabel = c.genres.join(' + ');
        const comboParam = c.genres.join('|');
        html += `<div style="margin:5px 0;font-size:0.9em">
          <button class="btn secondary small" onclick="filterByCombo('${comboParam}')" style="padding:6px 12px;font-size:0.85em">
            ${comboLabel} (${c.count} items)
          </button>
        </div>`;
      });
      html += `</div>`;
    }
    
    // Mood-based
    if(insights.moods && Object.keys(insights.moods).length > 0) {
      html += `<div style="background:rgba(255,107,157,0.1);padding:15px;border-radius:12px;margin-bottom:15px;border-left:4px solid #ff6b9d">`;
      html += `<h3 style="margin:0 0 10px 0;color:#ff6b9d;font-size:1em">🎭 Recommendations by Mood</h3>`;
      const moodIcons = {
        intense: '⚡',
        fun: '😄',
        emotional: '💔',
        exciting: '🚀',
        scary: '👻'
      };
      for(const [mood, info] of Object.entries(insights.moods)) {
        const icon = moodIcons[mood] || '🎬';
        html += `<div style="margin:8px 0">
          <button class="btn secondary small" onclick="filterByMood('` + mood + `')" style="padding:6px 12px;font-size:0.85em">
            ` + icon + ` Feeling ` + mood + `? (` + info.count + ` items)
          </button>
        </div>`;
      }
      html += `</div>`;
    }
  }
  
  // Genre list
  genres.forEach(g=>{
    const b=breakdown[g.genre]||{movies:[],shows:[]};
    const movieNames=b.movies.slice(0,3).map(m=>m.name).join(', ');
    const showNames=b.shows.slice(0,3).map(s=>s.series_name).join(', ');
    let items='';
    if(movieNames)items+=`<strong>Movies:</strong> ${movieNames}${b.movies.length>3?' & '+(b.movies.length-3)+' more':''}<br>`;
    if(showNames)items+=`<strong>Shows:</strong> ${showNames}${b.shows.length>3?' & '+(b.shows.length-3)+' more':''}`;
    
    html+=`<div class="genre-item" id="genre-item-${g.genre}">
      <div class="genre-header" style="cursor:pointer;display:flex;justify-content:space-between;align-items:center" onclick="toggleGenreDetails('${g.genre}')">
        <div style="display:flex;align-items:center;gap:10px">
          <span id="genre-arrow-${g.genre}" style="transition:transform 0.3s;display:inline-block">▶</span>
          <div class="genre-name">${g.genre}</div>
        </div>
        <div class="genre-count">${g.count} items</div>
      </div>
      <div id="genre-details-${g.genre}" class="genre-items" style="display:none;margin-top:10px">${items||'No items'}</div>
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
  loadRecommendations();
}

function scrollToGenre(genre) {
  const element = document.getElementById(`genre-item-${genre}`);
  if(element) {
    element.scrollIntoView({behavior: 'smooth', block: 'center'});
    toggleGenreDetails(genre);
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
  // Expand the relevant genres so the user can see which genres are
  // associated with the selected mood.  Use the global moodGenreMap
  // rather than a local copy to avoid duplication.
  const genres = moodGenreMap[mood] || [];
  genres.forEach(g => {
    const element = document.getElementById(`genre-item-${g}`);
    if(element) {
      const details = document.getElementById(`genre-details-${g}`);
      const arrow = document.getElementById(`genre-arrow-${g}`);
      if(details && details.style.display === 'none') {
        details.style.display = 'block';
        arrow.style.transform = 'rotate(90deg)';
      }
    }
  });
  // Scroll to first genre
  if(genres.length > 0) {
    scrollToGenre(genres[0]);
  }
  // Finally, reload recommendations to apply the filter
  loadRecommendations();
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
  // When selecting a genre combo, we can optionally expand the
  // corresponding genres in the list for context.  We'll open each
  // matching genre section.
  genres.forEach(g => {
    const element = document.getElementById(`genre-item-${g}`);
    if(element) {
      const details = document.getElementById(`genre-details-${g}`);
      const arrow = document.getElementById(`genre-arrow-${g}`);
      if(details && details.style.display === 'none') {
        details.style.display = 'block';
        arrow.style.transform = 'rotate(90deg)';
      }
    }
  });
  // Scroll to the first genre in the combo for user context
  if(genres.length > 0) {
    scrollToGenre(genres[0]);
  }
  // Reload recommendations to apply the filter
  loadRecommendations();
}

function toggleGenreDetails(genre) {
  const details = document.getElementById(`genre-details-${genre}`);
  const arrow = document.getElementById(`genre-arrow-${genre}`);
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
        const title = m.name || m.title || 'Unknown';
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
        const title = s.series_name || s.name || 'Unknown';
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
    arrow.style.transform = 'rotate(90deg)';
  } else {
    // Collapse the details
    details.style.display = 'none';
    arrow.style.transform = 'rotate(0deg)';
  }
}

async function loadRecommendations(section) {
  try {
    // Determine which offset to use based on the requested section.  When a
    // specific section is provided ('movies' or 'shows'), use that
    // category's offset to fetch a new batch of suggestions.  When
    // section is undefined (initial load or full refresh), use the
    // maximum of the two offsets so that both categories remain in sync.
    let offset;
    if(section === 'movies') {
      offset = recommendationOffsetMovies;
    } else if(section === 'shows') {
      offset = recommendationOffsetShows;
    } else {
      offset = Math.max(recommendationOffsetMovies, recommendationOffsetShows);
    }
    const resp = await fetch(`/api/genre_recommendations?offset=${offset}`);
    const recs = await resp.json();
    
    if(recs.success && recs.top_genre) {
      // Build filtered movie and show lists based on active mood or combo filters.
      let movies = Array.isArray(recs.movies) ? [...recs.movies] : [];
      let shows = Array.isArray(recs.shows) ? [...recs.shows] : [];
      let heading;
      if(currentMoodFilter) {
        const mg = moodGenreMap[currentMoodFilter] || [];
        const mgLower = mg.map(x => (x || '').toLowerCase());
        movies = movies.filter(m => {
          const gs = (m.genres || []).map(x => (x || '').toLowerCase());
          return gs.some(g => mgLower.includes(g));
        });
        shows = shows.filter(s => {
          const gs = (s.genres || []).map(x => (x || '').toLowerCase());
          return gs.some(g => mgLower.includes(g));
        });
        heading = `🎭 Recommendations for your ${currentMoodFilter} mood`;
      } else if(currentGenreComboFilter) {
        movies = movies.filter(m => {
          const gs = (m.genres || []).map(x => (x || '').toLowerCase());
          return currentGenreComboFilter.every(g => gs.includes((g || '').toLowerCase()));
        });
        shows = shows.filter(s => {
          const gs = (s.genres || []).map(x => (x || '').toLowerCase());
          return currentGenreComboFilter.every(g => gs.includes((g || '').toLowerCase()));
        });
        heading = `🎬 Recommendations: ${currentGenreComboFilter.join(' + ')}`;
      } else {
        heading = `🎯 Recommended for you (Based on ${recs.top_genre})`;
      }
      // Update stored recommendations only for the requested section (or both if
      // section is undefined).  This allows the user to refresh movies or
      // shows independently without overwriting the other category.
      if(!section || section === 'movies') {
        currentMoviesRecs = movies;
      }
      if(!section || section === 'shows') {
        currentShowsRecs = shows;
      }
      let recHtml = `<div id="recommendations-container" style="margin-top:30px;padding:20px;background:rgba(139,156,255,0.1);border-radius:16px;border:2px solid rgba(139,156,255,0.3)">`;
      // Heading row
      recHtml += `<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:20px">
        <h3 style="margin:0;color:#8b9cff">${heading}</h3>
      </div>`;
      recHtml += '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:20px">';
      // Movies column
      recHtml += '<div style="display:flex;flex-direction:column">';
      // Header row with a more button placed on the right so it appears above the list
      recHtml += `<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
        <h4 style="margin:0;color:rgba(255,255,255,0.8)">🎬 Movies</h4>
        <button class="btn secondary small" onclick="refreshMovieRecommendations()" style="padding:6px 12px">🔄 More Movies</button>
      </div>`;
      if(currentMoviesRecs.length === 0) {
        // Determine which message to show based on filters and offsets
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
            <div style="font-weight:700;margin-bottom:5px">${m.title} ${m.year ? '('+m.year+')' : ''}</div>
            <div style="font-size:0.85em;color:rgba(255,255,255,0.6);margin-bottom:8px">${m.overview}</div>
            <div>${(m.genres||[]).slice(0,3).map(g => `<span class="badge badge-info">${g}</span>`).join(' ')}</div>
          </div>`;
        });
      }
      // close movies column
      recHtml += '</div>';
      // Shows column
      recHtml += '<div style="display:flex;flex-direction:column">';
      // Header row with a more button on the right for shows
      recHtml += `<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
        <h4 style="margin:0;color:rgba(255,255,255,0.8)">📺 Shows</h4>
        <button class="btn secondary small" onclick="refreshShowRecommendations()" style="padding:6px 12px">🔄 More Shows</button>
      </div>`;
      if(currentShowsRecs.length === 0) {
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
            <div style="font-weight:700;margin-bottom:5px">${s.title} ${s.year ? '('+s.year+')' : ''}</div>
            <div style="font-size:0.85em;color:rgba(255,255,255,0.6);margin-bottom:8px">${s.overview}</div>
            <div>${(s.genres||[]).slice(0,3).map(g => `<span class="badge badge-info">${g}</span>`).join(' ')}</div>
          </div>`;
        });
      }
      recHtml += '</div>';
      recHtml += '</div>';
      recHtml += '</div>';
      // Remove old recommendations if they exist
      const oldRecs = document.getElementById('recommendations-container');
      if(oldRecs) oldRecs.remove();
      // Append the new recommendations container without re-rendering the entire
      // genres page.  Using insertAdjacentHTML avoids resetting the DOM and
      // inadvertently destroying the pie chart.
      const genresContainer = document.getElementById('genres-content');
      if(genresContainer) {
        genresContainer.insertAdjacentHTML('beforeend', recHtml);
      }
    }
  } catch(e) {
    console.error('Failed to load recommendations:', e);
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
  recommendationOffsetMovies++;
  loadRecommendations('movies');
}

// Fetch the next batch of show recommendations.  When the user clicks
// the "More Shows" button this function is invoked.  It increments
// the show offset and calls loadRecommendations() specifying the
// 'shows' section so that only the shows list is refreshed.
function refreshShowRecommendations() {
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
  statsHtml+=`<div class="quick-stat-card"><div class="quick-stat-title">⚡ Fastest Completion</div><div class="quick-stat-value">${quick.fastest_completion?.days||0}</div><div class="quick-stat-label">${quick.fastest_completion?.name||'N/A'}</div></div>`;
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
    const totalPossible=s.total_episodes_possible||0;
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
          const totalPossible = s.total_episodes_possible || 0;
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
          const totalPossible = s.total_episodes_possible || 0;
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
function navigateToMovie(movieName){
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
  // Scroll to the top of the page so the user sees the results
  window.scrollTo({ top: 0, behavior: 'smooth' });
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

load();
</script></body></html>"""

@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        payload = request.get_json(force=True, silent=True) or {}
        notification_type = (payload.get("NotificationType") or payload.get("Event") or "").lower()
        
        if "playbackstop" not in notification_type and "stop" not in notification_type:
            return jsonify({"success": True, "ignored": True})

        item_type = payload.get("ItemType") or payload.get("Type") or (payload.get("Item", {}) or {}).get("Type") or "Unknown"
        name = payload.get("Name") or (payload.get("Item", {}) or {}).get("Name") or "Unknown"
        year = payload.get("Year") or (payload.get("Item", {}) or {}).get("ProductionYear")
        series_name = payload.get("SeriesName") or (payload.get("Item", {}) or {}).get("SeriesName")
        season = payload.get("SeasonNumber") or (payload.get("Item", {}) or {}).get("ParentIndexNumber")
        episode = payload.get("EpisodeNumber") or (payload.get("Item", {}) or {}).get("IndexNumber")
        user = payload.get("NotificationUsername") or (payload.get("User", {}) or {}).get("Name") or "Unknown"
        genres = payload.get("Genres") or (payload.get("Item", {}) or {}).get("Genres") or []

        record = {
            "timestamp": datetime.now().isoformat(),
            "type": item_type,
            "name": name,
            "year": year,
            "series_name": series_name,
            "season": season,
            "episode": episode,
            "user": user,
            "genres": genres if isinstance(genres, list) else [],
            "source": "webhook",
        }

        append_record(record)
        return jsonify({"success": True})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/history")
def api_history():
    return jsonify(organize_data())

@app.route('/api/genre_recommendations', methods=['GET'])
def get_genre_recommendations():
    try:
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

        # Get recommendations from Radarr (movies)
        all_recommended_movies = []
        radarr_url = os.getenv("RADARR_URL")
        radarr_key = os.getenv("RADARR_API_KEY")
        
        if radarr_url and radarr_key:
            try:
                radarr_resp = requests.get(
                    f'{radarr_url}/api/v3/movie',
                    headers={'X-Api-Key': radarr_key},
                    timeout=10
                )
                if radarr_resp.ok:
                    all_movies = radarr_resp.json()
                    watched_movie_titles = {m['name'].lower() for m in org_data.get('movies', [])}
                    
                    for movie in all_movies:
                        movie_genres = movie.get('genres', [])
                        if movie_genres and top_genre in movie_genres:
                            title = movie.get('title', '')
                            if title.lower() not in watched_movie_titles:
                                all_recommended_movies.append({
                                    'title': title,
                                    'year': movie.get('year'),
                                    'overview': (movie.get('overview', '') or 'No description')[:150] + '...',
                                    'genres': movie_genres,
                                    'tmdbId': movie.get('tmdbId')
                                })
            except Exception as e:
                print(f"Radarr error: {e}")
                traceback.print_exc()
        
        # Get recommendations from Sonarr (shows)
        all_recommended_shows = []
        sonarr_url = os.getenv("SONARR_URL")
        sonarr_key = os.getenv("SONARR_API_KEY")
        
        if sonarr_url and sonarr_key:
            try:
                sonarr_resp = requests.get(
                    f'{sonarr_url}/api/v3/series',
                    headers={'X-Api-Key': sonarr_key},
                    timeout=10
                )
                if sonarr_resp.ok:
                    all_shows = sonarr_resp.json()
                    watched_show_titles = {s['series_name'].lower() for s in org_data.get('shows', [])}
                    
                    for show in all_shows:
                        show_genres = show.get('genres', [])
                        if show_genres and top_genre in show_genres:
                            title = show.get('title', '')
                            if title.lower() not in watched_show_titles:
                                all_recommended_shows.append({
                                    'title': title,
                                    'year': show.get('year'),
                                    'overview': (show.get('overview', '') or 'No description')[:150] + '...',
                                    'genres': show_genres,
                                    'tvdbId': show.get('tvdbId')
                                })
            except Exception as e:
                print(f"Sonarr error: {e}")
                traceback.print_exc()
        
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
        org_data = organize_data()
        records = get_all_records()
        
        # Time-based genre trends (this month vs last month)
        now = datetime.now()
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        last_month_start = (month_start - timedelta(days=1)).replace(day=1)
        
        this_month_genres = Counter()
        last_month_genres = Counter()
        
        for r in records:
            try:
                dt = datetime.fromisoformat(r["timestamp"].replace("Z", "").split("+")[0])
                genres = r.get("genres", [])
                
                if dt >= month_start:
                    for g in genres:
                        this_month_genres[g] += 1
                elif dt >= last_month_start and dt < month_start:
                    for g in genres:
                        last_month_genres[g] += 1
            except:
                pass
        
        # Calculate trends
        trends = []
        all_genres = set(this_month_genres.keys()) | set(last_month_genres.keys())
        
        for genre in all_genres:
            this_count = this_month_genres.get(genre, 0)
            last_count = last_month_genres.get(genre, 0)
            
            if last_count > 0:
                change_pct = round(((this_count - last_count) / last_count) * 100)
            elif this_count > 0:
                change_pct = 100  # New genre this month
            else:
                change_pct = 0
            
            if abs(change_pct) >= 20 and this_count >= 3:  # Significant changes only
                trends.append({
                    "genre": genre,
                    "this_month": this_count,
                    "last_month": last_count,
                    "change": change_pct,
                    "direction": "up" if change_pct > 0 else "down"
                })
        
        trends.sort(key=lambda x: abs(x["change"]), reverse=True)
        
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
        
        return jsonify({
            "success": True,
            "trends": trends[:5],  # Top 5 trends
            "combos": top_combos,
            "moods": moods
        })
    
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
        
        if record.get("type") == "Episode":
            if not record.get("series_name") or record.get("season") is None or record.get("episode") is None:
                return jsonify({"success": False, "error": "Episodes require series_name, season, and episode"}), 400
                        # Auto-fill episode title from TMDB if user didn't provide one (or provided a placeholder)
            try:
                incoming_name = (record.get("name") or "").strip()
                ep_num = record.get("episode")
                needs_title = (not incoming_name) or (incoming_name.lower() == "episode") or (incoming_name.lower().startswith("episode "))

                if needs_title:
                    tmdb_name = get_tmdb_episode_name(record.get("series_name"), record.get("season"), ep_num)
                    record["name"] = tmdb_name if tmdb_name else f"Episode {ep_num}"
            except Exception as e:
                # Never fail the request due to TMDB lookup issues
                ep_num = record.get("episode")
                record["name"] = record.get("name") or f"Episode {ep_num}"    
        
        append_record(record)
        print(f"✓ Manual entry added: {record.get('type')} - {record.get('name')}")
        
        return jsonify({"success": True, "message": "Watch record added successfully"})
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/add_episode", methods=["POST"])
def api_add_episode():
    try:
        data = request.get_json()
        series_name = data.get("series_name")
        season = data.get("season")
        episode = data.get("episode")
        date = data.get("date")
        skip_tmdb = data.get("skip_tmdb", False)
        
        if not all([series_name, season is not None, episode is not None, date]):
            return jsonify({"success": False, "error": "Missing required fields"}), 400
        
        if not skip_tmdb:
            episode_name = get_tmdb_episode_name(series_name, season, episode)
            if not episode_name:
                episode_name = f"Episode {episode}"
        else:
            episode_name = f"Episode {episode}"
        
        record = {
            "timestamp": datetime.fromisoformat(date + "T12:00:00").isoformat(),
            "type": "Episode",
            "name": episode_name,
            "year": None,
            "series_name": series_name,
            "season": season,
            "episode": episode,
            "user": "Manual Entry",
            "genres": [],
            "source": "manual_episode"
        }
        
        append_record(record)
        print(f"✓ Added episode: {series_name} S{season}E{episode} - {episode_name}")
        
        return jsonify({"success": True, "episode_name": episode_name})
    
    except Exception as e:
        print(f"✗ Add episode error: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/add_season", methods=["POST"])
def api_add_season():
    try:
        data = request.get_json()
        series_name = data.get("series_name")
        season = data.get("season")
        date = data.get("date")
        skip_tmdb = data.get("skip_tmdb", False)
        episode_count = data.get("episode_count")
        
        if not all([series_name, season is not None, date]):
            return jsonify({"success": False, "error": "Missing required fields"}), 400
        
        if not skip_tmdb:
            episodes = get_tmdb_season_episodes(series_name, season)
            
            if not episodes or len(episodes) == 0:
                if episode_count:
                    episodes = [{"episode_number": i, "name": f"Episode {i}"} for i in range(1, episode_count + 1)]
                else:
                    return jsonify({"success": False, "error": "Could not fetch episodes from TMDB. Try manual episode count or skip TMDB option."}), 400
        else:
            if not episode_count:
                return jsonify({"success": False, "error": "Episode count required when skipping TMDB"}), 400
            episodes = [{"episode_number": i, "name": f"Episode {i}"} for i in range(1, episode_count + 1)]
        
        count = 0
        for ep in episodes:
            record = {
                "timestamp": datetime.fromisoformat(date + "T12:00:00").isoformat(),
                "type": "Episode",
                "name": ep["name"],
                "year": None,
                "series_name": series_name,
                "season": season,
                "episode": ep["episode_number"],
                "user": "Manual Entry",
                "genres": [],
                "source": "manual_season"
            }
            append_record(record)
            count += 1
        
        print(f"✓ Added season: {series_name} S{season} - {count} episodes")
        
        return jsonify({"success": True, "count": count})
    
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
        date = (payload.get("date") or "").strip()
        skip_tmdb = bool(payload.get("skip_tmdb", False))

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

        if season is None:
            # ALL seasons + ALL episodes
            if skip_tmdb:
                return jsonify({"success": False, "error": "Cannot fetch all seasons when Skip TMDB is enabled"}), 400

            info = get_tmdb_series_info(series_name)
            if not info or not info.get("seasons"):
                return jsonify({"success": False, "error": "TMDB could not find this series"}), 400

            # Iterate seasons in TMDB info (season_number > 0 already in your cache builder)
            for snum in sorted(info["seasons"].keys()):
                eps = get_tmdb_season_episodes(series_name, snum)
                if not eps:
                    continue
                for ep in eps:
                    epno = ep.get("episode_number")
                    nm = ep.get("name") or f"Episode {epno}"
                    to_add.append((snum, epno, nm))

        else:
            # Season is provided
            if episodes_list:
                # Specific episodes in season
                if skip_tmdb:
                    for epno in episodes_list:
                        to_add.append((season, epno, f"Episode {epno}"))
                else:
                    for epno in episodes_list:
                        nm = get_tmdb_episode_name(series_name, season, epno) or f"Episode {epno}"
                        to_add.append((season, epno, nm))
            else:
                # ALL episodes in that season
                if skip_tmdb:
                    return jsonify({"success": False, "error": "Skip TMDB needs an episode list (e.g. 1,2,5-8)"}), 400

                eps = get_tmdb_season_episodes(series_name, season)
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
                "year": None,
                "series_name": series_name,
                "season": int(snum),
                "episode": int(epno),
                "user": "Manual Entry",
                "genres": [],
                "source": "manual_bulk_tv"
            }
            append_record(record)
            added += 1

        return jsonify({"success": True, "added": added})

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
        global series_cache, poster_cache
        series_cache = {}
        poster_cache = {}
        save_cache("series")
        save_cache("poster")
        
        cache["data"] = None
        cache["time"] = None
        
        print("✓ TMDB cache cleared")
        
        return jsonify({"success": True})
    
    except Exception as e:
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
        filtered = []
        deleted = 0
        
        for r in records:
            should_delete = False
            if r.get("type") == "Episode":
                for ep_data in episodes_to_delete:
                    if (r.get("series_name") == ep_data["series_name"] and 
                        r.get("season") == ep_data["season"] and 
                        r.get("episode") == ep_data["episode"]):
                        should_delete = True
                        deleted += 1
                        break
            
            if not should_delete:
                filtered.append(r)
        
        save_all_records(filtered)
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
        
        records = get_all_records()
        filtered = []
        deleted = 0
        
        for r in records:
            if r.get("type") == "Movie" and r.get("name") == movie_name and r.get("year") == movie_year:
                deleted += 1
            else:
                filtered.append(r)
        
        save_all_records(filtered)
        print(f"✓ Deleted movie: {movie_name} ({movie_year}) - {deleted} records")
        
        return jsonify({"success": True, "deleted": deleted})
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/delete_show", methods=["POST"])
def api_delete_show():
    try:
        data = request.get_json()
        series_name = data.get("series_name")
        
        if series_name in manual_complete:
            del manual_complete[series_name]
            save_cache("complete")
        
        records = get_all_records()
        filtered = []
        deleted = 0
        
        for r in records:
            if r.get("type") == "Episode" and r.get("series_name") == series_name:
                deleted += 1
            else:
                filtered.append(r)
        
        save_all_records(filtered)
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
        filtered = []
        deleted = 0
        
        for r in records:
            if r.get("type") == "Episode" and r.get("series_name") == series_name and r.get("season") == season:
                deleted += 1
            else:
                filtered.append(r)
        
        save_all_records(filtered)
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
        filtered = []
        deleted = 0
        
        for r in records:
            if (r.get("type") == "Episode" and 
                r.get("series_name") == series_name and 
                r.get("season") == season and 
                r.get("episode") == episode):
                deleted += 1
            else:
                filtered.append(r)
        
        save_all_records(filtered)
        print(f"✓ Deleted episode: {series_name} S{season}E{episode} - {deleted} records")
        
        return jsonify({"success": True, "deleted": deleted})
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/jellyfin_import", methods=["POST"])
def api_jellyfin_import():
    ok, msg, count = jellyfin_import()
    return jsonify({"success": ok, "message": msg if ok else None, "error": None if ok else msg, "imported": count})


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
        with open(DATA_FILE) as f:
            content = f.read()
        return send_file(io.BytesIO(content.encode()), mimetype="application/json", as_attachment=True, download_name=f"watch_history_{datetime.now().strftime('%Y%m%d')}.json")
    
    elif fmt == "csv":
        records = get_all_records()
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=["timestamp", "type", "name", "year", "series_name", "season", "episode", "user", "genres", "source"])
        writer.writeheader()
        for r in records:
            r_copy = r.copy()
            r_copy["genres"] = ", ".join(r_copy.get("genres", []))
            writer.writerow(r_copy)
        
        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype="text/csv",
            as_attachment=True,
            download_name=f"watch_history_{datetime.now().strftime('%Y%m%d')}.csv"
        )
    
    return jsonify({"error": "invalid format"}), 400


if __name__ == "__main__":
    os.makedirs("/data", exist_ok=True)
    os.makedirs(POSTER_DIR, exist_ok=True)
    if not os.path.exists(DATA_FILE):
        open(DATA_FILE, "a").close()
    load_caches()
    
    print("=" * 60)
    print("JELLYFIN WATCH TRACKER - VERSION 34")
    print("BULK ACTIONS & SEASON COMPLETE EDITION")
    print("=" * 60)
    print(f"UI:      http://0.0.0.0:5000")
    print(f"Webhook: http://0.0.0.0:5000/webhook")
    print(f"Records: {len(get_all_records())}")
    print("=" * 60)
    print("\n✨ NEW FEATURES:")
    print("  • ✅ Mark Season as 100% Complete")
    print("  • ☑️  Multi-Select Episodes with Checkboxes")
    print("  • 🗑️ Bulk Delete Selected Episodes")
    print("  • 📊 Visual Selection Counter")
    print("  • ⚡ Select All / Clear Selection")
    print("=" * 60 + "\n")
    
    app.run(host="0.0.0.0", port=5000, debug=False)