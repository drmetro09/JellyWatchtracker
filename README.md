# 🎬 Jellyfin Watch Tracker

JellyWatchTracker is a self-hosted web application that aggregates your watch history from Jellyfin, Sonarr and Radarr and wraps it in a modern dashboard. It exposes a simple Flask API backed by JSON files and a single-page web UI to help you explore your library, monitor viewing progress, discover genre and mood trends, and keep track of which shows and movies you've completed. Features include automatic Jellyfin polling, ratings & notes, undo functionality, collapsible progress sections, mood-based recommendations, genre combos, detailed charts, mobile-friendly design, manual progress controls and administrative tools to clear caches or rebuild insights.

![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)
![License](https://img.shields.io/badge/license-GPL--3.0-green.svg)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Docker](https://img.shields.io/badge/docker-ready-blue.svg)

---

## ✨ Features

### 📚 **Library Management**
- 🎬 Track movies and TV shows with watch counts
- 📺 Detailed episode-level tracking with season management
- 🖼️ Custom poster uploads for personalized library
- ✏️ Manual watch entry with bulk episode/season addition
- 🗑️ Selective deletion of movies, shows, seasons, or episodes
- ⭐ 5-star ratings and text notes for movies and shows
- ↩️ Undo functionality (last 20 actions)
- 🔗 Direct "Open in Jellyfin" links for all items

### 📊 **Advanced Analytics**
- 📈 Watch statistics and trends
- 🎭 Genre breakdown with visual pie charts
- 🔥 Trending content (most watched this week)
- 📉 Watch streaks (current and longest)
- ⚡ Quick stats (most rewatched, fastest completion, etc.)
- 🎯 Smart recommendations based on watch history

### 🎯 **Progress Tracking**
- 📈 Completion percentage for TV shows
- 🏆 In-progress vs. completed shows categorization
- ✅ Manual completion marking for shows/seasons
- 🔍 Search, sort, and filter progress lists
- 📊 TMDB integration for accurate episode counts
- 💾 Persistent show totals survive cache clears

### 🔄 **Automatic Sync & Import**
- 🔄 Automatic Jellyfin polling (configurable intervals)
- 📥 Jellyfin history import (full watch history sync)
- 🎬 Radarr integration (auto-import movie library)
- 📺 Sonarr integration (auto-import TV library)
- 🔔 Webhook deduplication (20-second window)
- ⚡ Background TMDB pre-warming for faster UI

### 🎨 **Beautiful UI**
- 🌙 Multiple themes (Dark, Light, AMOLED, Solarized, Nord)
- 📱 Fully responsive mobile design with optimized touch targets
- ⚡ Performance mode for low-end devices
- 🔍 Grid and list view modes with 3 layout densities (Compact, Comfortable, Spacious)
- 🎨 Smooth animations and modern design
- 📤 Export data (JSON/CSV)
- 🔎 Zoom control (70%-150%)
- 🔄 Auto-refresh with configurable intervals (1m, 5m, 10m, 15m, 30m)
- ♿ Accessibility support (reduced motion, focus indicators)

### 🚀 **Performance**
- 💾 Smart caching system (posters, series info)
- ⚡ Fast-path cache-only TMDB reads
- 🔄 Background TMDB fetching (4 parallel workers)
- 📦 Efficient data storage with smart invalidation
- 🎯 Content visibility optimization for large libraries
- 📱 Auto-optimization for mobile devices

---

## 🐳 Docker Installation

### Docker Compose (Recommended)

Create a `docker-compose.yml` file:

```yaml
version: '3.8'

services:
  jellyfin-watch-tracker:
    image: ghcr.io/drmetro09/jellyfin-watch-tracker:latest
    container_name: jellyfin-watch-tracker
    restart: unless-stopped
    ports:
      - "5000:5000"
    volumes:
      - ./data:/data
      - ./posters:/data/custom_posters
    environment:
      # Required: Jellyfin Connection
      - JELLYFIN_URL=http://your-jellyfin-server:8096
      - JELLYFIN_API_KEY=your_jellyfin_api_key

      # Required: TMDB API (for posters and metadata)
      - TMDB_API_KEY=your_tmdb_api_key

      # Optional: Sonarr Integration
      - SONARR_URL=http://your-sonarr-server:8989
      - SONARR_API_KEY=your_sonarr_api_key

      # Optional: Radarr Integration
      - RADARR_URL=http://your-radarr-server:7878
      - RADARR_API_KEY=your_radarr_api_key

      # Optional: Jellyfin Linking (for reverse proxy setups)
      - JELLYFIN_PUBLIC_URL=https://media.example.com/jellyfin

      # Optional: Polling Configuration
      - JELLYFIN_POLL_ENABLED=true           # Enable automatic polling (default: true)
      - JELLYFIN_POLL_INTERVAL_S=60          # Poll every N seconds (default: 60)
      - JELLYFIN_POLL_LIMIT=80               # Items to fetch per poll (default: 80)
      - JELLYFIN_POLL_LOOKBACK_S=604800      # Lookback period in seconds (default: 7 days)
```

Run with:
```bash
docker-compose up -d
```

### Docker Run Command

```bash
docker run -d \
  --name jellyfin-watch-tracker \
  -p 5000:5000 \
  -v $(pwd)/data:/data \
  -v $(pwd)/posters:/data/custom_posters \
  -e JELLYFIN_URL=http://your-jellyfin-server:8096 \
  -e JELLYFIN_API_KEY=your_jellyfin_api_key \
  -e TMDB_API_KEY=your_tmdb_api_key \
  -e SONARR_URL=http://your-sonarr-server:8989 \
  -e SONARR_API_KEY=your_sonarr_api_key \
  -e RADARR_URL=http://your-radarr-server:7878 \
  -e RADARR_API_KEY=your_radarr_api_key \
  -e JELLYFIN_PUBLIC_URL=https://media.example.com/jellyfin \
  -e JELLYFIN_POLL_ENABLED=true \
  -e JELLYFIN_POLL_INTERVAL_S=60 \
  --restart unless-stopped \
  ghcr.io/drmetro09/jellyfin-watch-tracker:latest
```

---

## 🖥️ Unraid Docker Template

### Installation via Community Applications

1. Go to **Apps** tab in Unraid
2. Search for "Jellyfin Watch Tracker"
3. Click **Install**
4. Configure the template (see below)

### Manual Template XML

Save this as a template in `/boot/config/plugins/dockerMan/templates-user/`:

```xml
<?xml version="1.0"?>
<Container version="2">
  <Name>JellyWatchTracker</Name>
  <Repository>ghcr.io/drmetro09/jellyfin-watch-tracker:latest</Repository>
  <Registry>https://hub.docker.com/</Registry>
  <Network>bridge</Network>
  <Privileged>false</Privileged>
  <Support>https://github.com/drmetro09/JellyWatchtracker/issues</Support>
  <Project>https://github.com/drmetro09/JellyfinWatchtracker</Project>
  <Overview>
    Beautiful web application for tracking Jellyfin watch history with advanced analytics, 
    progress tracking, TMDB integration, ratings &amp; notes, automatic polling, and undo functionality. 
    Features include episode-level tracking, custom posters, genre analytics, watch streaks, 
    multiple themes, performance mode, and import from Jellyfin/Sonarr/Radarr.
  </Overview>
  <Category>MediaApp:Video MediaServer:Video Status:Stable</Category>
  <WebUI>http://[IP]:[PORT:5000]</WebUI>
  <TemplateURL/>
  <Icon>https://raw.githubusercontent.com/drmetro09/JellyWatchtracker/main/icon.png</Icon>
  <ExtraParams/>
  <PostArgs/>
  <CPUset/>
  <DateInstalled></DateInstalled>
  <DonateText/>
  <DonateLink/>
  <Requires/>

  <Config Name="WebUI Port" Target="5000" Default="5000" Mode="tcp" Description="Web interface port" Type="Port" Display="always" Required="true" Mask="false">5000</Config>

  <Config Name="Data Directory" Target="/data" Default="/mnt/user/appdata/jellyfin-watch-tracker/data" Mode="rw" Description="Persistent data storage" Type="Path" Display="always" Required="true" Mask="false">/mnt/user/appdata/jellyfin-watch-tracker/data</Config>

  <Config Name="Custom Posters Directory" Target="/data/custom_posters" Default="/mnt/user/appdata/jellyfin-watch-tracker/posters" Mode="rw" Description="Custom poster uploads" Type="Path" Display="always" Required="true" Mask="false">/mnt/user/appdata/jellyfin-watch-tracker/posters</Config>

  <Config Name="Jellyfin URL" Target="JELLYFIN_URL" Default="http://192.168.1.100:8096" Mode="" Description="Your Jellyfin server URL (no trailing slash)" Type="Variable" Display="always" Required="true" Mask="false">http://192.168.1.100:8096</Config>

  <Config Name="Jellyfin API Key" Target="JELLYFIN_API_KEY" Default="" Mode="" Description="Jellyfin API key (Dashboard &gt; Advanced &gt; API Keys)" Type="Variable" Display="always" Required="true" Mask="true"></Config>

  <Config Name="TMDB API Key" Target="TMDB_API_KEY" Default="" Mode="" Description="TMDB API key for posters/metadata (get from themoviedb.org/settings/api)" Type="Variable" Display="always" Required="true" Mask="true"></Config>

  <Config Name="Jellyfin Public URL" Target="JELLYFIN_PUBLIC_URL" Default="" Mode="" Description="[OPTIONAL] Public Jellyfin URL for 'Open in Jellyfin' links (if behind reverse proxy)" Type="Variable" Display="always" Required="false" Mask="false"></Config>

  <Config Name="Polling Enabled" Target="JELLYFIN_POLL_ENABLED" Default="true" Mode="" Description="[OPTIONAL] Enable automatic Jellyfin polling (true/false)" Type="Variable" Display="always" Required="false" Mask="false">true</Config>

  <Config Name="Polling Interval" Target="JELLYFIN_POLL_INTERVAL_S" Default="60" Mode="" Description="[OPTIONAL] Poll Jellyfin every N seconds (default: 60)" Type="Variable" Display="always" Required="false" Mask="false">60</Config>

  <Config Name="Sonarr URL" Target="SONARR_URL" Default="" Mode="" Description="[OPTIONAL] Sonarr server URL for TV import" Type="Variable" Display="always" Required="false" Mask="false"></Config>

  <Config Name="Sonarr API Key" Target="SONARR_API_KEY" Default="" Mode="" Description="[OPTIONAL] Sonarr API key" Type="Variable" Display="always" Required="false" Mask="true"></Config>

  <Config Name="Radarr URL" Target="RADARR_URL" Default="" Mode="" Description="[OPTIONAL] Radarr server URL for movie import" Type="Variable" Display="always" Required="false" Mask="false"></Config>

  <Config Name="Radarr API Key" Target="RADARR_API_KEY" Default="" Mode="" Description="[OPTIONAL] Radarr API key" Type="Variable" Display="always" Required="false" Mask="true"></Config>
</Container>
```

---

## 🔧 Configuration

### Required Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `JELLYFIN_URL` | Jellyfin server URL (no trailing slash) | `http://192.168.1.100:8096` |
| `JELLYFIN_API_KEY` | Jellyfin API key | Get from Dashboard → Advanced → API Keys |
| `TMDB_API_KEY` | TMDB API key for posters/metadata | Get from themoviedb.org/settings/api |

### Optional Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `JELLYFIN_PUBLIC_URL` | Public Jellyfin URL for reverse proxy setups | None |
| `JELLYFIN_POLL_ENABLED` | Enable automatic polling fallback | `true` |
| `JELLYFIN_POLL_INTERVAL_S` | Polling interval in seconds | `60` |
| `JELLYFIN_POLL_LIMIT` | Items to fetch per poll | `80` |
| `JELLYFIN_POLL_LOOKBACK_S` | Lookback period in seconds | `604800` (7 days) |
| `SONARR_URL` | Sonarr server URL | None |
| `SONARR_API_KEY` | Sonarr API key | None |
| `RADARR_URL` | Radarr server URL | None |
| `RADARR_API_KEY` | Radarr API key | None |

### Getting API Keys

**Jellyfin:**
1. Go to Jellyfin Dashboard
2. Navigate to Advanced → API Keys
3. Create a new API key

**TMDB:**
1. Sign up at [themoviedb.org](https://www.themoviedb.org/)
2. Go to Settings → API
3. Request an API key (free)

**Sonarr/Radarr:**
1. Open Sonarr/Radarr web interface
2. Go to Settings → General
3. Copy the API Key

---

## 📁 Data Persistence

The application stores data in `/data`:

```
/data/
├── watch_history.json        # Main watch history database
├── poster_cache.json          # Cached TMDB posters
├── series_cache.json          # Cached TV series metadata
├── custom_posters.json        # Custom poster mappings
├── manual_complete.json       # Manually completed shows
├── season_complete.json       # Manually completed seasons
├── ratings.json               # User ratings and notes
├── action_history.json        # Undo history (last 20 actions)
├── user_preferences.json      # Theme/layout preferences
├── show_totals_cache.json     # TMDB episode count cache
├── jellyfin_poll_state.json   # Polling state tracker
└── custom_posters/            # Uploaded poster images
    ├── movie_Title_2023.jpg
    └── tv_SeriesName_2020.jpg
```

### Backup Recommendations

```bash
# Backup data directory
docker exec jellyfin-watch-tracker tar czf /data/backup.tar.gz /data/*.json

# Export to JSON
curl http://localhost:5000/api/export > backup.json

# Export to CSV
curl http://localhost:5000/api/export_csv > backup.csv
```

---

## 🎯 Usage

### First Time Setup

1. **Start the container** using Docker Compose or Docker Run
2. **Access the web interface** at `http://localhost:5000`
3. **Automatic polling starts immediately** (polls Jellyfin every 60 seconds by default)
4. **Optional: Manual import from Jellyfin**:
   - Go to **Settings** tab
   - Click **Import from Jellyfin**
   - Wait for import to complete (may take several minutes for large libraries)
5. **Optional: Import from Sonarr/Radarr** to pre-populate your library

### Ratings & Notes

**Add Ratings:**
- Click the **⭐** icon on any movie or show
- Select 1-5 stars
- Add optional text notes
- Ratings persist across devices

**View Ratings:**
- Ratings display inline on each item
- Filter/sort by rating (coming soon)

### Undo Functionality

**Undo Actions:**
- Click **↩️ Undo** button in header
- Reverses last action (delete, manual entry, etc.)
- Stores last 20 actions

### Adding Watch Entries

**Manual Entry:**
- Click **➕ Add Watch** button
- Select Movie or Episode
- Fill in details (bulk episode ranges supported: `1,2,5-8`)
- Click Submit

**Bulk Operations:**
- Add entire TV show: Enter show name only
- Add entire season: Enter show + season number
- Add specific episodes: Enter show + season + episodes (e.g., `1,2,5-8`)

### UI Customization

**Themes:**
- Click **🌙 Theme** button to cycle through:
  - Dark (default)
  - Light
  - AMOLED (pure black)
  - Solarized
  - Nord
- Preferences persist across sessions

**Layout Modes:**
- Choose **Compact**, **Comfortable**, or **Spacious**
- Works in both Grid and List views
- Mobile-optimized spacing

**Zoom Control:**
- Adjust UI scale from 70% to 150%
- Useful for high-DPI displays

**Performance Mode:**
- Click **⚡ Performance** button
- Disables animations and effects
- Optimizes for low-end devices

**Auto-Refresh:**
- Enable checkbox in header
- Choose interval: 1m, 5m, 10m, 15m, 30m
- Countdown timer shows next refresh

### Managing Shows

**Mark as Complete:**
- Hover over show/season
- Click **✓ Mark 100%**
- Useful for completed shows with missing metadata

**Custom Posters:**
- Hover over poster
- Click **📤** button
- Upload custom image

**Delete Operations:**
- Delete individual episodes
- Delete entire seasons
- Delete entire shows or movies
- Use **↩️ Undo** to reverse mistakes

### Viewing Statistics

**Library Tab:**
- View all movies and TV shows
- Filter by type (All, Movies, Shows, Incomplete)
- Switch between Grid and List views
- Search and sort options
- Click **🔗** to open in Jellyfin (if `JELLYFIN_PUBLIC_URL` set)

**Genres Tab:**
- See genre distribution
- Visual pie chart
- Movies and shows per genre

**Insights Tab:**
- Watch statistics and trends
- Quick stats (most rewatched, fastest completion, etc.)
- Watch streaks
- Smart recommendations based on history

**Progress Tab:**
- Track show completion
- Filter in-progress vs. completed
- Sort by various criteria
- Search specific shows

---

## 🚀 API Endpoints

### Watch History
- `GET /api/data` - Get organized watch data
- `GET /api/history` - Get organized watch data (alias)
- `GET /api/history_sig` - Fast data signature check
- `POST /api/webhook` - Jellyfin webhook endpoint
- `POST /api/manual_entry` - Add manual watch entry
- `POST /api/manual_tv_bulk` - Bulk add TV episodes

### Data Management
- `POST /api/delete_movie` - Delete movie entry
- `POST /api/delete_show` - Delete entire show
- `POST /api/delete_season` - Delete season
- `POST /api/delete_episode` - Delete episode
- `POST /api/bulk_delete` - Delete multiple entries

### Import/Export
- `POST /api/import_jellyfin` - Import from Jellyfin
- `POST /api/import_sonarr` - Import from Sonarr
- `POST /api/import_radarr` - Import from Radarr
- `GET /api/export` - Export JSON
- `GET /api/export_csv` - Export CSV

### Customization
- `POST /api/upload_poster` - Upload custom poster
- `POST /api/mark_complete` - Mark show complete
- `POST /api/mark_season_complete` - Mark season complete
- `POST /api/refresh_series` - Refresh TMDB metadata

### Ratings & Preferences
- `POST /api/rate` - Add/update rating and note
- `GET /api/ratings` - Get all ratings
- `POST /api/undo` - Undo last action
- `GET /api/preferences` - Get user preferences
- `POST /api/preferences` - Save user preferences
- `GET /api/jellyfin_link/{jellyfin_id}` - Generate Jellyfin direct link

---

## 🛠️ Development

### Build from Source

```bash
# Clone repository
git clone https://github.com/drmetro09/JellyWatchtracker.git
cd JellyWatchtracker/docker

# Build Docker image
docker build -t JellyfinWatchtracker .

# Run locally
docker run -p 5000:5000 -v $(pwd)/data:/data JellyfinWatchtracker
```

### Local Development (without Docker)

```bash
# Install dependencies
pip install flask requests

# Set environment variables
export JELLYFIN_URL=http://localhost:8096
export JELLYFIN_API_KEY=your_key
export TMDB_API_KEY=your_key

# Run application
python watch_tracker.py
```

---

## 📝 Jellyfin Integration

### Automatic Polling (Enabled by Default)

The application automatically polls Jellyfin every 60 seconds (configurable) to catch plays that webhooks might miss:

- **No webhook setup required**
- Catches "Next Episode" autoplay events
- Deduplicates against webhook data
- Configurable polling interval and lookback period
- Lightweight (only fetches recent plays)

### Webhook Setup (Optional - for real-time updates)

For instant watch history updates:

1. Install **Jellyfin Webhooks** plugin
2. Add webhook URL: `http://your-tracker-ip:5000/api/webhook`
3. Enable **Playback Stop** event
4. Watches will be tracked in real-time!

**Note:** Even without webhooks, the polling system ensures all plays are tracked.

---

## 🎨 Themes

- **🌙 Dark Mode** (default) - Easy on the eyes
- **☀️ Light Mode** - Bright and clean
- **🌑 AMOLED** - Pure black for OLED displays
- **🌅 Solarized** - Refined color palette
- **❄️ Nord** - Arctic-inspired theme

Toggle in header toolbar! Preferences persist across devices.

---

## 🐛 Troubleshooting

### Polling Issues

**Polling not working:**
- Check `JELLYFIN_POLL_ENABLED=true` in environment variables
- Verify Jellyfin URL and API key
- Check Docker logs: `docker logs jellyfin-watch-tracker`
- Look for "✓ Jellyfin poll sync" messages

**Duplicate entries:**
- Adjust `JELLYFIN_POLL_INTERVAL_S` (increase to 120+ seconds)
- The 20-second deduplication window should prevent most duplicates

### Import Issues

**Jellyfin import stuck:**
- Check API key validity
- Verify Jellyfin URL is accessible
- Check Docker logs: `docker logs jellyfin-watch-tracker`

**Missing posters:**
- Verify TMDB API key is set
- Check TMDB API rate limits (40 requests/10 seconds)
- Background pre-warming will gradually fetch missing posters
- Clear cache and refresh

**Episode counts wrong:**
- Click **Refresh Metadata** on show
- Verify show name matches TMDB exactly
- Manually mark season complete if needed
- Totals now persist across cache clears

### Performance Issues

**Slow loading:**
- Enable **⚡ Performance Mode** in header
- Use Grid view instead of List for large libraries
- Clear browser cache
- Reduce zoom level

**High memory usage:**
- Reduce cache sizes in `data/` directory
- Adjust `JELLYFIN_POLL_LIMIT` (lower = less memory)
- Restart container: `docker restart jellyfin-watch-tracker`

### Undo Not Working

**Undo button disabled:**
- Undo history stores last 20 actions only
- Some actions (like imports) cannot be undone
- Check `/data/action_history.json` for stored actions

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📜 License

This project is licensed under the **GNU General Public License v3.0** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Jellyfin** - Open-source media server
- **TMDB** - Movie and TV metadata
- **Flask** - Python web framework
- **Chart.js** - Beautiful charts

---

## 📧 Support

- 🐛 **Issues**: [GitHub Issues](https://github.com/drmetro09/JellyWatchtracker/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/drmetro09/JellyWatchtracker/discussions)
- 📖 **Wiki**: [Documentation](https://github.com/drmetro09/JellyWatchtracker/wiki)

---

## 🆕 What's New in v2.0

### 🔄 Automatic Polling
- No more missed plays! Automatic fallback polling every 60 seconds
- Catches "Next Episode" autoplay that webhooks miss
- Smart deduplication prevents duplicate entries

### ⭐ Ratings & Notes
- Rate movies and shows with 1-5 stars
- Add text notes and reviews
- Ratings persist across devices

### ↩️ Undo Functionality
- Reverse mistakes instantly
- Stores last 20 actions
- Works for deletions and manual entries

### 🎨 Enhanced UI
- 5 beautiful themes (Dark, Light, AMOLED, Solarized, Nord)
- 3 layout densities (Compact, Comfortable, Spacious)
- Zoom control (70%-150%)
- Performance mode for low-end devices
- Auto-refresh with configurable intervals
- Mobile-optimized touch targets

### ⚡ Performance Improvements
- Background TMDB pre-warming (4 parallel workers)
- Smart cache invalidation
- Content visibility optimization
- Persistent show totals survive cache clears
- Fast-path cache-only reads

### 🔗 Jellyfin Integration
- Direct "Open in Jellyfin" links
- Reverse proxy support via `JELLYFIN_PUBLIC_URL`
- Enhanced webhook deduplication

---

## 🤖 AI-Generated Disclaimer

This entire project was **vibe-coded** with AI assistance. It works great for my use case, but your mileage may vary. Code reviews, refactoring suggestions, and bug reports are not just welcome—they're encouraged! 🚀

Built with ☕ + 🤖 + 💾

## ⭐ Star History

If you find this project useful, please consider giving it a star!

[![Star History Chart](https://api.star-history.com/svg?repos=drmetro09/jellyfin-watch-tracker&type=Date)](https://star-history.com/#drmetro09/jellyfin-watch-tracker&Date)

---

**Made with ❤️ for the Jellyfin community**
