# 🎬 Jellyfin Watch Tracker

A beautiful, feature-rich web application for tracking your Jellyfin media watch history with advanced analytics, progress tracking, and TMDB integration.

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
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

### 🔄 **Import Capabilities**
- 📥 Jellyfin history import (full watch history sync)
- 🎬 Radarr integration (auto-import movie library)
- 📺 Sonarr integration (auto-import TV library)

### 🎨 **Beautiful UI**
- 🌙 Dark/Light theme toggle
- 📱 Fully responsive mobile design
- ⚡ Performance mode for low-end devices
- 🔍 Grid and list view modes
- 🎨 Smooth animations and modern design
- 📤 Export data (JSON/CSV)

### 🚀 **Performance**
- 💾 Smart caching system (posters, series info)
- ⚡ Lazy loading images
- 🔄 Auto-optimization for mobile devices
- 📦 Efficient data storage

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
  <Name>JellyfinWatchTracker</Name>
  <Repository>ghcr.io/drmetro09/jellyfin-watch-tracker:latest</Repository>
  <Registry>https://hub.docker.com/</Registry>
  <Network>bridge</Network>
  <Privileged>false</Privileged>
  <Support>https://github.com/drmetro09/jellyfin-watch-tracker/issues</Support>
  <Project>https://github.com/drmetro09/jellyfin-watch-tracker</Project>
  <Overview>
    Beautiful web application for tracking Jellyfin watch history with advanced analytics, 
    progress tracking, and TMDB integration. Features include episode-level tracking, 
    custom posters, genre analytics, watch streaks, and import from Jellyfin/Sonarr/Radarr.
  </Overview>
  <Category>MediaApp:Video MediaServer:Video Status:Stable</Category>
  <WebUI>http://[IP]:[PORT:5000]</WebUI>
  <TemplateURL/>
  <Icon>https://raw.githubusercontent.com/drmetro09/jellyfin-watch-tracker/main/icon.png</Icon>
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
3. **Import your Jellyfin history**:
   - Go to **Settings** tab
   - Click **Import from Jellyfin**
   - Wait for import to complete (may take several minutes for large libraries)
4. **Optional: Import from Sonarr/Radarr** to pre-populate your library

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

### Viewing Statistics

**Library Tab:**
- View all movies and TV shows
- Filter by type (All, Movies, Shows, Incomplete)
- Switch between Grid and List views
- Search and sort options

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

---

## 🛠️ Development

### Build from Source

```bash
# Clone repository
git clone https://github.com/drmetro09/jellyfin-watch-tracker.git
cd jellyfin-watch-tracker

# Build Docker image
docker build -t jellyfin-watch-tracker .

# Run locally
docker run -p 5000:5000 -v $(pwd)/data:/data jellyfin-watch-tracker
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

## 📝 Jellyfin Webhook Setup (Real-time Tracking)

For automatic watch history updates:

1. Install **Jellyfin Webhooks** plugin
2. Add webhook URL: `http://your-tracker-ip:5000/api/webhook`
3. Enable **Playback Stop** event
4. Watches will be tracked automatically!

---

## 🎨 Themes

- **🌙 Dark Mode** (default) - Easy on the eyes
- **☀️ Light Mode** - Bright and clean
- **⚡ Performance Mode** - Optimized for low-end devices

Toggle in header toolbar!

---

## 🐛 Troubleshooting

### Import Issues

**Jellyfin import stuck:**
- Check API key validity
- Verify Jellyfin URL is accessible
- Check Docker logs: `docker logs jellyfin-watch-tracker`

**Missing posters:**
- Verify TMDB API key is set
- Check TMDB API rate limits (40 requests/10 seconds)
- Clear cache and refresh

**Episode counts wrong:**
- Click **Refresh Metadata** on show
- Verify show name matches TMDB exactly
- Manually mark season complete if needed

### Performance Issues

**Slow loading:**
- Enable **Performance Mode** in header
- Use Grid view instead of List for large libraries
- Clear browser cache

**High memory usage:**
- Reduce cache sizes in `data/` directory
- Restart container: `docker restart jellyfin-watch-tracker`

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

- 🐛 **Issues**: [GitHub Issues](https://github.com/drmetro09/jellyfin-watch-tracker/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/drmetro09/jellyfin-watch-tracker/discussions)
- 📖 **Wiki**: [Documentation](https://github.com/drmetro09/jellyfin-watch-tracker/wiki)

---
---

## 🤖 AI-Generated Disclaimer

This entire project was **vibe-coded** with AI assistance. It works great for my use case, but your mileage may vary. Code reviews, refactoring suggestions, and bug reports are not just welcome—they're encouraged! 🚀

Built with ☕ + 🤖 + 💾

## ⭐ Star History

If you find this project useful, please consider giving it a star!

[![Star History Chart](https://api.star-history.com/svg?repos=your-drmetro09/jellyfin-watch-tracker&type=Date)](https://star-history.com/#your-username/jellyfin-watch-tracker&Date)

---

**Made with ❤️ for the Jellyfin community**