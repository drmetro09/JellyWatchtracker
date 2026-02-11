#!/bin/sh

PUID=${PUID:-99}
PGID=${PGID:-100}

echo "🚀 Jellyfin Watch Tracker"
echo "👤 PUID: $PUID | PGID: $PGID"

if ! getent group "$PGID" > /dev/null 2>&1; then
    addgroup -g "$PGID" appgroup
fi

GROUP_NAME=$(getent group "$PGID" | cut -d: -f1)

if ! getent passwd "$PUID" > /dev/null 2>&1; then
    adduser -D -u "$PUID" -G "$GROUP_NAME" appuser
fi

USER_NAME=$(getent passwd "$PUID" | cut -d: -f1)

mkdir -p /data
mkdir -p /data/posters
chown -R "$PUID:$PGID" /data 2>/dev/null || true
touch /data/watch_history.json 2>/dev/null || true
chown "$PUID:$PGID" /data/watch_history.json 2>/dev/null || true

echo "✅ Running as: $USER_NAME"
echo ""

exec su-exec "$USER_NAME" python3 -u /app/watch_tracker.py
