FROM python:3.11-alpine

RUN apk add --no-cache tzdata su-exec && \
    pip install --no-cache-dir flask==3.0.0 requests==2.31.0

ENV TZ=Asia/Kolkata

WORKDIR /app

COPY watch_tracker.py /app/
COPY entrypoint.sh /app/

RUN chmod +x /app/entrypoint.sh

VOLUME /data

EXPOSE 5000

ENTRYPOINT ["/app/entrypoint.sh"]
