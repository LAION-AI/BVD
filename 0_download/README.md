# Download

Distributed video downloader built on [Celery](https://docs.celeryq.dev/) and [yt-dlp](https://github.com/yt-dlp/yt-dlp).

Videos are downloaded from a parquet file of URLs, split into shards, and stored with separated audio, video, subtitles, and metadata.

## Architecture

- **Broker**: RabbitMQ
- **Backend**: Redis
- **Workers**: one Celery worker per download node, consuming the `bvd-downloads` queue
- **Orchestrator**: `CeleryDownloadManager` running on a management node

## Configuration

All connection settings are read from environment variables:

| Variable | Default | Description |
|---|---|---|
| `REDIS_HOST` | `localhost` | Redis host |
| `REDIS_PORT` | `6379` | Redis port |
| `REDIS_PASSWORD` | `` | Redis password |
| `REDIS_DB` | `0` | Redis DB number |
| `RABBITMQ_HOST` | `$REDIS_HOST` | RabbitMQ host |
| `RABBITMQ_PORT` | `5672` | RabbitMQ port |
| `RABBITMQ_USER` | `celery_user` | RabbitMQ username |
| `RABBITMQ_PASSWORD` | `$REDIS_PASSWORD` | RabbitMQ password |
| `RABBITMQ_VHOST` | `celery_host` | RabbitMQ vhost |
| `CELERY_USER` | `celery` | SSH user for worker management |
| `CELERY_PATH` | `celery` | Path to celery binary on workers |
| `CELERY_MODULE` | `celery_config` | Celery app module |
| `CELERY_ENV_SETUP` | `` | Optional shell setup run before launching workers |
| `CELERY_PROCESS_PATTERN` | `$CELERY_PATH -A $CELERY_MODULE worker` | Process pattern used to find/stop workers |
| `CELERY_SETUP_SCRIPT` | `setup_celery.sh` | Optional setup script path |
| `DOWNLOAD_REQUIRED_MOUNT` | `` | Optional mount path workers must see before downloading |

## Output structure

```
<download_path>/<parquet_name>/
    00000/
        000000.mp4       # video (no audio)
        000000.mp3       # audio
        000000.info.json # yt-dlp metadata
        000000.<lang>.vtt  # subtitles
        ...
    00000.parquet        # per-shard result log (presence marks shard complete)
    00001/
    ...
```

## Usage

Run from inside `0_download/`, or add this directory to `PYTHONPATH`.

```python
from download_manager import CeleryDownloadManager

manager = CeleryDownloadManager(
    parquet_file="path/to/shard.parquet",
    files_per_folder=1000,
    download_path="/path/to/downloads/",
)
manager.download_parallel(parallel=4)
```

The parquet file must have `url`, `uid`, `page_url`, and `domain` columns.

## Download task

Each URL is processed by `download_task` in `tasks.py`:

1. Downloads video and audio separately via yt-dlp (≤360p MP4 + MP3 at 64k)
2. Strips audio track from video file with ffmpeg
3. Copies subtitles (VTT) and info JSON to the output folder
4. Workers that receive too many rate-limit or block responses are automatically paused via the blocked-worker list

## Dependencies

```
celery
yt-dlp
ffmpeg  (system binary)
pandas
pyarrow
redis
rich
```
