import os
from celery import Celery

BACKEND_IP = os.environ.get("REDIS_HOST", "localhost")
BACKEND_PORT = os.environ.get("REDIS_PORT", "6379")
PASSWORD = os.environ.get("REDIS_PASSWORD", "")
DB_NUMBER = os.environ.get("REDIS_DB", "0")
BACKEND_URI = f"redis://:{PASSWORD}@{BACKEND_IP}:{BACKEND_PORT}/{DB_NUMBER}"

BROKER_IP = os.environ.get("RABBITMQ_HOST", BACKEND_IP)
BROKER_PORT = os.environ.get("RABBITMQ_PORT", "5672")
BROKER_USER = os.environ.get("RABBITMQ_USER", "celery_user")
BROKER_PASSWORD = os.environ.get("RABBITMQ_PASSWORD", PASSWORD)
BROKER_VHOST = os.environ.get("RABBITMQ_VHOST", "celery_host")
BROKER_URI = f"amqp://{BROKER_USER}:{BROKER_PASSWORD}@{BROKER_IP}:{BROKER_PORT}/{BROKER_VHOST}"

app = Celery(
    "downloader",
    broker=BROKER_URI,
    backend=BACKEND_URI,
    include=["tasks"],
)

app.conf.task_routes = {"tasks.download_task": {"queue": "bvd-downloads"}}
app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Europe/London",
    enable_utc=True,
)
