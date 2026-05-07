import os
import shutil
import subprocess
from celery_config import app
from celery.exceptions import SoftTimeLimitExceeded
from celery.utils.log import get_task_logger
from typing import Optional
from utils import (
    download_video_audio,
    FFMPEGException,
    TooManyRequestsException,
    DownloadException,
    LikelyBlockedException,
    NodeNotReadyException,
    sleep_timer,
    get_host_ip,
)
from config import (
    VEXTENSION,
    AEXTENSION,
    AQUALITY,
    REQUIRED_MOUNT_PATH,
)
from tempfile import TemporaryDirectory

logger = get_task_logger(__name__)


@app.task(
    rate_limit="130/h",
    acks_late=True,
    reject_on_worker_lost=True,
    soft_time_limit=60 * 4,
    time_limit=60 * 15,
    throws=(
        TooManyRequestsException,
        DownloadException,
        LikelyBlockedException,
        SoftTimeLimitExceeded,
    ),
    queue="bvd-downloads",
)
def download_task(
    url: str,
    output_path: str,
    out_file_name: str,
    yt_format: str = f"b[height<=360][ext={VEXTENSION}]",
    fall_back_format: Optional[str] = f"b[ext={VEXTENSION}]",
    keyframes_only: bool = False,
    audio_format: str = AEXTENSION,
    audio_quality: str = AQUALITY,
):
    if REQUIRED_MOUNT_PATH and not os.path.ismount(REQUIRED_MOUNT_PATH):
        raise NodeNotReadyException(
            f"{REQUIRED_MOUNT_PATH} is not mounted", get_host_ip()
        )
    with TemporaryDirectory("_celery") as tmp_path:
        try:
            files = download_video_audio(
                url,
                tmp_path,
                yt_format=yt_format,
                fall_back_format=fall_back_format,
                audio_quality=audio_quality,
                audio_format=audio_format,
            )

            meta_exts = (".json", ".vtt")
            for meta_file in (f for f in files if f.endswith(meta_exts)):
                logger.info("Copying Metadata " + meta_file)
                meta_ext = ".".join(meta_file.split(".")[1:])
                shutil.copy(
                    os.path.join(tmp_path, meta_file),
                    os.path.join(output_path, f"{out_file_name}.{meta_ext}"),
                )

            video_files = [f for f in files if f.endswith(VEXTENSION)]
            if len(video_files) == 0:
                raise DownloadException("No video files found")
            keyframe_args = ["-discard", "nokey"] if keyframes_only else []
            for video_file in video_files:
                logger.info("Remove audio %s", video_file)
                args = [
                    "ffmpeg",
                    "-y",
                    *keyframe_args,
                    "-i",
                    os.path.join(tmp_path, video_file),
                    "-c",
                    "copy",
                    "-copyts",
                    "-an",
                    os.path.join(output_path, f"{out_file_name}.{VEXTENSION}"),
                ]
                res = subprocess.run(args, capture_output=True, check=False)
                if res.returncode != 0:
                    raise FFMPEGException(res.stderr.decode("utf-8"))

            audio_files = [f for f in files if f.endswith(audio_format)]
            if len(audio_files) == 0:
                raise DownloadException("No audio files found")
            for audio_file in audio_files:
                shutil.copy(
                    os.path.join(tmp_path, audio_file),
                    os.path.join(output_path, f"{out_file_name}.{audio_format}"),
                )

            logger.info("Done videos %s", url)
            sleep_timer.sleep_normal()
        except Exception as e:
            logger.error("ERROR: %s: %s", url, e)
            sleep_timer.sleep_error()
            raise e
