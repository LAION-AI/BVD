import os
import socket
import subprocess
from time import sleep
from random import random
from celery.utils.log import get_task_logger
from typing import Optional
from config import (
    DOWNLOAD_THREADS,
    VEXTENSION,
    AEXTENSION,
    AQUALITY,
    TOO_MANY_REQUESTS_TIMEOUT,
    DEFAULT_TIMEOUT,
    DEFAULT_USER,
    DEFAULT_CELERY_PATH,
    DEFAULT_MODULE,
    DEFAULT_CELERY_ENV_SETUP,
    DEFAULT_CELERY_PROCESS_PATTERN,
)
from typing import List

logger = get_task_logger(__name__)


def run_ssh_command(
    host, user, command: str | List[str], timeout=None, check=True, ignore_key=False
):
    if isinstance(command, str):
        command = [command]
    ignore_host_key = []
    if ignore_key:
        ignore_host_key = [
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
        ]
    return subprocess.run(
        ["ssh", *ignore_host_key, f"{user}@{host}"] + command,
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )


def start_celery(
    host, user=DEFAULT_USER, session_name="celery_session", check=True, timeout=60
):
    command = (
        f"{DEFAULT_CELERY_ENV_SETUP} && " if DEFAULT_CELERY_ENV_SETUP else ""
    ) + (
        f"{DEFAULT_CELERY_PATH} -A {DEFAULT_MODULE} worker "
        "--without-mingle --without-gossip -c 1 "
        "--queues=bvd-downloads --loglevel INFO"
    )
    return run_ssh_command(
        host,
        user,
        f"tmux new-session -d -s {session_name} '{command}'",
        check=check,
        timeout=timeout,
    )


def stop_celery(
    host,
    user=DEFAULT_USER,
    session_name="celery_session",
    check=False,
    cold_stop=False,
    wait=False,
    wait_timeout=60,
    capture_logs: str | None = "~/celery_session.log",
):
    # return run_ssh_command(
    #     host, user, f"tmux kill-session -t {session_name}", check=check
    # )
    if capture_logs is not None:
        run_ssh_command(
            host,
            user,
            f"tmux capture-pane -t {session_name} -S - -p > {capture_logs}",
            check=False,
        )
    signal = "-KILL" if cold_stop else "-TERM"
    res = run_ssh_command(
        host,
        user,
        f"pkill {signal} -f '{DEFAULT_CELERY_PROCESS_PATTERN}'",
        check=check,
    )
    # Allow the worker to stop
    if wait:
        sleep(1)
        while is_celery_running(host, user):
            sleep(3)
            wait_timeout -= 3
            if wait_timeout < 0:
                raise Exception("Timeout waiting for celery to stop")
    return res


def is_celery_running(host, user=DEFAULT_USER):
    try:
        result = run_ssh_command(
            host,
            user,
            f"pgrep -f '{DEFAULT_CELERY_PROCESS_PATTERN}'",
            timeout=20,
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        return False


def random_sleep(time=DEFAULT_TIMEOUT, jitter=0.5):
    sleep(time * (1 + jitter * (2 * random() - 1)))


def restart_celery(
    host,
    user=DEFAULT_USER,
    session_name="celery_session",
    max_retries=3,
    cold_stop=False,
):
    stop_celery(host, user, session_name, check=False, wait=True, cold_stop=cold_stop)
    for i in range(max_retries):
        try:
            start_celery(host, user, session_name, check=True)
            break
        except subprocess.CalledProcessError as e:
            print(e.stderr.decode())
            sleep(10)
    else:
        raise Exception("Failed to restart celery")


class TooManyRequestsException(Exception):
    pass


class FFMPEGException(Exception):
    pass


class DownloadException(Exception):
    pass


class LikelyBlockedException(Exception):
    pass


class NodeNotReadyException(Exception):
    pass


def get_host_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    s.close()
    return ip


def download(
    url: str,
    output_path: str = "/tmp/",
    yt_format: Optional[str] = None,
    fall_back_format: Optional[str] = f"b[ext={VEXTENSION}]",
):
    """
    Download a video from YouTube using yt-dlp.

    Args:
        url (str): The URL of the YouTube video to download.
        output_path (str, optional): The path where the downloaded video will be saved. Defaults to "/tmp/".
        yt_format (str, optional): The video format code to download. See "FORMAT SELECTION" for all the info. Defaults to None.
        fall_back_format (str, optional): If the format given by "yt_format" is not available, download the best MP4 format.
                                          Defaults to f"b[ext={VEXTENSION}]".

    Returns:
        List[str]: A list of filenames of the downloaded videos.

    Raises:
        Exception: If the download process times out or encounters an error.

    """

    if yt_format is None:
        yt_format = f"b[height<=360][ext={VEXTENSION}]"

    logger.info("Downloading video from YouTube %s format:%s", url, yt_format)
    args = [
        "yt-dlp",
        "-N",
        str(DOWNLOAD_THREADS),
        "-R",
        "1",
        "--write-info-json",
        "--write-subs",
        "--write-auto-sub",
        "--sub-langs",
        "de,en,es,fr,it,ur,pl,ru,hi,zh-Hans",
        "--convert-subs",
        "vtt",
        "--sub-format",
        "vtt",
        "--embed-chapters",
        "--sleep-requests",
        "10",
        "--no-progress",
        "-q",
        "--format",
        yt_format,
        "--output",
        output_path + "%(id)s.%(ext)s",
        url,
    ]

    result = subprocess.run(args, capture_output=True, timeout=None)
    if result.returncode == 0:
        return os.listdir(output_path)

    std_err = result.stderr.decode("utf-8")

    if "Too Many Requests" in std_err:
        logger.error("Too Many Requests %s", url)
        raise TooManyRequestsException("Too Many Requests", get_host_ip())

    if "Forbidden" in std_err:
        logger.error("Forbidden %s", url)
        raise LikelyBlockedException(std_err, get_host_ip())

    if "Requested format is not available" in std_err and fall_back_format is not None:
        logger.warning(
            "Requested format is not available, retrying with %s", fall_back_format
        )
        return download(url, output_path, fall_back_format, None)

    if "Your IP is likely being blocked" in std_err:
        logger.error("Your IP is likely being blocked %s", url)
        raise LikelyBlockedException(std_err, get_host_ip())

    # logger.error("Error downloading video %s reason: %s", url, std_err)
    raise DownloadException(std_err)


def download_video_audio(
    url: str,
    output_path: str = "/tmp/",
    yt_format: Optional[str] = None,
    fall_back_format: Optional[str] = f"b[ext={VEXTENSION}]",
    audio_quality=AQUALITY,
    audio_format=AEXTENSION,
    download_threads=DOWNLOAD_THREADS,
):

    if yt_format is None:
        yt_format = f"b[height<=360][ext={VEXTENSION}]"

    logger.info("Downloading video from YouTube %s format:%s", url, yt_format)
    args = [
        "yt-dlp",
        "-N",
        str(download_threads),  # number of threads
        "--write-info-json",
        "--write-subs",
        "--write-auto-sub",
        "--sub-langs",
        "de,en,es,fr,it,ur,pl,ru,hi,zh-Hans",
        "--convert-subs",
        "vtt",
        "--sub-format",
        "vtt",
        "-x",
        "--audio-quality",
        audio_quality,
        "--audio-format",
        audio_format,
        "-k",
        "--embed-chapters",
        "--no-progress",
        "-q",
        "--format",
        yt_format,
        "--output",
        os.path.join(output_path, "%(id)s.%(ext)s"),
        url,
    ]

    result = subprocess.run(args, capture_output=True, timeout=None)
    if result.returncode == 0:
        return os.listdir(output_path)

    std_err = result.stderr.decode("utf-8")

    if "Too Many Requests" in std_err:
        logger.error("Too Many Requests %s", url)
        raise TooManyRequestsException("Too Many Requests", get_host_ip())

    if "Requested format is not available" in std_err:
        logger.warning(
            "Requested format is not available, retrying with %s", fall_back_format
        )
        if fall_back_format is not None:
            return download_video_audio(url, output_path, fall_back_format, None)
        raise DownloadException("Requested format is not available")

    if "Your IP is likely being blocked" in std_err:
        logger.error("Your IP is likely being blocked %s", url)
        raise LikelyBlockedException(std_err, get_host_ip())

    raise DownloadException(std_err)


class SleepTimer:
    def __init__(
        self,
        default_timeout=DEFAULT_TIMEOUT,
        max_error_count=3,
        max_too_many_requests_count=2,
    ):
        self.default_timeout = default_timeout
        self.current_timeout = default_timeout
        self.error_counter = 0
        self.too_many_requests_counter = 0
        self.max_error_count = max_error_count
        self.max_too_many_requests_count = max_too_many_requests_count

    def random_sleep(self, time=None, jitter=0.5):
        if time is None:
            time = self.current_timeout
        logger.info("Sleeping for %d seconds (with jitter)", time)
        random_sleep(time, jitter=jitter)

    def sleep_error(self):
        if self.error_counter < self.max_error_count:
            self.error_counter += 1
            self.current_timeout = self.current_timeout * 2
        self.random_sleep()

    def sleep_normal(self):
        self.error_counter = 0
        self.too_many_requests_counter = 0
        self.current_timeout = self.default_timeout
        self.random_sleep()

    def sleep_too_many_requests(self):
        if self.too_many_requests_counter < self.max_too_many_requests_count:
            self.too_many_requests_counter += 1
        self.random_sleep(TOO_MANY_REQUESTS_TIMEOUT * self.too_many_requests_counter)


sleep_timer = SleepTimer()
