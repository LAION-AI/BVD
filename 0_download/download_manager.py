import os
import re
import shutil
from math import ceil
from celery.result import AsyncResult
from celery.exceptions import SoftTimeLimitExceeded
import pandas as pd
from typing import List, Tuple, Optional
from tasks import download_task
from time import sleep, time
from utils import (
    LikelyBlockedException,
    TooManyRequestsException,
    NodeNotReadyException,
    stop_celery,
    start_celery,
)
from config import (
    BLOCKED_WORKER_FILE,
    MINIMUM_BLOCKED_TIME,
    KEYFRAMES_ONLY,
    YT_FORMAT,
    YT_FALLBACK_FORMAT,
    AEXTENSION,
    AQUALITY,
    MANAGEMENT_PROCESS_SHARD_TIMEOUT,
)
from datetime import datetime
import logging
from rich.progress import (
    Progress,
    MofNCompleteColumn,
    TimeElapsedColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)
from rich.console import Console
from multiprocessing.pool import ThreadPool


class CeleryDownloadManager:
    def __init__(
        self,
        parquet_file: str,
        files_per_folder: int,
        download_path: str,
        blocked_workers_file: str = BLOCKED_WORKER_FILE,
        yt_format: str = YT_FORMAT,
        yt_fallback: Optional[str] = YT_FALLBACK_FORMAT,
        keyframes_only: bool = KEYFRAMES_ONLY,
        allowed_domains: Optional[List[str]] = None,
    ):
        self.console = Console()
        self.parquet = pd.read_parquet(parquet_file)
        if allowed_domains is not None:
            self.console.log(f"Filtering allowed domains: {allowed_domains}")
            self.parquet = self.parquet[self.parquet["domain"].isin(allowed_domains)]
        self.folder_name = os.path.basename(parquet_file)
        self.files_per_folder = files_per_folder
        self.download_path = os.path.join(download_path, self.folder_name)
        self.blocked_workers_file = blocked_workers_file
        if os.path.exists(self.blocked_workers_file):
            self.blocked_workers = pd.read_csv(self.blocked_workers_file)
            self.blocked_workers["timestamp"] = pd.to_datetime(
                self.blocked_workers["timestamp"]
            )
        else:
            self.blocked_workers = pd.DataFrame(columns=["host", "timestamp", "type"])
            self.blocked_workers.to_csv(self.blocked_workers_file, index=False)
        self.yt_format = yt_format
        self.yt_fallback = yt_fallback
        self.keyframes_only = keyframes_only

    def log(self, msg: str, level=logging.INFO):
        # if self.progress_log:
        # self.progress_log(msg)
        self.console.log(msg)
        # Regular expression pattern to match style instructions
        pattern = r"\[\/*[^]]+\]"
        # Using re.sub() to replace the matched patterns with an empty string
        msg = re.sub(pattern, "", msg)
        logging.log(level, msg)

    def _file_id(self, i: int) -> str:
        return f"{i:06d}"

    def _folder_id(self, i: int) -> str:
        return f"{i:05d}"

    def _go_through_block_list(self, block=True):

        for i, row in self.blocked_workers.iterrows():
            time_diff = datetime.now() - row["timestamp"]
            if (
                time_diff.seconds > MINIMUM_BLOCKED_TIME
                and row["type"] == "TooManyRequestsException"
            ):
                self.log(f"[blue]Unblocking worker {row['host']}")
                self.blocked_workers.drop(i, inplace=True)
                start_celery(row["host"], check=False)
        self.blocked_workers.to_csv(self.blocked_workers_file, index=False)
        if block:
            blocked = self.blocked_workers["host"].tolist()
            self.log(
                f"[red]Stopping {len(blocked)} blocked workers: {', '.join(blocked)}"
            )
            return [stop_celery(host) for host in blocked]
        return []

    def _add_to_block_list(self, host: str, exception_type: str):
        if host in self.blocked_workers["host"].values:
            return
        self.log(f"Stopping worker {host} due to [yellow]{exception_type}")
        stop_celery(host)
        self.blocked_workers = pd.concat(
            [
                self.blocked_workers,
                pd.DataFrame(
                    [
                        {
                            "host": host,
                            "timestamp": datetime.now(),
                            "type": exception_type,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        self.blocked_workers.to_csv(self.blocked_workers_file, index=False)

    def __handle_result(self, result: AsyncResult, i: int, shard_parquet: pd.DataFrame):
        result_row = {
            "id": self._file_id(i),
            "uid": shard_parquet["uid"].iloc[i],
            "page_url": shard_parquet["page_url"].iloc[i],
            "url": shard_parquet["url"].iloc[i],
            "status": "success",
            "error_message": None,
        }
        try:
            result.get()
        except (LikelyBlockedException, TooManyRequestsException) as e:
            result_row["status"] = "blocked_error"
            result_row["error_message"] = e.args[0]
            self._add_to_block_list(e.args[1], type(e).__name__)
        except SoftTimeLimitExceeded as e:
            result_row["status"] = "timeout_error"
            result_row["error_message"] = str(e)
            # c_print(f"[red]SoftTimeLimitExceeded: {i}")
        except NodeNotReadyException as e:
            result_row["status"] = "not_ready_error"
            result_row["error_message"] = e.args[0]
            self._add_to_block_list(e.args[1], type(e).__name__)
        except Exception as e:
            result_row["status"] = "error"
            result_row["error_message"] = str(e)
        return result_row

    def download_parallel(self, parallel: int):
        """Download the shards in parallel
        Args:
            parallel (int): Number of shards to download in parallel
        """
        total_urls = len(self.parquet)
        total_shards = ceil(total_urls / self.files_per_folder)
        self._go_through_block_list(True)

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            # TimeRemainingColumn(),
            # transient=True,
            console=self.console,
        ) as progress:
            shards_progress = progress.add_task("Shards", total=total_shards)
            shards = list(range(total_shards))
            download_helper = lambda shard_id: self.download_shard(shard_id, progress)
            with ThreadPool(parallel) as pool:
                for shard in pool.imap_unordered(download_helper, shards):
                    self._go_through_block_list(False)
                    progress.update(shards_progress, advance=1)

            self.log(f"[green bold]Download finished: {self.folder_name}.")

    def download_shard(self, shard, progress):
        data = self._initialize_download(shard)
        if data is not None:
            self._finalise_download(shard, *data, progress)
        return shard

    def get_shard(self, shard: int):
        total_urls = len(self.parquet)
        start_index = shard * self.files_per_folder
        end_index = start_index + self.files_per_folder
        if end_index < total_urls:
            return self.parquet.iloc[start_index:end_index]
        return self.parquet.iloc[start_index:]

    def _initialize_download(self, shard: int):
        shard_folder = os.path.join(self.download_path, self._folder_id(shard))
        downloaded_marker = f"{shard_folder}.parquet"

        if os.path.exists(downloaded_marker):
            self.log(f"[blue]Shard {shard} already downloaded. Skipping...")
            return None

        # Remove old version of the folder
        if os.path.exists(shard_folder):
            self.log(f"[red]Removing old version of {shard_folder}")
            # with self.status(f"[red]Removing old version of {shard_folder}") as status:
            shutil.rmtree(shard_folder)
        os.makedirs(shard_folder, exist_ok=True)

        shard_parquet = self.get_shard(shard)
        shard_urls = shard_parquet["url"].tolist()
        results: List[Tuple[int, AsyncResult]] = [
            (
                i,
                download_task.delay(
                    url,
                    shard_folder,
                    self._file_id(i),
                    self.yt_format,
                    self.yt_fallback,
                    self.keyframes_only,
                    AEXTENSION,
                    AQUALITY,
                ),
            )
            for i, url in enumerate(shard_urls)
        ]
        return results, shard_parquet, shard_folder

    def _finalise_download(
        self,
        shard: int,
        results: List[Tuple[int, AsyncResult]],
        shard_parquet: pd.DataFrame,
        result_path: str,
        progress: Progress,
    ):
        start_time = time()
        videos_progress = progress.add_task(f"#{shard:05d}", total=len(results))
        result_df = pd.DataFrame(
            columns=["id", "uid", "page_url", "url", "status", "error_message"]
        )
        while len(results) > 0:
            for result_packed in results:
                i, result = result_packed
                if result.ready():
                    result_row = self.__handle_result(result, i, shard_parquet)
                    result_df = pd.concat(
                        [result_df, pd.DataFrame([result_row])],
                        ignore_index=True,
                    )
                    results.remove(result_packed)
                    progress.update(videos_progress, advance=1)
            if time() - start_time > MANAGEMENT_PROCESS_SHARD_TIMEOUT:
                break
            sleep(5)
        if len(results) > 0:
            timeout = time() - start_time
            self.log(
                f"[red]Shard {shard} timed out after {timeout:.0f}s. "
                + f"{len(results)} item(s) left."
            )

            result_df = pd.concat(
                [
                    result_df,
                    pd.DataFrame(
                        [
                            {
                                "id": self._file_id(i),
                                "uid": shard_parquet["uid"].iloc[i],
                                "page_url": shard_parquet["page_url"].iloc[i],
                                "url": shard_parquet["url"].iloc[i],
                                "status": "shard_timeout_error",
                                "error_message": f"Shard timed out after {timeout:.0f}s",
                            }
                            for i, _ in results
                        ]
                    ),
                ],
                ignore_index=True,
            )
        progress.remove_task(videos_progress)

        # result_df.set_index("id", inplace=True)
        result_df.sort_values("id").to_parquet(f"{result_path}.parquet", index=False)

        self.log(
            f"[bold]Shard {self._folder_id(shard)} downloaded successfully "
            + f"in {(time() - start_time)/60:.2f} min.\n"
            + f"[black]{self.folder_name}"
        )
        self.log(
            f"[green]Success: {len(result_df[result_df['status'] == 'success'])}\t[red]Failed: {len(result_df[result_df['status'] != 'success'])}\t"
            + f"[yellow]Ratio:  {100 * len(result_df[result_df['status'] == 'success']) / len(result_df):.1f}%\n"
            + f"\t\t[red not bold]:clock1: {len(result_df[result_df['status'] == 'timeout_error'])}, "
            + f":stop_sign: {len(result_df[result_df['status'] == 'blocked_error'])}"
        )
        return result_df
