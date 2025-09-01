import asyncio
import contextlib
import dataclasses
import enum
import json
import os
import pathlib
import shlex
import signal
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Union

# ---------------------------
# Job model & state machine
# ---------------------------
class JobState(enum.Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

@dataclass
class RetryPolicy:
    max_retries: int = 0
    backoff_seconds: float = 2.0  # exponential base backoff
    jitter_seconds: float = 0.5   # random jitter added to backoff


@dataclass
class JobSpec:
    """What to run.

    Exactly one of `cmd` or `callable` must be provided.
    """
    # Shell/exec style
    cmd: Optional[Union[str, List[str]]] = None
    # Python callable style (sync or async). Signature: (job_id: str, *args, **kwargs)
    callable: Optional[Callable[..., Any]] = None
    args: Tuple[Any, ...] = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)

    # Runtime controls
    env: Optional[Mapping[str, str]] = None
    cwd: Optional[Union[str, os.PathLike]] = None
    timeout_seconds: Optional[float] = None  # per-attempt timeout
    runtime_limit_seconds: Optional[float] = None  # total max wall time across retries
    retry: RetryPolicy = field(default_factory=RetryPolicy)

    # Metadata for your app
    name: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class Job:
    job_id: str
    spec: JobSpec
    state: JobState = JobState.PENDING
    attempts: int = 0
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    return_code: Optional[int] = None
    error: Optional[str] = None
    pid: Optional[int] = None
    stdout_path: Optional[str] = None
    stderr_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d["state"] = self.state.value
        return d

# ---------------------------
# Backend interface
# ---------------------------
class JobBackend:
    async def submit(self, job: Job, log_dir: pathlib.Path) -> None:
        raise NotImplementedError

    async def poll(self, job: Job) -> JobState:
        raise NotImplementedError

    async def cancel(self, job: Job) -> None:
        raise NotImplementedError


class LocalSubprocessBackend(JobBackend):
    """Runs jobs as local subprocesses using asyncio.create_subprocess_exec.
    Supports `cmd` list or string. Logs to files under log_dir/job_id.
    """

    def __init__(self) -> None:
        self._procs: Dict[str, asyncio.subprocess.Process] = {}

    async def submit(self, job: Job, log_dir: pathlib.Path) -> None:
        assert job.spec.cmd is not None, "LocalSubprocessBackend requires spec.cmd"
        cmd = job.spec.cmd
        if isinstance(cmd, str):
            # Respect shell-like string; split conservatively
            args = shlex.split(cmd)
        else:
            args = cmd

        job_dir = log_dir / job.job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        stdout_file = open(job_dir / "stdout.log", "wb")
        stderr_file = open(job_dir / "stderr.log", "wb")
        job.stdout_path = str(job_dir / "stdout.log")
        job.stderr_path = str(job_dir / "stderr.log")

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=stdout_file,
            stderr=stderr_file,
            cwd=str(job.spec.cwd) if job.spec.cwd else None,
            env=dict(os.environ, **(job.spec.env or {})),
        )
        self._procs[job.job_id] = proc
        job.pid = proc.pid
        job.state = JobState.RUNNING
        job.started_at = time.time()

        # Detach a waiter task to set final state when it exits
        async def _wait_and_finalize():
            try:
                rc = await asyncio.wait_for(proc.wait(), timeout=job.spec.timeout_seconds)
                job.return_code = rc
                job.ended_at = time.time()
                job.state = JobState.SUCCEEDED if rc == 0 else JobState.FAILED
            except asyncio.TimeoutError:
                job.error = f"Attempt timed out after {job.spec.timeout_seconds}s"
                job.state = JobState.FAILED
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                    await proc.wait()
                job.ended_at = time.time()
            finally:
                # Close file handles
                for f in (stdout_file, stderr_file):
                    with contextlib.suppress(Exception):
                        f.close()
                self._procs.pop(job.job_id, None)

        # It takes a coroutine object and schedules it to run in the event loop concurrently.
        # TODO: adding some handle here
        asyncio.create_task(_wait_and_finalize())

    async def poll(self, job: Job) -> JobState:
        return job.state

    async def cancel(self, job: Job) -> None:
        proc = self._procs.get(job.job_id)
        if proc and proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.send_signal(signal.SIGTERM)
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                    await proc.wait()
        job.state = JobState.CANCELLED
        job.ended_at = time.time()


class PythonCallableBackend(JobBackend):
    """Runs Python callables (sync or async) with per-attempt timeout in a thread.
    Captures text output to logs.
    """

    def __init__(self):
        self._tasks: Dict[str, asyncio.Task] = {}

    async def submit(self, job: Job, log_dir: pathlib.Path) -> None:
        assert job.spec.callable is not None
        job_dir = log_dir / job.job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        job.stdout_path = str(job_dir / "stdout.log")
        job.stderr_path = str(job_dir / "stderr.log")

        async def _runner():
            job.state = JobState.RUNNING
            job.started_at = time.time()
            loop = asyncio.get_running_loop()

            def _call_sync():
                return job.spec.callable(job.job_id, *job.spec.args, **job.spec.kwargs)

            try:
                if asyncio.iscoroutinefunction(job.spec.callable):
                    coro = job.spec.callable(job.job_id, *job.spec.args, **job.spec.kwargs)
                    result = await asyncio.wait_for(coro, timeout=job.spec.timeout_seconds)
                else:
                    result = await asyncio.wait_for(
                        loop.run_in_executor(None, _call_sync),
                        timeout=job.spec.timeout_seconds,
                    )
                # Persist result to stdout
                with open(job.stdout_path, "a", encoding="utf-8") as f:
                    f.write(str(result) + "\n")
                job.return_code = 0
                job.state = JobState.SUCCEEDED
            except asyncio.TimeoutError:
                job.error = f"Attempt timed out after {job.spec.timeout_seconds}s"
                job.state = JobState.FAILED
            except Exception as e:
                job.error = f"{type(e).__name__}: {e}\n" + traceback.format_exc()
                with open(job.stderr_path, "a", encoding="utf-8") as f:
                    f.write(job.error + "\n")
                job.return_code = 1
                job.state = JobState.FAILED
            finally:
                job.ended_at = time.time()

        t = asyncio.create_task(_runner())
        self._tasks[job.job_id] = t

    async def poll(self, job: Job) -> JobState:
        return job.state

    async def cancel(self, job: Job) -> None:
        task = self._tasks.get(job.job_id)
        if task and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        job.state = JobState.CANCELLED
        job.ended_at = time.time()
