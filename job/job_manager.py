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
from job_backend import Job, JobBackend, JobSpec, JobState

EventCallback = Callable[[str, Job], Union[None, Awaitable[None]]]

# ---------------------------
# JobManager with retries, events, persistence
# ---------------------------
class JobManager:
    def __init__(
        self,
        backend: JobBackend,
        log_root: Union[str, os.PathLike] = "./job_logs",
        persist_state_path: Optional[Union[str, os.PathLike]] = None,
        max_concurrent: int = 32,
    ) -> None:
        self.backend = backend
        self.log_root = pathlib.Path(log_root)
        self.log_root.mkdir(parents=True, exist_ok=True)
        self.persist_state_path = pathlib.Path(persist_state_path) if persist_state_path else None
        self.max_concurrent = max_concurrent

        self._jobs: Dict[str, Job] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._sem = asyncio.Semaphore(max_concurrent)
        self._event_handlers: Dict[str, List[EventCallback]] = {
            "on_submit": [],
            "on_start": [],
            "on_finish": [],
            "on_retry": [],
            "on_cancel": [],
        }
        self._runner_task: Optional[asyncio.Task] = None

    # ---- public API ----
    def register(self, event: str, callback: EventCallback) -> None:
        if event not in self._event_handlers:
            raise ValueError(f"Unknown event {event}")
        self._event_handlers[event].append(callback)

    async def submit(self, spec: JobSpec) -> str:
        job_id = self._new_job_id(spec)
        job = Job(job_id=job_id, spec=spec)
        self._jobs[job_id] = job
        await self._emit("on_submit", job)
        await self._queue.put(job_id)
        self._ensure_runner()
        self._persist()
        return job_id

    def status(self, job_id: str) -> Job:
        return self._jobs[job_id]

    def list(self, state: Optional[JobState] = None) -> List[Job]:
        jobs = list(self._jobs.values())
        return [j for j in jobs if state is None or j.state == state]

    async def cancel(self, job_id: str) -> None:
        job = self._jobs[job_id]
        await self.backend.cancel(job)
        await self._emit("on_cancel", job)
        self._persist()

    # ---- internal runner ----
    def _ensure_runner(self) -> None:
        if self._runner_task is None or self._runner_task.done():
            self._runner_task = asyncio.create_task(self._run_loop())

    async def _run_loop(self) -> None:
        while True:
            job_id = await self._queue.get()
            job = self._jobs.get(job_id)
            if job is None:
                continue
            await self._sem.acquire()
            asyncio.create_task(self._execute_with_retries(job))

    async def _execute_with_retries(self, job: Job) -> None:
        try:
            start_wall = time.time()
            attempts_left = 1 + job.spec.retry.max_retries
            attempt_idx = 0
            while attempts_left > 0:
                attempt_idx += 1
                job.attempts = attempt_idx
                await self._emit("on_start", job)

                # Choose backend based on spec
                await self.backend.submit(job, self.log_root)

                # Wait until job finishes (backend updates state)
                while True:
                    st = await self.backend.poll(job)
                    if st in (JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED):
                        break
                    await asyncio.sleep(0.2)

                if job.state == JobState.SUCCEEDED:
                    await self._emit("on_finish", job)
                    self._persist()
                    return

                # Failed: decide whether to retry
                attempts_left -= 1
                if attempts_left <= 0:
                    await self._emit("on_finish", job)
                    self._persist()
                    return

                # Check total runtime limit
                if job.spec.runtime_limit_seconds is not None:
                    if time.time() - start_wall >= job.spec.runtime_limit_seconds:
                        job.error = (job.error or "") + f"\nRuntime limit exceeded"
                        await self._emit("on_finish", job)
                        self._persist()
                        return

                # Backoff
                backoff = (job.spec.retry.backoff_seconds ** attempt_idx)
                if job.spec.retry.jitter_seconds:
                    backoff += (os.urandom(1)[0] / 255.0) * job.spec.retry.jitter_seconds
                await self._emit("on_retry", job)
                await asyncio.sleep(backoff)
        finally:
            self._sem.release()

    # ---- helpers ----
    async def _emit(self, event: str, job: Job) -> None:
        handlers = self._event_handlers.get(event, [])
        for h in handlers:
            res = h(event, job)
            if asyncio.iscoroutine(res):
                await res

    def _new_job_id(self, spec: JobSpec) -> str:
        ts = int(time.time() * 1000)
        base = spec.name or (spec.cmd[0] if isinstance(spec.cmd, list) else (spec.cmd or "job"))
        base = pathlib.Path(str(base)).name.replace(" ", "_")
        return f"{base}-{ts}-{os.getpid()}-{len(self._jobs)+1}"

    def _persist(self) -> None:
        if not self.persist_state_path:
            return
        state = {jid: j.to_dict() for jid, j in self._jobs.items()}
        tmp = str(self.persist_state_path) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, self.persist_state_path)


# ---------------------------
# Convenience sync wrappers
# ---------------------------
class SyncJobClient:
    """Blocking facade for quick scripts/tests."""

    def __init__(self, manager: JobManager):
        self.mgr = manager

    def submit(self, spec: JobSpec) -> str:
        return asyncio.run(self.mgr.submit(spec))

    def wait(self, job_id: str, poll_interval: float = 0.2) -> Job:
        while True:
            j = self.mgr.status(job_id)
            if j.state in (JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED):
                return j
            time.sleep(poll_interval)

    def cancel(self, job_id: str) -> None:
        asyncio.run(self.mgr.cancel(job_id))
