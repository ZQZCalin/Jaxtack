"""
jobrunner.py — A general-purpose job submit/monitor framework in pure Python.

Features
- Submit jobs (shell commands or Python callables)
- Track states: PENDING → RUNNING → {SUCCEEDED, FAILED, CANCELLED}
- Timeouts, retries with backoff, max runtime
- Persistent (optional) on-disk logs per job (stdout/stderr)
- Query status, list jobs, cancel jobs
- Callbacks and simple event bus
- Async-first (asyncio) with small sync wrappers
- Pluggable backends; LocalSubprocessBackend included

No external dependencies beyond the standard library.
"""
from job_backend import *
from job_manager import *

# ---------------------------
# Example usage
# ---------------------------
async def _demo():
    # Example 1: run a shell command
    backend = LocalSubprocessBackend()
    mgr = JobManager(backend, persist_state_path="./jobs_state.json", max_concurrent=4)

    def printer(event: str, job: Job):
        print(f"[{event}] {job.job_id} state={job.state.value} attempts={job.attempts}")

    for ev in ("on_submit", "on_start", "on_retry", "on_finish", "on_cancel"):
        mgr.register(ev, printer)

    spec = JobSpec(
        cmd=[sys.executable, "-c", "import time; print('hello'); time.sleep(1)\n"],
        timeout_seconds=5,
        retry=RetryPolicy(max_retries=1, backoff_seconds=1.5, jitter_seconds=0.2),
        name="hello_py",
        tags={"team": "ml"},
    )
    jid = await mgr.submit(spec)

    # # Example 2: run a Python callable
    async def my_task(job_id: str, x: int) -> str:
        await asyncio.sleep(0.5)
        return f"job {job_id} says {x*x}"

    py_backend = PythonCallableBackend()
    py_mgr = JobManager(py_backend)
    py_mgr.register("on_finish", printer)

    jid2 = await py_mgr.submit(JobSpec(callable=my_task, args=(7,), timeout_seconds=3, name="square"))

    # Wait for both jobs to complete
    while any(j.state not in (JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED)
          for j in [mgr.status(jid), py_mgr.status(jid2)]):
        await asyncio.sleep(0.2)

if __name__ == "__main__":
    # To run the demo, execute: python jobrunner.py
    asyncio.run(_demo())
