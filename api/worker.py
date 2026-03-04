from __future__ import annotations
import sys, queue, threading, traceback
from .job_manager import job_manager


def _check_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    print("[worker] WARNING: GPU not available, falling back to CPU", file=sys.stderr)
    return "cpu"


def _process_job(job_id: str, video_path: str) -> None:
    job_manager.update_job(job_id, status="processing", progress=0.0,
                           message="Starting pipeline")
    try:
        device = _check_device()
        from pipeline_web import WebPipeline

        def progress_cb(frac: float, msg: str = "") -> None:
            job_manager.update_job(job_id, progress=round(frac, 3), message=msg)

        pipeline = WebPipeline(job_id=job_id, device=device,
                               progress_callback=progress_cb)
        pipeline.run(video_path)
        job_manager.update_job(job_id, status="done", progress=1.0,
                               message="Complete",
                               result_path=f"outputs/{job_id}/result.json")
    except Exception:
        tb = traceback.format_exc()
        print(f"[worker] Job {job_id} failed:\n{tb}", file=sys.stderr)
        job_manager.update_job(job_id, status="failed", progress=0.0,
                               message=tb[:500])


_queue: queue.Queue[tuple[str, str]] = queue.Queue()


def _worker_loop() -> None:
    while True:
        job_id, video_path = _queue.get()
        try:
            _process_job(job_id, video_path)
        finally:
            _queue.task_done()


_thread = threading.Thread(target=_worker_loop, daemon=True, name="cutin-worker")
_thread.start()


def submit_job(job_id: str, video_path: str) -> None:
    _queue.put((job_id, video_path))
