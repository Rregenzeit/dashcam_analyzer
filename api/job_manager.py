from __future__ import annotations
import json, os, uuid, time
from pathlib import Path
from .schemas import JobInfo, JobResult, JobStatus

try:
    from config import OUTPUTS_DIR
except ImportError:
    OUTPUTS_DIR = "outputs"


class JobManager:
    def __init__(self, outputs_dir: str = OUTPUTS_DIR):
        self.outputs_dir = Path(outputs_dir)

    def _job_dir(self, job_id: str) -> Path:
        return self.outputs_dir / job_id

    def _job_file(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "job.json"

    def create_job(self, video_path: str) -> str:
        job_id = str(uuid.uuid4())
        d = self._job_dir(job_id)
        d.mkdir(parents=True, exist_ok=True)
        (d / "clips").mkdir(exist_ok=True)
        info = JobInfo(job_id=job_id, status="pending", created_at=time.time())
        self._job_file(job_id).write_text(info.model_dump_json(), encoding="utf-8")
        return job_id

    def get_job(self, job_id: str) -> JobInfo | None:
        p = self._job_file(job_id)
        if not p.exists():
            return None
        return JobInfo.model_validate_json(p.read_text(encoding="utf-8"))

    def update_job(self, job_id: str, **kwargs) -> None:
        info = self.get_job(job_id)
        if info is None:
            return
        updated = info.model_copy(update=kwargs)
        self._job_file(job_id).write_text(updated.model_dump_json(), encoding="utf-8")

    def get_result(self, job_id: str) -> JobResult | None:
        p = self._job_dir(job_id) / "result.json"
        if not p.exists():
            return None
        return JobResult.model_validate_json(p.read_text(encoding="utf-8"))

    def list_clips(self, job_id: str) -> list[str]:
        clips_dir = self._job_dir(job_id) / "clips"
        if not clips_dir.exists():
            return []
        return sorted(p.name for p in clips_dir.glob("*.mp4"))


job_manager = JobManager()
