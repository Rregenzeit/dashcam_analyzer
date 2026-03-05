from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field
import time

JobStatus = Literal["pending", "processing", "done", "failed"]

class CutInEvent(BaseModel):
    event_id: str
    frame_start: int
    frame_end: int
    track_id: int
    plate_text: Optional[str] = None
    turn_signal: Literal["on", "off", "unknown"] = "unknown"
    clip_filename: str

class JobResult(BaseModel):
    job_id: str
    status: JobStatus
    recording_date: Optional[str] = None
    events: list[CutInEvent] = Field(default_factory=list)
    created_at: float
    finished_at: Optional[float] = None

class JobInfo(BaseModel):
    job_id: str
    status: JobStatus
    progress: float = 0.0
    message: str = ""
    created_at: float = Field(default_factory=time.time)
    result_path: Optional[str] = None
