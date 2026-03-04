## Web API Server

The dashcam analyzer includes a REST API for processing dashcam videos asynchronously and detecting cut-in events.

### Start the server

```bash
# Install dependencies
pip install -r requirements.txt

# Start the API server (default: http://0.0.0.0:8000)
python server.py

# Or with uvicorn for development
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

### Submit a video for processing

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@/path/to/dashcam.mp4"
# → {"job_id": "3f2a1b4c-...", "status": "pending"}
```

### Check job status / progress

```bash
curl http://localhost:8000/jobs/{job_id}
# → {"job_id": "...", "status": "processing", "progress": 0.42, "message": "Frame 210/500"}
```

### Fetch detection results

```bash
# Once status == "done"
curl http://localhost:8000/jobs/{job_id}/result
# → {"job_id": "...", "events": [{"event_id": "cutin_0001", "track_id": 5, "plate_text": "ABC123", ...}], ...}
```

### List and download clips

```bash
# List available clips
curl http://localhost:8000/jobs/{job_id}/clips
# → {"clips": ["/jobs/{job_id}/clips/cutin_0001_track5.mp4"]}

# Download a clip
curl -O http://localhost:8000/jobs/{job_id}/clips/cutin_0001_track5.mp4
```

### Output structure

```
outputs/
  {job_id}/
    job.json          # live status + progress
    result.json       # final detection results (events, plates, turn signals)
    clips/
      cutin_*.mp4     # one clip per cut-in event (±3 s buffer)
```

### CLI (unchanged)

The original CLI continues to work exactly as before:

```bash
python main.py --input video.mp4 --output ./output --show
```
