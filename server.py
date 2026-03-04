"""
Dashcam Cut-In Analyzer — Web Server entry point.
Run: python server.py
Or:  uvicorn server:app --host 0.0.0.0 --port 8000
"""
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path
from api.routes import router

app = FastAPI(
    title="Dashcam Cut-In Analyzer API",
    description="Detect cut-in vehicles from dashcam footage via REST API",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
Path("static").mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = Path("static/index.html")
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>UI not found</h1>")

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
