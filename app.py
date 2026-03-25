"""
Web frontend for the LLAMMA Video pipeline.

Usage:
    uv run python app.py
    # Open http://localhost:8000
"""

import asyncio
import queue
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from visualize_loan import PipelineConfig, run_pipeline

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI()
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# ---------------------------------------------------------------------------
# Job management
# ---------------------------------------------------------------------------

class JobState:
    def __init__(self, config: "RunRequest | None" = None):
        self.queue: queue.Queue[tuple[str, str]] = queue.Queue()  # (event, data)
        self.video_path: Path | None = None
        self.running = True
        self.error: str | None = None
        self.config: "RunRequest | None" = config

jobs: dict[str, JobState] = {}
jobs_lock = threading.Lock()
active_job_id: str | None = None


class RunRequest(BaseModel):
    controller_address: str
    user_address: str
    start_block: int | str | None = None
    end_block: int | str | None = None
    duration: int | None = None
    block_step: int = 5
    auto_fetch_events: bool = True
    events: list[dict] = []
    etherscan_api_key: str | None = None
    rpc_url: str | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = BASE_DIR / "static" / "index.html"
    return HTMLResponse(
        html_path.read_text(),
        headers={"Cache-Control": "no-store"},
    )


@app.post("/api/run")
async def start_run(req: RunRequest):
    global active_job_id

    with jobs_lock:
        if active_job_id and active_job_id in jobs and jobs[active_job_id].running:
            raise HTTPException(409, "A job is already running")

        job_id = uuid.uuid4().hex[:12]
        state = JobState(config=req)
        jobs[job_id] = state
        active_job_id = job_id

    config = PipelineConfig(
        controller_address=req.controller_address,
        user_address=req.user_address,
        start_block=req.start_block,
        end_block=req.end_block,
        duration=req.duration,
        block_step=req.block_step,
        auto_fetch_events=req.auto_fetch_events,
        events=req.events,
        etherscan_api_key=req.etherscan_api_key,
        rpc_url=req.rpc_url,
        output_base=BASE_DIR / "output",
    )

    def _run():
        def log_fn(msg):
            state.queue.put(("log", str(msg)))

        try:
            video_path = run_pipeline(config, log_fn=log_fn)
            state.video_path = video_path
            fname = _build_video_filename(state)
            state.queue.put(("done", fname))
        except Exception as e:
            state.error = str(e)
            state.queue.put(("error", str(e)))
        finally:
            state.running = False

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return {"job_id": job_id}


@app.get("/api/stream/{job_id}")
async def stream(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    state = jobs[job_id]

    async def event_generator():
        while True:
            sent = False
            # Drain all available messages in one go
            while True:
                try:
                    event, data = state.queue.get_nowait()
                    # SSE format: newlines in data need separate "data:" lines
                    data_lines = "\n".join(f"data: {line}" for line in data.split("\n"))
                    yield f"event: {event}\n{data_lines}\n\n"
                    sent = True
                    if event in ("done", "error"):
                        return
                except queue.Empty:
                    break
            if not sent:
                if not state.running:
                    return
                yield ": keepalive\n\n"
            await asyncio.sleep(0.15)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _build_video_filename(state: JobState) -> str:
    cfg = state.config
    if not cfg:
        return "video.mp4"
    ctrl = cfg.controller_address[:10]
    user = cfg.user_address[:10]
    parts = [ctrl, user]
    if cfg.start_block is not None:
        parts.append(f"b{cfg.start_block}")
    if cfg.end_block is not None:
        parts.append(f"to{cfg.end_block}")
    elif cfg.duration is not None:
        parts.append(f"d{cfg.duration}")
    return f"llamma_{'_'.join(parts)}.mp4"


@app.get("/api/video/{job_id}")
async def get_video(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    state = jobs[job_id]
    if not state.video_path or not state.video_path.exists():
        raise HTTPException(404, "Video not ready")
    filename = _build_video_filename(state)
    return FileResponse(state.video_path, media_type="video/mp4", filename=filename)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
