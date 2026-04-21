"""
FastAPI entry point for the river-extraction MVP backend.

The API is deliberately small: health, create task, inspect task, and cancel
task. This keeps the first runnable version focused on long-running workflow
management and progress feedback.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from urllib.parse import unquote

from fastapi import FastAPI, HTTPException
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .logging_utils import configure_logging, get_logger
from .models import RiverTaskRequest, RiverTaskSnapshot
from .storage import list_uploaded_input_files, save_uploaded_input_file
from .task_runner import InMemoryTaskRunner

configure_logging()
logger = get_logger("river.api")

app = FastAPI(
    title="River Extraction MVP",
    version="0.1.0",
    summary="Task-oriented backend for terrain-to-river-channel processing.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

app.mount("/data", StaticFiles(directory=Path(__file__).resolve().parents[3] / "data"), name="data")

task_runner = InMemoryTaskRunner()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Log every API request with timing information.

    This makes backend output useful even before we have a full observability
    stack and helps diagnose stalls during long-running interactive sessions.
    """

    start_time = perf_counter()
    logger.info("Request started: %s %s", request.method, request.url.path)

    response = await call_next(request)

    duration_ms = (perf_counter() - start_time) * 1000
    logger.info(
        "Request finished: %s %s -> %s in %.2fms",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.get("/health")
def get_health() -> dict[str, str]:
    """Simple health probe used during local development."""

    logger.info("Health endpoint called.")
    return {"status": "ok"}


@app.post("/tasks", response_model=dict[str, str])
def create_task(request: RiverTaskRequest) -> dict[str, str]:
    """Create a new background task and return its identifier."""

    task_id = task_runner.create_task(request)
    logger.info("Task created via API: %s", task_id)
    return {"task_id": task_id}


@app.post("/files/upload", response_model=dict[str, str | int])
async def upload_input_file(request: Request) -> dict[str, str | int]:
    """
    Save one uploaded input file into the project workspace.

    The frontend sends raw file bytes and passes the original filename in the
    `X-Filename` header. This avoids introducing multipart parsing while the MVP
    is still focused on core architecture.
    """

    filename = unquote(request.headers.get("X-Filename", "").strip())
    if not filename:
        raise HTTPException(status_code=400, detail="Missing X-Filename header.")

    content = await request.body()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file content is empty.")

    absolute_path, relative_path = save_uploaded_input_file(filename, content)
    size_bytes = len(content)

    logger.info(
        "Saved uploaded input file '%s' to '%s' (%s bytes).",
        filename,
        absolute_path,
        size_bytes,
    )
    return {
        "filename": Path(filename).name,
        "stored_path": relative_path,
        "size_bytes": size_bytes,
    }


@app.get("/files/uploaded", response_model=list[dict[str, str | int]])
def get_uploaded_files() -> list[dict[str, str | int]]:
    """List previously uploaded files so the frontend can reuse them."""

    uploaded_files = list_uploaded_input_files()
    logger.info("Uploaded file list served with %s entries.", len(uploaded_files))
    return uploaded_files


@app.get("/tasks/{task_id}", response_model=RiverTaskSnapshot)
def get_task(task_id: str) -> RiverTaskSnapshot:
    """Fetch the latest snapshot for one task."""

    snapshot = task_runner.get_task(task_id)
    if snapshot is None:
        logger.warning("Task lookup failed: %s", task_id)
        raise HTTPException(status_code=404, detail="Task not found.")

    logger.info("Task snapshot served: %s", task_id)
    return snapshot


@app.post("/tasks/{task_id}/cancel", response_model=RiverTaskSnapshot)
def cancel_task(task_id: str) -> RiverTaskSnapshot:
    """Request cancellation for a task and return the newest snapshot."""

    snapshot = task_runner.cancel_task(task_id)
    if snapshot is None:
        logger.warning("Task cancellation failed because task was missing: %s", task_id)
        raise HTTPException(status_code=404, detail="Task not found.")

    logger.info("Task cancel request accepted: %s", task_id)
    return snapshot
