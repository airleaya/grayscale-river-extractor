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

from fastapi import FastAPI, HTTPException, Query
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .logging_utils import configure_logging, get_logger
from .models import (
    ContinueTaskRequest,
    DeleteUploadedFileRequest,
    DraftTaskState,
    DraftTaskRequest,
    RenameTaskRequest,
    RenameUploadedFileRequest,
    RiverTaskRequest,
    RiverTaskSnapshot,
    UploadedFileInfo,
    UploadedFileKind,
)
from .storage import delete_uploaded_file, list_uploaded_files, rename_uploaded_file, save_uploaded_file
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


@app.post("/tasks/draft", response_model=RiverTaskSnapshot)
def create_task_draft(request: DraftTaskRequest) -> RiverTaskSnapshot:
    """Create one editable task draft without starting execution."""

    snapshot = task_runner.create_draft_task(request.name)
    logger.info("Task draft created via API: %s", snapshot.task_id)
    return snapshot


@app.get("/tasks", response_model=list[RiverTaskSnapshot])
def list_tasks() -> list[RiverTaskSnapshot]:
    """Return all known task snapshots so the frontend can rebuild its task center."""

    snapshots = task_runner.list_tasks()
    logger.info("Task list served with %s entries.", len(snapshots))
    return snapshots


@app.post("/files/upload", response_model=UploadedFileInfo)
async def upload_input_file(
    request: Request,
    kind: UploadedFileKind = Query(default=UploadedFileKind.INPUT),
) -> UploadedFileInfo:
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

    file_info = save_uploaded_file(kind, filename, content)

    logger.info(
        "Saved uploaded %s file '%s' to '%s' (%s bytes).",
        kind.value,
        filename,
        file_info.stored_path,
        file_info.size_bytes,
    )
    return file_info


@app.get("/files/uploaded", response_model=list[UploadedFileInfo])
def get_uploaded_files(
    kind: UploadedFileKind = Query(default=UploadedFileKind.INPUT),
) -> list[UploadedFileInfo]:
    """List previously uploaded files in one category for frontend reuse."""

    uploaded_files = list_uploaded_files(kind)
    logger.info("Uploaded %s file list served with %s entries.", kind.value, len(uploaded_files))
    return uploaded_files


@app.post("/files/rename", response_model=UploadedFileInfo)
def rename_file(request: RenameUploadedFileRequest) -> UploadedFileInfo:
    """Rename one uploaded input or mask asset."""

    try:
        file_info = rename_uploaded_file(request.kind, request.stored_path, request.name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.info("Uploaded %s file renamed to %s.", request.kind.value, file_info.stored_path)
    return file_info


@app.post("/files/delete", response_model=dict[str, bool])
def delete_file(request: DeleteUploadedFileRequest) -> dict[str, bool]:
    """Delete one uploaded input or mask asset."""

    try:
        delete_uploaded_file(request.kind, request.stored_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.info("Uploaded %s file deleted: %s", request.kind.value, request.stored_path)
    return {"deleted": True}


@app.get("/tasks/{task_id}", response_model=RiverTaskSnapshot)
def get_task(task_id: str) -> RiverTaskSnapshot:
    """Fetch the latest snapshot for one task."""

    snapshot = task_runner.get_task(task_id)
    if snapshot is None:
        logger.warning("Task lookup failed: %s", task_id)
        raise HTTPException(status_code=404, detail="Task not found.")

    logger.info("Task snapshot served: %s", task_id)
    return snapshot


@app.post("/tasks/{task_id}/start", response_model=RiverTaskSnapshot)
def start_task(task_id: str, request: RiverTaskRequest) -> RiverTaskSnapshot:
    """Start one existing task draft with the provided payload."""

    snapshot = task_runner.start_task(task_id, request)
    if snapshot is None:
        logger.warning("Task draft start failed because task was missing: %s", task_id)
        raise HTTPException(status_code=404, detail="Task not found.")

    logger.info("Task draft start request accepted: %s", task_id)
    return snapshot


@app.post("/tasks/{task_id}/draft-state", response_model=RiverTaskSnapshot)
def update_task_draft_state(task_id: str, request: DraftTaskState) -> RiverTaskSnapshot:
    """Persist editable draft configuration without starting execution."""

    snapshot = task_runner.update_draft_state(task_id, request)
    if snapshot is None:
        logger.warning("Task draft update failed because task was missing: %s", task_id)
        raise HTTPException(status_code=404, detail="Task not found.")

    logger.info("Task draft state updated: %s", task_id)
    return snapshot


@app.post("/tasks/{task_id}/rename", response_model=RiverTaskSnapshot)
def rename_task(task_id: str, request: RenameTaskRequest) -> RiverTaskSnapshot:
    """Rename one existing task."""

    try:
        snapshot = task_runner.rename_task(task_id, request.name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if snapshot is None:
        logger.warning("Task rename failed because task was missing: %s", task_id)
        raise HTTPException(status_code=404, detail="Task not found.")

    logger.info("Task rename request accepted: %s -> %s", task_id, request.name)
    return snapshot


@app.delete("/tasks/{task_id}", response_model=dict[str, bool])
def delete_task(task_id: str) -> dict[str, bool]:
    """Delete one non-running task and its artifacts."""

    try:
        snapshot = task_runner.delete_task(task_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    if snapshot is None:
        logger.warning("Task deletion failed because task was missing: %s", task_id)
        raise HTTPException(status_code=404, detail="Task not found.")

    logger.info("Task deleted via API: %s", task_id)
    return {"deleted": True}


@app.post("/tasks/{task_id}/cancel", response_model=RiverTaskSnapshot)
def cancel_task(task_id: str) -> RiverTaskSnapshot:
    """Request cancellation for a task and return the newest snapshot."""

    snapshot = task_runner.cancel_task(task_id)
    if snapshot is None:
        logger.warning("Task cancellation failed because task was missing: %s", task_id)
        raise HTTPException(status_code=404, detail="Task not found.")

    logger.info("Task cancel request accepted: %s", task_id)
    return snapshot


@app.post("/tasks/{task_id}/pause", response_model=RiverTaskSnapshot)
def pause_task(task_id: str) -> RiverTaskSnapshot:
    """Request pausing for a task and return the newest snapshot."""

    snapshot = task_runner.pause_task(task_id)
    if snapshot is None:
        logger.warning("Task pause failed because task was missing: %s", task_id)
        raise HTTPException(status_code=404, detail="Task not found.")

    logger.info("Task pause request accepted: %s", task_id)
    return snapshot


@app.post("/tasks/{task_id}/resume", response_model=RiverTaskSnapshot)
def resume_task(task_id: str) -> RiverTaskSnapshot:
    """Resume a paused task from the next stage-safe boundary."""

    snapshot = task_runner.resume_task(task_id)
    if snapshot is None:
        logger.warning("Task resume failed because task was missing: %s", task_id)
        raise HTTPException(status_code=404, detail="Task not found.")

    logger.info("Task resume request accepted: %s", task_id)
    return snapshot


@app.post("/tasks/{task_id}/continue", response_model=RiverTaskSnapshot)
def continue_task(task_id: str, request: ContinueTaskRequest) -> RiverTaskSnapshot:
    """Continue one persisted task to the requested stage boundary."""

    snapshot = task_runner.continue_task(
        task_id,
        end_stage=request.end_stage,
        inherit_intermediates=request.inherit_intermediates,
        inherit_stage_outputs=request.inherit_stage_outputs,
    )
    if snapshot is None:
        logger.warning("Task continue failed because task was missing: %s", task_id)
        raise HTTPException(status_code=404, detail="Task not found.")

    logger.info(
        "Task continue request accepted: %s end_stage=%s inherit=%s",
        task_id,
        request.end_stage.value if request.end_stage is not None else "default",
        request.inherit_intermediates,
    )
    return snapshot


@app.post("/tasks/{task_id}/rerun", response_model=RiverTaskSnapshot)
def rerun_task(task_id: str) -> RiverTaskSnapshot:
    """Reset one task and queue it from the beginning again."""

    snapshot = task_runner.rerun_task(task_id)
    if snapshot is None:
        logger.warning("Task rerun failed because task was missing: %s", task_id)
        raise HTTPException(status_code=404, detail="Task not found.")

    logger.info("Task rerun request accepted: %s", task_id)
    return snapshot
