"""
Storage helpers for uploaded input files and generated project paths.

The helpers in this module keep filesystem rules out of the API layer so later
changes to storage layout do not ripple through unrelated modules.
"""

from __future__ import annotations

from pathlib import Path
from shutil import rmtree
from uuid import uuid4

from .models import RiverTaskRecord, UploadedFileInfo, UploadedFileKind


PROJECT_ROOT = Path(__file__).resolve().parents[3]
UPLOAD_DIRECTORY_ROOT = PROJECT_ROOT / "data" / "input" / "uploads"
TASK_DIRECTORY_ROOT = PROJECT_ROOT / "data" / "output" / "tasks"

_UPLOAD_DIRECTORY_BY_KIND = {
    UploadedFileKind.INPUT: UPLOAD_DIRECTORY_ROOT / "inputs",
    UploadedFileKind.MASK: UPLOAD_DIRECTORY_ROOT / "masks",
}


def get_upload_directory(kind: UploadedFileKind) -> Path:
    """Return the stable upload directory for one asset category."""

    return _UPLOAD_DIRECTORY_BY_KIND[kind]


def ensure_upload_directory(kind: UploadedFileKind) -> None:
    """Create the upload directory used by the frontend asset manager."""

    get_upload_directory(kind).mkdir(parents=True, exist_ok=True)


def ensure_task_directory(task_id: str) -> Path:
    """Create and return the output directory for one task."""

    task_directory = TASK_DIRECTORY_ROOT / task_id
    task_directory.mkdir(parents=True, exist_ok=True)
    return task_directory


def task_record_path(task_id: str) -> Path:
    """Return the JSON record path for one persisted task."""

    return ensure_task_directory(task_id) / "task_record.json"


def sanitize_filename(filename: str) -> str:
    """
    Reduce a user-provided filename to a safe local basename.

    The rule is intentionally simple for the MVP: keep only the basename and
    replace spaces with underscores. A unique prefix prevents collisions.
    """

    basename = Path(filename).name.strip()
    basename = basename or "upload.bin"
    return basename.replace(" ", "_")


def resolve_uploaded_file_path(kind: UploadedFileKind, stored_path: str) -> Path:
    """Resolve and validate one stored relative upload path."""

    expected_root = get_upload_directory(kind).resolve()
    resolved_path = (PROJECT_ROOT / stored_path).resolve()
    try:
        resolved_path.relative_to(expected_root)
    except ValueError as exc:
        raise ValueError(f"Uploaded file path is outside the managed {kind.value} directory.") from exc

    return resolved_path


def save_uploaded_file(
    kind: UploadedFileKind,
    filename: str,
    content: bytes,
) -> UploadedFileInfo:
    """
    Persist an uploaded input file under the project input directory.

    Returns both the absolute path and the project-relative path string used by
    the task API contract.
    """

    ensure_upload_directory(kind)
    safe_name = sanitize_filename(filename)
    stored_name = f"{uuid4().hex}_{safe_name}"
    absolute_path = get_upload_directory(kind) / stored_name
    absolute_path.write_bytes(content)

    return UploadedFileInfo(
        kind=kind,
        filename=Path(filename).name,
        stored_path=absolute_path.relative_to(PROJECT_ROOT).as_posix(),
        size_bytes=len(content),
    )


def list_uploaded_files(kind: UploadedFileKind) -> list[UploadedFileInfo]:
    """
    Return uploaded files in newest-first order for frontend reuse.

    The response intentionally mirrors the upload response shape so the frontend
    can treat freshly uploaded files and historical files through one contract.
    """

    ensure_upload_directory(kind)
    upload_directory = get_upload_directory(kind)
    entries: list[UploadedFileInfo] = []

    for file_path in sorted(
        upload_directory.iterdir(),
        key=lambda candidate: candidate.stat().st_mtime,
        reverse=True,
    ):
        if not file_path.is_file():
            continue

        entries.append(
            UploadedFileInfo(
                kind=kind,
                filename=file_path.name,
                stored_path=file_path.relative_to(PROJECT_ROOT).as_posix(),
                size_bytes=file_path.stat().st_size,
            )
        )

    return entries


def rename_uploaded_file(
    kind: UploadedFileKind,
    stored_path: str,
    next_name: str,
) -> UploadedFileInfo:
    """Rename one uploaded file while keeping it in the same managed bucket."""

    source_path = resolve_uploaded_file_path(kind, stored_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Uploaded file does not exist: {stored_path}")

    sanitized_name = sanitize_filename(next_name)
    target_path = source_path.with_name(f"{uuid4().hex}_{sanitized_name}")
    source_path.rename(target_path)
    return UploadedFileInfo(
        kind=kind,
        filename=target_path.name,
        stored_path=target_path.relative_to(PROJECT_ROOT).as_posix(),
        size_bytes=target_path.stat().st_size,
    )


def delete_uploaded_file(kind: UploadedFileKind, stored_path: str) -> None:
    """Delete one uploaded file from the managed asset library."""

    target_path = resolve_uploaded_file_path(kind, stored_path)
    if not target_path.exists():
        raise FileNotFoundError(f"Uploaded file does not exist: {stored_path}")

    target_path.unlink()


def save_task_record(record: RiverTaskRecord) -> None:
    """Persist one task record under its task directory."""

    record_path = task_record_path(record.task_id)
    record_path.write_text(
        record.model_dump_json(indent=2),
        encoding="utf-8",
    )


def load_task_records() -> list[RiverTaskRecord]:
    """Load previously persisted task records from disk."""

    if not TASK_DIRECTORY_ROOT.exists():
        return []

    records: list[RiverTaskRecord] = []
    for task_directory in sorted(TASK_DIRECTORY_ROOT.iterdir(), key=lambda item: item.name):
        if not task_directory.is_dir():
            continue

        record_path = task_directory / "task_record.json"
        if not record_path.exists():
            continue

        records.append(RiverTaskRecord.model_validate_json(record_path.read_text(encoding="utf-8")))

    return records


def clear_task_directory(task_id: str) -> None:
    """Remove all task-local artifacts so the task can be recomputed cleanly."""

    task_directory = TASK_DIRECTORY_ROOT / task_id
    if task_directory.exists():
        rmtree(task_directory)


def delete_task_directory(task_id: str) -> None:
    """Delete one task directory and all persisted task artifacts."""

    clear_task_directory(task_id)
